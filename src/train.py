"""Train an AirResNet on CIFAR-10 with all data and augmentation kept on the MPS device.

Recipe follows airbench96 (Jordan, 2024): frozen whitening stem, Nesterov SGD with decoupled
hyperparameters, label smoothing, flip + 4px translate + cutout, Lookahead-style EMA and
flip/translate test-time augmentation. Optional knowledge distillation from teacher checkpoints
that were themselves trained from scratch by this script.

Only `x_train`/`y_train` are ever used for training or model selection. The test images are
only passed through the finished model to write predictions.
"""
import argparse
import json
import os
import time
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models import AirResNet, count_params, init_whitening

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MEAN = torch.tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1)
STD = torch.tensor((0.2470, 0.2435, 0.2616)).view(1, 3, 1, 1)
PARAM_LIMIT = 5_000_000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--widths", type=int, nargs=3, default=[128, 256, 352])
    p.add_argument("--depth", type=int, nargs=3, default=[1, 1, 1])
    p.add_argument("--epochs", type=float, default=40)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=9.0)  # per 1024 examples, decoupled from momentum
    p.add_argument("--momentum", type=float, default=0.85)
    p.add_argument("--wd", type=float, default=0.012)  # per 1024 examples, decoupled from lr
    p.add_argument("--bias-scaler", type=float, default=64.0)
    p.add_argument("--label-smoothing", type=float, default=0.2)
    p.add_argument("--warmup", type=float, default=0.1)  # fraction of steps
    p.add_argument("--translate", type=int, default=4)
    p.add_argument("--cutout", type=int, default=12)
    p.add_argument("--whiten-bias-epochs", type=int, default=3)
    p.add_argument("--val", type=int, default=5000, help="holdout size from train; 0 = train on all 50k")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--teachers", nargs="*", default=[], help="run dirs of teacher models for distillation")
    p.add_argument("--kd-alpha", type=float, default=0.5, help="weight of the distillation loss")
    p.add_argument("--kd-temp", type=float, default=4.0)
    p.add_argument("--allow-big", action="store_true", help="allow >5M params (teachers only)")
    p.add_argument("--eval-every", type=int, default=5)
    return p.parse_args()


# ---------------------------------------------------------------- data / augmentation


def load_data(device):
    d = np.load(os.path.join(ROOT, "data", "cifar10_cache.npz"))
    to_t = lambda a: torch.from_numpy(a).to(device).permute(0, 3, 1, 2).float().div(255)
    norm = lambda x: (x - MEAN.to(device)) / STD.to(device)
    return (norm(to_t(d["x_train"])), torch.from_numpy(d["y_train"]).to(device),
            norm(to_t(d["x_test"])), d["test_ids"])


def split(n, val, seed):
    """Stratified holdout split over the training set, fixed by seed."""
    if val == 0:
        return np.arange(n), np.array([], dtype=np.int64)
    y = np.load(os.path.join(ROOT, "data", "cifar10_cache.npz"))["y_train"]
    rng = np.random.default_rng(1000 + seed)
    val_idx = np.concatenate([rng.permutation(np.where(y == c)[0])[: val // 10] for c in range(10)])
    train_idx = np.setdiff1d(np.arange(n), val_idx)
    return train_idx, np.sort(val_idx)


def batch_crop(padded, size):
    r = (padded.size(-1) - size) // 2
    n = len(padded)
    shifts = torch.randint(-r, r + 1, size=(n, 2), device=padded.device)
    tmp = torch.empty((n, 3, size, size + 2 * r), device=padded.device, dtype=padded.dtype)
    for s in range(-r, r + 1):
        m = shifts[:, 0] == s
        tmp[m] = padded[m, :, r + s : r + s + size, :]
    out = torch.empty((n, 3, size, size), device=padded.device, dtype=padded.dtype)
    for s in range(-r, r + 1):
        m = shifts[:, 1] == s
        out[m] = tmp[m, :, :, r + s : r + s + size]
    return out


def batch_cutout(x, size):
    n, _, h, w = x.shape
    cy = torch.randint(0, h - size + 1, (n,), device=x.device).view(-1, 1, 1, 1)
    cx = torch.randint(0, w - size + 1, (n,), device=x.device).view(-1, 1, 1, 1)
    ys = torch.arange(h, device=x.device).view(1, 1, h, 1) - cy
    xs = torch.arange(w, device=x.device).view(1, 1, 1, w) - cx
    mask = (ys >= 0) & (ys < size) & (xs >= 0) & (xs < size)
    return x.masked_fill(mask, 0)


class Loader:
    """Augmented epochs built on-device. Flip is random per image on epoch 0, then every image's
    flip alternates each epoch (airbench's 'alternating flip')."""

    def __init__(self, images, labels, batch_size, translate, cutout):
        n = len(images)
        flip = (torch.rand(n, device=images.device) < 0.5).view(-1, 1, 1, 1)
        base = torch.where(flip, images.flip(-1), images)
        self.padded = F.pad(base, (translate,) * 4, mode="reflect") if translate else base
        self.labels, self.bs, self.cutout, self.translate = labels, batch_size, cutout, translate
        self.size = images.shape[-1]
        self.epoch = 0

    def __len__(self):
        return len(self.labels) // self.bs

    def __iter__(self):
        x = batch_crop(self.padded, self.size) if self.translate else self.padded
        if self.epoch % 2 == 1:
            x = x.flip(-1)
        if self.cutout:
            x = batch_cutout(x, self.cutout)
        self.epoch += 1
        idx = torch.randperm(len(x), device=x.device)
        for i in range(len(self)):
            j = idx[i * self.bs : (i + 1) * self.bs]
            yield x[j], self.labels[j]


# ---------------------------------------------------------------- model helpers


class Lookahead:
    def __init__(self, net):
        self.ema = {k: v.clone() for k, v in net.state_dict().items()}

    @torch.no_grad()
    def update(self, net, decay):
        for e, p in zip(self.ema.values(), net.state_dict().values()):
            if p.dtype.is_floating_point:
                e.lerp_(p, 1 - decay)
                p.copy_(e)


@torch.no_grad()
def predict(net, images, tta=2, bs=2000):
    """Logits with test-time augmentation: 0 none, 1 mirror, 2 mirror + 1px translations."""
    net.eval()

    def mirror(x):
        return 0.5 * net(x) + 0.5 * net(x.flip(-1))

    out = []
    for x in images.split(bs):
        if tta == 0:
            out.append(net(x))
            continue
        logits = mirror(x)
        if tta >= 2:
            p = F.pad(x, (1,) * 4, mode="reflect")
            shifted = 0.5 * (mirror(p[:, :, :32, :32]) + mirror(p[:, :, 2:, 2:]))
            logits = 0.5 * logits + 0.5 * shifted
        out.append(logits)
    return torch.cat(out).float()


def load_run(run_dir, device):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    net = AirResNet(tuple(cfg["widths"]), tuple(cfg["depth"])).to(device)
    net.load_state_dict(torch.load(os.path.join(run_dir, "model.pt"), map_location=device))
    return net.eval()


# ---------------------------------------------------------------- training


def main():
    args = parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(args.seed)
    out_dir = os.path.join(ROOT, "runs", args.name)
    os.makedirs(out_dir, exist_ok=True)

    x_all, y_all, x_test, test_ids = load_data(device)
    tr_idx, va_idx = split(len(y_all), args.val, args.seed)
    tr_t = torch.from_numpy(tr_idx).to(device)
    x_tr, y_tr = x_all[tr_t], y_all[tr_t]
    if len(va_idx):
        va_t = torch.from_numpy(va_idx).to(device)
        x_va, y_va = x_all[va_t], y_all[va_t]

    net = AirResNet(tuple(args.widths), tuple(args.depth)).to(device)
    n_params = count_params(net)
    n_total = count_params(net, trainable_only=False)
    print(f"params: {n_params:,} trainable, {n_total:,} total", flush=True)
    if not args.allow_big:
        assert n_total <= PARAM_LIMIT, f"{n_total} params exceeds the 5M limit"
    init_whitening(net.whiten, x_tr[:5000])

    teachers = [load_run(t, device) for t in args.teachers]
    for t in teachers:
        for p in t.parameters():
            p.requires_grad = False

    loader = Loader(x_tr, y_tr, args.batch_size, args.translate, args.cutout)
    total_steps = ceil(len(loader) * args.epochs)

    kilostep = 1024 * (1 + 1 / (1 - args.momentum))
    lr = args.lr / kilostep
    wd = args.wd * args.batch_size / kilostep
    lr_bias = lr * args.bias_scaler
    norm_biases = [p for n, p in net.named_parameters() if "norm" in n and p.requires_grad]
    others = [p for n, p in net.named_parameters() if "norm" not in n and p.requires_grad]
    opt = torch.optim.SGD([dict(params=norm_biases, lr=lr_bias, weight_decay=wd / lr_bias),
                           dict(params=others, lr=lr, weight_decay=wd / lr)],
                          momentum=args.momentum, nesterov=True)
    warm = int(total_steps * args.warmup)

    def lr_mult(step):
        if step < warm:
            f = step / warm
            return 0.2 * (1 - f) + f
        return (total_steps - step) / (total_steps - warm)

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_mult)
    alpha = 0.95**5 * (torch.arange(total_steps + 1) / total_steps) ** 3
    lookahead = Lookahead(net)
    ce = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, reduction="none")

    history, step, t0 = [], 0, time.time()
    for epoch in range(ceil(args.epochs)):
        net.whiten.bias.requires_grad = epoch < args.whiten_bias_epochs
        net.train()
        for x, y in loader:
            logits = net(x)
            loss = ce(logits, y)
            if teachers:
                with torch.no_grad():
                    for t in teachers:
                        t.eval()
                    t_logits = torch.stack([t(x) for t in teachers]).mean(0)
                T = args.kd_temp
                kd = F.kl_div(F.log_softmax(logits / T, 1), F.softmax(t_logits / T, 1),
                              reduction="none").sum(1) * T * T
                loss = (1 - args.kd_alpha) * loss + args.kd_alpha * kd
            loss = loss.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % 5 == 0:
                lookahead.update(net, decay=alpha[step].item())
            if step >= total_steps:
                lookahead.update(net, decay=1.0)
                break
        rec = dict(epoch=epoch + 1, loss=loss.item() / len(y), time=round(time.time() - t0, 1))
        if len(va_idx) and ((epoch + 1) % args.eval_every == 0 or step >= total_steps):
            rec["val_acc"] = (predict(net, x_va, tta=0).argmax(1) == y_va).float().mean().item()
        history.append(rec)
        print(json.dumps(rec), flush=True)
        if step >= total_steps:
            break

    result = dict(vars(args), params=n_params, params_total=n_total, train_time=time.time() - t0)
    if len(va_idx):
        va_logits = predict(net, x_va, tta=2)
        result["val_acc_tta"] = (va_logits.argmax(1) == y_va).float().mean().item()
        torch.save(va_logits.cpu(), os.path.join(out_dir, "val_logits.pt"))
        np.save(os.path.join(out_dir, "val_idx.npy"), va_idx)
    torch.save(predict(net, x_test, tta=2).cpu(), os.path.join(out_dir, "test_logits.pt"))
    torch.save(net.state_dict(), os.path.join(out_dir, "model.pt"))
    json.dump(result, open(os.path.join(out_dir, "config.json"), "w"), indent=1)
    json.dump(history, open(os.path.join(out_dir, "history.json"), "w"))
    print(json.dumps({k: result[k] for k in ("name", "params", "train_time") + (("val_acc_tta",) if len(va_idx) else ())}))


if __name__ == "__main__":
    main()
