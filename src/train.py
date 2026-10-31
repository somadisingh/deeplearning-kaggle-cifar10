"""Train an AirResNet on CIFAR-10 with all data and augmentation kept on the MPS device.

Recipe follows airbench96 (Jordan, 2024): frozen whitening stem, Nesterov SGD with decoupled
hyperparameters, label smoothing, flip + 4px translate + cutout, Lookahead-style EMA and
flip/translate test-time augmentation. Optional knowledge distillation from teacher checkpoints
that were themselves trained from scratch by this script.

The data pipeline (batch_crop, batch_cutout), the decoupled SGD hyperparameters, the Lookahead EMA and the
flip / translate TTA are adapted from cifar10-airbench by Keller Jordan (MIT License); see THIRD_PARTY_NOTICES.md.

Only `x_train`/`y_train` are ever used for training or model selection. The test images are
only passed through the finished model to write predictions.
"""
import argparse
import copy
import json
import math
import os
import time
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models import AirResNet, PreActResNet, ShakeResNet, count_params, init_whitening

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
    p.add_argument("--trivialaugment", action="store_true")
    p.add_argument("--sam-rho", type=float, default=0.0, help="SAM neighborhood radius; 0 = plain SGD")
    p.add_argument("--arch", choices=["air", "preact", "shake"], default="air",
                   help="air: airbench-style AirResNet with its decoupled SGD recipe; preact: standard "
                        "pre-activation ResNet (shake: Shake-Shake ResNet) trained with plain SGD + cosine schedule")
    p.add_argument("--blur", action="store_true", help="preact: anti-aliased (BlurPool) downsampling")
    p.add_argument("--sgd-lr", type=float, default=0.1, help="preact: peak lr per 128 examples")
    p.add_argument("--sgd-wd", type=float, default=5e-4, help="preact: weight decay")
    p.add_argument("--sgd-warmup-epochs", type=float, default=5)
    p.add_argument("--ssl", choices=["none", "sla", "sla_cp", "rot"], default="none",
                   help="rotation self-supervision: SLA+SD joint labels, or an auxiliary rotation head")
    p.add_argument("--two-view", action="store_true",
                   help="two augmented views per image + Jensen-Shannon consistency between their predictions")
    p.add_argument("--cons-weight", type=float, default=12.0, help="weight of the two-view JS loss (AugMix uses 12)")
    p.add_argument("--cutmix", type=float, default=0.0,
                   help="SLA only: probability per batch of CutMix (box from a shuffled partner, area ~ Beta(1,1)), "
                        "pasted before the rotations")
    p.add_argument("--sla-percopy", action="store_true",
                   help="SLA only: each rotated copy gets its own crop, flip and cutout (copy 0 stays upright)")
    p.add_argument("--ema", type=float, default=0.0,
                   help="decay of a weight average kept beside the live network (0 = off); evaluated with BN "
                        "statistics recomputed on augmented training images")
    p.add_argument("--ema-epoch", type=int, default=0,
                   help="also save the weight average at this epoch (for a 50k retrain, the epoch picked on 45k)")
    p.add_argument("--sla-beta", type=float, default=1.0, help="weight of the plain-head CE in SLA+SD")
    p.add_argument("--ckpt-every", type=int, default=10, help="save a resumable checkpoint every N epochs")
    p.add_argument("--resume", action="store_true", help="continue from runs/<name>/ckpt.pt if it exists")
    p.add_argument("--rot-weight", type=float, default=0.5, help="lambda for the auxiliary rotation loss")
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


_TA_IMAGES = None


def _ta_init(images):
    global _TA_IMAGES
    torch.set_num_threads(1)
    _TA_IMAGES = images


def _ta_chunk(start, end, seed):
    from torchvision.transforms import v2
    torch.manual_seed(seed)
    ta = v2.TrivialAugmentWide()
    return torch.stack([ta(img) for img in _TA_IMAGES[start:end]]).numpy()


class TrivialAugmentSource:
    """Per-image TrivialAugment (Müller & Hutter, 2021) on CPU worker processes. The next epoch's
    images are computed in the background while the GPU trains on the current one."""

    def __init__(self, images_uint8, device, seed, workers=None, chunk=1000):
        from concurrent.futures import ProcessPoolExecutor
        self.n, self.device, self.seed, self.chunk = len(images_uint8), device, seed, chunk
        self.pool = ProcessPoolExecutor(workers or max(1, len(os.sched_getaffinity(0)) - 1 if hasattr(os, "sched_getaffinity") else os.cpu_count() - 2), initializer=_ta_init,
                                        initargs=(images_uint8,))
        self.epoch = 0
        self.pending = self._submit()

    def _submit(self):
        s = self.seed * 100_000 + self.epoch * 1000
        return [self.pool.submit(_ta_chunk, i, min(i + self.chunk, self.n), s + k)
                for k, i in enumerate(range(0, self.n, self.chunk))]

    def next(self):
        x = np.concatenate([f.result() for f in self.pending])
        self.epoch += 1
        self.pending = self._submit()
        x = torch.from_numpy(x).to(self.device).float().div(255)
        return (x - MEAN.to(self.device)) / STD.to(self.device)


class Loader:
    """Augmented epochs built on-device. Flip is random per image on epoch 0, then every image's
    flip alternates each epoch (airbench's 'alternating flip'). With a `source` (TrivialAugment),
    fresh base images arrive every epoch and flips are random per image."""

    def __init__(self, images, labels, batch_size, translate, cutout, source=None, two_view=False, percopy=False):
        n = len(images)
        flip = (torch.rand(n, device=images.device) < 0.5).view(-1, 1, 1, 1)
        base = torch.where(flip, images.flip(-1), images)
        self.padded = F.pad(base, (translate,) * 4, mode="reflect") if translate else base
        self.labels, self.bs, self.cutout, self.translate = labels, batch_size, cutout, translate
        self.size = images.shape[-1]
        self.source = source
        self.two_view = two_view
        self.percopy = percopy
        self.epoch = 0

    def __len__(self):
        return len(self.labels) // self.bs

    def __iter__(self):
        if self.source is not None:
            base = self.source.next()
            flip = (torch.rand(len(base), device=base.device) < 0.5).view(-1, 1, 1, 1)
            base = torch.where(flip, base.flip(-1), base)
            self.padded = F.pad(base, (self.translate,) * 4, mode="reflect") if self.translate else base
        if self.percopy:
            # Four views per image for the four SLA copies: own crop, flip and cutout each (shape B, 4, C, H, W).
            views = []
            for _ in range(4):
                v = batch_crop(self.padded, self.size) if self.translate else self.padded
                flip = (torch.rand(len(v), device=v.device) < 0.5).view(-1, 1, 1, 1)
                v = torch.where(flip, v.flip(-1), v)
                views.append(batch_cutout(v, self.cutout) if self.cutout else v)
            x = torch.stack(views, 1)
        else:
            x = batch_crop(self.padded, self.size) if self.translate else self.padded
            if self.source is None and self.epoch % 2 == 1:
                x = x.flip(-1)
            if self.cutout:
                x = batch_cutout(x, self.cutout)
        x2 = None
        if self.two_view:
            # Second view of the same images: its own crop, flip and cutout (the TrivialAugment op is shared).
            x2 = batch_crop(self.padded, self.size) if self.translate else self.padded
            flip = (torch.rand(len(x2), device=x2.device) < 0.5).view(-1, 1, 1, 1)
            x2 = torch.where(flip, x2.flip(-1), x2)
            if self.cutout:
                x2 = batch_cutout(x2, self.cutout)
        self.epoch += 1
        idx = torch.randperm(len(x), device=x.device)
        for i in range(len(self)):
            j = idx[i * self.bs : (i + 1) * self.bs]
            if x2 is None:
                yield x[j], self.labels[j]
            else:
                yield torch.cat([x[j], x2[j]]), self.labels[j]


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


def rotate4(x):
    """Stack the 4 rotations of each image, interleaved: row 4*i + k is image i rotated by k*90 degrees."""
    return torch.stack([torch.rot90(x, k, (2, 3)) for k in range(4)], 1).flatten(0, 1)


def cutmix_box(x, lam):
    """CutMix (Yun et al., 2019): paste one box from a shuffled partner into every image. Works on (B, C, H, W)
    and on per-copy views (B, 4, C, H, W). Returns the mixed images, the partner order and the kept area."""
    h, w = x.shape[-2:]
    r = math.sqrt(1 - lam)
    bh, bw = int(h * r), int(w * r)
    cy, cx = np.random.randint(h), np.random.randint(w)
    y1, y2 = max(cy - bh // 2, 0), min(cy + bh // 2, h)
    x1, x2 = max(cx - bw // 2, 0), min(cx + bw // 2, w)
    partner = torch.randperm(len(x), device=x.device)
    x = x.clone()
    x[..., y1:y2, x1:x2] = x[partner][..., y1:y2, x1:x2]
    return x, partner, 1 - (y2 - y1) * (x2 - x1) / (h * w)


@torch.no_grad()
def recompute_bn(net, images, bs=500):
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None
    net.train()
    for x in images.split(bs):
        net(x)
    net.eval()


PERMS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]  # RGB channel orders


def colour_permute(x, perm_idx):
    """Apply channel permutation PERMS[perm_idx[i]] to image i."""
    order = torch.tensor(PERMS, device=x.device)[perm_idx]  # (n, 3)
    return torch.gather(x, 1, order.view(-1, 3, 1, 1).expand_as(x))


def sla_aggregate(net, x):
    """SLA+AG (Lee et al., 2020, eq. 4): for the copy rotated by k, keep the joint logits of labels
    (i, k), then average those logits over the 4 rotations. With rotation x colour-permutation labels
    (24 transformations), the copies keep the original colour order (permutation 0)."""
    joint = net.joint_head(net.features(rotate4(x))) * net.scale
    m = joint.shape[1] // 10
    joint = joint.view(len(x), 4, 10, m)
    step = m // 4
    return torch.stack([joint[:, k, :, k * step] for k in range(4)]).mean(0)


@torch.no_grad()
def predict(net, images, tta=2, bs=2000, fn=None):
    """Logits with test-time augmentation: 0 none, 1 mirror, 2 mirror + 1px translations.
    `fn(net, x)` replaces the plain forward pass (e.g. sla_aggregate)."""
    net.eval()
    fn = fn or (lambda m, x: m(x))

    def mirror(x):
        return 0.5 * fn(net, x) + 0.5 * fn(net, x.flip(-1))

    out = []
    for x in images.split(bs):
        if tta == 0:
            out.append(fn(net, x))
            continue
        logits = mirror(x)
        if tta >= 2:
            p = F.pad(x, (1,) * 4, mode="reflect")
            shifted = 0.5 * (mirror(p[:, :, :32, :32]) + mirror(p[:, :, 2:, 2:]))
            logits = 0.5 * logits + 0.5 * shifted
        out.append(logits)
    return torch.cat(out).float()


def build_net(arch, widths, depth, ssl, blur=False):
    if arch == "preact":
        return PreActResNet(tuple(widths), tuple(depth), ssl=ssl, blur=blur)
    cls = ShakeResNet if arch == "shake" else AirResNet
    return cls(tuple(widths), tuple(depth), ssl=ssl)


def load_run(run_dir, device):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    net = build_net(cfg.get("arch", "air"), cfg["widths"], cfg["depth"], cfg.get("ssl", "none"),
                    cfg.get("blur", False)).to(device)
    net.load_state_dict(torch.load(os.path.join(run_dir, "model.pt"), map_location=device))
    return net.eval()


# ---------------------------------------------------------------- training


def main():
    args = parse_args()
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
    else:
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

    assert not (args.ssl != "none" and args.teachers), "SSL heads and teacher distillation are not combined"
    net = build_net(args.arch, args.widths, args.depth, args.ssl, args.blur).to(device)
    n_params = count_params(net)
    n_total = count_params(net, trainable_only=False)
    print(f"params: {n_params:,} trainable, {n_total:,} total", flush=True)
    if not args.allow_big:
        assert n_total <= PARAM_LIMIT, f"{n_total} params exceeds the 5M limit"
    if args.arch == "air":
        init_whitening(net.whiten, x_tr[:5000])

    teachers = [load_run(t, device) for t in args.teachers]
    for t in teachers:
        for p in t.parameters():
            p.requires_grad = False

    source = None
    if args.trivialaugment:
        raw = np.load(os.path.join(ROOT, "data", "cifar10_cache.npz"))["x_train"][tr_idx]
        source = TrivialAugmentSource(torch.from_numpy(raw).permute(0, 3, 1, 2).contiguous(), device, args.seed)
    assert not args.two_view or args.ssl in ("sla", "sla_cp"), "--two-view is implemented on top of SLA"
    assert not ((args.cutmix or args.sla_percopy) and args.ssl not in ("sla", "sla_cp")), "--cutmix/--sla-percopy need SLA"
    assert not (args.sla_percopy and args.two_view), "--sla-percopy and --two-view are separate tests"
    loader = Loader(x_tr, y_tr, args.batch_size, args.translate, args.cutout, source, two_view=args.two_view,
                    percopy=args.sla_percopy)
    total_steps = ceil(len(loader) * args.epochs)

    if args.arch == "air":
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
    else:
        # Standard recipe: Nesterov SGD (momentum 0.9), lr scaled linearly with batch size, weight decay
        # on conv/linear weights only, linear warmup then cosine decay to zero; loss is the batch mean.
        decay = [p for p in net.parameters() if p.ndim > 1]
        no_decay = [p for p in net.parameters() if p.ndim <= 1]
        opt = torch.optim.SGD([dict(params=decay, weight_decay=args.sgd_wd), dict(params=no_decay, weight_decay=0)],
                              lr=args.sgd_lr * args.batch_size / 128, momentum=0.9, nesterov=True)
        warm = int(len(loader) * args.sgd_warmup_epochs)

        def lr_mult(step):
            if step < warm:
                return (step + 1) / warm
            return 0.5 * (1 + math.cos(math.pi * (step - warm) / (total_steps - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_mult)
    alpha = 0.95**5 * (torch.arange(total_steps + 1) / total_steps) ** 3
    lookahead = Lookahead(net)
    ema_net = copy.deepcopy(net) if args.ema else None
    ema_best = (-1.0, 0, None)  # (holdout accuracy, epoch, state) of the best weight average
    ce = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, reduction="none")

    def ema_eval_copy():
        """The weight average with BN statistics recomputed on augmented training images (crop, flip, cutout)."""
        e = copy.deepcopy(ema_net)
        imgs = batch_crop(loader.padded[:10000], 32) if args.translate else loader.padded[:10000]
        if args.cutout:
            imgs = batch_cutout(imgs, args.cutout)
        recompute_bn(e, imgs)
        return e

    history, step, t0, start_epoch = [], 0, time.time(), 0
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        lookahead.ema = ck["lookahead"]
        if ema_net is not None and ck.get("ema") is not None:
            ema_net.load_state_dict(ck["ema"])
            ema_best = ck["ema_best"]
        history, step, start_epoch = ck["history"], ck["step"], ck["epoch"]
        t0 -= ck["elapsed"]
        loader.epoch = start_epoch
        torch.manual_seed(args.seed * 1000 + start_epoch)
        print(f"resumed from epoch {start_epoch}", flush=True)
    for epoch in range(start_epoch, ceil(args.epochs)):
        if args.arch == "air":
            net.whiten.bias.requires_grad = epoch < args.whiten_bias_epochs
        net.train()
        for x, y in loader:
            t_logits = None
            if teachers:
                with torch.no_grad():
                    for t in teachers:
                        t.eval()
                    t_logits = torch.stack([t(x) for t in teachers]).mean(0)

            norm_by = 1 if args.arch == "air" else len(y)
            y_b, lam = None, 1.0
            if args.cutmix and np.random.rand() < args.cutmix:
                x, partner, lam = cutmix_box(x, np.random.beta(1.0, 1.0))
                y_b = y[partner]

            def compute_loss():
                if args.ssl in ("sla", "sla_cp"):
                    # SLA+SD (Lee et al., 2020, eqs. 3 and 5; hankook/SLA create_sla_sd_trainer):
                    # joint (class, transformation) cross-entropy averaged over the 4 rotated copies, plus
                    # self-distillation of the aggregated joint prediction (held constant) into the
                    # plain head on the unrotated image, plus beta * its usual cross-entropy.
                    # sla_cp: the transformation is rotation x RGB permutation (24); each image gets one
                    # random permutation, shared by its 4 rotated copies, so the cost stays at 4x.
                    xs, ys = x, y
                    if args.two_view:
                        ys = torch.cat([y, y])
                    n = len(xs)
                    rots = torch.arange(4, device=y.device).repeat(n, 1)  # (n, 4)
                    if args.sla_percopy:
                        xr = torch.stack([torch.rot90(xs[:, k], k, (2, 3)) for k in range(4)], 1).flatten(0, 1)
                    else:
                        xr = rotate4(xs)
                    m = 4
                    t = rots
                    if args.ssl == "sla_cp":
                        m = 24
                        perm = torch.randint(0, 6, (n,), device=y.device)
                        xr = colour_permute(xr, perm.repeat_interleave(4))
                        t = rots * 6 + perm.view(-1, 1)
                    z = net.features(xr)
                    joint = net.joint_head(z) * net.scale
                    joint_y = (ys.view(-1, 1) * m + t).flatten()
                    joint_loss = F.cross_entropy(joint, joint_y, reduction="none").view(n, 4).mean(1)
                    single = net.head(z[::4]) * net.scale
                    plain = ce(single, ys)
                    if y_b is not None:
                        # CutMix: mix the joint and plain-head losses by area; the aggregate teacher sees the
                        # same mixed image, so the self-distillation term is unchanged.
                        joint_b = (y_b.view(-1, 1) * m + t).flatten()
                        joint_loss = lam * joint_loss + (1 - lam) * F.cross_entropy(
                            joint, joint_b, reduction="none").view(n, 4).mean(1)
                        plain = lam * plain + (1 - lam) * ce(single, y_b)
                    j = joint.view(n, 4, 10, m)
                    agg = torch.gather(j, 3, t.view(n, 4, 1, 1).expand(n, 4, 10, 1)).squeeze(3).mean(1).detach()
                    sd = F.kl_div(F.log_softmax(single, 1), F.softmax(agg, 1), reduction="none").sum(1)
                    per_img = joint_loss + sd + args.sla_beta * plain
                    if args.two_view:
                        # Jensen-Shannon consistency between the two views (Hendrycks et al., 2020, AugMix).
                        h = n // 2
                        p1, p2 = F.softmax(single[:h], 1), F.softmax(single[h:], 1)
                        mix = ((p1 + p2) / 2).clamp_min(1e-7).log()
                        js = 0.5 * (F.kl_div(mix, p1, reduction="none").sum(1) + F.kl_div(mix, p2, reduction="none").sum(1))
                        return (0.5 * (per_img[:h] + per_img[h:]) + args.cons_weight * js).sum() / norm_by
                    return per_img.sum() / norm_by
                if args.ssl == "rot":
                    # Auxiliary rotation prediction (Hendrycks et al., 2019; hendrycks/ss-ood):
                    # classification loss on the unrotated images only, plus lambda times the
                    # 4-way rotation loss averaged over all 4 rotations.
                    n = len(x)
                    z = net.features(rotate4(x))
                    rot_y = torch.arange(4, device=y.device).repeat(n)
                    rot_loss = F.cross_entropy(net.rot_head(z) * net.scale, rot_y,
                                               reduction="none").view(n, 4).mean(1)
                    return (ce(net.head(z[::4]) * net.scale, y) + args.rot_weight * rot_loss).sum() / norm_by
                logits = net(x)
                loss = ce(logits, y)
                if t_logits is not None:
                    T = args.kd_temp
                    kd = F.kl_div(F.log_softmax(logits / T, 1), F.softmax(t_logits / T, 1),
                                  reduction="none").sum(1) * T * T
                    loss = (1 - args.kd_alpha) * loss + args.kd_alpha * kd
                return loss.sum() / norm_by

            loss = compute_loss()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.sam_rho > 0:
                # SAM (Foret et al., 2021): step with the gradient taken at the worst-case point
                # within an L2 ball of radius rho around the current weights.
                params = [p for p in net.parameters() if p.grad is not None]
                with torch.no_grad():
                    norm = torch.norm(torch.stack([p.grad.norm() for p in params])) + 1e-12
                    eps = [p.grad * (args.sam_rho / norm) for p in params]
                    for p, e in zip(params, eps):
                        p.add_(e)
                opt.zero_grad(set_to_none=True)
                compute_loss().backward()
                with torch.no_grad():
                    for p, e in zip(params, eps):
                        p.sub_(e)
            opt.step()
            sched.step()
            step += 1
            if ema_net is not None:
                with torch.no_grad():
                    for e, p_ in zip(ema_net.state_dict().values(), net.state_dict().values()):
                        if p_.dtype.is_floating_point:
                            e.lerp_(p_, 1 - args.ema)
            if args.arch == "air" and step % 5 == 0:
                lookahead.update(net, decay=alpha[step].item())
            if step >= total_steps:
                if args.arch == "air":
                    lookahead.update(net, decay=1.0)
                break
        rec = dict(epoch=epoch + 1, loss=loss.item() * norm_by / len(y), time=round(time.time() - t0, 1))
        if len(va_idx) and ((epoch + 1) % args.eval_every == 0 or step >= total_steps):
            rec["val_acc"] = (predict(net, x_va, tta=0).argmax(1) == y_va).float().mean().item()
            if ema_net is not None:
                e = ema_eval_copy()
                rec["val_acc_ema"] = (predict(e, x_va, tta=0).argmax(1) == y_va).float().mean().item()
                if rec["val_acc_ema"] > ema_best[0]:
                    ema_best = (rec["val_acc_ema"], epoch + 1, copy.deepcopy(e.state_dict()))
        if ema_net is not None and args.ema_epoch == epoch + 1:
            e = ema_eval_copy()
            tag = f"ema_e{epoch + 1}"
            torch.save(e.state_dict(), os.path.join(out_dir, f"model_{tag}.pt"))
            torch.save(predict(e, x_test, tta=2).cpu(), os.path.join(out_dir, f"test_logits_{tag}.pt"))
            if args.ssl in ("sla", "sla_cp"):
                torch.save(predict(e, x_test, tta=2, fn=sla_aggregate).cpu(),
                           os.path.join(out_dir, f"test_logits_{tag}_sla_agg.pt"))
            net.train()
        history.append(rec)
        print(json.dumps(rec), flush=True)
        if args.ckpt_every and (epoch + 1) % args.ckpt_every == 0 and step < total_steps:
            torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                            lookahead=lookahead.ema, history=history, step=step, epoch=epoch + 1,
                            ema=ema_net.state_dict() if ema_net is not None else None, ema_best=ema_best,
                            elapsed=time.time() - t0), ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)
        if step >= total_steps:
            break

    if source is not None:
        source.pool.shutdown(cancel_futures=True)
    result = dict(vars(args), params=n_params, params_total=n_total, train_time=time.time() - t0)
    if len(va_idx):
        va_logits = predict(net, x_va, tta=2)
        result["val_acc_tta"] = (va_logits.argmax(1) == y_va).float().mean().item()
        torch.save(va_logits.cpu(), os.path.join(out_dir, "val_logits.pt"))
        np.save(os.path.join(out_dir, "val_idx.npy"), va_idx)
        if args.ssl in ("sla", "sla_cp"):
            va_agg = predict(net, x_va, tta=2, fn=sla_aggregate)
            result["val_acc_tta_sla_agg"] = (va_agg.argmax(1) == y_va).float().mean().item()
            torch.save(va_agg.cpu(), os.path.join(out_dir, "val_logits_sla_agg.pt"))
    torch.save(predict(net, x_test, tta=2).cpu(), os.path.join(out_dir, "test_logits.pt"))
    if args.ssl in ("sla", "sla_cp"):
        torch.save(predict(net, x_test, tta=2, fn=sla_aggregate).cpu(), os.path.join(out_dir, "test_logits_sla_agg.pt"))
    torch.save(net.state_dict(), os.path.join(out_dir, "model.pt"))
    if ema_net is not None:
        finals = [("ema", ema_eval_copy())]
        if ema_best[2] is not None:
            b = copy.deepcopy(net)
            b.load_state_dict(ema_best[2])
            finals.append(("ema_best", b.eval()))
            result["ema_best_epoch"] = ema_best[1]
        for tag, e in finals:
            torch.save(e.state_dict(), os.path.join(out_dir, f"model_{tag}.pt"))
            torch.save(predict(e, x_test, tta=2).cpu(), os.path.join(out_dir, f"test_logits_{tag}.pt"))
            if args.ssl in ("sla", "sla_cp"):
                torch.save(predict(e, x_test, tta=2, fn=sla_aggregate).cpu(),
                           os.path.join(out_dir, f"test_logits_{tag}_sla_agg.pt"))
            if len(va_idx):
                result[f"val_acc_tta_{tag}"] = (predict(e, x_va, tta=2).argmax(1) == y_va).float().mean().item()
                if args.ssl in ("sla", "sla_cp"):
                    result[f"val_acc_tta_{tag}_sla_agg"] = (predict(e, x_va, tta=2, fn=sla_aggregate).argmax(1)
                                                            == y_va).float().mean().item()
    json.dump(result, open(os.path.join(out_dir, "config.json"), "w"), indent=1)
    json.dump(history, open(os.path.join(out_dir, "history.json"), "w"))
    print(json.dumps({k: result[k] for k in ("name", "params", "train_time") + (("val_acc_tta",) if len(va_idx) else ())}))


if __name__ == "__main__":
    main()
