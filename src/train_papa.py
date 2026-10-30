"""PopulAtion Parameter Averaging (Jolicoeur-Martineau et al., TMLR 2024) for the pre-act / Shake-Shake recipes.

p copies of the same network train side by side, each with its own initialization, data order, augmentation
draws and label smoothing (drawn from {0, 0.05, 0.1}, as in the paper's random regularizations). Every
`--papa-every` steps each copy is pulled toward the population mean (Algorithm 1):

    theta_j <- a' * theta_j + (1 - a') * mean(theta),   1 - a' = (lr / lr0) * (1 - alpha_papa)

with alpha_papa = 0.99 and f = 10 at batch 64 (the paper's from-scratch defaults); at larger batches f is scaled
down so each copy gets the same number of pulls per epoch (f = 1 at batch 512). At the end the copies are averaged into
one network (the "average soup"); its BatchNorm statistics are recomputed on un-augmented training images.
Only that single averaged network (<= 5M params) produces predictions.

Deviations from the paper: no Mixup / CutMix / Random Erasing draws (our recipe already uses TrivialAugment +
Cutout, and SLA joint labels do not mix), no REPAIR, no greedy soup.
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
from torch import nn

from models import count_params
from train import (PARAM_LIMIT, ROOT, Loader, TrivialAugmentSource, build_net, load_data, predict, rotate4,
                   sla_aggregate, split)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--arch", choices=["preact", "shake"], default="preact")
    p.add_argument("--widths", type=int, nargs=3, default=[64, 128, 256])
    p.add_argument("--depth", type=int, nargs=3, default=[4, 4, 3])
    p.add_argument("--blur", action="store_true")
    p.add_argument("--ssl", choices=["none", "sla"], default="none")
    p.add_argument("--population", type=int, default=4)
    p.add_argument("--papa-alpha", type=float, default=0.99)
    p.add_argument("--papa-every", type=int, default=None,
                   help="steps between pulls; default keeps the paper's rate per sample (every 10 steps at batch 64)")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--sgd-lr", type=float, default=0.1)
    p.add_argument("--sgd-wd", type=float, default=5e-4)
    p.add_argument("--sgd-warmup-epochs", type=float, default=5)
    p.add_argument("--translate", type=int, default=4)
    p.add_argument("--cutout", type=int, default=12)
    p.add_argument("--trivialaugment", action="store_true")
    p.add_argument("--val", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--ckpt-every", type=int, default=10)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def make_optimizer(net, args):
    decay = [p for p in net.parameters() if p.ndim > 1]
    no_decay = [p for p in net.parameters() if p.ndim <= 1]
    return torch.optim.SGD([dict(params=decay, weight_decay=args.sgd_wd), dict(params=no_decay, weight_decay=0)],
                           lr=args.sgd_lr * args.batch_size / 128, momentum=0.9, nesterov=True)


@torch.no_grad()
def average_into(target, nets):
    """target <- mean of nets (parameters and floating-point buffers)."""
    states = [n.state_dict() for n in nets]
    for k, v in target.state_dict().items():
        if v.dtype.is_floating_point:
            v.copy_(torch.stack([s[k] for s in states]).mean(0))


@torch.no_grad()
def recompute_bn(net, images, bs=500):
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None  # cumulative average over all batches
    net.train()
    for x in images.split(bs):
        net(x)
    net.eval()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    out_dir = os.path.join(ROOT, "runs", args.name)
    os.makedirs(out_dir, exist_ok=True)

    x_all, y_all, x_test, _ = load_data(device)
    tr_idx, va_idx = split(len(y_all), args.val, args.seed)
    tr_t = torch.from_numpy(tr_idx).to(device)
    x_tr, y_tr = x_all[tr_t], y_all[tr_t]
    if len(va_idx):
        va_t = torch.from_numpy(va_idx).to(device)
        x_va, y_va = x_all[va_t], y_all[va_t]

    if args.papa_every is None:
        args.papa_every = max(1, round(10 * 64 / args.batch_size))
    P = args.population
    rng = np.random.default_rng(args.seed)
    smoothing = [float(rng.choice([0.0, 0.05, 0.1])) for _ in range(P)]
    nets = []
    for j in range(P):
        torch.manual_seed(args.seed * 100 + j)
        nets.append(build_net(args.arch, args.widths, args.depth, args.ssl, args.blur).to(device))
    n_total = count_params(nets[0], trainable_only=False)
    assert n_total <= PARAM_LIMIT, f"{n_total} params exceeds the 5M limit"
    print(f"population {P} x {n_total:,} params; label smoothing {smoothing}", flush=True)

    source = None
    if args.trivialaugment:
        raw = np.load(os.path.join(ROOT, "data", "cifar10_cache.npz"))["x_train"][tr_idx]
        source = TrivialAugmentSource(torch.from_numpy(raw).permute(0, 3, 1, 2).contiguous(), device, args.seed)
    # One augmented copy of the data per member per epoch: the shared TrivialAugment pool hands out a fresh
    # draw on every call, and each member's Loader adds its own crops, flips, cutout and order.
    loaders = [Loader(x_tr, y_tr, args.batch_size, args.translate, args.cutout, source) for _ in range(P)]
    steps_per_epoch = len(loaders[0])
    total_steps = steps_per_epoch * args.epochs
    warm = int(steps_per_epoch * args.sgd_warmup_epochs)

    def lr_mult(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / (total_steps - warm)))

    opts = [make_optimizer(n, args) for n in nets]
    scheds = [torch.optim.lr_scheduler.LambdaLR(o, lr_mult) for o in opts]
    ces = [nn.CrossEntropyLoss(label_smoothing=s, reduction="none") for s in smoothing]
    mean_net = copy.deepcopy(nets[0])

    def loss_fn(net, ce, x, y):
        if args.ssl == "sla":
            n = len(x)
            z = net.features(rotate4(x))
            joint = net.joint_head(z) * net.scale
            joint_y = (y.view(-1, 1) * 4 + torch.arange(4, device=y.device)).flatten()
            joint_loss = nn.functional.cross_entropy(joint, joint_y, reduction="none").view(n, 4).mean(1)
            single = net.head(z[::4]) * net.scale
            j = joint.view(n, 4, -1)
            agg = torch.stack([j[:, k, k::4] for k in range(4)]).mean(0).detach()
            sd = nn.functional.kl_div(nn.functional.log_softmax(single, 1), nn.functional.softmax(agg, 1),
                                      reduction="none").sum(1)
            return (joint_loss + sd + ce(single, y)).mean()
        return ce(net(x), y).mean()

    history, step, start_epoch, t0 = [], 0, 0, time.time()
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        for n, o, s, st in zip(nets, opts, scheds, ck["members"]):
            n.load_state_dict(st["net"]); o.load_state_dict(st["opt"]); s.load_state_dict(st["sched"])
        history, step, start_epoch = ck["history"], ck["step"], ck["epoch"]
        t0 -= ck["elapsed"]
        print(f"resumed from epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        for n in nets:
            n.train()
        iters = [iter(l) for l in loaders]
        for _ in range(steps_per_epoch):
            for n, o, s, ce, it in zip(nets, opts, scheds, ces, iters):
                x, y = next(it)
                loss = loss_fn(n, ce, x, y)
                o.zero_grad(set_to_none=True)
                loss.backward()
                o.step()
                s.step()
            step += 1
            if step % args.papa_every == 0:
                # Pull toward the population mean; the pull scales with the current lr (Algorithm 1).
                pull = scheds[0].get_last_lr()[0] / (args.sgd_lr * args.batch_size / 128) * (1 - args.papa_alpha)
                average_into(mean_net, nets)
                mean_state = mean_net.state_dict()
                with torch.no_grad():
                    for n in nets:
                        for k, v in n.state_dict().items():
                            if v.dtype.is_floating_point:
                                v.lerp_(mean_state[k], pull)
        rec = dict(epoch=epoch + 1, loss=loss.item(), time=round(time.time() - t0, 1))
        if len(va_idx) and ((epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs):
            average_into(mean_net, nets)
            recompute_bn(mean_net, x_tr[:10000])
            rec["val_acc_soup"] = (predict(mean_net, x_va, tta=0).argmax(1) == y_va).float().mean().item()
            rec["val_acc_member0"] = (predict(nets[0], x_va, tta=0).argmax(1) == y_va).float().mean().item()
        history.append(rec)
        print(json.dumps(rec), flush=True)
        if args.ckpt_every and (epoch + 1) % args.ckpt_every == 0 and epoch + 1 < args.epochs:
            members = [dict(net=n.state_dict(), opt=o.state_dict(), sched=s.state_dict())
                       for n, o, s in zip(nets, opts, scheds)]
            torch.save(dict(members=members, history=history, step=step, epoch=epoch + 1,
                            elapsed=time.time() - t0), ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)

    if source is not None:
        source.pool.shutdown(cancel_futures=True)
    average_into(mean_net, nets)
    recompute_bn(mean_net, x_tr[:10000])
    net = mean_net
    result = dict(vars(args), params=count_params(net), params_total=n_total, smoothing=smoothing,
                  train_time=time.time() - t0)
    if len(va_idx):
        va_logits = predict(net, x_va, tta=2)
        result["val_acc_tta"] = (va_logits.argmax(1) == y_va).float().mean().item()
        torch.save(va_logits.cpu(), os.path.join(out_dir, "val_logits.pt"))
        np.save(os.path.join(out_dir, "val_idx.npy"), va_idx)
        result["val_acc_members_tta"] = [(predict(n, x_va, tta=2).argmax(1) == y_va).float().mean().item()
                                         for n in nets]
        if args.ssl == "sla":
            va_agg = predict(net, x_va, tta=2, fn=sla_aggregate)
            result["val_acc_tta_sla_agg"] = (va_agg.argmax(1) == y_va).float().mean().item()
            torch.save(va_agg.cpu(), os.path.join(out_dir, "val_logits_sla_agg.pt"))
    torch.save(predict(net, x_test, tta=2).cpu(), os.path.join(out_dir, "test_logits.pt"))
    if args.ssl == "sla":
        torch.save(predict(net, x_test, tta=2, fn=sla_aggregate).cpu(), os.path.join(out_dir, "test_logits_sla_agg.pt"))
    torch.save(net.state_dict(), os.path.join(out_dir, "model.pt"))
    json.dump(result, open(os.path.join(out_dir, "config.json"), "w"), indent=1)
    json.dump(history, open(os.path.join(out_dir, "history.json"), "w"))
    print(json.dumps({k: result.get(k) for k in ("name", "params", "train_time", "val_acc_tta", "val_acc_members_tta")}))


if __name__ == "__main__":
    main()
