"""Train the final recipe: PreActResNet (64/128/256, 4/4/3) with SLA+SD rotation labels and RegMixup.

Recipe (docs/LOGBOOK.md, Section 8):
- Augmentation: TrivialAugmentWide, random flip, 4-pixel translation, 12-pixel cutout.
- SLA+SD (Lee et al., 2020): each image goes in as its 4 rotations. Loss = joint (class x rotation) cross-entropy
  averaged over the 4 copies + cross-entropy of the 10-class head on the upright copy + KL self-distillation of the
  aggregated joint prediction into the 10-class head.
- RegMixup (Pinto et al., 2022): each batch also carries a mixed copy of itself (one lambda ~ Beta(20, 20) per batch).
  The mixed copy takes both labels on both heads, weight 1; self-distillation stays on the clean images.
- Nesterov SGD, momentum 0.9, lr 0.1 per 128 images, weight decay 5e-4 on conv and linear weights,
  5 warmup epochs then cosine to zero, label smoothing 0.1 on the 10-class head.

Only the training images are used for training and for the holdout. The test images only go through the
finished network. Usage (from the repository root):
  python src/train.py --name regmix_e300_45k            # 45,000 train / 5,000 holdout images
  python src/train.py --name regmix_e300_50k --val 0    # all 50,000 images (the submission model)
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from data import CACHE, ROOT, Loader, TrivialAugment, load_data, split
from model import PARAM_LIMIT, PreActResNet, count_params
from predict import device, predict_probs, rotate4, write_submission


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="output folder runs/<name>")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=256, help="clean images per step (plus as many mixed images)")
    p.add_argument("--lr", type=float, default=0.1, help="peak learning rate per 128 images")
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--warmup-epochs", type=float, default=5)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--regmixup-alpha", type=float, default=20.0)
    p.add_argument("--regmixup-eta", type=float, default=1.0, help="weight of the mixed-image loss")
    p.add_argument("--se-reduction", type=int, default=0, help="squeeze-and-excitation in each block; 0 = off")
    p.add_argument("--val", type=int, default=5000, help="holdout images from the training set; 0 = train on all 50k")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--ckpt-every", type=int, default=10, help="save runs/<name>/ckpt.pt every N epochs")
    p.add_argument("--resume", action="store_true", help="continue from runs/<name>/ckpt.pt")
    p.add_argument("--subset", type=int, default=0, help="smoke test only: train on the first N training images")
    return p.parse_args()


def sla_regmixup_loss(net, x, y, x_mix, y_mix, lam, ce, eta):
    """Summed loss of one batch. x, y: the clean images and labels; x_mix = lam * x + (1 - lam) * x[partner],
    y_mix = y[partner]. Rows 0..B-1 of every tensor are the clean copy, rows B..2B-1 the mixed copy."""
    B = len(y)
    n = 2 * B
    rots = torch.arange(4, device=y.device).repeat(n, 1)  # (n, 4)
    z = net.features(rotate4(torch.cat([x, x_mix])))
    joint = net.joint_head(z)  # (4n, 40); row 4 * i + k = image i rotated by k
    single = net.head(z[::4])  # 10-class head on the upright copies

    def joint_ce(labels, rows):
        target = (labels.view(-1, 1) * 4 + rots[rows]).flatten()
        rows4 = slice(4 * rows.start, 4 * rows.stop)
        return F.cross_entropy(joint[rows4], target, reduction="none").view(-1, 4).mean(1)

    clean, mixed = slice(0, B), slice(B, n)
    joint_loss = torch.cat([joint_ce(y, clean), lam * joint_ce(y, mixed) + (1 - lam) * joint_ce(y_mix, mixed)])
    plain = torch.cat([ce(single[:B], y), lam * ce(single[B:], y) + (1 - lam) * ce(single[B:], y_mix)])

    # Self-distillation: the mean over the 4 copies of the joint logits of (class, own rotation), held constant.
    j = joint.view(n, 4, 10, 4)
    agg = torch.gather(j, 3, rots.view(n, 4, 1, 1).expand(n, 4, 10, 1)).squeeze(3).mean(1).detach()
    sd = F.kl_div(F.log_softmax(single[:B], 1), F.softmax(agg[:B], 1), reduction="none").sum(1)

    hard = joint_loss + plain
    return torch.cat([hard[:B] + sd, eta * hard[B:]]).sum()


def main():
    args = parse_args()
    dev = device()
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    out_dir = os.path.join(ROOT, "runs", args.name)
    os.makedirs(out_dir, exist_ok=True)

    x_all, y_all, x_test, _ = load_data(dev)
    tr_idx, va_idx = split(y_all.cpu().numpy(), args.val)
    if args.subset:
        tr_idx = tr_idx[: args.subset]
    np_rng = np.random.default_rng(args.seed)
    x_va, y_va = x_all[torch.from_numpy(va_idx).to(dev)], y_all[torch.from_numpy(va_idx).to(dev)]
    y_tr = y_all[torch.from_numpy(tr_idx).to(dev)]

    net = PreActResNet(se_reduction=args.se_reduction).to(dev)
    n_params = count_params(net)
    print(f"trainable parameters: {n_params:,}", flush=True)
    assert n_params <= PARAM_LIMIT

    raw = np.load(CACHE)["x_train"][tr_idx]
    source = TrivialAugment(torch.from_numpy(raw).permute(0, 3, 1, 2).contiguous(), dev, args.seed)
    loader = Loader(source, y_tr, args.batch_size)
    total_steps = len(loader) * args.epochs
    warm = int(len(loader) * args.warmup_epochs)

    decay = [p for p in net.parameters() if p.ndim > 1]
    no_decay = [p for p in net.parameters() if p.ndim <= 1]
    opt = torch.optim.SGD([dict(params=decay, weight_decay=args.wd), dict(params=no_decay, weight_decay=0)],
                          lr=args.lr * args.batch_size / 128, momentum=0.9, nesterov=True)

    def lr_mult(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / (total_steps - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_mult)
    ce = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, reduction="none")

    history, step, start_epoch, t0 = [], 0, 0, time.time()
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        history, step, start_epoch = ck["history"], ck["step"], ck["epoch"]
        t0 -= ck["elapsed"]
        source.seek(ck["source_epoch"])
        torch.set_rng_state(ck["rng"]["torch"].cpu())
        if ck["rng"]["cuda"] is not None and dev.type == "cuda":
            torch.cuda.set_rng_state_all([r.cpu() for r in ck["rng"]["cuda"]])
        np_rng.bit_generator.state = ck["rng"]["numpy"]
        print(f"resumed from epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        net.train()
        for x, y in loader:
            lam = float(np_rng.beta(args.regmixup_alpha, args.regmixup_alpha))
            partner = torch.randperm(len(x), device=x.device)
            x_mix = lam * x + (1 - lam) * x[partner]
            loss = sla_regmixup_loss(net, x, y, x_mix, y[partner], lam, ce, args.regmixup_eta) / len(y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
        rec = dict(epoch=epoch + 1, loss=loss.item(), time=round(time.time() - t0, 1))
        if len(va_idx) and ((epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs):
            with torch.no_grad():
                net.eval()
                rec["val_acc"] = (net(x_va).argmax(1) == y_va).float().mean().item()  # one view, head only
        history.append(rec)
        print(json.dumps(rec), flush=True)
        if args.ckpt_every and (epoch + 1) % args.ckpt_every == 0 and epoch + 1 < args.epochs:
            rng = dict(torch=torch.get_rng_state(), numpy=np_rng.bit_generator.state,
                       cuda=torch.cuda.get_rng_state_all() if dev.type == "cuda" else None)
            torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), history=history,
                            step=step, epoch=epoch + 1, source_epoch=source.epoch, rng=rng,
                            elapsed=time.time() - t0), ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)
    source.close()

    torch.save(net.state_dict(), os.path.join(out_dir, "model.pt"))
    result = dict(vars(args), params=n_params, train_time=round(time.time() - t0, 1))
    if len(va_idx):
        result["val_acc_head_vote"] = (predict_probs(net, x_va).argmax(1) == y_va.cpu()).float().mean().item()
        np.save(os.path.join(out_dir, "val_idx.npy"), va_idx)
    write_submission(predict_probs(net, x_test).argmax(1).numpy(), os.path.join(out_dir, "submission.csv"))
    json.dump(result, open(os.path.join(out_dir, "config.json"), "w"), indent=1)
    json.dump(history, open(os.path.join(out_dir, "history.json"), "w"))
    print(json.dumps({k: result[k] for k in result if k in ("name", "params", "train_time", "val_acc_head_vote")}))


if __name__ == "__main__":
    main()
