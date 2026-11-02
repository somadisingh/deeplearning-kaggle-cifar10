"""Checks of the RegMixup and SLA distillation losses in src/train.py, with no training.

Usage (from the repo root or /scratch/$USER/cifar10): python scripts/check_losses.py [runs dir]
The runs dir (default runs) must hold the teachers preact_sla_ta_e200, preact_sla_ta_e300 and preact_wide_sla_ta_e200.
Runs on the CPU in about a minute. Every check prints PASS or stops with an error.
"""
import copy, os, sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import train
from models import PreActResNet

RUNS = sys.argv[1] if len(sys.argv) > 1 else "runs"
TEACHERS = ["preact_sla_ta_e200", "preact_sla_ta_e300", "preact_wide_sla_ta_e200"]
torch.manual_seed(0)
args = SimpleNamespace(ssl="sla", two_view=False, sla_percopy=False, sla_beta=1.0, kd_alpha=0.5, kd_temp=4.0,
                       regmixup_eta=1.0)
ce = nn.CrossEntropyLoss(label_smoothing=0.1, reduction="none")
net = PreActResNet((8, 16, 32), (1, 1, 1), ssl="sla").eval()  # eval: BN uses fixed statistics, so calls agree
B = 16
x, y = torch.randn(B, 3, 32, 32), torch.randint(0, 10, (B,))
close = lambda a, b: torch.allclose(a, b, rtol=1e-5, atol=1e-5)


def reference(xs, ys, ys_b=None, lam=1.0):
    """SLA+SD per image, written out from Lee et al. (2020) with a mixed label pair (ys, ys_b, lam)."""
    n = len(xs)
    z = net.features(train.rotate4(xs))
    joint, single = net.joint_head(z), net.head(z[::4])
    r = torch.arange(4).repeat(n)
    jl = lambda yy: F.cross_entropy(joint, yy.repeat_interleave(4) * 4 + r, reduction="none").view(n, 4).mean(1)
    joint_loss, plain = jl(ys), ce(single, ys)
    if ys_b is not None:
        joint_loss = lam * joint_loss + (1 - lam) * jl(ys_b)
        plain = lam * plain + (1 - lam) * ce(single, ys_b)
    agg = torch.stack([joint.view(n, 4, 10, 4)[:, k, :, k] for k in range(4)], 1).mean(1).detach()
    sd = (F.softmax(agg, 1) * (F.log_softmax(agg, 1) - F.log_softmax(single, 1))).sum(1)
    return joint_loss + plain, sd, joint, single


# 1. Without the new options, sla_loss is the SLA+SD loss.
hard, sd, _, _ = reference(x, y)
assert close(train.sla_loss(net, x, y, args, ce), (hard + sd).sum())
print("PASS 1: sla_loss without RegMixup or teachers = SLA+SD loss")

# 2. RegMixup = clean SLA+SD + eta * mixed loss with both labels; self-distillation on the clean half only.
perm, lam = torch.randperm(B), 0.37
xm = lam * x + (1 - lam) * x[perm]
hard_m, _, _, _ = reference(xm, y, y[perm], lam)
for eta in (1.0, 0.5):
    args.regmixup_eta = eta
    got = train.sla_loss(net, x, y, args, ce, x_mix=xm, partner_m=perm, lam_m=lam)
    assert close(got, (hard + sd).sum() + eta * hard_m.sum()), (got, eta)
args.regmixup_eta = 1.0
print("PASS 2: RegMixup loss matches the paper's eq. 3 on top of SLA+SD (eta 1 and 0.5)")

# 3. With lam = 1 and the identity partner, the mixed half is a copy of the clean half.
got = train.sla_loss(net, x, y, args, ce, x_mix=x.clone(), partner_m=torch.arange(B), lam_m=1.0)
assert close(got, (hard + sd).sum() + hard.sum())
print("PASS 3: lam = 1 gives clean loss + a second clean hard loss")

# 4. Distillation from the student itself: the KD term is zero, the loss is (1 - alpha) * hard + SD.
xr = train.rotate4(x)
targets = train.sla_teacher_targets([copy.deepcopy(net)], xr, args.kd_temp)
got = train.sla_loss(net, x, y, args, ce, kd_targets=targets)
assert close(got, ((1 - args.kd_alpha) * hard + sd).sum())
# Hand-written KD against a different teacher, for both heads.
other = PreActResNet((8, 16, 32), (1, 1, 1), ssl="sla").eval()
pj, pp = train.sla_teacher_targets([other], xr, args.kd_temp)
_, _, joint, single = reference(x, y)
T = args.kd_temp
kl = lambda s, p: (p * (p.log() - F.log_softmax(s / T, 1))).sum(1) * T * T
kd = kl(joint, pj).view(B, 4).mean(1) + kl(single, pp)
got = train.sla_loss(net, x, y, args, ce, kd_targets=(pj, pp))
assert close(got, ((1 - args.kd_alpha) * hard + args.kd_alpha * kd + sd).sum())
print("PASS 4: KD term is 0 for a self-teacher and matches a hand-written KL on both heads")

# 5. Teacher targets: probabilities, the mean over teachers, no change to the teachers' BN or weights.
two = [copy.deepcopy(net), other]
before = [copy.deepcopy(t.state_dict()) for t in two]
pj2, pp2 = train.sla_teacher_targets(two, xr, T)
assert pj2.shape == (4 * B, 40) and pp2.shape == (B, 10)
assert close(pj2.sum(1), torch.ones(4 * B)) and close(pp2.sum(1), torch.ones(B))
assert close(pp2, 0.5 * (targets[1] + pp))
assert all(all(torch.equal(a, b) for a, b in zip(s.values(), t.state_dict().values())) for s, t in zip(before, two))
assert not pj2.requires_grad
print("PASS 5: teacher targets are averaged probabilities; teachers are unchanged and get no gradient")

# 6. Gradients flow into the student with RegMixup and KD together.
s = PreActResNet((8, 16, 32), (1, 1, 1), ssl="sla").train()
xin = torch.cat([x, xm])
loss = train.sla_loss(s, x, y, args, ce, x_mix=xm, partner_m=perm, lam_m=lam,
                      kd_targets=train.sla_teacher_targets([other], train.rotate4(xin), T))
loss.backward()
assert torch.isfinite(loss) and all(p.grad is not None for p in s.parameters())
print("PASS 6: RegMixup + KD loss is finite and every student parameter gets a gradient")

# 7. The real teachers: their saved holdout equals split(seed 0), and their targets on holdout images are right.
d = np.load(os.path.join("data", "cifar10_cache.npz"))
_, va = train.split(50000, 5000, 0)
xs = torch.from_numpy(d["x_train"][va[:200]]).permute(0, 3, 1, 2).float().div(255)
xs, ys = (xs - train.MEAN) / train.STD, torch.from_numpy(d["y_train"][va[:200]])
for name in TEACHERS:
    r = os.path.join(RUNS, name)
    assert np.array_equal(np.load(os.path.join(r, "val_idx.npy")), va), f"{name} holdout differs from split(seed 0)"
    t = train.load_run(r, "cpu")
    pj, pp = train.sla_teacher_targets([t], train.rotate4(xs), 1.0)
    acc = (pp.argmax(1) == ys).float().mean().item()
    joint_acc = (pj.argmax(1) == ys.repeat_interleave(4) * 4 + torch.arange(4).repeat(200)).float().mean().item()
    assert acc > 0.95 and joint_acc > 0.9, (name, acc, joint_acc)
    print(f"  {name}: plain head {100 * acc:.1f}%, joint head {100 * joint_acc:.1f}% on 200 holdout images")
print("PASS 7: teachers use the student's holdout split (seed 0) and their targets are correct")
print("ALL CHECKS PASSED")
