"""Photometric-shift check: re-render the 5k training holdout with the evaluation set's per-channel
mean/std (unlabeled pixel statistics only) and measure each saved model's accuracy drop."""
import sys, os, json, numpy as np, torch
sys.path.insert(0, "/Users/somaditya/dl-kaggle/cifar10/src")
from train import load_run, predict, split, MEAN, STD
dev = "mps"
d = np.load("/Users/somaditya/dl-kaggle/cifar10/data/cifar10_cache.npz")
_, va = split(50000, 5000, 0)
x = torch.from_numpy(d["x_train"][va]).permute(0, 3, 1, 2).float().div(255).to(dev)
y = torch.from_numpy(d["y_train"][va]).to(dev)
tr = d["x_train"].astype(np.float64) / 255; te = d["x_test"].astype(np.float64) / 255
m_tr, s_tr = [torch.tensor(a, dtype=torch.float32, device=dev).view(1, 3, 1, 1) for a in (tr.mean((0, 1, 2)), tr.std((0, 1, 2)))]
m_te, s_te = [torch.tensor(a, dtype=torch.float32, device=dev).view(1, 3, 1, 1) for a in (te.mean((0, 1, 2)), te.std((0, 1, 2)))]
del tr, te
def shift(x, k):  # k=1 matches the evaluation-set statistics, k=2 doubles the shift
    m = m_tr + k * (m_te - m_tr); s = s_tr * (1 + k * (s_te / s_tr - 1))
    return ((x - m_tr) / s_tr * s + m).clamp(0, 1)
def per_image_brightness(x, sd):  # widen per-image brightness spread (test 0.144 vs train 0.125)
    g = torch.randn(len(x), 1, 1, 1, device=dev, generator=torch.Generator(dev).manual_seed(0)) * sd
    return (x + g).clamp(0, 1)
norm = lambda z: (z - MEAN.to(dev)) / STD.to(dev)
conds = {"clean": x, "test-stats x1": shift(x, 1), "test-stats x2": shift(x, 2),
         "brightness spread +0.07": per_image_brightness(x, 0.07), "contrast x1.15": ((x - x.mean((1,2,3), keepdim=True)) * 1.15 + x.mean((1,2,3), keepdim=True)).clamp(0,1)}
runs = sys.argv[1:]
print("model".ljust(28), *[c.rjust(24) for c in conds])
for r in runs:
    net = load_run(f"/Users/somaditya/dl-kaggle/cifar10/runs/{r}", dev)
    accs = [(predict(net, norm(c), tta=2).argmax(1) == y).float().mean().item() * 100 for c in conds.values()]
    print(r.ljust(28), *[f"{a:24.2f}" for a in accs], flush=True)
