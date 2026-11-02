"""Paired comparison of two runs on the same holdout, with the submitted predictor (head + 4-rotation vote).

Usage (from the repo root or /scratch/$USER/cifar10): python scripts/compare_runs.py <baseline run> <new run>
Prints both accuracies, the images the new run fixes and breaks, and an exact two-sided McNemar p-value.
Stops if the two runs were not evaluated on the same holdout images.
"""
import math, os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from predictor import load_probs

y_all = np.load(os.path.join("data", "cifar10_cache.npz"))["y_train"]
runs = [os.path.join("runs", r) for r in sys.argv[1:3]]
idx = [np.load(os.path.join(r, "val_idx.npy")) for r in runs]
assert np.array_equal(idx[0], idx[1]), "the two runs used different holdout images"
y = y_all[idx[0]]
ok = [load_probs(r, "val").argmax(1).numpy() == y for r in runs]
fixes, breaks = int((~ok[0] & ok[1]).sum()), int((ok[0] & ~ok[1]).sum())
k, n = min(fixes, breaks), fixes + breaks
p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
for r, o in zip(sys.argv[1:3], ok):
    print(f"{r}: {100 * o.mean():.2f}% (head + vote, {len(y)} holdout images)")
print(f"new run fixes {fixes}, breaks {breaks}, net {fixes - breaks:+d} images "
      f"({100 * (fixes - breaks) / len(y):+.2f} points), McNemar p = {p:.3f}")
