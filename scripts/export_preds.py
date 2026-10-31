"""Print a run's test predictions as comma-separated rows plus a SHA-256, for copying back from the HPC node.

Usage (from /scratch/$USER/cifar10): python scripts/export_preds.py <run> [tag]
  tag = "" (live network), "ema", "ema_best" or "ema_e<epoch>" (weight average). SLA runs combine the softmax of the
  normal head and of the 4-rotation vote. Rebuild the CSV locally with src/import_preds.py.
"""
import hashlib, json, os, sys

import torch

run = sys.argv[1]
tag = sys.argv[2] if len(sys.argv) > 2 else ""
d = os.path.join("runs", run)
suffix = f"_{tag}" if tag else ""
p = torch.load(os.path.join(d, f"test_logits{suffix}.pt")).softmax(1)
agg = os.path.join(d, f"test_logits{suffix}_sla_agg.pt")
if os.path.exists(agg):
    p = p + torch.load(agg).softmax(1)
s = "".join(map(str, p.argmax(1).tolist()))
c = json.load(open(os.path.join(d, "config.json")))
val = c.get(f"val_acc_tta{suffix}", c.get("val_acc_tta", 0))
print("RUN", run, tag or "live", "params", c["params_total"], "val", round(100 * val, 2))
print("SHA", hashlib.sha256(s.encode()).hexdigest()[:40])
for i in range(0, 2000, 50):
    print("ROW" + str(i // 50) + ":" + ",".join(s[i:i + 50]))
