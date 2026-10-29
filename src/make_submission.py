"""Write submission.csv from one run's saved test logits (TTA already applied)."""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--out", default=os.path.join(ROOT, "submission.csv"))
    args = p.parse_args()
    run_dir = os.path.join(ROOT, "runs", args.run)
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    assert cfg["params_total"] <= 5_000_000, "final model must have at most 5M parameters"
    logits = torch.load(os.path.join(run_dir, "test_logits.pt"))
    test = pd.read_csv(os.path.join(ROOT, "data", "test.csv"))
    sample = pd.read_csv(os.path.join(ROOT, "data", "sample_submission.csv"))
    ids = np.load(os.path.join(ROOT, "data", "cifar10_cache.npz"))["test_ids"]
    assert (ids == test["id"].to_numpy()).all() and len(logits) == len(test)
    sub = pd.DataFrame({"id": ids, "label": logits.argmax(1).numpy().astype(int)})
    assert list(sub["id"]) == list(sample["id"]) and sub["label"].between(0, 9).all()
    sub.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(sub)} rows, class counts {np.bincount(sub['label'], minlength=10).tolist()}")
