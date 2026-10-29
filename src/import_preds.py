"""Rebuild a submission CSV from predictions exported on a remote GPU node (export_preds.py prints
the 2,000 test labels as comma-separated rows plus a SHA-256 of the label string)."""
import argparse
import hashlib
import os
import re

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("rows_file")
    p.add_argument("sha_prefix")
    p.add_argument("out")
    args = p.parse_args()
    rows = re.findall(r"ROW(\d+):([0-9,]+)", open(args.rows_file).read())
    s = "".join(r.replace(",", "") for _, r in sorted(rows, key=lambda t: int(t[0])))
    assert len(s) == 2000, len(s)
    assert hashlib.sha256(s.encode()).hexdigest().startswith(args.sha_prefix), "hash mismatch"
    ids = pd.read_csv(os.path.join(ROOT, "data", "sample_submission.csv"))["id"]
    sub = pd.DataFrame({"id": ids, "label": [int(c) for c in s]})
    sub.to_csv(os.path.join(ROOT, args.out), index=False)
    for ref in ["submission.csv", "submission_sla.csv"]:
        path = os.path.join(ROOT, ref)
        if os.path.exists(path):
            agree = (pd.read_csv(path)["label"].to_numpy() == sub["label"].to_numpy()).mean()
            print(f"agreement with {ref}: {agree * 100:.1f}%")
    print("wrote", args.out)
