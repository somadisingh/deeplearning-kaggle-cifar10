"""The one predictor used for validation and for every submission: the softmax of the plain head, combined with
the softmax of the 4-rotation vote (SLA+AG) for SLA runs. Runs without a vote use the plain head."""
import os

import torch

HEAD_WEIGHT = 0.5  # fixed on 2026-10-31; averaged over three 45k SLA runs, 0.25 to 0.75 differ by <= 0.02 points


def combine(head_logits, vote_logits=None, head_weight=HEAD_WEIGHT):
    p = head_logits.float().softmax(1)
    if vote_logits is None:
        return p
    return head_weight * p + (1 - head_weight) * vote_logits.float().softmax(1)


def load_probs(run_dir, split="test", tag=""):
    """Combined probabilities from a run's saved logits. split = "test" or "val"; tag = "" (live network),
    "ema", "ema_best" or "ema_e<epoch>"."""
    suffix = f"_{tag}" if tag else ""
    head = torch.load(os.path.join(run_dir, f"{split}_logits{suffix}.pt"), map_location="cpu")
    vote_path = os.path.join(run_dir, f"{split}_logits{suffix}_sla_agg.pt")
    vote = torch.load(vote_path, map_location="cpu") if os.path.exists(vote_path) else None
    return combine(head, vote)
