#!/bin/bash
# HPC queue (run from /scratch/$USER/cifar10 with: nohup bash scripts/hpc_next.sh > logs/hpc_next.log 2>&1 &).
# Every run checkpoints every 10 epochs and resumes from runs/<name>/ckpt.pt, so re-running this script in a new
# session continues where the previous job stopped and skips runs that already finished.
# Earlier queues (SAM, PAPA, Shake-Shake, anti-aliasing, 400 epochs, distillation) are done; see docs/LOGBOOK.md.
cd /scratch/$USER/cifar10/src
P=../.venv/bin/python
# Final recipe (PreActResNet + SLA+SD + TrivialAugment) with a weight average (EMA 0.999) evaluated beside it.
SLA="--arch preact --widths 64 128 256 --depth 4 4 3 --ssl sla --batch-size 512 --label-smoothing 0.1 \
  --trivialaugment --eval-every 10 --ckpt-every 10 --resume --ema 0.999"

run() {  # run <name> <args...>; skips finished runs
  name=$1; shift
  if [ -f ../runs/$name/config.json ]; then echo "skip $name (done)"; return; fi
  $P -u train.py --name $name "$@" >> ../logs/$name.log 2>&1
}

# Baseline to beat: preact_sla_ta_e200 = 97.56% (seed 0), preact_sla_ta_e200_s1 = 97.78% (seed 1).

# 1. CutMix inside SLA: box pasted before the rotations, losses mixed by area, p = 0.5 per batch. ~85 min.
run preact_sla_cutmix_e200 --cutmix 0.5 --epochs 200 $SLA

# Paused by request (2026-10-31): only the CutMix run is queued for now. The other prepared runs:
# run preact_sla_percopy_e200 --sla-percopy --epochs 200 $SLA                      # per-copy augmentation, ~85 min
# run preact_sla_ta_e300 --epochs 300 $SLA                                          # SLA-300, ~2.1 h
# two-view consistency: --two-view --cons-weight 12 --batch-size 256 (see docs/experiments.md), ~2.8 h
