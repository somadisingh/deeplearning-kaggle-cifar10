#!/bin/bash
# HPC queue (run from /scratch/$USER/cifar10 with: nohup bash scripts/hpc_next.sh > logs/hpc_next.log 2>&1 &).
# Every run checkpoints every 10 epochs and resumes from runs/<name>/ckpt.pt, so re-running this script in a new
# session continues where the previous job stopped and skips runs that already finished.
# Earlier queues (SAM, PAPA, Shake-Shake, anti-aliasing, 400 epochs, distillation) are done; see docs/LOGBOOK.md.
cd /scratch/$USER/cifar10/src
P=../.venv/bin/python
# Final recipe (PreActResNet + SLA+SD + TrivialAugment). The weight average is off: it trailed the live network by
# 0.4 points in the CutMix run.
SLA="--arch preact --widths 64 128 256 --depth 4 4 3 --ssl sla --batch-size 512 --label-smoothing 0.1 \
  --trivialaugment --eval-every 10 --ckpt-every 10 --resume"

run() {  # run <name> <args...>; skips finished runs
  name=$1; shift
  if [ -f ../runs/$name/config.json ]; then echo "skip $name (done)"; return; fi
  $P -u train.py --name $name "$@" >> ../logs/$name.log 2>&1
}

# Earlier queues (CutMix-300, SLA-300, per-copy, two-view, wide 80/160/320) are done; see docs/LOGBOOK.md.
# Baselines on the 45k holdout (seed-0 split), head + vote: SLA-200 97.54%, SLA-300 97.74%.
# Before the first run of this queue: bash scripts/hpc_preflight.sh (loss checks + 2-step smoke runs).
# Compare each finished run with: python scripts/compare_runs.py <baseline> <run> (paired fixes/breaks, McNemar).
# Keep a change only if it beats its baseline by >= 0.3 points (head + vote) on the same holdout.

# 1. RegMixup inside SLA (alpha 20, eta 1), 200 epochs. Batch 256 clean + 256 mixed, so each step has 2,048 rotated
#    images as in the baseline; lr scales with the clean batch (0.2). ~2.8 h. Baseline: preact_sla_ta_e200.
run preact_sla_regmix_e200 --regmixup 20 --regmixup-eta 1 --epochs 200 $SLA --batch-size 256

# 2. Distillation of 3 SLA teachers (SLA-200, SLA-300, wide SLA-200; all seed-0 split, all have 50k versions) into an
#    SLA student, both heads, alpha 0.5, T 4, 200 epochs. ~2.8 h. Baseline: preact_sla_ta_e200.
run preact_sla_kd3_e200 --teachers ../runs/preact_sla_ta_e200 ../runs/preact_sla_ta_e300 ../runs/preact_wide_sla_ta_e200 \
  --kd-alpha 0.5 --kd-temp 4 --epochs 200 $SLA

# 3. Final recipe, 600 epochs (resumes across sessions). ~4.2 h. Baseline: preact_sla_ta_e300.
run preact_sla_ta_e600 --epochs 600 $SLA

# Final: retrain the winner on all 50k with --val 0 (KD: teachers ../runs/preact_sla_ta_e200_full
# ../runs/preact_sla_ta_e300_full ../runs/preact_wide_sla_ta_e200_full) and export it with scripts/export_preds.py <run>.
