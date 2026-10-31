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

# Baseline to beat: preact_sla_ta_e200 = 97.56% (seed 0), preact_sla_ta_e200_s1 = 97.78% (seed 1).
# Done: preact_sla_cutmix_e200 (CutMix, 200 epochs) = 97.42% head / 97.52% vote, a tie.
# Ranked on the latest results (2026-10-31): longer training is the only change with our own evidence (+0.30 for 400
# epochs without SLA), and CutMix usually gains more from long schedules, so the two 300-epoch runs come first.

# 1. CutMix inside SLA, 300 epochs (fresh cosine schedule). ~2.1 h.
run preact_sla_cutmix_e300 --cutmix 0.5 --epochs 300 $SLA

# 2. Final recipe, 300 epochs (control for #1, and the SLA-300 test). ~2.1 h.
run preact_sla_ta_e300 --epochs 300 $SLA

# 3. Per-copy augmentation, 200 epochs (~85 min). Enable when approved:
# run preact_sla_percopy_e200 --sla-percopy --epochs 200 $SLA
# 4. Two views + JS consistency (~2.8 h), only if #3 reaches 97.76%:
# run preact_sla2v_ta_e200 --two-view --cons-weight 12 --epochs 200 --arch preact --widths 64 128 256 --depth 4 4 3 \
#   --ssl sla --batch-size 256 --label-smoothing 0.1 --trivialaugment --eval-every 10 --ckpt-every 10 --resume
# Removed: baseline + weight average (the average trailed the live network by 0.4 in the CutMix run).
# Final: retrain the winner on all 50k with --val 0 and export its CSV with scripts/export_preds.py <run>.
