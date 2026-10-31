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

# Ranked training priorities (2026-10-31). Only #1 is enabled; uncomment the next one when it is approved.
# 2. Baseline recipe + weight average on 45k (~85 min): does the weight average beat the live network?
# run preact_sla_ta_e200_ema --epochs 200 $SLA
# 3. Per-copy augmentation (~85 min)
# run preact_sla_percopy_e200 --sla-percopy --epochs 200 $SLA
# 4. SLA-300 (~2.1 h)
# run preact_sla_ta_e300 --epochs 300 $SLA
# 5. Two views + JS consistency (~2.8 h), only if #3 reaches 97.76%:
# run preact_sla2v_ta_e200 --two-view --cons-weight 12 --epochs 200 --arch preact --widths 64 128 256 --depth 4 4 3 \
#   --ssl sla --batch-size 256 --label-smoothing 0.1 --trivialaugment --eval-every 10 --ckpt-every 10 --resume --ema 0.999
# Final: retrain the winner on all 50k with --val 0 --ema-epoch <best weight-average epoch on 45k>, and export two
# CSVs (live network and weight average) with export_preds.py <run> and export_preds.py <run> ema_e<epoch>.
