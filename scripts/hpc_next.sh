#!/bin/bash
# Next HPC session queue (run from /scratch/$USER/cifar10 with: nohup bash scripts/hpc_next.sh > logs/hpc_next.log 2>&1 &).
# Every run checkpoints every 10 epochs and resumes from runs/<name>/ckpt.pt, so re-running this script in a new
# session continues where the previous job stopped and skips runs that already finished.
cd /scratch/$USER/cifar10/src
P=../.venv/bin/python
PA="--arch preact --widths 64 128 256 --depth 4 4 3 --batch-size 512 --label-smoothing 0.1 --trivialaugment --eval-every 10 --ckpt-every 10 --resume"

run() {  # run <name> <args...>; skips finished runs
  name=$1; shift
  if [ -f ../runs/$name/config.json ]; then echo "skip $name (done)"; return; fi
  $P -u train.py --name $name "$@" >> ../logs/$name.log 2>&1
}

# 1. SAM + SLA+SD on the PreActResNet, 45k/5k holdout, compared with preact_sla_ta_e200 (97.56%). ~2.8 h.
run preact_sla_ta_e200_sam05 --ssl sla --sam-rho 0.05 --epochs 200 $PA
