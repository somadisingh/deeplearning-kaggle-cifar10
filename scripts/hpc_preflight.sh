#!/bin/bash
# Checks before the queue in scripts/hpc_next.sh (run from /scratch/$USER/cifar10). No real training: it runs the loss
# checks, then each new configuration for 2 steps with a tiny network, then deletes the smoke runs. Stops on any error.
set -euo pipefail
cd /scratch/$USER/cifar10
P=.venv/bin/python
for r in preact_sla_ta_e200 preact_sla_ta_e300 preact_wide_sla_ta_e200 preact_sla_ta_e200_full preact_sla_ta_e300_full \
  preact_wide_sla_ta_e200_full; do
  test -f runs/$r/model.pt -a -f runs/$r/config.json || { echo "missing teacher runs/$r"; exit 1; }
done
$P scripts/check_losses.py runs
cd src
TINY="--arch preact --widths 8 16 32 --depth 1 1 1 --ssl sla --batch-size 512 --label-smoothing 0.1 --trivialaugment \
  --epochs 0.02 --eval-every 1"
$P -u train.py --name smoke_regmix $TINY --regmixup 20 --batch-size 256
$P -u train.py --name smoke_kd3 $TINY --teachers ../runs/preact_sla_ta_e200 ../runs/preact_sla_ta_e300 \
  ../runs/preact_wide_sla_ta_e200
$P -u train.py --name smoke_kd3_full $TINY --val 0 --teachers ../runs/preact_sla_ta_e200_full \
  ../runs/preact_sla_ta_e300_full ../runs/preact_wide_sla_ta_e200_full
# A 45k teacher with a 50k student must be refused.
if $P -u train.py --name smoke_bad $TINY --val 0 --teachers ../runs/preact_sla_ta_e200 > ../logs/smoke_bad.log 2>&1 \
  || ! grep -q "holdout split" ../logs/smoke_bad.log; then
  echo "FAIL: a 45k teacher was not refused for a 50k student (see logs/smoke_bad.log)"; exit 1; fi
rm -rf ../runs/smoke_regmix ../runs/smoke_kd3 ../runs/smoke_kd3_full ../runs/smoke_bad
echo "PREFLIGHT PASSED"
