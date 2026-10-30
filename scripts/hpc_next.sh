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

# 2. PAPA: 4 PreActResNets pulled toward their mean, averaged into one network; compared with preact_ta_e200
#    (96.82%) and at about equal compute with preact_sla_ta_e200 (97.56%). ~95 min.
if [ ! -f ../runs/papa4_preact_ta_e200/config.json ]; then
  $P -u train_papa.py --name papa4_preact_ta_e200 --population 4 --epochs 200 --batch-size 512 --trivialaugment \
    --eval-every 10 --ckpt-every 10 --resume >> ../logs/papa4_preact_ta_e200.log 2>&1
fi

# 3. Shake-Shake-26 2x40d (4.56M), 600 epochs, resumable across sessions (Gastaldi: lr 0.2 per 128, wd 1e-4).
run shake2x40d_ta_e600 --arch shake --widths 40 80 160 --depth 4 4 4 --batch-size 256 --sgd-lr 0.2 --sgd-wd 1e-4 \
  --label-smoothing 0.1 --trivialaugment --epochs 600 --eval-every 20 --ckpt-every 10 --resume

# 4. Anti-aliased PreActResNet (BlurPool downsampling, 0 extra params), 200 epochs; compare with preact_ta_e200 (96.82%).
run preact_blur_ta_e200 --blur --epochs 200 $PA

# 5. PreActResNet for 400 epochs; compare with preact_ta_e200 (96.82%, 200 epochs).
run preact_ta_e400 --epochs 400 $PA

# 6. Distillation: PreActResNet student taught by the two best SLA teachers (both trained on the same 45k split,
#    so the holdout stays clean), same augmented view for teacher and student; compare with 97.56%.
run preact_kd_sla2_ta_e200 --epochs 200 --teachers ../runs/preact_sla_ta_e200 ../runs/preact_sla_ta_e200_sam05 \
  --kd-alpha 0.5 --kd-temp 4 $PA
