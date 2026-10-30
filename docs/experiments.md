# Experiment log

All runs: AirResNet (src/models.py), airbench96-style recipe (src/train.py defaults), batch 1024,
label smoothing 0.2, flip + 4px translate + 12px cutout, Lookahead EMA. Validation is a fixed
stratified 5,000-image holdout from the training set (seed 0); models trained on the other 45,000.
"TTA" = mirror + 1px translate test-time augmentation. Hardware: M4 Pro, MPS, fp32.

## Sweep 1: architecture at 40 epochs (2026-10-29)

| run | widths | res pairs/stage | params | val acc | val acc (TTA) | train time |
|---|---|---|---|---|---|---|
| w128-256-352_d111_e40 | 128/256/352 | 1/1/1 | 4.84M | 94.90% | **95.58%** | 13.6 min |
| w96-256-352_d211_e40 | 96/256/352 | 2/1/1 | 4.80M | 95.06% | 95.36% | 13.2 min |
| w96-192-384_d111_e40 | 96/192/384 | 1/1/1 | 4.34M | 94.60% | 95.10% | 9.1 min |

## Sweep 2: TrivialAugment and longer training (2026-10-29)

Same 4.84M model (widths 128/256/352, 1 residual pair per stage) and recipe as sweep 1.
TrivialAugment (Müller & Hutter, 2021) is applied per image on CPU worker processes before
flip / translate / cutout; epoch time is unchanged (~20 s).

| run | epochs | TrivialAugment | val acc | val acc (TTA) | train time |
|---|---|---|---|---|---|
| w128-256-352_d111_e40 | 40 | no | 94.90% | 95.58% | 13.6 min |
| w128-256-352_ta_e40 | 40 | yes | 94.06% | 94.98% | 13.6 min |
| w128-256-352_e120 | 120 | no | 95.42% | 96.02% | 40.6 min |
| w128-256-352_ta_e120 | 120 | yes | 95.64% | **96.20%** | 40.9 min |

Longer training is worth +0.4 to +0.6 points. TrivialAugment hurts at 40 epochs and helps slightly at
120 (+0.2, within one standard error of a 5k holdout, ~0.27 points), so it needs long schedules.

## Diagnostic: photometric shift check (2026-10-29)

`src/shift_check.py` re-renders the 5k holdout with the evaluation set's per-channel mean/std
(unlabeled pixel statistics only; test images are not classified) and measures the accuracy drop (TTA on).

| model | clean | test stats ×1 | test stats ×2 | per-image brightness spread +0.07 | contrast ×1.15 |
|---|---|---|---|---|---|
| w128-256-352_d111_e40 | 95.58 | 95.46 | 95.16 | 95.42 | 95.36 |
| w96-256-352_d211_e40 | 95.36 | 95.14 | 95.08 | 95.06 | 95.04 |
| w128-256-352_e120 | 96.02 | 95.94 | 95.68 | 95.78 | 95.78 |
| w128-256-352_ta_e120 | 96.20 | 96.16 | 96.16 | 95.96 | 96.00 |

Drops are at most 0.4 points, so brightness/contrast differences do not explain the Kaggle gap;
the frozen whitening stem is not fragile to this shift. TrivialAugment makes the model the most stable.

SAM smoke test (TrivialAugment recipe, 2 epochs, rho 0.05): 46 s/epoch vs 23 s for SGD (1.97×);
holdout accuracy after 2 epochs 39.7% vs 48.7% for SGD.

SLA+SD smoke test (TrivialAugment recipe, 2 epochs): batch 1024 (4096 rotated images per step) runs out of
memory on the 24 GB M4 Pro and swaps; batch 512 works at 95 s/epoch (4.1× SGD). After 2 epochs the plain
(self-distilled) head reaches 58.2% and the 4-rotation aggregate 53.7% on the holdout (SGD smoke at batch 1024: 48.7%,
not directly comparable because the batch size differs). Model has 4.86M params.

## Sweep 3: SLA+SD on an NYU HPC A100 (2026-10-29)

Same 4.84M model + TrivialAugment, 120 epochs, batch 1024, 45k/5k split (seed 0), run on an A100-SXM4-40GB
(torch 2.15+cu132). SLA+SD (Lee et al., 2020) adds a 40-way joint (class × rotation) head; the model has 4.86M params.
The control is the same recipe without SLA on the same GPU.

| run | val acc (TTA) | train time |
|---|---|---|
| ta_e120_b1024_a100 (control) | 95.90% | 4.0 min |
| sla_ta_e120_b1024: self-distilled head | 96.30% | 12.4 min |
| sla_ta_e120_b1024: 4-rotation aggregate | 96.32% | |
| sla_ta_e120_b1024: head + aggregate (mean of softmaxes) | **96.40%** | |

SLA+SD beats the same-GPU control by +0.4 to +0.5 points. The Mac run of the control recipe scored 96.20%,
so run-to-run variation is about ±0.3; the gain is suggestive, not certain. `submission_sla.csv` holds the
head + aggregate predictions of this 45k-trained model.

## Sweep 4: standard pre-activation ResNet (2026-10-29, A100)

PreActResNet (3×3 stem, pre-activation basic blocks, stages 4/4/3 at widths 64/128/256, global average
pooling; 4.70M params) trained with plain Nesterov SGD (lr 0.4 at batch 512, wd 5e-4, 5-epoch warmup,
cosine), label smoothing 0.1, TrivialAugment + flip + 4px translate + 12px cutout, 200 epochs, 22 min.

| run | holdout val (TTA) | Kaggle public | Recht line prediction |
|---|---|---|---|
| w128-256-352_d111_e40 (AirResNet) | 95.58% | 0.85 | 89.3% |
| w128-256-352_ta_e120 (AirResNet) | 96.20% | 0.86 | 90.3% |
| sla_ta_e120_b1024 (AirResNet + SLA) | 96.40% | 0.863 | 90.7% |
| ta_e120_b1024_s1 (AirResNet, seed 1) | 96.28% | 0.857 | 90.5% |
| preact_ta_e200 (PreActResNet) | **96.82%** | **0.884** | 91.3% |

| preact_sla_ta_e200 (PreActResNet + SLA+SD, 4.71M) | **97.56%** (head; aggregate 97.42%) | pending | 92.5% |

The airbench-style models sat about 4 points below the CIFAR-10.1 line; the standard ResNet sits about 3 below,
so it gains more on the hidden set (+2.1) than its holdout gain (+0.4) predicts. The architecture family
matters for this shift, as Recht et al. also found for width vs depth and Cutout.

Seed 1 of the AirResNet SLA comparison (different holdout split, seed 1): SLA+SD self-distilled head 96.68%,
rotation aggregate 96.30%; control without SLA 96.28%. Across two seeds SLA adds +0.4 to +0.5 on the holdout.

## Resume list (HPC job ends ~2026-10-29 23:50; continue in the next A100 session)

State on the A100 (persists in `/scratch/ss20288/cifar10`): `.venv` (torch 2.15+cu132), `data/cifar10_cache.npz`
(hash verified), `src/` (current code), `runs/` and `logs/` for every A100 run, `export_preds.py`.
Archives of finished runs are in `~/cifar10_results/` (with `SHA256SUMS`).

Next steps, in order:
1. ~~Copy back the PreActResNet + SLA run~~ done: 97.56% holdout, `submission_preact_sla_ta_e200.csv`; it is the new winner.
2. Retrain the winning recipe (PreActResNet + SLA+SD + TrivialAugment, 200 epochs, batch 512) on all 50k images
   (`--val 0`, ~95 min on the A100) and write the final `submission.csv`.
3. Seeds of the winning PreActResNet recipe to confirm the choice on the holdout.
4. Possible improvements on the PreActResNet: SAM at equal wall-clock (implemented, `--sam-rho`), width/depth variants
   under 5M, longer schedule.

## Final retrain on all 50k (2026-10-30, A100)

`preact_sla_ta_e200_full`: PreActResNet + SLA+SD + TrivialAugment, 200 epochs, batch 512, all 50,000 training
images (`--val 0`), 92 min. Predictions (head + rotation aggregate) are `submission.csv`
(= `submission_preact_sla_ta_e200_full.csv`); they agree with the 45k-trained version on 96.2% of test images.
**Kaggle public: 0.909** (up from 0.884 for the 45k PreActResNet without SLA).

## SAM on the PreActResNet (2026-10-30, A100)

`preact_ta_e200_sam05`: same recipe as `preact_ta_e200` (96.82%) plus SAM, rho 0.05, ascent on the full batch
of 512; 200 epochs, 13.6 s/epoch (2x SGD), 44 min. Holdout (TTA): **97.18%** (+0.36), one seed.
Meets the +0.3 bar for one seed; a second seed is needed before stacking SAM on SLA (~3 h per 200-epoch run).
CSV: `submission_preact_ta_e200_sam05.csv` (45k-trained, no SLA).

## Resume list (after HPC session 2, 2026-10-30)

Done: all-50k final retrain (Kaggle 0.909, `submission.csv`); SAM one seed (+0.36).
Next, in order:
1. SAM + SLA on PreActResNet (user approved stacking): `scripts/hpc_next.sh` (resumable; started at the end of
   session 2, ~20 epochs done). Re-run the script next session; compare with 97.56%; if it wins, retrain on all 50k.
2. PAPA (`src/train_papa.py`, p=4, PreActResNet, no SLA first, ~90 min); compare at equal compute with 97.56%.
3. Shake-Shake-26 2x40d pilot (`--arch shake --widths 40 80 160 --depth 4 4 4 --sgd-lr 0.2 --sgd-wd 1e-4
   --batch-size 128`): measure epoch time, then 600 epochs with `--resume` across jobs.
4. Anti-aliasing (`--blur`) and 400-epoch / wider variants, one run each.
5. Retrain whatever wins on all 50k for the final submission.
Rule: stack only changes that win by >= 0.3 on the holdout across 2 seeds.
