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
