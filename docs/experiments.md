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
