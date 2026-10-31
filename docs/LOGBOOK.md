# Logbook: Efficient ResNet for CIFAR-10 (5M parameter limit)

This logbook records all the approaches that we tried and all the results that we got.
The language follows ASD-STE100 Simplified Technical English. Sentences are short. Each sentence gives one idea.

## 1. Task and rules

The task is a Kaggle competition for a course. The rules are:

1. Train a modified ResNet. The network must have skip connections.
2. The model must have a maximum of 5,000,000 trainable parameters.
3. Train from zero. Do not use pre-trained weights.
4. Use only the CIFAR-10 training data. Do not use other datasets.
5. Do not label the test images by hand. Do not train on the test images.
6. Optimizers, augmentation, regularization, schedules and distillation are permitted.
7. The competition closes on 2026-11-06 at 23:59. The report is due on 2026-11-09.

The metric is classification accuracy. Kaggle shows a public score on half of the hidden images.
The private score on the other half sets the final rank.

## 2. Data

The `data/` folder contains these files:

| File | Contents |
|---|---|
| `train_labels.csv` | 50,000 rows: `id`, `filename`, `label` (0 to 9) |
| `train_images/` | 50,000 RGB PNG images, 32 × 32 pixels |
| `test.csv` | 2,000 rows: `id`, `filename` |
| `test_images/` | 2,000 RGB PNG images, 32 × 32 pixels |
| `sample_submission.csv` | 2,000 rows: `id`, `label` |

Each class has exactly 5,000 training images. The test IDs are 0 to 1999. They match the sample submission.

`src/prepare_data.py` decodes all PNG files one time. It writes `data/cifar10_cache.npz`.
The channel means of the training images are 0.4914, 0.4822 and 0.4465. These are the standard CIFAR-10 values.

The test set has 2,000 images. This is the same size as CIFAR-10.1 (Recht et al., 2018).
CIFAR-10.1 is a new test set that has a small distribution shift from CIFAR-10.
Models usually get 4 to 10 points less accuracy on CIFAR-10.1 than on CIFAR-10.
Our Kaggle scores agree with this. Thus, we think that the hidden set is similar to CIFAR-10.1.

## 3. Method for evaluation

We did not select models with the public Kaggle score. The public split has only about 1,000 images.
One score has a standard error of about 1.1 points. Selection on that split can overfit.

We use a holdout set from the training data. The holdout has 5,000 images: 500 from each class.
`src/train.py` makes the split with a fixed seed (`--val 5000 --seed 0`).
The model trains on the other 45,000 images.

All holdout accuracies in this logbook use test-time augmentation (TTA), unless the text says otherwise.
TTA averages the outputs for the image, its mirror, and two 1-pixel shifts.

For the final submission, we train the selected recipe again on all 50,000 images (`--val 0`).

Rule for a change: we keep a change only if it adds at least 0.3 points on the holdout.

## 4. Hardware

| Hardware | Use | Speed |
|---|---|---|
| Apple M4 Pro, MPS backend | Phase 1 and 2 | About 23 s per epoch for a 4.8M AirResNet |
| NVIDIA A100 40 GB, NYU HPC | Phase 3 to 6 | About 6.6 s per epoch for the 4.7M PreActResNet |

All data stays on the GPU. The GPU does the crop, flip and cutout operations.
TrivialAugment runs on CPU worker processes. The workers prepare the next epoch during training.

## 5. Literature research

We did two literature surveys. We used three research skill packages together:

- `academic-research-plugin` (literature-survey): search on arXiv, themes, gaps, proposals, BibTeX.
- `academic-research-skills` (deep-research): WHY / HOW / WHAT notes, method weaknesses, a devil's-advocate check.
- `scientific-agent-skills` (literature-review): search log, screening, citation check.

The full reports are in `output/literature-survey/` and in `docs/RESEARCH_REPORT.md`.

Main results from the research:

1. Accuracy on a shifted test set follows accuracy on the original test set on a straight line.
   For CIFAR-10.1: new accuracy = 1.62 × CIFAR-10 accuracy − 65.51 (Recht et al., 2018).
   Thus, each +1 point on the holdout gives about +1.6 points on the hidden set.
2. Almost no method moves a model above this line without more training data (Taori et al., 2020).
3. Cutout and wider networks help more on CIFAR-10.1 than on CIFAR-10.
4. Test-time adaptation (BN adaptation, TENT) uses the test images. The rules do not permit it.
   It also makes CIFAR-10.1 accuracy worse (Zhao et al., 2023).

## 6. Experiments in time order

### Phase 1: AirResNet on the M4 Pro (2026-10-29)

The first model is "AirResNet". It comes from airbench96 (Jordan, 2024).
It has a frozen 2×2 whitening layer, three stages of conv → max-pool → BN → GELU, and residual pairs of 3×3 convs.
The recipe uses Nesterov SGD, label smoothing 0.2, flip, 4-pixel translate, 12-pixel cutout, and a Lookahead EMA.

We tested three width and depth settings for 40 epochs:

| Run | Widths | Residual pairs | Params | Holdout |
|---|---|---|---|---|
| `w128-256-352_d111_e40` | 128/256/352 | 1/1/1 | 4.84M | **95.58%** |
| `w96-256-352_d211_e40` | 96/256/352 | 2/1/1 | 4.80M | 95.36% |
| `w96-192-384_d111_e40` | 96/192/384 | 1/1/1 | 4.34M | 95.10% |

The widest model was best. Its Kaggle public score was **0.850**.

### Phase 2: longer training and TrivialAugment (2026-10-29)

| Run | Epochs | TrivialAugment | Holdout | Kaggle |
|---|---|---|---|---|
| `w128-256-352_d111_e40` | 40 | no | 95.58% | 0.850 |
| `w128-256-352_ta_e40` | 40 | yes | 94.98% | |
| `w128-256-352_e120` | 120 | no | 96.02% | |
| `w128-256-352_ta_e120` | 120 | yes | **96.20%** | 0.860 |

Longer training added 0.4 to 0.6 points. TrivialAugment decreased accuracy at 40 epochs. It added 0.2 points at 120 epochs.

Diagnostic: we made the holdout images brighter and with more contrast. This matched the pixel statistics of the test images.
The accuracy decreased by a maximum of 0.4 points (`src/shift_check.py`). Thus, brightness and contrast do not cause the Kaggle gap.

### Phase 3: rotation self-supervision (SLA) on the A100 (2026-10-29)

SLA (Lee et al., 2020) trains the network on the 4 rotations of each image. The labels are joint labels (class × rotation), 40 classes.
A second head learns from the averaged rotation prediction (self-distillation).
At test time, we use the normal head, the 4-rotation vote, or the mean of the two.
We checked our code against the paper and the official code (hankook/SLA).

| Run | Seed | Holdout (head / vote / both) | Control without SLA | Kaggle |
|---|---|---|---|---|
| `sla_ta_e120_b1024` | 0 | 96.30% / 96.32% / **96.40%** | 95.90% (`ta_e120_b1024_a100`) | 0.863 |
| `sla_ta_e120_b1024_s1` | 1 | **96.68%** / 96.30% / – | 96.28% (`ta_e120_b1024_s1`) | 0.857 (control) |

SLA added 0.4 to 0.5 points on two seeds. But all AirResNet models stayed near 0.86 on Kaggle.

### Phase 4: standard pre-activation ResNet (2026-10-29)

All AirResNet models were about 4 points below the CIFAR-10.1 line. Thus, we changed the architecture family.

The new model is a standard pre-activation ResNet (He et al., 2016b):

- A 3×3 stem conv with 64 channels.
- Three stages with 4, 4 and 3 basic blocks. The widths are 64, 128 and 256.
- Each block is BN → ReLU → conv 3×3 → BN → ReLU → conv 3×3, plus a skip connection.
- The skip is the identity, or a 1×1 conv when the shape changes.
- Global average pooling and one linear layer.

The recipe: Nesterov SGD, lr 0.4 at batch 512, weight decay 5e-4, 5 warmup epochs, cosine schedule, label smoothing 0.1,
TrivialAugment, flip, 4-pixel translate, 12-pixel cutout, 200 epochs.

| Run | Params | Holdout | Kaggle |
|---|---|---|---|
| `preact_ta_e200` | 4,696,394 | 96.82% | **0.884** |

The holdout gain was 0.4 points. The Kaggle gain was 2.1 points. The architecture family has an effect on the shift.

### Phase 5: PreActResNet + SLA and the final model (2026-10-29 and 2026-10-30)

| Run | Data | Holdout (head / vote / both) | Kaggle |
|---|---|---|---|
| `preact_sla_ta_e200` | 45,000 | **97.56%** / 97.42% / 97.54% | |
| `preact_sla_ta_e200_full` | 50,000 | – | **0.909** |

SLA added 0.74 points on the PreActResNet. This is our best recipe.
A second seed (`preact_sla_ta_e200_s1`, seed 1, its own holdout split) got **97.78%** (head) and 97.54% (vote).
Thus, the recipe gives 97.56% to 97.78% on the holdout. The difference between seeds is about 0.2 points.
We trained it again on all 50,000 images (92 minutes on the A100). This is the final `submission.csv`.
The prediction is the mean of the softmax of the normal head and the softmax of the 4-rotation vote.

### Phase 6: other methods from the research report (2026-10-30)

We tested each method against the correct baseline. No method beat the final recipe.

| Run | Method | Holdout | Baseline | Result |
|---|---|---|---|---|
| `preact_ta_e200_sam05` | SAM, ρ = 0.05 (Foret et al., 2021) | 97.18% | 96.82% | +0.36 |
| `preact_sla_ta_e200_sam05` | SAM + SLA | 97.58% (both) | 97.54% (both) | +0.04, not kept (2× cost) |
| `papa4_preact_ta_e200` | PAPA, 4 copies (Jolicoeur-Martineau et al., 2024) | 97.18–97.20% per copy | 96.82% | below 97.56% |
| `shake2x40d_ta_e600` | Shake-Shake-26 2x40d, 600 epochs (Gastaldi, 2017) | 97.16% | 97.56% | below |
| `preact_blur_ta_e200` | Anti-aliased downsampling (Zhang, 2019) | 97.00% | 96.82% | +0.18, not kept |
| `preact_ta_e400` | 400 epochs | 97.12% | 96.82% | +0.30, below 97.56% |
| `preact_kd_sla2_ta_e200` | Distillation from 2 SLA teachers (Hinton et al., 2015) | 97.18% | 97.56% | below |

Notes:

- SAM helps the plain ResNet. SAM does not help on top of SLA.
- In the PAPA run, the averaged model got only 96.20%. The cause was a bug: the code calculated the BN statistics again on clean images.
  With BN statistics from augmented images, the averaged model got 96.94%. We changed the code. The averaged model now uses the mean of the BN statistics of the copies.
- Shake-Shake needs a long schedule. We trained it for 600 epochs over two HPC sessions with checkpoints (`--resume`).

### Phase 7: CutMix inside SLA (2026-10-31)

`preact_sla_cutmix_e200`: the final recipe with CutMix (Yun et al., 2019). With probability 0.5, each batch gets one box from a shuffled partner image.
The box goes in before the four rotations, and the losses are mixed by area. The run also kept a weight average (EMA, decay 0.999).

| Prediction | CutMix run | Final recipe (seed 0) |
|---|---|---|
| Normal head | 97.42% | 97.56% |
| 4-rotation vote | 97.52% | 97.42% |
| Weight average, normal head | 96.98% | – |

CutMix did not add accuracy. The weight average was 0.4 points worse than the live network at the end of the cosine schedule. We queued a 300-epoch CutMix run for the next session.

`preact_sla_cutmix_e300`: the same CutMix run for 300 epochs (seed 0), without the weight average. Training time was 124 minutes on the A100.

| Prediction | CutMix, 300 epochs | Final recipe (seed 0, 200 epochs) |
|---|---|---|
| Normal head | 97.66% | 97.56% |
| 4-rotation vote | 97.64% | 97.42% |
| Head + vote | 97.70% | – |

The gain is +0.10 points. This is less than our limit of +0.3 points (the seed spread is approximately 0.2 points). Thus, we did not change the final model.
We saved the predictions in `submissions/submission_preact_sla_cutmix_e300.csv`. We did the 300-epoch run without CutMix in the next HPC session (see below).

### Phase 8: Per-copy augmentation and 300 epochs (2026-10-31)

We did two more runs on the A100 with seed 0:

- `preact_sla_percopy_e200`: each of the four rotated copies of an image gets its own random augmentation.
- `preact_sla_ta_e300`: the final recipe for 300 epochs, without CutMix. This run is the control for the CutMix run.

| Prediction | Per-copy, 200 epochs | 300 epochs | CutMix, 300 epochs | Final recipe, 200 epochs |
|---|---|---|---|---|
| Normal head | 97.58% | 97.68% | 97.66% | 97.56% |
| 4-rotation vote | 97.52% | 97.68% | 97.64% | 97.42% |
| Head + vote | 97.66% | 97.74% | 97.70% | – |

Per-copy augmentation did not change the accuracy. Thus, we did not do the two-view run.
The 300-epoch runs got approximately +0.1 points, with or without CutMix. Thus, CutMix does not add accuracy.
All gains are less than our limit of +0.3 points. We did not change the final model.
We saved the predictions of the two runs in `submissions/`.

## 7. Distillation result

The student is a PreActResNet without SLA. The teachers are `preact_sla_ta_e200` and `preact_sla_ta_e200_sam05`.
Both teachers trained on the same 45,000 images. Thus, the holdout stays clean.
The teacher and the student see the same augmented image (Beyer et al., 2022). The loss is 0.5 × cross-entropy + 0.5 × KL divergence at temperature 4.

Result: the student got **97.18%** on the holdout. This is +0.36 points more than the same ResNet without distillation (96.82%).
But it is 0.38 points less than the teachers' recipe (PreActResNet + SLA, 97.56%). Thus, we did not keep it.

## 8. Kaggle public scores

| Submission | Model | Holdout | Kaggle public |
|---|---|---|---|
| 1 | AirResNet, 40 epochs | 95.58% | 0.850 |
| 2 | AirResNet + TrivialAugment, 120 epochs | 96.20% | 0.860 |
| 3 | AirResNet + SLA, 120 epochs | 96.40% | 0.863 |
| 4 | AirResNet, seed 1, 120 epochs | 96.28% | 0.857 |
| 5 | PreActResNet, 200 epochs | 96.82% | 0.884 |
| 6 | PreActResNet + SLA, all 50k (final) | 97.56% (45k version) | **0.909** |

## 9. Final model

| Item | Value |
|---|---|
| Architecture | Pre-activation ResNet, stages 4/4/3 blocks, widths 64/128/256, average pooling |
| Heads | 10-class head + 40-class joint head (class × rotation) |
| Trainable parameters (torchsummary, forward pass) | 4,696,394 |
| Trainable parameters (all, with the 40-class head) | 4,706,674 |
| Training data | All 50,000 CIFAR-10 training images |
| Training | 200 epochs, batch 512, Nesterov SGD, cosine schedule, SLA+SD, TrivialAugment, cutout |
| Test-time augmentation | Mirror and 1-pixel shifts; normal head + 4-rotation vote |
| Holdout accuracy (same recipe on 45,000 images) | 97.56% |
| Kaggle public accuracy | 0.909 |

## 10. Lessons

1. Do not select models with the public leaderboard. Use a holdout from the training data.
2. The architecture family has an effect on accuracy under shift. A fast-training airbench net was 1.4 points worse on Kaggle than a standard ResNet with the same holdout accuracy.
3. SLA (rotation labels) gave the largest gain for its cost: +0.4 to +0.7 points.
4. SAM, PAPA, distillation, longer schedules and anti-aliasing each gave +0.2 to +0.4 points on the plain ResNet. None of them beat SLA.
5. Calculate BN statistics on data that is similar to the training data. Clean-image statistics decreased the PAPA model by 1 point.
6. Make long runs resumable. HPC jobs stop after 3 to 5 hours.
