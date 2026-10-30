# Efficient ResNet for CIFAR-10 (≤ 5M parameters)

Course Kaggle challenge: train a modified ResNet from scratch on the CIFAR-10 training set (50,000 images) with at
most 5,000,000 trainable parameters, and predict 2,000 held-out evaluation images.

## Result

| Item | Value |
|---|---|
| Model | Pre-activation ResNet, 3 stages of 4 / 4 / 3 basic blocks, widths 64 / 128 / 256, global average pooling |
| Training method | SLA+SD rotation self-supervision (Lee et al., 2020), TrivialAugment, Cutout, Nesterov SGD, cosine schedule, 200 epochs |
| Trainable parameters | **4,696,394** (torchsummary); 4,706,674 including the 40-way SLA head used in training and in the rotation vote |
| Holdout accuracy (same recipe, 45k train / 5k holdout) | **97.56%** (normal head), 97.54% (head + rotation vote) |
| Kaggle public accuracy (trained on all 50k) | **0.909** |
| Final submission | `submission.csv` (also `submissions/submission_preact_sla_ta_e200_full.csv`) |

The notebook [`notebooks/final_model.ipynb`](notebooks/final_model.ipynb) loads the final weights, prints the parameter
count with torchsummary, recomputes the holdout accuracy, rebuilds the submission, and plots the training curves and
the results of every experiment. Its saved outputs show the run.

## Documents

- [`docs/LOGBOOK.md`](docs/LOGBOOK.md): every approach we tried, in time order, with results and lessons (ASD-STE100 Simplified Technical English).
- [`docs/RESEARCH_REPORT.md`](docs/RESEARCH_REPORT.md): the literature research and the ranked plan, with the result of each step (ASD-STE100).
- [`docs/experiments.md`](docs/experiments.md): the raw experiment log.
- [`results/runs/`](results/runs/): `config.json` (all hyperparameters and final accuracies) and `history.json` (per-epoch loss and holdout accuracy) for every run.
- [`output/literature-survey/`](output/literature-survey/): survey reports, search logs and BibTeX.

## Rules we follow

- Training is from scratch. No pre-trained weights. Only the CIFAR-10 training images are used.
- Test images are never used for training, adaptation or model selection, and never labelled by hand.
- Model selection uses a fixed, stratified 5,000-image holdout from the training set, not the public leaderboard.
- Every training run asserts the 5M parameter limit before training.
- Distillation teachers were also trained here, from scratch, on the same training images.

## Reproduce

```bash
pip install -r requirements.txt torchsummary matplotlib jupyter
# put the Kaggle files in data/ (train_labels.csv, train_images/, test.csv, test_images/, sample_submission.csv)
python src/prepare_data.py

# final recipe on the 45k/5k split (holdout accuracy), then on all 50k images (submission)
cd src
python train.py --name preact_sla_ta_e200 --arch preact --widths 64 128 256 --depth 4 4 3 \
  --ssl sla --batch-size 512 --label-smoothing 0.1 --trivialaugment --epochs 200
python train.py --name preact_sla_ta_e200_full --val 0 --arch preact --widths 64 128 256 --depth 4 4 3 \
  --ssl sla --batch-size 512 --label-smoothing 0.1 --trivialaugment --epochs 200
cd .. && jupyter notebook notebooks/final_model.ipynb
```

Training uses CUDA when available, else Apple MPS. On one A100 the 200-epoch SLA run takes about 85 minutes
(45k images) or 92 minutes (50k images). Runs save a checkpoint every 10 epochs and continue with `--resume`.
`scripts/hpc_next.sh` is the resumable queue we used on the NYU HPC A100.

## Code

| File | Purpose |
|---|---|
| `src/prepare_data.py` | Decodes the PNG files once into `data/cifar10_cache.npz` |
| `src/models.py` | `PreActResNet` (final), `AirResNet`, `ShakeResNet`, BlurPool blocks, SLA / rotation heads |
| `src/train.py` | Training: GPU-side augmentation, TrivialAugment workers, SLA+SD, SAM, distillation, TTA, checkpoints |
| `src/train_papa.py` | PAPA population training averaged into one network |
| `src/make_submission.py`, `src/import_preds.py`, `src/shift_check.py` | Submission writing and diagnostics |
| `models/` | Final weights (`final_preact_sla_50k.pt`) and the 45k holdout model |

## Credits

Parts of `src/models.py` and `src/train.py` are adapted from
[cifar10-airbench](https://github.com/KellerJordan/cifar10-airbench) by Keller Jordan (MIT License); see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). SLA follows Lee et al. (2020) and was checked against
[hankook/SLA](https://github.com/hankook/SLA). Other methods: pre-activation ResNet (He et al., 2016), TrivialAugment
(Müller and Hutter, 2021), Cutout (DeVries and Taylor, 2017), SAM (Foret et al., 2021), PAPA (Jolicoeur-Martineau
et al., 2024), Shake-Shake (Gastaldi, 2017), BlurPool (Zhang, 2019), knowledge distillation (Hinton et al., 2015).
Full references are in `docs/RESEARCH_REPORT.md`.
