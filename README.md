# Efficient ResNet for CIFAR-10 (≤ 5M parameters)

Course Kaggle challenge: train a modified ResNet from scratch on the CIFAR-10 training set (50,000 images) with at
most 5,000,000 trainable parameters, and predict 2,000 held-out evaluation images.

## Result

| Item | Value |
|---|---|
| Model | Pre-activation ResNet, 3 stages of 4 / 4 / 3 basic blocks, widths 64 / 128 / 256, global average pooling |
| Training method | SLA+SD rotation labels (Lee et al., 2020), RegMixup (Pinto et al., 2022), TrivialAugment, Cutout, Nesterov SGD, cosine schedule, 300 epochs |
| Trainable parameters | **4,696,394** (torchsummary); 4,706,674 with the 40-way SLA head used in training and in the rotation vote |
| Holdout accuracy (same recipe, 45k train / 5k holdout) | **97.94%** (head + rotation vote) |
| Kaggle public accuracy (trained on all 50k) | **0.912** |
| Submission | [`submissions/submission_preact_sla_regmix_e300_50k.csv`](submissions/submission_preact_sla_regmix_e300_50k.csv) |
| Weights | `models/regmix_e300_50k.pt` (submission), `models/regmix_e300_45k.pt` (holdout) |

## Reproduce

```bash
pip install -r requirements.txt
# put the Kaggle files in data/: train_labels.csv, train_images/, test.csv, test_images/, sample_submission.csv
jupyter notebook notebooks/final_model.ipynb   # then Run All
```

The notebook [`notebooks/final_model.ipynb`](notebooks/final_model.ipynb) decodes the images, counts the parameters,
loads the saved weights, recomputes the holdout accuracy, writes `runs/submission.csv` and checks that it is
byte-identical to the submitted CSV. It also plots the training curves and lists every experiment. Set `TRAIN = True`
in its first cell to train both models from scratch instead (about 4.2 h + 4.6 h on one A100; GPU training is not
bit-exact, so a retrained model gives a slightly different CSV).

The same steps from the command line:

```bash
python src/prepare_data.py                                   # data/cifar10_cache.npz
python src/predict.py models/regmix_e300_50k.pt submission.csv
python src/train.py --name regmix_e300_45k                   # 45k train / 5k holdout, writes runs/<name>/
python src/train.py --name regmix_e300_50k --val 0           # all 50k images
```

Training uses CUDA when available, else Apple MPS, else the CPU. It saves a checkpoint every 10 epochs and continues
with `--resume`.

## Code

| File | Purpose |
|---|---|
| `src/prepare_data.py` | Decodes the PNG files once into `data/cifar10_cache.npz` |
| `src/model.py` | `PreActResNet` with the 10-class head and the 40-class SLA joint head |
| `src/data.py` | Data loading, the stratified holdout split, TrivialAugment workers, GPU flip / translate / cutout |
| `src/train.py` | Training: SLA+SD + RegMixup loss, SGD with warmup and cosine schedule, checkpoints |
| `src/predict.py` | Test-time augmentation, the 4-rotation vote, head + vote prediction, the submission CSV |

## Documents

- [`docs/LOGBOOK.md`](docs/LOGBOOK.md): every experiment, in time order, with results and lessons (ASD-STE100 Simplified Technical English).
- [`docs/RESEARCH_REPORT.md`](docs/RESEARCH_REPORT.md): the literature research and the ranked plan (ASD-STE100).
- [`results/runs/`](results/runs/): `config.json` (hyperparameters and final accuracies) and `history.json` (per-epoch loss and holdout accuracy) for every run.
- [`output/literature-survey/`](output/literature-survey/): survey reports, search logs and BibTeX.

## Rules we follow

- Training is from scratch. No pre-trained weights. Only the CIFAR-10 training images are used.
- Test images are never used for training or model selection, and never labelled by hand.
- Model selection uses a fixed, stratified 5,000-image holdout from the training set, not the public leaderboard.
- Training asserts the 5M parameter limit before it starts.

## Credits

`src/data.py` (crop and cutout) and `src/predict.py` (flip / translate TTA) adapt code from
[cifar10-airbench](https://github.com/KellerJordan/cifar10-airbench) by Keller Jordan (MIT License); see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). SLA follows Lee et al. (2020) and was checked against
[hankook/SLA](https://github.com/hankook/SLA). Other methods: pre-activation ResNet (He et al., 2016), RegMixup
(Pinto et al., 2022), TrivialAugment (Müller and Hutter, 2021), Cutout (DeVries and Taylor, 2017).
Full references are in `docs/RESEARCH_REPORT.md`.
