# Efficient ResNet for CIFAR-10 (≤ 5M parameters)

Course Kaggle challenge: train a modified ResNet from scratch on the CIFAR-10 training set (50,000 images) with at
most 5,000,000 trainable parameters, and predict 2,000 held-out evaluation images.

## Result

We selected two final submissions on Kaggle. Both use the same network and the same prediction.

| Item | RegMixup-300 | RegMixup-300 + mild RandConv |
|---|---|---|
| Model | Pre-activation ResNet, 3 stages of 4 / 4 / 3 basic blocks, widths 64 / 128 / 256, global average pooling | same |
| Training method | SLA+SD rotation labels (Lee et al., 2020), RegMixup (Pinto et al., 2022), TrivialAugment, Cutout, Nesterov SGD, cosine schedule, 300 epochs | same, plus mild RandConv (Xu et al., 2021) on 25% of each batch |
| Trainable parameters | **4,696,394** (torchsummary); 4,706,674 with the 40-way SLA head | same |
| Holdout accuracy (45k train / 5k holdout, head + vote) | **97.94%** | 97.82% (equal, McNemar p = 0.46) |
| Robustness probes on the holdout | grayscale 94.86%, blur σ = 1 91.02% (¹) | grayscale **95.52%**, blur σ = 1 **92.46%** (both p < 0.01) |
| Kaggle public accuracy (trained on all 50k) | 0.912 | **0.913** |
| Submission | [`submission_preact_sla_regmix_e300_50k.csv`](submissions/submission_preact_sla_regmix_e300_50k.csv) | [`submission_preact_sla_regmix_randconv_e300_50k.csv`](submissions/submission_preact_sla_regmix_randconv_e300_50k.csv) |
| Weights | `models/regmix_e300_50k.pt` (submission), `models/regmix_e300_45k.pt` (holdout) | `models/regmix_randconv_e300_50k.pt` (submission), `models/regmix_randconv_e300_45k.pt` (holdout) |

(¹) The notebook shows 91.04% for this probe. This is 1 image of 5,000. The notebook does not clamp the pixels
after the float conversion.

## Reproduce

```bash
pip install -r requirements.txt
# put the Kaggle files in data/: train_labels.csv, train_images/, test.csv, test_images/, sample_submission.csv
jupyter notebook notebooks/final_model.ipynb   # then Run All
```

The notebook [`notebooks/final_model.ipynb`](notebooks/final_model.ipynb) decodes the images, counts the parameters,
loads the four saved weight files, recomputes the holdout accuracy of both models, and writes the two submission
CSVs. It checks that each CSV is byte-identical to the submitted CSV. It also plots the training curves and lists
every experiment. Set `TRAIN = True` in its first cell to train the models from zero instead (about 4.2 h per 45k model
and 4.7 h per 50k model on one A100; GPU training is not bit-exact, so a retrained model gives a slightly different CSV).

The same steps from the command line:

```bash
python src/prepare_data.py                                           # data/cifar10_cache.npz
python src/predict.py models/regmix_e300_50k.pt submission_regmix.csv
python src/predict.py models/regmix_randconv_e300_50k.pt submission_randconv.csv
python src/train.py --name regmix_e300_45k                           # 45k train / 5k holdout, writes runs/<name>/
python src/train.py --name regmix_e300_50k --val 0                   # all 50k images
python src/train.py --name regmix_randconv_e300_45k --randconv-prob 0.25
python src/train.py --name regmix_randconv_e300_50k --val 0 --randconv-prob 0.25
```

Training uses CUDA when available, else Apple MPS, else the CPU. It saves a checkpoint every 10 epochs and continues
with `--resume`. `--se-reduction 16` gives the squeeze-and-excitation ablation of docs/LOGBOOK.md, Phase 12.

## Code

| File | Purpose |
|---|---|
| `src/prepare_data.py` | Decodes the PNG files once into `data/cifar10_cache.npz` |
| `src/model.py` | `PreActResNet` with the 10-class head and the 40-class SLA joint head (optional squeeze-and-excitation) |
| `src/data.py` | Data loading, the stratified holdout split, TrivialAugment workers, GPU flip / translate / cutout |
| `src/train.py` | Training: SLA+SD + RegMixup loss, optional mild RandConv, SGD with warmup and cosine schedule, checkpoints |
| `src/predict.py` | Test-time augmentation, the 4-rotation vote, head + vote prediction, the submission CSV |

## Documents

- [`docs/LOGBOOK.md`](docs/LOGBOOK.md): every experiment, in time order, with results and lessons (ASD-STE100 Simplified Technical English).
- [`docs/RESEARCH_REPORT.md`](docs/RESEARCH_REPORT.md): the literature research and the ranked plan (ASD-STE100).
- [`results/runs/`](results/runs/): `config.json` (hyperparameters and final accuracies) and `history.json` (per-epoch loss and holdout accuracy) for the runs.
- [`results/analysis/`](results/analysis/): paired holdout comparisons (fixed / broken images, McNemar test) and the test-time augmentation study.

## Rules we follow

- Training is from scratch. No pre-trained weights. Only the CIFAR-10 training images are used.
- Test images are never used for training or model selection, and never labelled by hand.
- Model selection uses a fixed, stratified 5,000-image holdout from the training set, not the public leaderboard.
- Training asserts the 5M parameter limit before it starts.

## License

The code in this repository has the MIT License (see [`LICENSE`](LICENSE)). The adapted airbench code keeps its own
MIT License (see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)).

## Credits

`src/data.py` (crop and cutout) and `src/predict.py` (flip / translate TTA) adapt code from
[cifar10-airbench](https://github.com/KellerJordan/cifar10-airbench) by Keller Jordan (MIT License); see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). SLA follows Lee et al. (2020) and was checked against
[hankook/SLA](https://github.com/hankook/SLA). RandConv follows Xu et al. (2021, ICLR). Other methods: pre-activation
ResNet (He et al., 2016), RegMixup (Pinto et al., 2022), TrivialAugment (Müller and Hutter, 2021), Cutout (DeVries
and Taylor, 2017). The ablations also used squeeze-and-excitation (Hu et al., 2018), progressive RandConv (Choi et al.,
2023, CVPR) and Fourier amplitude mixing (Xu et al., 2021, FACT; Chen et al., 2021, APR).
Full references are in `docs/RESEARCH_REPORT.md` and `THIRD_PARTY_NOTICES.md`.
