# Literature Survey: Closing the CIFAR-10 → CIFAR-10.1-style gap with a ≤5M-parameter ResNet trained from scratch (round 2)

**Date:** 2026-10-29 | **Records retrieved:** 194 (192 unique, arXiv) + 50 key papers verified by arXiv ID | **Included:** 45 | **Read in full:** 24 | **Date range:** 2016–2026 | **Search queries:** 22

Reader-facing version (ASD-STE100): `docs/RESEARCH_REPORT.md`.
Round 1 (augmentation, SAM, SWA/EMA, distillation, the Recht line) is in `../2026-10-29-1530/survey_report.md`;
this round skips those queries and targets what round 1 left open: effective robustness, architecture and
aliasing, ensembles inside a parameter budget, weight averaging from scratch, and test-time methods.

## How the three skills were used

| Skill | Use in this round |
|---|---|
| **academic-research-plugin / literature-survey** | Workflow and output layout (decompose → search → seminal papers → themes → gaps → cross-domain → proposals → BibTeX). All 22 searches ran through its `paper_search.py`, using arXiv field syntax (`abs:"..." AND ...`) because plain-word queries returned mostly off-topic hits. |
| **academic-research-skills / deep-research** | `three-way-scan` mode: WHY / HOW / WHAT per key paper, method weaknesses on the empirical-paper checklist (`academic-paper-reviewer/references/review_criteria_framework.md` §2.1) with each item marked author-acknowledged or reader-inferred. ARS arXiv protocol for verification (one `id_list` request, title similarity ≥ 0.70, ≥3 s between requests). "Gray zone = FAIL": the CIFAR-10.2 paper (Lu et al. 2020) could not be found on arXiv or OpenAlex, so it is cited only through Miller et al. Devil's-advocate checkpoints 2 and 3 (end of report). |
| **scientific-agent-skills / literature-review** | Reproducibility: every query, start date and count in `search_log.json`; raw results in `raw/`, merged records in `papers_raw.json`; title → abstract → full-text screening; citation check (`seminal/arxiv_id_check.json`, BibTeX from arxiv.org/bibtex); limitations section. |

Inclusion: image classification under natural distribution shift or CIFAR-scale accuracy per parameter; usable from scratch, CIFAR-10 train only, ≤5M parameters in the final model. Exclusion: methods that need pretrained weights, external or generated data, or any use of the evaluation images for training, fine-tuning or selection (competition rule); off-topic arXiv matches (astronomy, chemistry, NLP, vision-language test-time tuning).

## Project facts this survey builds on (from `docs/experiments.md`)

| Model | Holdout (5k, TTA) | Kaggle public | Recht fit | Gap |
|---|---|---|---|---|
| AirResNet, 40 epochs | 95.58% | 0.850 | 89.3% | −4.3 |
| AirResNet + TrivialAugment, 120 epochs | 96.20% | 0.86 | 90.3% | −4.3 |
| AirResNet + TrivialAugment + SLA+SD, 120 epochs | 96.40% | 0.863 | 90.7% | −4.4 |
| Pre-act ResNet 64/128/256, 4/4/3 blocks, TrivialAugment + Cutout, 200 epochs | 96.82% | 0.884 | 91.3% | −2.9 |

Recht fit: CIFAR-10.1 accuracy = 1.62 × CIFAR-10 accuracy − 65.51. The public split has about 1,000 images, so one score carries a standard error of about 1.1 points and a paired difference about 1 point.

## Paper Summary Table

Citations: Semantic Scholar batch lookup, 2026-10-29 (counts can lag or split across versions). "—" = not in the batch.

| # | Paper | First author | Year | Venue | Citations | Role here |
|---|---|---|---|---|---|---|
| 1 | [Do CIFAR-10 Classifiers Generalize to CIFAR-10?](https://arxiv.org/abs/1806.00451) | Recht | 2018 | arXiv | 486 | **Seminal.** The fit; per-model table; Cutout and width help the new set more |
| 2 | [Accuracy on the Line](https://arxiv.org/abs/2107.04649) | Miller | 2021 | ICML | 357 | **Seminal.** Lines hold over 1,000+ models; describes CIFAR-10.2 |
| 3 | [Measuring Robustness to Natural Distribution Shifts](https://arxiv.org/abs/2007.00644) | Taori | 2020 | NeurIPS | 742 | Few interventions beat the line; only more data does |
| 4 | [Models Out of Line: A Fourier Lens on Distribution Shift Robustness](https://arxiv.org/abs/2207.04075) | Fridovich-Keil | 2022 | NeurIPS | 2 | CIFAR-10.1 shift is low-frequency; pruning and AugMix stay on its line |
| 5 | [Deep Ensembles Work, But Are They Necessary?](https://arxiv.org/abs/2202.06985) | Abe | 2022 | NeurIPS | 102 | Ensembles' shifted-set gains = their clean gains |
| 6 | [Identifying Statistical Bias in Dataset Replication](https://arxiv.org/abs/2005.09619) | Engstrom | 2020 | ICML | 56 | Replication bias explains ~8 of ImageNetV2's 11.7-point drop |
| 7 | [Effective Robustness against Natural Distribution Shifts for Models with Different Training Data](https://arxiv.org/abs/2302.01381) | Shi | 2023 | NeurIPS | 17 | Effective robustness is about training data |
| 8 | [On the Variance of Neural Network Training w.r.t. Test Sets and Distributions](https://arxiv.org/abs/2304.01910) | Jordan | 2024 | ICLR | 24 | Seed-to-seed test differences are mostly sampling noise |
| 9 | [94% on CIFAR-10 in 3.29 Seconds on a Single GPU](https://arxiv.org/abs/2404.00498) | Jordan | 2024 | arXiv | 24 | Our former base; no shifted-set evaluation in the paper |
| 10 | [AutoAugment](https://arxiv.org/abs/1805.09501) | Cubuk | 2019 | CVPR | 2026 | Shake-Shake 26 2x32d (2.9M): 97.5% with AutoAugment |
| 11 | [Shake-Shake regularization](https://arxiv.org/abs/1705.07485) | Gastaldi | 2017 | arXiv | 398 | 2.9M params, 96.45% with standard augmentation, 1,800 epochs |
| 12 | [ShakeDrop Regularization for Deep Residual Learning](https://arxiv.org/abs/1802.02375) | Yamada | 2019 | IEEE Access | 180 | Single-branch stochastic regularizer |
| 13 | [Deep Pyramidal Residual Networks](https://arxiv.org/abs/1610.02915) | Han | 2017 | CVPR | 771 | PyramidNet-110 α=84: 95.74% at 3.8M |
| 14 | [PopulAtion Parameter Averaging (PAPA)](https://arxiv.org/abs/2304.03094) | Jolicoeur-Martineau | 2024 | TMLR | 27 | One averaged ResNet-18 at 97.4% vs 96.8% best single |
| 15 | [Linear Mode Connectivity and the Lottery Ticket Hypothesis](https://arxiv.org/abs/1912.05671) | Frankle | 2020 | ICML | 875 | Runs split after ~3% of training stay linearly connected |
| 16 | [Lookaround Optimizer](https://arxiv.org/abs/2306.07684) | Zhang | 2023 | NeurIPS | 12 | ~+0.2 over Lookahead on CIFAR-10 ResNet-50 |
| 17 | [DART: Diversify-Aggregate-Repeat Training](https://arxiv.org/abs/2302.14685) | Jain | 2023 | CVPR | 29 | Periodic averaging of differently-augmented copies |
| 18 | [Packed-Ensembles for Efficient Uncertainty Estimation](https://arxiv.org/abs/2210.09184) | Laurent | 2023 | ICLR | 66 | Budget ensembles; BatchEnsemble and MIMO don't beat single ResNet-18 |
| 19 | [Is BatchEnsemble a Single Model?](https://arxiv.org/abs/2601.16936) | Zamyatin | 2026 | arXiv | 4 | BatchEnsemble tracks a single model |
| 20 | [On Power Laws in Deep Ensembles](https://arxiv.org/abs/2007.08483) | Lobacheva | 2020 | NeurIPS | 50 | Memory-split advantage at fixed parameter budget |
| 21 | [Training independent subnetworks for robust prediction (MIMO)](https://arxiv.org/abs/2010.06610) | Havasi | 2021 | ICLR | 252 | Free ensembles need spare capacity |
| 22 | [BatchEnsemble](https://arxiv.org/abs/2002.06715) | Wen | 2020 | ICLR | 610 | Rank-1 ensembles |
| 23 | [Impact of Aliasing on Generalization in Deep Convolutional Networks](https://arxiv.org/abs/2108.03489) | Vasconcelos | 2021 | ICCV | 41 | Anti-aliasing strided skips: ImageNet +0.8, ImageNetV2 +0.6 |
| 24 | [Making Convolutional Networks Shift-Invariant Again](https://arxiv.org/abs/1904.11486) | Zhang | 2019 | ICML | 1018 | BlurPool; CIFAR-10 effect within ±0.6 with augmentation |
| 25 | [Truly shift-invariant convolutional neural networks](https://arxiv.org/abs/2011.14214) | Chaman | 2021 | CVPR | 95 | Adaptive polyphase sampling |
| 26 | [On Pitfalls of Test-Time Adaptation](https://arxiv.org/abs/2306.03536) | Zhao | 2023 | ICML | 91 | BN adaptation and TENT hurt on CIFAR-10.1 |
| 27 | [Improving robustness against common corruptions by covariate shift adaptation](https://arxiv.org/abs/2006.16971) | Schneider | 2020 | NeurIPS | 708 | BN statistics adaptation |
| 28 | [Tent: Fully Test-time Adaptation by Entropy Minimization](https://arxiv.org/abs/2006.10726) | Wang | 2021 | ICLR | 71 | Entropy-based test-time training |
| 29 | [Better Aggregation in Test-Time Augmentation](https://arxiv.org/abs/2011.11156) | Shanmugam | 2021 | ICCV | 264 | TTA can flip correct predictions |
| 30 | [Greedy Policy Search](https://arxiv.org/abs/2002.09103) | Molchanov | 2020 | UAI | 116 | Learned TTA policies |
| 31 | [Deep Nets with Subsampling Layers Unwittingly Discard Useful Activations at Test-Time](https://arxiv.org/abs/2410.01083) | Yang | 2024 | ECCV | 1 | Test-time search over discarded activations |
| 32 | [Agreement-on-the-Line](https://arxiv.org/abs/2206.13089) | Baek | 2022 | NeurIPS | 125 | Unlabeled OOD accuracy estimation (not usable under the rules) |
| 33 | [When Does Label Smoothing Help?](https://arxiv.org/abs/1906.02629) | Müller | 2019 | NeurIPS | 2575 | Smoothed teachers distil worse |
| 34 | [Does Knowledge Distillation Really Work?](https://arxiv.org/abs/2106.05945) | Stanton | 2021 | NeurIPS | 334 | Students match teachers poorly; gains still possible |
| 35 | [Human uncertainty makes classification more robust](https://arxiv.org/abs/1908.07086) | Peterson | 2019 | ICCV | 422 | Soft labels help CIFAR-10.1 cross-entropy; needs external labels |
| 36 | [Whitening and second order optimization ... can reduce or prevent generalization](https://arxiv.org/abs/2008.07545) | Wadia | 2021 | ICML | 21 | Theory against full-input whitening |
| 37 | [Do We Train on Test Data? Purging CIFAR of Near-Duplicates](https://arxiv.org/abs/1902.00423) | Barz | 2020 | J. Imaging | 139 | Near-duplicate inflation of CIFAR scores |
| 38 | [Sharpness-Aware Minimization](https://arxiv.org/abs/2010.01412) | Foret | 2021 | ICLR | — | Round 1; WRN CIFAR-10 2.2% → 1.6% error |
| 39 | [A good teacher is patient and consistent](https://arxiv.org/abs/2106.05237) | Beyer | 2022 | CVPR | — | Round 1; distillation recipe |
| 40 | [Self-supervised Label Augmentation via Input Transformations](https://arxiv.org/abs/1910.05872) | Lee | 2020 | ICML | 68 | Implemented (SLA+SD) |
| 41 | [SWAD](https://arxiv.org/abs/2102.08604) | Cha | 2021 | NeurIPS | 650 | Dense weight averaging for domain generalization |
| 42 | [Model soups](https://arxiv.org/abs/2203.05482) | Wortsman | 2022 | ICML | 1893 | Weight averaging needs a shared basin |
| 43 | [Distilling OOD Robustness from Vision-Language Foundation Models](https://arxiv.org/abs/2311.01441) | Zhou | 2023 | NeurIPS | 14 | Distillation can carry robustness (CLIP teachers, not allowed) |
| 44 | [PixMix](https://arxiv.org/abs/2112.05135) | Hendrycks | 2022 | CVPR | 202 | Uses external fractal images (not allowed) |
| 45 | [Benchmarking Robustness to Common Corruptions](https://arxiv.org/abs/1903.12261) | Hendrycks | 2019 | ICLR | 4961 | CIFAR-10-C |

---

## Theme Clusters

### Theme 1: Shifted-set accuracy follows clean accuracy, with small per-family offsets

**Summary.** Recht et al. (#1) fit 30 models with one line, and Miller et al. (#2) show the same in probit space for over 1,000 models; Taori et al. (#3) find only more diverse training data lifts models above such lines. Fridovich-Keil et al. (#4) find that pruning, AugMix and Gaussian-noise training stay on the CIFAR-10.1 line (in-distribution accuracy predicts it with R² about 0.86–0.93), and Abe et al. (#5) find ensembles gain on shifted sets only what they gain in-distribution. Families still differ a little: residuals in Recht's table run from −1.6 (DenseNet-BC) to +1.2 (Shake-Shake-64d with Cutout), and our two families differ by about 1.4 points.

**Three-way scan.**

#### Models Out of Line (Fridovich-Keil et al., 2022)
Source: arXiv / NeurIPS | Year: 2022 | Link: https://arxiv.org/abs/2207.04075
- WHY: effectively robust models are rare and unexplained; which data or model properties produce them?
- HOW: CIFAR-10 models (Conv8, ResNet-18, VGG16) with five pruning families at 50–95% sparsity and three augmentation regimes; ImageNet CLIP interpolations; power-spectral-density analysis of each shift; Fourier interpolation sensitivity metrics.
- WHAT: CIFAR-10.1's shift from CIFAR-10 is concentrated in low frequencies, like brightness and contrast. On CIFAR-10.1, in-distribution accuracy stays a strong predictor across pruning and augmentation; no single metric explains robustness everywhere.
  - **Method weaknesses** (empirical; read: §1–4.3, Tables 1–2)
    - checked: found: external validity, conclusion conservatism | checked: none found: statistical reporting completeness | not checked: research hypothesis clarity, variable operational definitions, internal validity
    - Three small architectures trained on CIFAR-10 only; distorts the result when generalized to stronger nets near 97%, where the line's local slope and family offsets could differ. [reader-inferred]
    - Shift spectra are averages over whole test sets; distorts the result when a shift is driven by a small subset of hard images rather than spectral content. [reader-inferred]

#### Do CIFAR-10 Classifiers Generalize to CIFAR-10? (Recht et al., 2018)
Source: arXiv | Year: 2018 | Link: https://arxiv.org/abs/1806.00451
- WHY: has a decade of test-set reuse overfit CIFAR-10?
- HOW: new 2,000-image test set from Tiny Images; ~30 public models; cross-validation retraining (§5.6).
- WHAT: 4–10 point drops on a common line; Cutout and width help the new set more than the old. Retraining on train-split holdouts gave accuracies 0.1 to 0.8 points above the official test set (Table 2), which suggests our holdout reads slightly high.
  - **Method weaknesses** (empirical; read: §4, §5.6, Tables 1–2; round 1 read the full text)
    - checked: found: internal validity, statistical reporting completeness | checked: none found: conclusion conservatism | not checked: variable operational definitions
    - Models come from public repositories with different recipes; distorts the Cutout and width comparisons when the paired models also differ in schedule or code. [reader-inferred]

#### Deep Ensembles Work, But Are They Necessary? (Abe et al., 2022)
Source: arXiv / NeurIPS | Year: 2022 | Link: https://arxiv.org/abs/2202.06985
- WHY: do ensembles offer robustness beyond what their accuracy explains?
- HOW: deep ensembles vs single larger models on CIFAR-10 → CIFAR-10.1, CINIC-10, CIFAR-10-C and ImageNet shifts; effective-robustness analysis on accuracy and probabilistic metrics.
- WHAT: ensembles' shifted-set performance is determined by in-distribution performance; a single larger model replicates it.
  - **Method weaknesses**: not assessed (read scope: abstract, §1, §5 headings)

#### Identifying Statistical Bias in Dataset Replication (Engstrom et al., 2020)
Source: arXiv / ICML | Year: 2020 | Link: https://arxiv.org/abs/2005.09619
- WHY: is ImageNetV2's 11.7-point drop real model fragility?
- HOW: statistical model of selection-frequency bias in replication.
- WHAT: after correction, 3.6 ± 1.5 of the 11.7 ± 1.0 points remain unexplained.
  - **Method weaknesses**: not assessed (read scope: abstract_only)

**Contribution.** Our goal is clean accuracy; family choice is worth about 1–2 points at most, and the remaining offset is likely a property of the hidden set.

### Theme 2: Accuracy per parameter under 5M

**Summary.** The strongest documented model under 5M is Shake-Shake 26 2x32d: 2.9M parameters, 96.45% with standard augmentation (Gastaldi, #11, Table 1), 97.0% with Cutout and 97.5% ± 0.1 with AutoAugment (Cubuk et al., #10, Table 2, replicated), all over 1,800 epochs. PyramidNet-110 with α=84 reaches 95.74% at 3.8M with standard augmentation over 300 epochs (#13). Our pre-act ResNet reaches 96.82% on the holdout at 4.70M in 200 epochs. In Recht's table, stochastic-branch models with Cutout sit above the fit (Shake-Shake-64d + Cutout +1.2, ShakeDrop +0.8), single models with intervals of about ±1.2.

#### Shake-Shake regularization (Gastaldi, 2017) and AutoAugment Table 2 (Cubuk et al., 2019)
Source: arXiv | Years: 2017, 2019 | Links: https://arxiv.org/abs/1705.07485, https://arxiv.org/abs/1805.09501
- WHY: overfitting in multi-branch residual nets.
- HOW: replace the fixed sum of two residual branches with per-image random convex weights, independently drawn for forward and backward passes; lr 0.2 cosine, batch 128, 1,800 epochs.
- WHAT: 2.9M-parameter 26 2x32d at 3.55% error (Shake-Shake-Image) to 3.47% (Shake-Even-Image); with AutoAugment 2.5%.
  - **Method weaknesses** (empirical; read: §2.1–2.3, Table 1; AutoAugment Table 2)
    - checked: found: statistical reporting completeness (3-run averages, AutoAugment ±0.1) | checked: none found: internal validity | not checked: external validity beyond CIFAR
    - Results use 1,800 epochs; distorts the expected gain when the schedule is cut to fit a deadline, since stochastic branches converge slowly. [reader-inferred]
    - Errors are on the official test set used during development; distorts comparisons with our train-split holdout by a few tenths. [reader-inferred]

**Contribution.** Architecture and schedule, not tricks, set most of the clean accuracy at this budget.

### Theme 3: Ensemble gains inside one network

**Summary.** PAPA (#14) trains a population and pulls each copy toward the population mean; on CIFAR-10 with ResNet-18 at 300 epochs and 10 copies, the final average reached 97.4% vs 96.8% for the best single model, close to the 97.5% ensemble. It works because copies stay in one basin; Frankle et al. (#15) show ResNet-20 runs become linearly connected after about 3% of training. Lookaround (#16) adds about 0.2 over Lookahead on CIFAR-10 ResNet-50. Parameter-sharing ensembles do not help at our size: in Packed-Ensembles' own Table 1 (#18), BatchEnsemble scores 92.9% and MIMO 94.0% against 94.0% for a single ResNet-18, and Packed-Ensembles adds 0.3; a 2026 study (#19) finds BatchEnsemble behaves like one model. Lobacheva et al. (#20) find an ensemble of medium nets can beat one large net at a fixed budget, including budgets below standard size.

#### PopulAtion Parameter Averaging (Jolicoeur-Martineau et al., 2024)
Source: arXiv / TMLR | Year: 2024 | Link: https://arxiv.org/abs/2304.03094
- WHY: weight averaging is cheap at inference but usually fails between independently trained nets.
- HOW: population of p models with different data orders, augmentations and regularizations; slow pull toward the average (PAPA), or periodic full averaging (PAPA-all, PAPA-2).
- WHAT: CIFAR-10 ResNet-18, 300 epochs, p=10: one averaged model at 97.4% vs best single 96.8%; ensemble 97.5%.
  - **Method weaknesses** (empirical; read: abstract, Tables 1–3)
    - checked: found: statistical reporting completeness (std over runs) | checked: none found: conclusion conservatism | not checked: internal validity
    - Baseline "single" is the best of 10 independently trained models; distorts the gain when compared with a tuned single run under equal compute (10× fewer epochs per copy would be fairer). [reader-inferred]

#### Packed-Ensembles (Laurent et al., 2023)
Source: arXiv / ICLR | Year: 2023 | Link: https://arxiv.org/abs/2210.09184
- WHY: deep ensembles are too large for constrained settings.
- HOW: grouped convolutions pack M subnetworks into one backbone; compared with BatchEnsemble, MIMO, Masksembles and deep ensembles over 5 runs.
- WHAT: CIFAR-10 ResNet-18: single 94.0, BatchEnsemble 92.9, MIMO 94.0, Packed 94.3 (8.2M), deep ensemble 95.1; gains grow with backbone size.
  - **Method weaknesses** (empirical; read: §4 text, Table 1)
    - checked: found: external validity (authors note ResNet-18 lacks capacity to split, §4) | not checked: internal validity
    - Baselines near 94% for ResNet-18 are weak; distorts transfer to strong recipes near 97%. [reader-inferred]

**Contribution.** Get ensemble gains through PAPA or distillation into one ≤5M network, not through parameter-sharing ensembles.

### Theme 4: Architecture details and optimization

**Summary.** Anti-aliasing strided skips and convs with fixed low-pass filters raised ResNet-50 from 76.7% to 77.5% on ImageNet and 64.6% to 65.2% on ImageNetV2 (Vasconcelos et al., #23, Table 7), so its shift gain matches its clean gain. On CIFAR-10 with augmentation, BlurPool changed accuracy by −0.3 to +0.6 points depending on filter and architecture (Zhang, #24, Table 5). SAM (round 1) reduced WRN error on CIFAR-10 from 2.2% to 1.6%. Whitening of the full input can harm generalization in theory (Wadia, #36), but our photometric check found the airbench stem stable, and the family is now retired.

#### Impact of Aliasing on Generalization in Deep Convolutional Networks (Vasconcelos et al., 2021)
Source: arXiv / ICCV | Year: 2021 | Link: https://arxiv.org/abs/2108.03489
- WHY: data augmentation cannot remove aliasing built into strided layers.
- HOW: non-trainable low-pass filters at chosen strided points of ResNet-50 and EfficientNet; ImageNet, ImageNet-C, ImageNet-R, ImageNetV2, Meta-Dataset.
- WHAT: biggest gain from filtering the strided 1×1 skip connections; ImageNet 77.47% vs 76.49% baseline; ImageNetV2 +0.6.
  - **Method weaknesses** (empirical; read: abstract, §5.1, Appendix E)
    - checked: found: internal validity (ERF control, Table 1) | not checked: statistical reporting completeness
    - Single 90-epoch ImageNet runs without seeds reported; distorts the size of sub-point gains. [reader-inferred]

### Theme 5: Test-time methods and selection

**Summary.** In a fair benchmark (Zhao et al., #26, Table 2), CIFAR-10.1 error was 12.8% for the source model; BN adaptation raised it to 19.0%, TENT to 17.9–18.6%, SAR to 17.1–18.0%; only per-image MEMO helped (10.8%), and all of these train on test images, which the rules forbid. Plain TTA helps modestly but can flip correct predictions (#29). Seed-to-seed differences on a test set are mostly sampling noise: the distribution-wise standard deviation is about 0.03 points for CIFAR-10, higher under shift (Jordan, #8).

#### On Pitfalls of Test-Time Adaptation (Zhao et al., 2023)
Source: arXiv / ICML | Year: 2023 | Link: https://arxiv.org/abs/2306.03536
- WHY: TTA papers compare under inconsistent protocols.
- HOW: TTAB benchmark with fixed hyperparameter selection, episodic and online evaluation across shift types.
- WHAT: on CIFAR-10.1, BN adaptation, TENT, SAR and CoTTA all increase error; no method helps on every shift.
  - **Method weaknesses** (empirical; read: Table 2, §6)
    - checked: found: statistical reporting completeness (± over seeds) | not checked: external validity to non-ResNet-26 backbones
    - One backbone per dataset; distorts the conclusion for architectures without batch norm. [author-acknowledged, section: §6 ("prefers ... group norm")]

#### On the Variance of Neural Network Training (Jordan, 2024)
Source: arXiv / ICLR | Year: 2024 | Link: https://arxiv.org/abs/2304.01910
- WHY: test-set variance between runs hampers comparisons.
- HOW: thousands of CIFAR-10 and ImageNet runs; independent-errors model; distribution-wise variance estimator.
- WHAT: test-set spread is mostly finite-sample noise; distribution-wise std ~0.03%; shifted sets (e.g., ImageNet-Sketch 0.257%) vary more between runs.
  - **Method weaknesses** (empirical; read: abstract, §4.3–5)
    - checked: found: statistical reporting completeness | not checked: internal validity
    - Fast-training CIFAR nets dominate the CIFAR experiments; distorts transfer to 200-epoch strong-augmentation runs. [reader-inferred]

---

## Research Gaps

1. **No published shifted-set evaluation of fast-training (airbench-style) nets.** The airbench paper reports no CIFAR-10.1 numbers; our data suggests the family sits about 1.4 points below a standard ResNet at equal holdout accuracy.
2. **Family offsets near the top of the line are thinly measured.** Recht's table has one or two models per family above 96%, each with ±1.2-point intervals, so we cannot tell which modern families sit above the line.
3. **Ensemble-in-one-network methods have not been tested on natural shifts at ≤5M.** PAPA reports clean accuracy only.
4. **Long-schedule stochastic-branch nets (Shake-Shake, ShakeDrop) have no published short-schedule numbers.** Deadline-limited users cannot tell how much of the 97.5% survives at 600 epochs.
5. **The hidden set's identity is unknown.** Every one of our models sits further below the CIFAR-10.1 fit than any published model, which fits a harder reproduction set; this cannot be checked without breaking the rules.

## Cross-Domain Findings

- **ImageNet → ImageNetV2 replication bias (Engstrom et al.).** Part of a reproduction gap can come from how the replica was built, so a shared offset across all our models is expected and not fixable by model choice.
- **Domain generalization → our shift (SWAD, flat-minima work).** Dense weight averaging gives large out-of-domain gains on DomainBed with pretrained backbones; from scratch on CIFAR, the measured effect is the smaller in-distribution gain of averaging optimizers (Lookaround +0.2 over Lookahead).
- **CLIP robustness distillation (Zhou et al.).** Distillation can carry robustness from a robust teacher, but the robust teachers there are pretrained on external data; for us distillation transfers accuracy, which is still the main lever.

## Innovation Proposals

### Proposal 1: Shake-Shake pre-act family within 5M, on a deadline schedule

**Description.** Replace the pre-act basic blocks with Shake-Shake blocks (26 2x32d at 2.9M, or 2x40d at about 4.5M), keep TrivialAugment + Cutout, and measure how much of the published 97.5% survives at 600 epochs before committing to 1,800.

**Feasibility.** Data: CIFAR-10 train only. Compute: estimated 3–5 h per 1,800-epoch run on the A100 (not measured; the pilot measures it), so checkpoint and resume across Jupyter jobs. Novelty: low; this is reproduction under a deadline. Timeline: 2–3 days.

**Potential Weaknesses.** Published gains depend on 1,800 epochs. SLA+SD (4× cost) does not fit on top. Family offsets for Shake-Shake rest on single models in Recht's table.

**Landing Plan.**
1. Implement the block (per-image α forward, β backward); 600-epoch pilot on 45k/5k.
2. If the pilot beats the pre-act net on the holdout across 2 seeds, run 1,800 epochs with checkpoints.
3. One public submission as a coarse check of the family offset.
- Success: holdout ≥ 97.2%.
- Fallback: keep the pre-act family; spend the time on Proposal 2.

### Proposal 2: Ensemble gains inside one ≤5M network (PAPA plus distillation)

**Description.** Train 4–6 pre-act copies with different data order and augmentation under PAPA, then distil the population (or a seed ensemble) into the final student with consistent views, mixup/CutMix inputs and no label smoothing.

**Feasibility.** Compute: 4–6× one 25-minute run for PAPA; 1–2 h for the student. Code: `--teachers` exists; PAPA needs a population loop. Timeline: 2 days.

**Potential Weaknesses.** PAPA's gain was measured against the best of 10 independent runs; at equal compute it may be smaller. KD gains on CIFAR-10 have been small with single teachers. The two methods overlap.

**Landing Plan.**
1. PAPA with p=4 on the pre-act recipe; compare the averaged model with a single run of 4× epochs.
2. Distil from the best population into one student; keep only if ≥ +0.3 over the best non-distilled run.
- Success: holdout ≥ 97.3%.
- Fallback: a late-branch soup (split after ~5% of training, average at the end).

### Proposal 3: Anti-aliased pre-act ResNet with SAM

**Description.** Add fixed [1, 2, 1] blur before every stride-2 conv and strided skip, then train with SAM (ρ 0.05, small-batch ascent).

**Feasibility.** Zero extra parameters; SAM doubles cost (~45 min per 200-epoch run). Timeline: 1 day.

**Potential Weaknesses.** CIFAR-10 effects of anti-aliasing are within about half a point; SAM's gain shrinks for small nets and large ascent batches.

**Landing Plan.**
1. Anti-aliasing alone, 2 seeds.
2. SAM alone, 2 seeds.
3. Both together if each wins.
- Success: +0.3 holdout each.
- Fallback: drop whichever does not win.

---

## Ranked next steps (same as the reader-facing doc)

| Rank | Step | Expected hidden-set gain (points) | A100 cost | Rule check |
|---|---|---|---|---|
| 1 | Make the pre-act ResNet the base | Done: +2.1 observed | None | OK: also wins on holdout |
| 2 | Retrain the final recipe on all 50k | +0.2 to +0.4 | ~25 min | OK |
| 3 | Pre-act + SLA+SD, 200 epochs (queued) | +0.3 to +0.8 | ~95 min | OK |
| 4 | Shake-Shake-26 2x32d or 2x40d, TA + Cutout, 600 → 1,800 epochs | +0.6 to +1.2 | 3–5 h per long run | OK |
| 5 | SAM on the pre-act recipe | +0.3 to +0.6 | ~45 min | OK |
| 6 | Ensemble (4 seeds + wide net) distilled into the student | +0.3 to +0.8 | 2–3 h | OK, explicitly allowed |
| 7 | PAPA over 4–6 copies | +0.5 to +1.0 | 4–6× one run | OK, one final network |
| 8 | 400 epochs; wider-shallower pre-act (3/3/2 at 80/160/320, ~4.9M) | +0.2 to +0.6 | 1–2 runs | OK |
| 9 | Anti-alias strided convs and skips | 0 to +0.5 | 1 run | OK, 0 params |
| 10 | Wider TTA chosen on holdout | +0.1 to +0.3 | Inference | OK |

Hidden-set gains assume ≈1.6 × the holdout gain (Recht slope) and do not add linearly.

## Rejected for rules or evidence

- BN-statistics adaptation, TENT, SAR, CoTTA, MEMO, pseudo-labels on the test images: rule (evaluation images may not be used for training or fine-tuning); most also hurt on CIFAR-10.1 (#26).
- Agreement-on-the-line, entropy or confidence on test images to choose models: "unauthorized model selection".
- Forcing balanced class counts on test predictions: relies on the hidden labels' distribution.
- CIFAR-10.1/10.2, CINIC-10, Tiny Images, PixMix fractals, generated images: external data; CIFAR-10.1 labels could expose hidden labels.
- Large frozen layers to exploit "trainable parameters": against the spirit of the cap.
- BatchEnsemble, MIMO: no gain at this size (#18, #19). Small-net ensembles summing to 5M: unclear under the one-model rule.
- AugMix / noise augmentation for the shift (#4), temperature scaling (does not change the argmax), ensembles for robustness (#5).

---

## Devil's Advocate Report — Checkpoints 2 and 3

### Verdict: PASS

### Critical Issues
No critical issues identified.

### Major Issues
1. **Transferred effect sizes.**
   - Type: Evidence. Location: ranked table.
   - Problem: gains come from ResNet-18, WRN-28-10 or 1,800-epoch Shake-Shake runs; a 4.7M net on a deadline may get half.
   - Recommendation: keep the "≥ +0.3 holdout across 2 seeds" bar before stacking.
2. **Family effect from one pair of submissions.**
   - Type: Evidence. Location: "What the leaderboard says".
   - Problem: +1.4 points beyond the holdout prediction is about 1.3 standard errors.
   - Recommendation: treat as suggestive; the switch is justified by the holdout anyway.
3. **Unexplained shared offset.**
   - Type: Alternative explanation.
   - Problem: the ~3-point offset shared by both families could still hide architecture effects not yet tested.
   - Recommendation: one public check when a new family wins on the holdout (step 4).

### Minor Issues
- Semantic Scholar counts may undercount (e.g., #4 shows 2).
- Holdout may read 0.2–0.8 above the official test set (Recht Table 2), so plotted gaps are slightly overstated.

### Strongest Counter-Argument
"Nothing here moves a model above the line, so at most +1–2 public points remain; time may be better spent on the report." The list is ordered by evidence and cost for that reason, and the cheapest steps (2, 3, 5) come first.

### Stress Test Results
| Test | Result |
|---|---|
| Remove strongest source (Recht) — does the argument hold? | Yes: Miller, Taori, Abe and Fridovich-Keil carry the same conclusion |
| Flip the question — is "beat the line without data" credible? | No: no CIFAR-10-only method in the corpus shows reliable effective robustness on CIFAR-10.1 |
| Apply to a different context — does it generalize? | Partly: same law on ImageNetV2; offsets differ per shift |
| "So what?" — is the significance justified? | Yes: ranks 2–10 total a realistic +1 to +3 public points |

## Limitations

- arXiv-only retrieval: Semantic Scholar search was rate-limited and DBLP returned a bot check (not bypassed); venue-only papers may be missed.
- One reviewer; narrative/rapid scope, not a PRISMA systematic review. 24 papers read in full, the rest from abstracts.
- The CIFAR-10.2 paper could not be verified and is cited only through Miller et al.

## References

See `references.bib` (BibTeX from arxiv.org/bibtex). Search log: `search_log.json`. Raw records: `raw/`, `papers_raw.json`. ID check and Semantic Scholar metadata: `seminal/`.

---

**Generated**: 2026-10-29 | **Tool**: literature-survey skill v1.0 + ARS deep-research (three-way-scan, devil's advocate) + K-Dense literature-review
