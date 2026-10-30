---
title: "Research report: how to close the CIFAR-10.1 gap with a ResNet of 5M parameters or less"
subtitle: "Written in ASD-STE100 Simplified Technical English"
date: 2026-10-30
---

# Research report: how to close the CIFAR-10.1 gap with a ResNet of 5M parameters or less

This report gives the results of two literature surveys (2026-10-29). It also gives the results of our tests of the ranked methods (2026-10-30).
The language follows ASD-STE100 Simplified Technical English.

## 1. Purpose

The hidden Kaggle test set has 2,000 images. Our scores on it are lower than our holdout scores.
The purpose of this research is to find methods that increase the score on the hidden set.
Each method must obey the competition rules:

- The model is a ResNet with a maximum of 5,000,000 trainable parameters.
- The model trains from zero. It does not use pre-trained weights.
- The model uses only the CIFAR-10 training images.
- The method does not use the test images for training, adaptation or model selection.

## 2. Method of the survey

We used three research skill packages together:

| Package | What it did |
|---|---|
| `academic-research-plugin` (literature-survey) | Search on arXiv, theme clusters, gaps, proposals, BibTeX |
| `academic-research-skills` (deep-research) | WHY / HOW / WHAT notes for each paper, method weaknesses, devil's-advocate check |
| `scientific-agent-skills` (literature-review) | Search log, screening, citation check |

Survey 1 used 8 topic searches and 22 title searches (168 records). We read 6 papers in full.
Survey 2 used 22 searches (192 records). We included 45 papers and read 24 papers in full.
We checked all references against arXiv. We did not cite a paper that we could not find.

Limits of the survey:

- Semantic Scholar and DBLP did not let us search (rate limits and a bot check). Thus, the search used only arXiv.
- One reviewer did the screening. This is a narrative review, not a systematic review.

## 3. Main findings

### 3.1 Accuracy on a shifted test set follows a line

Recht et al. (2018) made CIFAR-10.1. It is a new test set of 2,000 images from the same source as CIFAR-10.
All models lost 4 to 10 points on it. Better models lost fewer points.
The relation is a straight line:

> CIFAR-10.1 accuracy = 1.62 × CIFAR-10 accuracy − 65.51

Miller et al. (2021) found the same line for more than 1,000 models (R² = 0.995).
Taori et al. (2020) found that almost no method moves a model above the line. Only more training data does this.
The rules do not permit more data.

Result for us: increase the clean accuracy. Each +1 point on the holdout gives about +1.6 points on the hidden set.

### 3.2 Some methods help more on the shifted set

In the table of Recht et al., two changes helped more on CIFAR-10.1 than on CIFAR-10:

- Cutout added 1 point on CIFAR-10 and 2.2 points on CIFAR-10.1 (WRN-28-10).
- Wider networks helped more than deeper networks.

Different architecture families sit at different distances from the line (−1.6 to +1.2 points).

### 3.3 Accuracy per parameter below 5M

| Model | Parameters | CIFAR-10 accuracy | Schedule | Source |
|---|---|---|---|---|
| Shake-Shake-26 2x32d + AutoAugment | 2.9M | 97.5% | 1,800 epochs | Cubuk et al., 2019 |
| Shake-Shake-26 2x32d + Cutout | 2.9M | 97.0% | 1,800 epochs | Cubuk et al., 2019 |
| PyramidNet-110, α = 84 | 3.8M | 95.7% | 300 epochs | Han et al., 2017 |
| "Efficient ResNets" (course project) | 4.7M | 96.0% | 200 epochs | Thakur et al., 2023 |

Caution: Thakur et al. used the test set for validation. Thus, their number is too high.

### 3.4 Augmentation and regularization

- TrivialAugment applies one random operation to each image. It is equal to or better than AutoAugment on CIFAR-10 (Müller and Hutter, 2021).
- SAM finds flat minima. It decreased the error of a WideResNet from 2.2% to 1.6% (Foret et al., 2021). It costs 2 times the compute.
- SLA trains on joint labels (class × rotation). It does not force the network to ignore rotation (Lee et al., 2020).
  On WRN-40-2 with Cutout, it decreased the error from 4.33% to 3.36%.

### 3.5 Ensembles inside one network

- PAPA trains a population of networks and pulls them to their mean. One averaged ResNet-18 got 97.4%. The best single network got 96.8% (Jolicoeur-Martineau et al., 2024).
- BatchEnsemble and MIMO do not help at this size (Laurent et al., 2023).
- Distillation can move the accuracy of an ensemble into one network. The teacher and the student must see the same augmented image, and the schedule must be long (Beyer et al., 2022).
  But a careful 2026 study found only +0.2 points on CIFAR-10.

### 3.6 Anti-aliasing

A fixed low-pass filter before each stride-2 layer added 0.8 points on ImageNet and 0.6 points on ImageNetV2 (Vasconcelos et al., 2021).
On CIFAR-10 with augmentation, the change was between −0.3 and +0.6 points (Zhang, 2019).

### 3.7 Methods that we must not use

| Method | Reason |
|---|---|
| BN adaptation, TENT, SAR, CoTTA, MEMO on the test images | The rules do not permit training or adaptation on test images. Also, most of these methods increase the CIFAR-10.1 error (from 12.8% to 17–19%) (Zhao et al., 2023). |
| Model selection with test-image confidence or agreement | "Unauthorized model selection" |
| CIFAR-10.1, CIFAR-10.2, CINIC-10, PixMix fractals, generated images | External data |
| Large frozen layers | Against the purpose of the parameter limit |
| Balancing the class counts of the test predictions | Uses the label distribution of the hidden set |

## 4. Ranked plan and the results of our tests

The survey ranked ten steps. We tested all ten. The table gives the result of each step.
"Holdout" is the accuracy on 5,000 training images that the model did not train on, with test-time augmentation.

| Rank | Step | Expected gain on hidden set | Our result | Decision |
|---|---|---|---|---|
| 1 | Use a standard pre-activation ResNet | – | Holdout 96.82%, Kaggle 0.884 (+2.1) | Kept |
| 2 | Train the final recipe on all 50,000 images | +0.2 to +0.4 | Kaggle 0.909 | Kept (final) |
| 3 | PreActResNet + SLA, 200 epochs | +0.3 to +0.8 | Holdout 97.56% (+0.74) | Kept (best) |
| 4 | Shake-Shake-26 2x40d, long schedule | +0.6 to +1.2 | Holdout 97.16% (600 epochs) | Not kept |
| 5 | SAM on the PreActResNet | +0.3 to +0.6 | +0.36 alone; +0.04 with SLA | Not kept |
| 6 | Distillation into one network | +0.3 to +0.8 | See `docs/LOGBOOK.md` | – |
| 7 | PAPA with 4 copies | +0.5 to +1.0 | 97.2% per copy, below 97.56% | Not kept |
| 8 | 400 epochs | +0.2 to +0.6 | Holdout 97.12% (+0.30) | Not kept |
| 9 | Anti-aliasing | 0 to +0.5 | Holdout 97.00% (+0.18) | Not kept |
| 10 | Test-time augmentation chosen on the holdout | +0.1 to +0.3 | Mirror + shifts + rotation vote | Kept |

## 5. Gap to the CIFAR-10.1 line

| Model | Holdout | Kaggle public | Line prediction | Gap |
|---|---|---|---|---|
| AirResNet, 40 epochs | 95.58% | 0.850 | 89.3% | −4.3 |
| AirResNet + TrivialAugment, 120 epochs | 96.20% | 0.860 | 90.3% | −4.3 |
| AirResNet + SLA, 120 epochs | 96.40% | 0.863 | 90.7% | −4.4 |
| PreActResNet, 200 epochs | 96.82% | 0.884 | 91.3% | −2.9 |
| PreActResNet + SLA, all 50k | 97.56% (45k) | 0.909 | 92.5% | −1.6 |

The airbench-style family was about 1.4 points further below the line than the standard ResNet.
Some of the remaining gap can come from the hidden set. A replicated test set can be harder by construction (Engstrom et al., 2020).
Also, our holdout can read 0.1 to 0.8 points high (Recht et al., 2018, Table 2).

## 6. Devil's-advocate check

- The expected gains come from bigger models and longer schedules. A 4.7M network on a deadline can get less. Our results agree: most methods gave +0.2 to +0.4.
- The family effect comes from a small number of public submissions. Each public score has a standard error of about 1.1 points. But the holdout also supported the change.
- No method in the survey moves a model above the line without more data. Thus, the realistic target was +1 to +3 public points. We got +5.9 public points (0.850 to 0.909).

## 7. References

All references are on arXiv. The BibTeX is in `output/literature-survey/2026-10-29-1705/references.bib`.

- Beyer, L. et al. (2022). Knowledge distillation: A good teacher is patient and consistent. CVPR. arXiv:2106.05237
- Cubuk, E. D. et al. (2019). AutoAugment: Learning augmentation policies from data. CVPR. arXiv:1805.09501
- DeVries, T. and Taylor, G. W. (2017). Improved regularization of CNNs with Cutout. arXiv:1708.04552
- Engstrom, L. et al. (2020). Identifying statistical bias in dataset replication. ICML. arXiv:2005.09619
- Foret, P. et al. (2021). Sharpness-aware minimization. ICLR. arXiv:2010.01412
- Gastaldi, X. (2017). Shake-Shake regularization. arXiv:1705.07485
- Han, D. et al. (2017). Deep pyramidal residual networks. CVPR. arXiv:1610.02915
- He, K. et al. (2016). Identity mappings in deep residual networks. ECCV. arXiv:1603.05027
- Hendrycks, D. et al. (2019). Using self-supervised learning can improve model robustness and uncertainty. NeurIPS. arXiv:1906.12340
- Hinton, G. et al. (2015). Distilling the knowledge in a neural network. arXiv:1503.02531
- Jolicoeur-Martineau, A. et al. (2024). Population parameter averaging (PAPA). TMLR. arXiv:2304.03094
- Jordan, K. (2024). 94% on CIFAR-10 in 3.29 seconds on a single GPU. arXiv:2404.00498
- Laurent, O. et al. (2023). Packed-Ensembles for efficient uncertainty estimation. ICLR. arXiv:2210.09184
- Lee, H., Hwang, S. J. and Shin, J. (2020). Self-supervised label augmentation via input transformations. ICML. arXiv:1910.05872
- Miller, J. et al. (2021). Accuracy on the line. ICML. arXiv:2107.04649
- Müller, S. G. and Hutter, F. (2021). TrivialAugment. ICCV. arXiv:2103.10158
- Recht, B. et al. (2018). Do CIFAR-10 classifiers generalize to CIFAR-10? arXiv:1806.00451
- Taori, R. et al. (2020). Measuring robustness to natural distribution shifts in image classification. NeurIPS. arXiv:2007.00644
- Thakur, A. et al. (2023). Efficient ResNets: Residual network design. arXiv:2306.12100
- Vasconcelos, C. et al. (2021). Impact of aliasing on generalization in deep convolutional networks. ICCV. arXiv:2108.03489
- Zhang, R. (2019). Making convolutional networks shift-invariant again. ICML. arXiv:1904.11486
- Zhao, H. et al. (2023). On pitfalls of test-time adaptation. ICML. arXiv:2306.03536

