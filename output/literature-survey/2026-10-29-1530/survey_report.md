# Literature Survey: Generalizing to a CIFAR-10.1-style held-out set with a ≤5M-parameter ResNet trained from scratch

**Date:** 2026-10-29 | **Records retrieved:** 168 unique (arXiv) | **Screened in:** 25 | **Read in full text:** 6 | **Date range:** 2015–2026 | **Search queries:** 8 topic + 22 seminal-title lookups

## How this survey was run (three skills in tandem)

| Skill | What it contributed here |
|---|---|
| **academic-research-plugin / literature-survey** | Workflow skeleton (query decomposition → parallel search → seminal papers → themes → gaps → cross-domain → proposals → BibTeX) and its `paper_search.py` for all retrieval. Output layout follows its spec. |
| **academic-research-skills / deep-research** (`three-way-scan` mode) | Per-paper WHY / HOW / WHAT extraction with *method weaknesses* labelled author-acknowledged vs reader-inferred; the "gray zone = FAIL" rule for references (nothing cited that wasn't verified on arXiv); a devil's-advocate checkpoint on the conclusions (end of report). |
| **scientific-agent-skills / literature-review** | Reproducibility discipline: every query, date and result count is logged in `search_log.json`; title → abstract → full-text screening; citation verification (all 25 references re-fetched from arXiv's BibTeX endpoint into `references.bib`); explicit limitations section. |

Screening criteria. *Include:* image classification on CIFAR-10 or natural distribution shift; techniques usable with ≤5M params, from scratch, CIFAR-10 train data only. *Exclude:* methods that need pretrained weights, external/unlabeled extra data, or test-time access to labels; off-topic hits (≈140 records — the arXiv relevance ranking returned astronomy, power-grid and SDE papers for generic terms; recorded in `papers_raw.json`).

## Paper Summary Table

| # | Paper | Year | Venue | Role here |
|---|---|---|---|---|
| 1 | [Do CIFAR-10 Classifiers Generalize to CIFAR-10?](https://arxiv.org/abs/1806.00451) (Recht et al.) | 2018 | arXiv / ICML'19 line of work | **Seminal.** Builds CIFAR-10.1 (2,000 images). Linear fit, Table 1 |
| 2 | [Measuring Robustness to Natural Distribution Shifts](https://arxiv.org/abs/2007.00644) (Taori et al.) | 2020 | NeurIPS | Most robustness interventions give no "effective robustness" on natural shift |
| 3 | [Accuracy on the Line](https://arxiv.org/abs/2107.04649) (Miller et al.) | 2021 | ICML | **Seminal.** ID→OOD linear in probit space; CIFAR-10→10.1 R²=0.995 over 1,060 models |
| 4 | [Accuracy on the wrong line](https://arxiv.org/abs/2406.19049) (Sanyal et al.) | 2024 | arXiv | When the line breaks: label noise / nuisance features |
| 5 | [CIFAR-10-Warehouse](https://arxiv.org/abs/2310.04414) | 2023 | ICLR | 180 shifted CIFAR-10 test sets; shift is broad, not one direction |
| 6 | [Cutout](https://arxiv.org/abs/1708.04552) (DeVries & Taylor) | 2017 | arXiv | Augmentation with outsized CIFAR-10.1 benefit (via #1) |
| 7 | [AutoAugment](https://arxiv.org/abs/1805.09501) | 2018 | CVPR'19 | Learned policies |
| 8 | [RandAugment](https://arxiv.org/abs/1909.13719) | 2019 | NeurIPS'20 | 2-parameter policy |
| 9 | [TrivialAugment](https://arxiv.org/abs/2103.10158) (Müller & Hutter) | 2021 | ICCV | Tuning-free; ≥ AA/RA on CIFAR-10 |
| 10 | [AugMix](https://arxiv.org/abs/1912.02781) | 2019 | ICLR'20 | Corruption robustness |
| 11 | [mixup](https://arxiv.org/abs/1710.09412) | 2017 | ICLR'18 | Interpolation regularizer |
| 12 | [CutMix](https://arxiv.org/abs/1905.04899) | 2019 | ICCV | Patch mixing regularizer |
| 13 | [Sharpness-Aware Minimization](https://arxiv.org/abs/2010.01412) (Foret et al.) | 2020 | ICLR'21 | Flat minima; WRN CIFAR-10 2.2%→1.6% error |
| 14 | [SWA](https://arxiv.org/abs/1803.05407) (Izmailov et al.) | 2018 | UAI | Weight averaging → wider optima |
| 15 | [EMA of weights: dynamics and benefits](https://arxiv.org/abs/2411.18704) | 2024 | arXiv | EMA generalizes better, more robust |
| 16 | [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) (Hinton et al.) | 2015 | NIPS-W | **Seminal** KD |
| 17 | [Born-Again Networks](https://arxiv.org/abs/1805.04770) | 2018 | ICML | Same-size students beat teachers |
| 18 | [A good teacher is patient and consistent](https://arxiv.org/abs/2106.05237) (Beyer et al.) | 2021 | CVPR'22 | KD recipe: same augmented view, very long training |
| 19 | [Understanding ensemble, KD and self-distillation](https://arxiv.org/abs/2012.09816) (Allen-Zhu & Li) | 2020 | ICLR'23 | Why seed-ensembles distil into one model |
| 20 | [Revisiting KD via label smoothing](https://arxiv.org/abs/1909.11723) (Yuan et al.) | 2019 | CVPR'20 | KD ≈ learned label smoothing; weak teachers still help |
| 21 | [Student capacity moderates KD (ResNet, CIFAR-10)](https://arxiv.org/abs/2605.31191) | 2026 | arXiv | Careful protocol: KD gains on CIFAR-10 are small (+0.2 pp) |
| 22 | [Multi-teacher KD for compressing ensembles](https://arxiv.org/abs/2302.07215) | 2023 | arXiv | Ensemble→single-model KD |
| 23 | [Efficient ResNets: Residual Network Design](https://arxiv.org/abs/2306.12100) (Thakur et al.) | 2023 | arXiv | Same 5M-param brief; 96.04% |
| 24 | [94% on CIFAR-10 in 3.29 s](https://arxiv.org/abs/2404.00498) (Jordan) | 2024 | arXiv | airbench; our current base |
| 25 | [Wide Residual Networks](https://arxiv.org/abs/1605.07146) | 2016 | BMVC | Width vs depth |

Citation counts are not shown: Semantic Scholar was rate-limited for the whole session, so the counts the tool returned are all 0 and meaningless.

---

## Theme Clusters

### Theme 1: The hidden set is a natural distribution shift, and it follows a line

**Summary.** Recht et al. (#1) re-collected a 2,000-image CIFAR-10 test set from the same source and saw every model drop 4–10 points, with the drop smaller for better models. Their least-squares fit over ~30 models is **acc_new = 1.62·acc_orig − 65.51%** (±0.04 slope, ±3.16 intercept). Miller et al. (#3) confirm the trend over 1,060 models (probit-space R² = 0.995 for CIFAR-10→10.1 and 0.997 for →10.2) and show it holds across architecture, hyperparameters, training-set size and training duration. Taori et al. (#2) find that almost no intervention moves models *above* the line on natural shifts; only more diverse training data does, which the rules forbid.

**Three-way scan (key items).**
- **#1 Recht 2018.** WHY: are CIFAR-10 results overfit to the test set? HOW: new 2,000-image test set from Tiny Images, same labelling protocol; ~30 published models. WHAT: 4–10 pt drop, explained by distribution shift, not adaptive overfitting. Two findings matter directly for us: **Cutout adds more on the new set than on the old one** (shake-shake-64d: +0.12 orig vs ≈+1.5 new; WRN-28-10: ≈+1 orig vs +2.2 new), and **widening a ResNet helps the new set more than deepening it**. *Method weaknesses:* (checked: found) models taken from public repos with heterogeneous training recipes, so the Cutout/width effects are single-pair comparisons, not controlled ablations [reader-inferred].
- **#3 Miller 2021.** WHY: when does ID accuracy predict OOD accuracy? HOW: 1,000+ models, probit-scaled scatter plots over many shifts. WHAT: tight linear trends for CIFAR-10.1/10.2/CINIC/STL. Strong augmentation *reduces* OOD variation around the line (Appendix C.1). *Weakness:* linear trends say what is typical, not what a specific new technique will do [author-acknowledged, section 1].
- **#4 Sanyal 2024.** WHAT: label noise and nuisance features can break the line. This is a reason to avoid shortcut-prone tricks, not something we can measure on CIFAR-10.

**Contribution.** It gives a quantitative target. Each +1 pt of clean CIFAR-10 accuracy should give about **+1.6 pt** on a CIFAR-10.1-style set.

### Theme 2: Data augmentation

**Summary.** Cutout (#6), AutoAugment (#7), RandAugment (#8), TrivialAugment (#9), mixup (#11) and CutMix (#12) are the standard CIFAR-10 regularizers. TrivialAugment applies one random op at a random strength per image, with no search. On CIFAR-10 it reaches **WRN-40-2 96.32 ± .05 (default aug 96.16), WRN-28-10 97.46 ± .06 (97.03), ShakeShake-26-2x96d 98.21 (97.54), PyramidNet 98.58 (97.95)** (Table 2, 10 runs each), matching or beating AA/RA. AugMix (#10) targets synthetic corruptions. Taori (#2) found that corruption robustness transfers poorly to natural shift, so AugMix is lower priority here.

**Contribution.** Cutout, which we already use, is the best-documented CIFAR-10.1 lever. TrivialAugment is the cheapest way to strengthen augmentation further. The published gains come from 200-epoch schedules, so it needs longer training than our 40 epochs.

### Theme 3: Optimization geometry and weight averaging

**Summary.** SAM (#13) cuts WRN CIFAR-10 error from 2.2% to 1.6% and stacks with strong regularizers, but costs two forward/backward passes per step. SWA (#14) and EMA (#15) average weights along the trajectory to reach flatter optima. #15 also reports better robustness and calibration for EMA models. Our training already applies an airbench Lookahead-style EMA.

**Contribution.** SAM is the one large, architecture-independent gain left, at 2× compute.

### Theme 4: Knowledge distillation from from-scratch teachers (explicitly allowed by the rules)

**Summary.** Hinton (#16) introduced KD. BAN (#17) and Yuan (#20) show that even same-size or weaker teachers help, because KD acts as learned label smoothing. Allen-Zhu & Li (#19) explain why an ensemble of differently-seeded nets can be distilled into one network. Beyer et al. (#18) give the practical recipe: **teacher and student see the same augmented view ("consistent"), with mixup-style function matching and very long schedules ("patient")**. Their Figure 1 shows precomputed or fixed teacher targets doing much worse. A careful 2026 CIFAR-10 study (#21) finds **small** gains: +0.2 pp, significant only in some ResNet pairs. *Weakness of #21:* its teachers are plain ResNet-50/101 trained the standard way; ensemble or larger-gap teachers are not tested [reader-inferred from abstract; full text not read].

**Contribution.** KD is legal and well-founded, but the CIFAR-10 evidence says to expect tenths of a point unless the teacher is clearly stronger, e.g. an ensemble. Our `train.py` already computes the teacher online on the same augmented batch, so it is "consistent" in Beyer's sense.

### Theme 5: Architecture under a 5M budget

**Summary.** Efficient ResNets (#23) is the same assignment and reports 96.04% at 4.70M params: 3 stages × [4,4,3] basic blocks, widths 64/128/256, squeeze-excitation, SGD+Lookahead, cosine LR, 200 epochs, only crop and flip augmentation. *Method weakness:* "Test set used as validation set" (their Table 1), so 96.04% is selected on test and optimistic [author-acknowledged, Table 1]. airbench96 (#24) gets 96.0% in 40 epochs with width 128/384/512 (9.9M params, over our budget). WRN (#25) and Recht (#1) both favour width.

**Contribution.** About 96% is the realistic clean-accuracy ceiling for ≤5M params with plain recipes. Width-heavy designs are preferred for the shifted set.

---

## Research Gaps (as they bear on this project)

1. **No controlled study of fast-training (airbench-style) nets on CIFAR-10.1.** It is unknown whether 40-epoch, max-pool-heavy, frozen-whitening nets sit on the Recht/Miller line or below it.
2. **Parameter-budgeted effective robustness.** Recht's and Miller's model zoos are mostly >10M params. Nobody reports where ≤5M models fall.
3. **KD + strong augmentation + SAM under a fixed small budget** is untested as a combination on CIFAR-10.1. Each piece has been studied separately.
4. **Low-level statistics shift.** Our own unlabeled check found the evaluation images brighter (mean RGB 0.513/0.498/0.461 vs 0.491/0.482/0.447), higher-contrast (std ≈0.265 vs ≈0.247) and more varied in brightness per image (0.144 vs 0.125). No surveyed paper ties CIFAR-10.1's drop to photometric statistics.

## Cross-Domain Findings

- **ImageNet → ImageNet-V2 (#3).** The same linear-trend law holds (R² 0.996) for a different dataset family. The CIFAR result is not a quirk of one dataset, so "raise clean accuracy" is the dependable lever.
- **Large-scale KD (#18) → small CIFAR students.** The "consistent + patient" recipe comes from ImageNet-scale compression. It transfers in principle, but #21 shows the CIFAR-10 headroom is much smaller.
- **Corruption robustness (CIFAR-10-C, AugMix) → natural shift.** Per #2 and #3 (CIFAR-10-C brightness R² only 0.94), this does **not** transfer reliably. One exception is cheap enough to try: photometric jitter (brightness/contrast) matches the measured statistics gap in Gap 4.

---

## Innovation Proposals (= concrete next steps for this competition)

### Proposal 1: Longer schedule + TrivialAugment on the current 4.84M AirResNet

**Description.** Keep the widest config from Sweep 1 (128/256/352). Add TrivialAugment, which includes brightness and contrast ops, on top of flip, translate and Cutout, and train 120–200 epochs instead of 40.

**Feasibility.** Data: CIFAR-10 train only. Compute: ~23 s/epoch on the M4 Pro, so 120 epochs ≈ 46 min and 200 ≈ 77 min. TrivialAugment needs per-image PIL-style ops, so it runs in CPU DataLoader workers on all 12 cores; the GPU may then wait on augmentation, so throughput must be measured first. Timeline: 1 day.

**Weaknesses.** airbench's LR/WD were tuned for 40 epochs, so the schedule needs one short re-tune. Stronger augmentation can lower clean accuracy at short schedules.

**Landing plan.** (1) A 40-epoch A/B with and without TA on the 45k/5k split, to check the pipeline and throughput. (2) 120-epoch run. (3) If it helps, a 200-epoch run. Success = holdout val ≥ 96.2% (predicts ≈ +1 pt on the hidden set). Fallback = only brightness/contrast jitter plus Cutout.

### Proposal 2: SAM on top of Proposal 1

**Description.** Swap SGD for SAM (ρ≈0.05) for the final recipe.

**Feasibility.** 2× compute per epoch. SAM at 100 epochs costs about the same as SGD at 200, so compare at equal wall-clock.

**Weaknesses.** Interaction with airbench's BN-bias LR scaling and Lookahead is untested. Gains reported on WRN may shrink on a 5M net.

**Landing plan.** Equal-time A/B against the best Proposal 1 run. Keep SAM only if holdout val improves by ≥0.2 pt.

### Proposal 3: Ensemble teacher → ≤5M student distillation ("consistent and patient")

**Description.** Train 2–3 teachers from scratch on CIFAR-10 train only, e.g. full-width airbench96 (9.9M) plus our best 4.84M net with different seeds. Then distil their averaged logits into the ≤5M student on identical augmented views, over a long schedule. Only the student is submitted.

**Feasibility.** Allowed by the rules ("Knowledge distillation using models trained from scratch on the permitted data"). Compute: about 25 min per teacher at 40 epochs, plus a student run ~1.3× normal cost. `train.py --teachers` already implements online, consistent-view KD.

**Weaknesses.** The careful CIFAR-10 study (#21) suggests only +0.2 pp. Teachers must use the same holdout split, or the student's validation score leaks.

**Landing plan.** Only after Proposals 1–2 settle the student recipe. Success = student holdout val ≥ best non-KD run + 0.3 pt. Fallback: skip KD.

### Evaluation protocol (applies to all three)

- Select models **only** on the 5k training-split holdout, never on the public leaderboard. It has ~1,000 images (95% CI ≈ ±2.2 pt), and the rules forbid "unauthorized model selection" on evaluation data.
- Use the Recht line to translate holdout accuracy into an expected hidden-set accuracy, and treat leaderboard moves under ~2 pt as noise.
- For the final submission, retrain the chosen recipe on all 50k images with the same hyperparameters.

---

## Devil's-advocate checkpoint (deep-research skill)

- **"Our 85% is 4 pt below the line, so something is wrong with the model."** Not established. The line predicts 1.62×95.6 − 65.5 ≈ **89.3%**, with an intercept uncertainty of ±3.2 pt. The public score uses ~1,000 images (±2.2 pt at 95%). Our 95.6% is measured on a train-split holdout and includes TTA, so the true CIFAR-10-test-equivalent may be a few tenths lower. The hidden set could also be CIFAR-10.2's 2,000-image test split rather than CIFAR-10.1. Its line differs, and nothing in the competition text says which one it is. Together these cover most of the gap. A genuine below-line effect for fast-training nets (Gap 1) is possible but unproven.
- **Cherry-picking check.** Counter-evidence is reported: #2 (most interventions don't help natural shift) and #21 (KD gains are small).
- **Rule-compliance check.** None of the proposals use pretrained weights, external data, test labels, or test images for training or selection. The unlabeled pixel statistics in Gap 4 are a description of the input format, not a training or selection signal.

## Limitations

- Retrieval was arXiv-only: Semantic Scholar returned HTTP 429 and DBLP returned errors for the whole session, so coverage of venue-only papers is incomplete, and citation counts are unavailable.
- Ten seminal-title lookups (e.g. PyramidNet, ShakeDrop, Bag of Tricks) hit arXiv rate limits. They are cited only through Recht et al.'s Table 1 or not at all ("gray zone = FAIL").
- Single reviewer, a narrative/rapid scope rather than a PRISMA systematic review. Full text was read for 6 papers; the rest are assessed from abstracts.
- **AI disclosure:** this survey was produced by Claude using the three named skill packages; all 25 references were re-verified against arXiv records.

## References

See `references.bib` (25 entries, fetched from arxiv.org/bibtex). Search log: `search_log.json`. Raw records: `papers_raw.json`.

---
**Generated:** 2026-10-29 | **Tool:** literature-survey skill v1.0 + ARS deep-research (three-way-scan) + K-Dense literature-review
