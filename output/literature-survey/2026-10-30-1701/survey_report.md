# Literature Survey: Methods that improve on SLA for a ≤5M-parameter ResNet trained from scratch on CIFAR-10 (round 3)

**Date:** 2026-10-30 | **Records retrieved:** 261 from 26 arXiv queries (254 unique) + forward citations of 7 seed papers, 2,344 unique records in all | **Title keyword match:** 892 | **Kept after title screen:** 50 | **Read in full:** 24 + 2 code benchmarks | **Date range of papers read:** 2018–2025 | **Search queries:** 26

Round 1 is in `../2026-10-29-1530/survey_report.md` and round 2 in `../2026-10-29-1705/survey_report.md`.
This round does not repeat their topics (augmentation policies, SAM, SWA, Shake-Shake, anti-aliasing, PAPA, the Recht line),
except where a method must be checked against SLA.

## How the three skills were used

| Skill | Use in this round |
|---|---|
| **academic-research-plugin / literature-survey** | Workflow and output layout. All 26 searches ran through its `scripts/paper_search.py` (`--sources arxiv --max-results 30`) with arXiv field syntax, on the Mac, because the arXiv and Semantic Scholar APIs are blocked from the cloud container. Forward citations of 7 seed papers (SLA, Hendrycks et al. 2019, HSAKD, CSI, Gated SSL, Soft Augmentation, Label Augmentation): Semantic Scholar first, OpenAlex as the fallback, saved in `citing/`. |
| **academic-research-skills / deep-research** | `three-way-scan` mode: WHY / HOW / WHAT for each paper read in full, with weaknesses marked (A) author-acknowledged or (R) reader-inferred. ARS arXiv protocol: one `id_list` request for 20 ids, title similarity ≥ 0.70 (20 passed, 0 failed, `seminal/arxiv_id_check.json`); BibTeX from arxiv.org/bibtex (`references.bib`, 21 entries). "Gray zone = FAIL": the 7 papers that could not be read (Decoupled SLA among them) are listed but carry no weight in the ranking. Devil's-advocate check on the ranking and on the report. |
| **scientific-agent-skills / literature-review** | Reproducibility: every query with its start date and count in `search_log.json`; raw results in `raw/`, merged records in `papers_raw.json`; keyword screen in `keyword_hits.txt`; title → abstract → full-text screening (counts in "How the survey was run"); 16 PDFs in `fulltext/`; limitations section. |

Inclusion: supervised image-classification methods that add auxiliary or joint labels, extra augmented views, self-distillation,
label softening, feature-space augmentation or weight averaging to a standard training recipe, and that can be used from scratch
on the CIFAR-10 training set with a final ResNet of ≤5M trainable parameters. Exclusion: methods that need pretrained weights,
outside or generated data, or any use of the test images for training, labeling or adaptation (competition rule); papers on
semi-supervised learning, out-of-distribution detection only, wide-network ensembles, or theory only (removed after the abstract).

---

## Answer: what to try next

**No published successor of SLA is a clear upgrade for us.** HSAKD improves distillation teachers, LoRot and the gated-SSL papers target imbalanced data, and the feature-level SLA variants are cheaper but slightly weaker. Where they report gains over SLA, the gains are 0.1 to 0.4 points from one research group each. The better path is to strengthen SLA itself: add a method that is known to stack with it (CutMix), and make better use of its four rotated copies.

No paper tests any change on top of SLA + TrivialAugment near 97.6%, so every gain below is an estimate. The estimates scale published gains to our 2.4% error. On the Kaggle hidden set, expect about 1.6 times the holdout gain (the Recht line).

| Rank | Method | Expected holdout gain | A100 time per seed (45k) | Evidence | Rules |
|---|---|---|---|---|---|
| 1 | CutMix inside SLA | +0.1 to +0.4 | about 85 min, no extra cost | Moderate. CutMix + SLA was the best CIFAR-10 result in the SLA paper (1.80% error), but without TrivialAugment. | OK |
| 2 | Own crop, flip and Cutout for each rotated copy | 0 to +0.3 | about 85 min, no extra cost | Indirect. Batch-augmentation papers; never tested with SLA. | OK |
| 3 | Weight average (EMA), evaluated beside the live network | 0 to +0.15 | free in every run | ResNet-18 on CIFAR-10: 95.25% to 95.62%, but at an earlier epoch with BatchNorm recomputed; equal at the end of the schedule. | OK |
| 4 | Joint-label head on stage 2 | 0 to +0.3 | about 90 min | Weak to moderate. HSAKD and the feature-level SLA papers, CIFAR-100 only. | OK, 4.86M even if counted |
| 5 | SLA-to-SLA distillation | 0 to +0.3 | about 2 h with one teacher | Moderate in papers, weak for us: SAM and our first distillation run added nothing. | OK, own teachers, cite |
| Queued | SLA for 300 epochs | 0 to +0.3 | about 2.1 h | Our log: +0.30 for 400 epochs without SLA. | OK |
| Queued | Two-view consistency | 0 to +0.3 | about 2.8 h | Close to rank 2, at twice the cost. | OK |

### Recommended order before the Nov 6 close

1. **Next HPC session (about 5 hours).** Add the flags from the method details. Run CutMix and per-copy augmentation at seed 0 (about 85 min each), then the queued 300-epoch SLA run (about 2.1 h). Every run also saves the weight-averaged network.
2. **Confirm.** Repeat at seed 1 any run that beats 97.56% by 0.2 points or more.
3. **Combine and retrain.** If two seeds confirm a gain of about 0.3, combine the winners in one 45k run, then retrain on all 50k (1.5 to 2.5 h). Submit by Nov 5 to keep one day of slack.
4. **Only if a session is left:** ranks 4 and 5. Run the queued two-view job only if per-copy augmentation helps, because the two test similar ideas.

## Where we stand

The recipe to beat is a pre-activation ResNet with SLA+SD rotation labels: 97.56% and 97.78% on two 45k holdout seeds, and 0.909 on Kaggle when trained on all 50k images.

| Part | Current setting |
|---|---|
| Network | PreActResNet, widths 64/128/256, 4/4/3 blocks, 4,706,674 parameters |
| SLA+SD | 40-way joint head (class x 4 rotations), joint loss averaged over the 4 copies, KL from the 4-rotation aggregate into the plain head, plain-head CE with label smoothing 0.1 |
| Data | TrivialAugment, flip, 4 px translate, 12 px Cutout |
| Optimizer | Nesterov SGD, lr 0.4 at batch 512, wd 5e-4, 5 warmup epochs, cosine, 200 epochs, no weight averaging |
| Inference | plain head + rotation aggregate, with mirror and 1 px shift TTA |
| A100 time | about 83 min on 45k, 92 min on 50k |

Already tried without a gain over SLA: SAM + SLA (97.58%), PAPA without SLA (members 97.2%), Shake-Shake 2x40d for 600 epochs (97.16%), anti-aliasing (97.00% vs 96.82%), 400 epochs without SLA (97.12%), distillation from two SLA teachers (97.18%). Queued: SLA for 300 epochs, and two-view consistency on SLA. Dropped by you: rotation x colour-permutation labels.

A change is kept only if it adds about 0.3 points on the holdout across two seeds, because the seed-to-seed spread is about 0.2.

## Method details and how to add them to train.py

All five changes go into the SLA branch of `compute_loss` or into the `Loader` in `src/train.py`. Each one gets its own flag, so one script can run them separately and then together.

### 1. CutMix inside SLA (`--cutmix 0.5`)

**How it works.** With probability 0.5 per batch, paste one box from a shuffled partner image into each image, with the box area set by λ ~ Beta(1, 1). Paste the box **before** `rotate4`, so the box turns with the image and the four copies share one rotation label for both parts. Mix the joint loss and the plain-head loss by area. The self-distillation term needs no change, because the aggregate teacher sees the same mixed image.

```python
lam, yb = 1.0, None
if args.cutmix > 0 and torch.rand(()) < args.cutmix:
    perm = torch.randperm(n, device=xs.device)
    lam = float(np.random.beta(1.0, 1.0))
    s = xs.size(-1); r = int(s * (1 - lam) ** 0.5)
    cy, cx = np.random.randint(s), np.random.randint(s)
    y0, y1 = max(cy - r // 2, 0), min(cy + r // 2, s)
    x0, x1 = max(cx - r // 2, 0), min(cx + r // 2, s)
    xs = xs.clone()
    xs[:, :, y0:y1, x0:x1] = xs[perm, :, y0:y1, x0:x1]
    lam = 1 - (y1 - y0) * (x1 - x0) / (s * s)
    yb = ys[perm]
# joint CE with labels (y*4 + k) for ys, and again for yb; plain CE likewise
# loss = lam * loss(ys) + (1 - lam) * loss(yb); the SD term is unchanged
```

**Cost:** none. **Risk:** it may only repeat what TrivialAugment + Cutout already do (see the devil's-advocate check). Settings p = 0.5 and Beta(1, 1) are the usual CIFAR settings reported in the DLB paper.

### 2. Independent augmentation for each rotated copy (`--sla-percopy`)

**Why.** Today `rotate4` rotates one augmented image, so the four copies share the crop, flip and Cutout (checked in `train.py`). If each copy gets its own crop, flip and Cutout, SLA becomes "batch augmentation" with four views at no extra cost, and the self-distillation teacher (the mean over the four copies) becomes an average over four different views.

**How.** In `Loader.__iter__`, build four views from the same TrivialAugment image (`self.padded`): four calls of `batch_crop`, four random flips, four `batch_cutout` calls, stacked to shape (B, 4, 3, 32, 32). In `compute_loss`, replace `rotate4(xs)` with:

```python
xr = torch.stack([torch.rot90(x[:, k], k, (2, 3)) for k in range(4)], 1).flatten(0, 1)
```

Copy 0 stays upright, so `z[::4]` still feeds the plain head. Flips happen before rotation, so the rotation labels stay well defined. The epoch tensor grows to about 2.2 GB in fp32 on 45k images, which fits on the A100.

**Cost:** none. **Relation to the queued two-view run:** two-view also gives each view its own crop, flip and Cutout, but doubles the compute and adds a consistency loss. Per-copy augmentation tests the cheaper half of that idea first.

### 3. Weight average for evaluation (`--ema 0.999`)

Keep a second copy of the network. After each optimizer step, move it toward the live weights: `e.lerp_(p, 1 - decay)` for every floating-point tensor, BatchNorm running statistics included (the same update as the `Lookahead` class, but **without** copying back into the live network). Decay 0.999 averages about the last 11 epochs at 88 steps per epoch. At every evaluation, recompute the BatchNorm statistics of the averaged copy with one pass over the training images (training data only, so it is allowed), then evaluate and save both networks with the same TTA. The published gain came from the best averaged epoch, near epoch 150 of 200, so log the averaged accuracy at every evaluation. For the 50k retrain, reuse the epoch chosen on the 45k run. **Cost:** under 1% (one forward pass over the training images at each evaluation). Add it to every run from now on.

### 4. Joint-label head on stage 2 (`--aux-joint 0.5`)

**Why.** HSAKD puts joint-label heads after every stage, and the feature-level SLA papers found that two joint heads beat one. Our joint head sees only the last stage.

**How.** Return the stage-2 output (128 channels, 16 x 16) from the PreActResNet forward pass. Add a small head: BatchNorm, ReLU, one 3 x 3 convolution 128 to 128 with stride 2, BatchNorm, ReLU, global pooling, and a 128 to 40 linear layer (about 0.15M parameters). Add `0.5 x` its joint cross-entropy, averaged over the four copies. Drop the head at inference, or keep it: 4.71M + 0.15M is still under 5M, so the count is safe either way.

**Cost:** a few percent.

### 5. SLA-to-SLA distillation (`--teachers` with `--ssl sla`)

**How.** Remove the `assert not (args.ssl != "none" and args.teachers)`. Run each teacher without gradients on the same rotated batch `xr`. Add α T² KL(teacher joint ‖ student joint), averaged over the copies, plus the same term for the plain heads, with T = 4 (the current `--kd-temp`) and α = 0.5.

**Teachers.** For the 45k test, use `preact_sla_ta_e200` and `preact_sla_ta_e200_sam05`; both trained on the seed-0 split, so the seed-0 holdout stays unseen. For the final 50k student, use `preact_sla_ta_e200_full`.

**Cost:** one teacher forward pass of 4 x 512 images per step, about +35% time per teacher. **Rules:** the teachers are our own CIFAR-10 networks, and the submitted student has 4.7M parameters. Cite HSAKD and the self-distillation papers in the report.

### Tier 3, only if a session is left

- **Soft crop labels** (Soft Augmentation): lower the target confidence by how much of the image the crop hides. With TrivialAugment the gain was only +0.12.
- **LoRot head**: one extra head that predicts the rotation of one image quadrant, weight 0.1. It was never tested together with SLA.
- **PS-KD**: mix the label with the model's own prediction from the last epoch (weight up to 0.8). Free, but it overlaps with SLA+SD and label smoothing.
- **ISDA**: class-wise feature-space augmentation. +0.1 to +0.2 on CIFAR-10 with AutoAugment.
- **Feature-level joint labels** (quadrant or channel masks on stage-2 and stage-3 features): +0.2 to +0.3 over SLA on CIFAR-100 in one paper.

## Methods rejected

Two filters removed methods before ranking: the competition rules, then the evidence and the A100 cost.

### Out by the rules

| Method | Why it is out |
|---|---|
| Test-time training or adaptation on the Kaggle images (TTT, TENT, MEMO, re-estimating BatchNorm statistics on test data) | It updates the model with test images. The brief forbids training on test images. |
| Pseudo-labels or semi-supervised learning on the Kaggle images (for example FixMatch) | It labels test images. |
| Self-supervised pretraining on outside images; pretrained teachers (ImageNet, CLIP) | Outside data and pretrained weights are not allowed. |
| PixMix, DeepAugment, images from diffusion models | They bring in outside images or pretrained generators. |
| Test-time ensembles of several networks | The submitted predictor would have more than 5M parameters. Rotation aggregation of one network is allowed and is already in use. |
| Full HSAKD auxiliary branches; MixSKD branches and discriminators | The extra branches are copies of later stages, so training-time parameters reach about 3 times the backbone. The brief counts trainable parameters, so this is a gray zone. Only the slim stage-2 head in the ranking fits under 5M when it is counted. |

### Out by evidence or cost

| Method | Evidence against it | Source |
|---|---|---|
| Colour-permutation labels, or rotation x colour labels | They lower CIFAR-10 accuracy: SLA+SD 91.51% against a 92.39% baseline. You also dropped this idea. | SLA, Table 3 |
| Rotation as plain augmentation, or as a separate rotation head | 90.44% (augmentation) and 90.79% (separate head) against a 92.39% baseline. SLA's joint labels are the form that works. | SLA, Table 1 |
| Gated or mixture-of-SSL heads (LoRot + flip + channel shuffle with a learned gate) | The best variant is +0.09 over LoRot alone on imbalanced CIFAR-10, from single runs. No balanced CIFAR-10 result. Clean accuracy fell in their adversarial setting. | Gated SSL; Mixture of SSL |
| Supervised contrastive learning, supervised CSI | Two training stages and 700 epochs. ResNet-18 reaches 94.8% on CIFAR-10. | CSI, Table 3 |
| MixSKD | At least 2 times the cost (an extra forward pass of the mixed images, branches, discriminators). No CIFAR-10 result. Not tested with strong augmentation. | MixSKD |
| Label Augmentation (2024) | 25-epoch training. Its baselines are far below ours. | Label Augmentation |
| AutoLabel | Needs held-out images to set the labels. On CIFAR-10 it adds +0.2 only on top of mixup. It is mainly a calibration method. | AutoLabel, Table V |
| Learned mixing (PuzzleMix, AutoMix, SAMix) | Extra modules or saliency passes. At 200 epochs they add +0.3 to +0.6 over CutMix on ResNet-18, and that benchmark has no TrivialAugment. | OpenMixup benchmark |
| Meta-auxiliary learning (MAXL) | Needs a second label-generator network and meta-gradients. CIFAR-10 gain is 92.77% to 93.49% on VGG-16 without regularisation. | MAXL, Table 1 |
| PAPA + SLA; online peer distillation | 2 to 4 times the compute. PAPA without SLA already lost to SLA. | Our log |

### Not assessed (no full text)

These were found by the citation search but could not be read, so they carry no weight in the ranking: Decoupled SLA (Knowledge-Based Systems 2022, closed, no abstract in OpenAlex), "A good data augmentation policy is not all you need" (IEEE TCSVT 2022, closed, abstract only), the rotation + multi-input multi-output ensemble (IEEE Access 2024, PDF fetch failed), HSSAKD (IEEE TNNLS), EntAugment, SAFLEX, and Stochastic Batch Augmentation.

## Evidence by theme

Each paper read in full got a WHY / HOW / WHAT scan. Numbers are copied from the papers' tables. Weaknesses are marked **(A)** when the authors state them and **(R)** when they are the reader's inference.

### Joint-label methods: SLA and its successors

- **SLA** (Lee, Hwang and Shin, ICML 2020). WHY: rotation used as plain augmentation hurts, because a rotated image is a different input; joint labels keep that information. HOW: N x 4 joint labels, self-distillation into a plain head, rotation aggregation at test time. WHAT: CIFAR-10 ResNet-32 92.39% to 93.26% (self-distilled head) and 94.50% (aggregation). With strong augmentation (Table 5, error): WRN-40-2 + Cutout 4.33% to 3.36%; + AutoAugment 3.70% to 2.95%; PyramidNet-200 + CutMix 2.88% to 1.80%. Weaknesses: the gain shrinks as accuracy rises, 95.26% to 96.19% for the stronger CIFAR-10 baseline in the appendix (R); SLA forwards four copies, so its comparisons are not at equal compute (R); ImageNet results come from one trial (A).
- **HSAKD** (Yang et al., IJCAI 2021). WHY: give the student richer teacher knowledge through joint-label distributions at every stage. HOW: joint-label heads after each stage, built from copies of the later stages, removed at test time. WHAT: CIFAR-100 teachers gain 4.09 points on average (WRN-40-2: 76.45% to 80.70%); ImageNet ResNet-34 73.31% to 75.48%. Weaknesses: no CIFAR-10 (R); the effects of deep supervision and of joint labels are not separated (R); the branches roughly triple the training parameters (R, from the code).
- **Feature-level SLA** (Yang et al. and Ding et al., arXiv 2021, one group). WHY: SLA costs about 3.9 times the training time; transforming feature maps is cheaper. HOW: channel-group or quadrant masks on the stage-2 and stage-3 features, with joint labels on two heads. WHAT (CIFAR-100, ResNet-110): about 0.4 points below SLA when used alone; on top of SLA 19.71% to 19.53% error, and 80.30% to 80.61% with aggregation; two joint heads beat one (76.80% against 75.79%); time +12% to +24%. Weaknesses: the group re-ran SLA itself; no CIFAR-10 result with SLA; no significance tests (R).
- **LoRot** (Moon, Kim and Heo, ECCV 2022). WHY: a pretext task for supervised learning should focus on the object and shift the inputs less than a full rotation. HOW: rotate one patch or quadrant, one extra head, weight 0.1, no extra forward pass. WHAT: imbalanced CIFAR-10 SLA+SD 80.24% against LoRot 81.82%; supervised contrastive learning on CIFAR-10 95.75% to 96.79%; on ImageNet no gain on top of CutMix. Weaknesses: never combined with SLA, and no TrivialAugment (R).
- Gated or mixture-of-SSL heads and Decoupled SLA: see Methods rejected.

### CutMix and SLA

- SLA Table 5 shows that the two stack: PyramidNet-200 + CutMix 2.88% to 1.80% error with SLA+SD on CIFAR-10, and 14.47% to 12.24% on CIFAR-100.
- OpenMixup benchmark (ResNet-18, crop and flip only): 200 epochs 94.87%, with CutMix 96.11%; 800 epochs 95.50%, with CutMix 96.68%.
- DLB paper, CIFAR-10, small networks, 3 runs: CutMix about equal to Cutout (WRN20-8 4.89% against 4.88% error; ResNet-110 4.82% against 4.64%).
- PS-KD paper, CIFAR-100, PyramidNet-200: Cutout 16.05% against CutMix 15.62% error.
- Weaknesses: no paper tests CutMix on top of TrivialAugment + Cutout near 97.5% on CIFAR-10 (R). The SLA paper does not say how it applied CutMix, and the official SLA code has none (checked) (R).

### Several augmented views of each image

- **Batch Augmentation** (Hoffer et al., CVPR 2020): WRN-28-10 + Cutout 96.6% to 97.15% with 6 views per image, and 96.95% against a baseline that sees the same number of images (Table 4); DARTS 97.11% to 97.64%. Mostly single runs (R).
- **Fort et al. (2021):** at the same number of updates, 4 views per image raise WRN-16-4 on CIFAR-100 from 72.1% to 73.7%; the gain stops growing near 4 views on CIFAR-100 (A).
- Link to our code: SLA already forwards four copies, but they differ only by rotation (checked in `train.py`). Neither paper uses rotation labels (R).

### Distillation and self-distillation

- **Revisiting self-distillation** (Pham et al., 2022): ResNet-18, CIFAR-10, Cutout + AutoAugment, 600 epochs: teacher 97.15% to student 97.40% ± 0.04 (3 runs); SAM alone 97.42%. The authors link both to flatter minima (A). Checkpoints were picked on test accuracy (R).
- **Born-again networks** (Furlanello et al., ICML 2018): CIFAR-10 changes from -0.17 to +0.31 points, single runs, no strong augmentation; some large networks got worse.
- **PS-KD** (Kim et al., ICCV 2021): CIFAR-100 PyramidNet-200: with Cutout 16.05% to 14.82% error; with CutMix 15.62% to 15.03%. No CIFAR-10; no extra compute.
- **DLB** (Shen et al., CVPR 2022): CIFAR-10 gains of 0.37 to 1.01 points, but on baselines of 93% to 95%.
- **MixSKD** (Yang et al., ECCV 2022): CIFAR-100 ResNet-18 76.24% to 80.32% (Mixup 78.68%). No CIFAR-10; training cost not reported (R).
- Our log: distillation from two SLA teachers into a student without SLA reached 97.18%, below SLA alone (97.56%), and SAM on top of SLA added nothing.

### Softer labels for strongly augmented images

- **Soft Augmentation** (Liu et al., CVPR 2023), CIFAR-10 error, 3 runs: WRN-28 with RandAugment 3.26% to 2.45%; PyramidNet + ShakeDrop with RandAugment 2.32% to 2.02%. On ImageNet-V2 the gain is larger than on ImageNet (ResNet-50: 1.65 against 0.96 points of error).
- **Siedel et al. (2025)** test the same idea on top of TrivialAugment (WRN-28-4, 5 runs): soft crop 96.64% to 96.76% (+0.12), but accuracy on corrupted images -1.17 points; a fully soft stack loses 0.46 points. They advise it only when a few homogeneous transforms dominate the training data (A).
- **AutoLabel** (Qin et al., 2023): CIFAR-10 with mixup 96.5% to 96.7%; CIFAR-100 with RandAugment 82.0% to 82.7%. It needs held-out images to set the labels.

### Feature-space augmentation

- **ISDA** (Wang et al., NeurIPS 2019), CIFAR-10 error with AutoAugment: WRN-28-10 2.65% to 2.56%; Shake-Shake 26 2x32d 2.89% to 2.68%.

### Weight averaging

- **Morales-Brotons et al. (TMLR 2024)**, CIFAR-10, ResNet-18, 200 epochs, cosine schedule, 3 runs: last weights 95.25% ± 0.11; weight average with BatchNorm recomputed 95.62% ± 0.11, and 95.39% without recomputing. The averaged model was best near epochs 143 to 155 and lost ground as the learning rate decayed further (A); with step decay it did not beat the baseline (A). So the gain needs BatchNorm recomputation and an earlier epoch picked on held-out data (R).
- **Rebuffi et al. (NeurIPS 2021)** pair CutMix with weight averaging only under adversarial training. Clean accuracy there moves from -2.55 to +2.00 points depending on the architecture, so it tells us nothing about standard training (R).

### The Kaggle hidden set

- **Hendrycks et al. (2019):** an auxiliary rotation head keeps CIFAR-10 clean accuracy (94.7% to 95.5%) and lifts accuracy on corrupted images from 72.3% to 76.9%.
- Our log: the airbench-style models sit about 4 points below the CIFAR-10.1 line on Kaggle, and the plain PreActResNet about 3 points below. SLA moved the public score from 0.884 to 0.909, while the line predicted about +1.2 points. Part of that gain comes from training on all 50k images, and the public set has only about 1,000 images, so this is a hint, not proof, that SLA-like changes pay off more on the hidden set than the holdout shows.

## Devil's-advocate check

**Verdict: pass, with major caveats.** The ranking is a sound order for cheap experiments. But no expected gain is proven for our recipe, and the most likely result of any one run is a change inside the seed noise.

**Critical issues:** none. No method in the top five uses test images or outside data, and each stays under 5M parameters, also when the stage-2 head is counted.

**Major issues**

1. **Transfer gap.** Most evidence comes from CIFAR-100, or from CIFAR-10 models at 93% to 96%. Gains get smaller near 97.6%. The estimates here keep the relative error cut, not the absolute gain. Example: self-distillation took ResNet-18 from 97.15% to 97.40%, a 9% cut in error; the same cut on our 2.4% error is about +0.2 points.
2. **Overlap with SLA.** SLA+SD already has a self-distillation term and four copies of each image, and SAM added nothing on top of SLA (97.58% against 97.56%). Distillation (rank 5) and per-copy augmentation (rank 2) act through the same paths, so they can give less than the papers show.
3. **Seed noise.** The seed-to-seed spread is about 0.2 points, so a +0.3 keep decision needs two seeds per method: about 3 hours on the A100, plus 1.5 to 2.5 hours to retrain the winner on all 50k.
4. **CutMix may only repeat Cutout.** In the DLB paper's CIFAR-10 table, CutMix was no better than Cutout on small networks (WRN20-8: 4.89% against 4.88% error). If CutMix only does what TrivialAugment + Cutout already do, its gain is zero.

**Minor issues**

- The SLA paper does not say how it combined CutMix with rotations, and the official code has no CutMix. Pasting one box before the four rotations is our design choice.
- An average of the weights usually adds little at the end of a cosine schedule.
- The Kaggle public set (about 1,000 images) cannot resolve a 0.3-point change; its standard error is about 1 point. Decisions must rest on the holdout.

**Strongest counter-argument.** "With a week left, only the queued 300-epoch SLA run has support from our own data (+0.30 for 400 epochs without SLA). New ideas risk ending the week with nothing confirmed." The answer: CutMix and per-copy augmentation each cost one normal run and fit in the same HPC session as the 300-epoch run, and the weight average is free. The plan keeps the 300-epoch run in the first session.

### Stress test of the main claims

| Claim | Test | Result |
|---|---|---|
| CutMix adds to SLA | SLA Table 5: CutMix + SLA+SD 1.80% error against CutMix alone 2.88% (PyramidNet-200) | Shows that SLA adds to CutMix, not that CutMix adds to SLA + TrivialAugment. Holds only in part. |
| Per-copy augmentation is free | Read `train.py`: `rotate4` rotates one augmented image, so the four copies share crop, flip and Cutout | Holds. The forward count stays the same; only the crop and Cutout ops repeat. |
| Distillation adds to SLA | Our own runs: distillation into a plain student 97.18%; SAM + SLA 97.58% | Weak. Kept at rank 5. |
| A stage-2 joint head helps | SFE Table 7: two joint heads 76.80% against one 75.79% (CIFAR-100, crop and flip only) | Weak to moderate, one research group. |
| Soft labels help with strong augmentation | Siedel et al.: on TrivialAugment, +0.12 accuracy but -1.17 corruption accuracy | Fails in our setting, so it is in tier 3. |
| Kaggle gain is about 1.6 times the holdout gain | Recht line and our three public scores | Holds roughly. SLA itself did better than the line; that this carries over to ranks 2 and 4 is a guess. |

## How the survey was run

The same three research skills as rounds 1 and 2 were used. No training ran and no HPC job was started.

1. **Literature survey** (academic-research-plugin). 26 arXiv field-syntax queries through `paper_search.py`, run on your Mac because the arXiv and Semantic Scholar APIs are blocked from the cloud container. One request every 4 s or more, no errors. 261 records, 254 unique (by arXiv id, then by normalized title). Forward citations of 7 seed papers through Semantic Scholar, with OpenAlex as the fallback: SLA 310 citing papers, Hendrycks et al. 2019 1,103, HSAKD 117, CSI 801, Gated SSL 10, Soft Augmentation 23, Label Augmentation 2.
2. **ARS deep-research.** The arXiv protocol (one connection, 3 s or more between requests). 20 of 20 arXiv ids passed the id check (title similarity 0.70 or more); `references.bib` has 21 entries from arxiv.org/bibtex. Every paper read in full got a WHY / HOW / WHAT scan with weaknesses marked as author-acknowledged or reader-inferred. Gray-zone rule: a reference that was not verified by an id check or a direct fetch is left out. Devil's-advocate checks ran on the ranking and on this report.
3. **K-Dense literature review.** Every query is logged with its date and count, and the screening counts are below.

| Screening stage | Papers |
|---|---|
| Unique records (26 searches + 7 citation lists) | 2,344 |
| Title has a survey keyword | 892 |
| Kept after title screening (37 from the lists, 13 seeds and planned reads) | 50 |
| Removed after the abstract (semi-supervised, out-of-distribution detection, wide-network ensembles, theory only) | 19 |
| Could not be read (closed or the fetch failed) | 7 |
| Read in full, plus 2 code-repository benchmarks | 24 |

The raw files are on your Mac in `output/literature-survey/2026-10-30-1701/`: the 26 query results, `papers_raw.json`, `search_log.json`, the 7 citation lists, `keyword_hits.txt`, `references.bib` and 16 PDFs. They are not committed.

### Limitations

- Only arXiv was searched by query. Journal-only papers were found only through the citation lists, and 7 relevant ones could not be read.
- Semantic Scholar rate-limited two seeds (it worked on the second attempt). OpenAlex now needs an API key for cursor paging, so page-number paging was used.
- Long PDFs were read as text, which can garble tables. Numbers were checked against the paper text, and garbled cells were left out.
- One reader screened and scanned the papers; there was no second reviewer.
- No paper tests any of these methods on top of SLA + TrivialAugment near 97.6% on CIFAR-10. Every expected gain in this report is an extrapolation.
- The view of the Kaggle hidden set comes from three public-leaderboard scores on about 1,000 images.

## References

Every arXiv id below passed the id check or was fetched and read in this survey. Items marked "record only" were found and their metadata was checked, but the paper could not be read.

### SLA and joint-label methods

- Lee, Hwang, Shin. Self-supervised Label Augmentation via Input Transformations. ICML 2020. [arXiv:1910.05872](https://arxiv.org/abs/1910.05872); official code [hankook/SLA](https://github.com/hankook/SLA)
- Yang et al. Hierarchical Self-supervised Augmented Knowledge Distillation. IJCAI 2021. [arXiv:2107.13715](https://arxiv.org/abs/2107.13715); code [winycg/HSAKD](https://github.com/winycg/HSAKD)
- Moon, Kim, Heo. Tailoring Self-Supervision for Supervised Learning (LoRot). ECCV 2022. [arXiv:2207.10023](https://arxiv.org/abs/2207.10023)
- Yang, Ding, Cheng, Wang, Liu. Self-supervised Feature Enhancement: Applying Internal Pretext Task to Supervised Learning. [arXiv:2106.04921](https://arxiv.org/abs/2106.04921)
- Ding, Yang, Cheng, Wang, Liu. Self-supervision of Feature Transformation for Further Improving Supervised Learning. [arXiv:2106.04922](https://arxiv.org/abs/2106.04922)
- Fuadi et al. Gated Self-supervised Learning for Improving Supervised Learning. [arXiv:2301.05865](https://arxiv.org/abs/2301.05865)
- Ruslim, Yudistira, Setiawan. Mixture of Self-Supervised Learning. [arXiv:2307.14897](https://arxiv.org/abs/2307.14897)
- Gao, Wu, Lam, Xia, Zou. Decoupled self-supervised label augmentation for fully-supervised image classification. Knowledge-Based Systems 235, 2022. [doi:10.1016/j.knosys.2021.107605](https://doi.org/10.1016/j.knosys.2021.107605) (record only)
- Hendrycks, Mazeika, Kadavath, Song. Using Self-Supervised Learning Can Improve Model Robustness and Uncertainty. NeurIPS 2019. [arXiv:1906.12340](https://arxiv.org/abs/1906.12340)
- Tack, Mo, Jeong, Shin. CSI: Novelty Detection via Contrastive Learning on Distributionally Shifted Instances. NeurIPS 2020. [arXiv:2007.08176](https://arxiv.org/abs/2007.08176)

### Mixing and multiple views

- Yun et al. CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features. ICCV 2019. [arXiv:1905.04899](https://arxiv.org/abs/1905.04899)
- OpenMixup CIFAR-10/100 mixup benchmark. [openmixup.readthedocs.io](https://openmixup.readthedocs.io/en/latest/mixup_benchmarks/Mixup_cifar.html)
- Hoffer et al. Augment Your Batch: Better Training with Larger Batches. [arXiv:1901.09335](https://arxiv.org/abs/1901.09335)
- Fort et al. Drawing Multiple Augmentation Samples Per Image During Training Efficiently Decreases Test Error. [arXiv:2105.13343](https://arxiv.org/abs/2105.13343)

### Distillation and self-distillation

- Pham et al. Revisiting Self-Distillation. [arXiv:2206.08491](https://arxiv.org/abs/2206.08491)
- Furlanello et al. Born-Again Neural Networks. ICML 2018. [arXiv:1805.04770](https://arxiv.org/abs/1805.04770)
- Kim et al. Self-Knowledge Distillation with Progressive Refinement of Targets (PS-KD). ICCV 2021. [arXiv:2006.12000](https://arxiv.org/abs/2006.12000)
- Shen et al. Self-Distillation from the Last Mini-Batch for Consistency Regularization (DLB). CVPR 2022. [arXiv:2203.16172](https://arxiv.org/abs/2203.16172)
- Yang et al. MixSKD: Self-Knowledge Distillation from Mixup for Image Recognition. ECCV 2022. [arXiv:2208.05768](https://arxiv.org/abs/2208.05768)
- Self-KD-Lib CIFAR-100 benchmark. [winycg/Self-KD-Lib](https://github.com/winycg/Self-KD-Lib)

### Label softening

- Liu et al. Soft Augmentation for Image Classification. CVPR 2023. [arXiv:2211.04625](https://arxiv.org/abs/2211.04625)
- Siedel, Gupta, Shao, Vock, Morozov. Combined Image Data Augmentations diminish the benefits of Adaptive Label Smoothing. [arXiv:2507.16427](https://arxiv.org/abs/2507.16427)
- Qin et al. What Are Effective Labels for Augmented Data? Improving Calibration and Robustness with AutoLabel. [arXiv:2302.11188](https://arxiv.org/abs/2302.11188)
- Amerehi, Healy. Label Augmentation for Neural Networks Robustness. CoLLAs 2024. [arXiv:2408.01977](https://arxiv.org/abs/2408.01977)

### Other methods

- Wang et al. Implicit Semantic Data Augmentation for Deep Networks (ISDA). NeurIPS 2019. [arXiv:1909.12220](https://arxiv.org/abs/1909.12220)
- Liu, Davison, Johns. Self-Supervised Generalisation with Meta Auxiliary Learning (MAXL). NeurIPS 2019. [arXiv:1901.08933](https://arxiv.org/abs/1901.08933)
- Morales-Brotons, Vogels, Hendrikx. Exponential Moving Average of Weights in Deep Learning: Dynamics and Benefits. TMLR 2024. [arXiv:2411.18704](https://arxiv.org/abs/2411.18704)
- Rebuffi et al. Data Augmentation Can Improve Robustness. NeurIPS 2021. [arXiv:2111.05328](https://arxiv.org/abs/2111.05328)
- Zhang, Ma. A Good Data Augmentation Policy is not All You Need: A Multi-Task Learning Perspective. IEEE TCSVT, 2023. [doi:10.1109/TCSVT.2022.3219339](https://doi.org/10.1109/TCSVT.2022.3219339) (record only)
- Park. Efficient Ensemble via Rotation-Based Self-Supervised Learning Technique and Multi-Input Multi-Output Network. IEEE Access 12, 2024. [doi:10.1109/ACCESS.2024.3373692](https://doi.org/10.1109/ACCESS.2024.3373692) (record only)

### Distribution shift and our earlier reports

- Recht, Roelofs, Schmidt, Shankar. Do CIFAR-10 Classifiers Generalize to CIFAR-10? [arXiv:1806.00451](https://arxiv.org/abs/1806.00451)
