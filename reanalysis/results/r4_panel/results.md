# Round 4, A2: expert panel with matched models, reference as a rater, selection check, form vs scan

> **Public-release copy.** The expert-identity section was removed. Per-evaluator rows and counts were removed, and per-leaf panel grades were removed (the sorted four-evaluator grades, set against the paper's five-evaluator per-leaf figure, would reveal the added evaluator's grades). Every number shown is unchanged; local paths are written as `<ROOT>` (the working tree) and `<USER_HOME>`.

PG-26038 major revision, 28 Sep 2026. Scripts in this folder: `run_forms.py` (CPU inference with onnxruntime on the 60 form images), `panel_analysis.py` (all statistics, `results.json`), `make_md.py` (this file). Form predictions: `form_predictions.csv`; inference metadata and hashes: `run_forms.json`; console logs: `run_forms.log`, `panel_analysis.log`. No training of any kind was run.

**Protocol (unchanged from `validation/`).** Form images are converted to the training format exactly as `validation/run_models.py` does (face cut out, pasted on black in a 1583 × 2048 frame at the median dataset leaf-height ratio R = 0.562, n = 46 scans), then letterboxed to 1024 × 1024, ImageNet-normalised, raw argmax; severity = lesion / (predicted leaf + lesion) × 100, as in `coffee-app/backend/infer.py`. Rust grade = abaxial face; leaf spot = mean of both faces; cut-points 0.5, 5 and 20%. Primary set = the 53 form leaves unseen in training (30 leaf spot, 23 rust; the 7 rust-training healthy leaves are excluded). The matched models were trained on the same packs and splits, and the check in `panel_analysis.py` confirms that the same 7 form leaves (and no others) are in their training split, so the same 53 leaves are unseen by them too. The 3 healthy leaves shown in the leaf-spot form (ojo11, ojo24, ojo27) are rust-pack training leaves; they were never seen by any leaf-spot model and are kept, as in the paper.

**Statistics (as `validation/tiefree.py` and `results/panel_stats`).** Leave-one-evaluator-out (LOO) QWK: for each evaluator j the reference is the median of the other three; rater value = mean over j of QWK(rater, reference j); the evaluators' value = mean over j of QWK(evaluator j, reference j). Contrast = rater minus evaluators. Bootstrap: 10,000 leaf resamples, seed 20260928, one resample vector per disease shared by all raters, evaluators held fixed (inference conditional on these four readers). Two-sided p = min(1, 2 min(P*(d ≤ 0), P*(d ≥ 0))). Published SegFormer-B5 = pre-specified primary (unadjusted). Secondary family = published DeepLabV3+ and UNet++ plus matched DeepLabV3+ and UNet++ (8 contrasts, Holm). A matched contrast is the seed-mean contrast: the mean over seeds 0, 1 and 2 of the model's LOO QWK, minus the evaluators' LOO QWK, bootstrapped as one statistic. Per-seed values are descriptive. Seed SD is the sample SD (ddof = 1). Evaluators are labelled E1–E4 in the column order of `validation/expert_consensus.csv`; no name is written next to any grade or statistic.

## 0. [Section omitted from the public release]

This section listed the expert-panel identities (names, occupations, affiliations, years of experience). Identities are not part of the code release.

## Reproduction checks

- Published models re-run by `run_forms.py` vs `validation/model_predictions.csv` (60 leaves, both faces): ojo_deeplab max |Δseverity| = 0.00 pp (n = 30); ojo_segformer max |Δseverity| = 0.00 pp (n = 30); ojo_unetpp max |Δseverity| = 0.00 pp (n = 30); roya_deeplab max |Δseverity| = 0.00 pp (n = 30); roya_segformer max |Δseverity| = 0.00 pp (n = 30); roya_unetpp max |Δseverity| = 0.00 pp (n = 30).
- ONNX files in `modal-recovery/serving/` are byte-identical (SHA-256) to `coffee-app/models/` used by `run_models.py`: all 6 identical.
- American leaf spot: published contrasts recomputed with the tiefree seed 7 give CIs SegFormer-B5 [0.061, 0.188] (tiefree [0.061, 0.188]); DeepLabV3+ [0.02, 0.175] (tiefree [0.02, 0.175]); UNet++ [-0.131, 0.093] (tiefree [-0.131, 0.093]).
- Coffee leaf rust: published contrasts recomputed with the tiefree seed 7 give CIs SegFormer-B5 [-0.015, 0.218] (tiefree [-0.015, 0.218]); DeepLabV3+ [-0.077, 0.263] (tiefree [-0.077, 0.263]); UNet++ [-0.035, 0.204] (tiefree [-0.035, 0.204]).

## 1. Matched models against the panel (53 unseen leaves)

Table 1. LOO QWK against the panel and contrast with the evaluators. Matched rows: mean ± SD over seeds 0–2 (range); contrast CI and p are for the seed-mean contrast.

| Disease (n) | Rater | LOO QWK | Difference vs evaluators [95% CI] | p (two-sided) | Holm, 8 secondary | Holm, 4 matched | Exact vs LOO | Inside panel range |
|---|---|---|---|---|---|---|---|---|
| American leaf spot (30) | Evaluators (mean; range across evaluators) | 0.765 (0.67–0.82) | — | — | — | — | 61% | — |
| | SegFormer-B5 published (primary) | 0.884 | +0.119 [0.061, 0.187] | <0.001 | — (primary) | — | 82% | 30/30 |
| | DeepLabV3+ published (secondary) | 0.860 | +0.095 [0.018, 0.172] | 0.018 | 0.15 | — | 78% | 30/30 |
| | UNet++ published (secondary) | 0.747 | −0.018 [−0.132, 0.091] | 0.75 | 1.00 | — | 65% | 30/30 |
| | DeepLabV3+ matched (secondary) | 0.808 ± 0.112 (0.680–0.884) | +0.043 [−0.023, 0.118] | 0.20 | 1.00 | 0.59 | 73% ± 13 | 29.7 ± 0.6 (29–30) |
| | UNet++ matched (secondary) | 0.842 ± 0.015 (0.834–0.860) | +0.077 [−0.004, 0.160] | 0.061 | 0.43 | 0.24 | 76% ± 2 | 30.0 ± 0.0 (30–30) |
| Coffee leaf rust (23) | Evaluators (mean; range across evaluators) | 0.728 (0.69–0.75) | — | — | — | — | 55% | — |
| | SegFormer-B5 published (primary) | 0.815 | +0.087 [−0.019, 0.220] | 0.100 | — (primary) | — | 72% | 21/23 |
| | DeepLabV3+ published (secondary) | 0.777 | +0.049 [−0.078, 0.263] | 0.47 | 1.00 | — | 72% | 21/23 |
| | UNet++ published (secondary) | 0.794 | +0.066 [−0.037, 0.206] | 0.18 | 1.00 | — | 70% | 20/23 |
| | DeepLabV3+ matched (secondary) | 0.763 ± 0.055 (0.704–0.814) | +0.035 [−0.086, 0.217] | 0.57 | 1.00 | 0.61 | 70% ± 2 | 20.3 ± 0.6 (20–21) |
| | UNet++ matched (secondary) | 0.792 ± 0.023 (0.769–0.815) | +0.064 [−0.068, 0.222] | 0.30 | 1.00 | 0.61 | 69% ± 3 | 20.3 ± 0.6 (20–21) |

Per-seed matched values (descriptive, unadjusted; each seed bootstrapped separately with the shared resamples):

| Disease | Model | Seed | LOO QWK | Difference [95% CI] | p | Exact vs LOO | Bias vs 4-median (ties up; down) | Inside range |
|---|---|---|---|---|---|---|---|---|
| American leaf spot | DeepLabV3+ | 0 | 0.884 | +0.119 [0.061, 0.187] | <0.001 | 82% | +0.03; +0.27 | 30/30 |
| American leaf spot | DeepLabV3+ | 1 | 0.860 | +0.095 [0.018, 0.172] | 0.018 | 78% | +0.07; +0.30 | 30/30 |
| American leaf spot | DeepLabV3+ | 2 | 0.680 | −0.085 [−0.208, 0.036] | 0.16 | 58% | +0.27; +0.50 | 29/30 |
| American leaf spot | UNet++ | 0 | 0.834 | +0.069 [−0.021, 0.158] | 0.12 | 75% | +0.10; +0.33 | 30/30 |
| American leaf spot | UNet++ | 1 | 0.860 | +0.095 [0.018, 0.172] | 0.018 | 78% | +0.07; +0.30 | 30/30 |
| American leaf spot | UNet++ | 2 | 0.834 | +0.069 [−0.021, 0.158] | 0.12 | 75% | +0.10; +0.33 | 30/30 |
| Coffee leaf rust | DeepLabV3+ | 0 | 0.773 | +0.044 [−0.105, 0.208] | 0.50 | 67% | −0.13; +0.26 | 20/23 |
| Coffee leaf rust | DeepLabV3+ | 1 | 0.814 | +0.086 [−0.050, 0.253] | 0.20 | 72% | −0.13; +0.26 | 21/23 |
| Coffee leaf rust | DeepLabV3+ | 2 | 0.704 | −0.024 [−0.195, 0.252] | 0.83 | 70% | −0.30; +0.09 | 20/23 |
| Coffee leaf rust | UNet++ | 0 | 0.769 | +0.041 [−0.115, 0.215] | 0.54 | 67% | −0.04; +0.35 | 20/23 |
| Coffee leaf rust | UNet++ | 1 | 0.815 | +0.087 [−0.019, 0.220] | 0.100 | 72% | +0.00; +0.39 | 21/23 |
| Coffee leaf rust | UNet++ | 2 | 0.792 | +0.064 [−0.089, 0.241] | 0.37 | 67% | −0.04; +0.35 | 20/23 |

Matched vs published (paired on the same bootstrap resamples; exploratory, unadjusted):

| Disease | Comparison | Difference in LOO QWK [95% CI] | p |
|---|---|---|---|
| American leaf spot | matched DeepLabV3+ (seed mean) − published DeepLabV3+ | −0.052 [−0.098, −0.006] | 0.023 |
| American leaf spot | matched DeepLabV3+ (seed mean) − published SegFormer-B5 | −0.076 [−0.129, −0.036] | <0.001 |
| American leaf spot | matched UNet++ (seed mean) − published UNet++ | +0.095 [0.012, 0.199] | 0.025 |
| American leaf spot | matched UNet++ (seed mean) − published SegFormer-B5 | −0.042 [−0.114, 0.000] | 0.23 |
| Coffee leaf rust | matched DeepLabV3+ (seed mean) − published DeepLabV3+ | −0.014 [−0.110, 0.063] | 0.70 |
| Coffee leaf rust | matched DeepLabV3+ (seed mean) − published SegFormer-B5 | −0.052 [−0.145, 0.058] | 0.32 |
| Coffee leaf rust | matched UNet++ (seed mean) − published UNet++ | −0.002 [−0.118, 0.109] | 0.91 |
| Coffee leaf rust | matched UNet++ (seed mean) − published SegFormer-B5 | −0.023 [−0.087, 0.021] | 0.70 |

Published contrasts with the round-4 seed and the 8-contrast Holm family (for comparison with `results/panel_stats`, which used seed 7 and Holm over 4): American leaf spot DeepLabV3+ +0.095 [0.018, 0.172], p 0.018, Holm-4 0.073, Holm-8 0.15; American leaf spot UNet++ −0.018 [−0.132, 0.091], p 0.75, Holm-4 0.94, Holm-8 1.00; Coffee leaf rust DeepLabV3+ +0.049 [−0.078, 0.263], p 0.47, Holm-4 0.94, Holm-8 1.00; Coffee leaf rust UNet++ +0.066 [−0.037, 0.206], p 0.18, Holm-4 0.54, Holm-8 1.00.

**Where the models differ (American leaf spot).** The 9 model grade vectors (3 published, 6 matched) differ on only 7 of 30 leaves: ojo01, ojo05, ojo10, ojo11, ojo22, ojo24, ojo27. Identical grade vectors (hence identical statistics): SegFormer-B5 pub. = DeepLabV3+ matched s0; DeepLabV3+ pub. = DeepLabV3+ matched s1; DeepLabV3+ pub. = UNet++ matched s1; DeepLabV3+ pub. = Reference (mask); DeepLabV3+ matched s1 = UNet++ matched s1; DeepLabV3+ matched s1 = Reference (mask); UNet++ matched s0 = UNet++ matched s2; UNet++ matched s1 = Reference (mask).

**Where the models differ (Coffee leaf rust).** The 9 model grade vectors (3 published, 6 matched) differ on only 10 of 23 leaves: roya07, roya09, roya10, roya13, roya15, roya17, roya21, roya23, roya27, roya29. Identical grade vectors (hence identical statistics): SegFormer-B5 pub. = UNet++ matched s1.

Panel-G0 leaves graded ≥ G1 (panel median G0 with ties rounded up; with ties rounded down) and grades outside the panel range:

| Rater | Leaf spot: panel-G0 graded ≥ G1 (up; down) | Leaf spot: below min / above max | Rust: panel-G0 graded ≥ G1 (up; down) | Rust: below min / above max |
|---|---|---|---|---|
| SegFormer-B5 pub. | 0/7; 3/10 | 0 / 0 | 0/3; 4/7 | 0 / 2 |
| DeepLabV3+ pub. | 1/7; 4/10 | 0 / 0 | 0/3; 3/7 | 2 / 0 |
| UNet++ pub. | 5/7; 8/10 | 0 / 0 | 0/3; 3/7 | 0 / 3 |
| DeepLabV3+ matched s0 | 0/7; 3/10 | 0 / 0 | 0/3; 3/7 | 1 / 2 |
| DeepLabV3+ matched s1 | 1/7; 4/10 | 0 / 0 | 0/3; 3/7 | 1 / 1 |
| DeepLabV3+ matched s2 | 7/7; 10/10 | 0 / 1 | 0/3; 3/7 | 3 / 0 |
| UNet++ matched s0 | 2/7; 5/10 | 0 / 0 | 0/3; 4/7 | 1 / 2 |
| UNet++ matched s1 | 1/7; 4/10 | 0 / 0 | 0/3; 4/7 | 0 / 2 |
| UNet++ matched s2 | 2/7; 5/10 | 0 / 0 | 0/3; 3/7 | 1 / 2 |
| Reference (mask) | 1/7; 4/10 | 0 / 0 | 0/3; 4/7 | 2 / 1 |
| Reference (visual) | 4/7; 7/10 | 0 / 0 | — | — |

## 2. The reference grade as a rater

Reference grades: rust = grade of the abaxial reference mask (detector + SAM 2, spot-checked); leaf spot = grade of the mean reference-mask severity of both faces (primary) and the annotator's visual grade (secondary). All 12 healthy form leaves have an empty reference mask on both faces in a local pack (10 in `pg26038-roya-real-pack`, roya02 and roya08 in `pg26038-ojo-lesions`), so their reference is G0.

Table 2a. Reference scored against the LOO consensus exactly like a model (primary set).

| Disease (n) | Rater | LOO QWK | Difference vs evaluators [95% CI] | p | Exact vs LOO | Pairwise QWK with evaluators (mean) | Inside panel range | Bias vs 4-median (ties up; down) | SegFormer-B5 minus reference, LOO QWK [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| American leaf spot (30) | Evaluators | 0.765 | — | — | 61% | 0.721 | — | — | — |
| | Reference, mask-derived (primary) | 0.860 | +0.095 [0.018, 0.172] | 0.018 | 78% | 0.792 | 30/30 | +0.07; +0.30 | +0.025 [0.000, 0.088] |
| | Reference, visual (secondary) | 0.778 | +0.013 [−0.097, 0.115] | 0.79 | 68% | 0.742 | 30/30 | +0.17; +0.40 | +0.107 [0.025, 0.220] |
| | SegFormer-B5 published (for comparison) | 0.884 | see Table 1 | | 82% | 0.808 | 30/30 | +0.03; +0.27 | — |
| Coffee leaf rust (23) | Evaluators | 0.728 | — | — | 55% | 0.769 | — | — | — |
| | Reference, mask-derived | 0.743 | +0.015 [−0.140, 0.237] | 0.81 | 67% | 0.698 | 20/23 | −0.13; +0.26 | +0.072 [−0.063, 0.182] |
| | SegFormer-B5 published (for comparison) | 0.815 | see Table 1 | | 72% | 0.778 | 21/23 | +0.00; +0.39 | — |

Paired reference contrast (V2, same bootstrap as Table 2a): leaf-spot mask-derived minus visual reference LOO QWK +0.082 [0.000, 0.183], two-sided p = 0.088 (10,000 leaf resamples, seed 20260928). The mask-derived grade's higher agreement is therefore descriptive, not significant.

Table 2b. Per-evaluator agreement with the reference grade (primary set; QWK CI from 10,000 leaf resamples, seed 20260928). 'Higher' = evaluator grade above the reference.

| Disease (n) | Reference | Evaluator | Exact | Within one | Higher / lower | Mean (evaluator − reference) | QWK [95% CI] |
|---|---|---|---|---|---|---|---|
| | mask | 4-evaluator median, ties up / down | 26/30 (87%) / 21/30 (70%) | 30/30 (100%) / 30/30 (100%) | | | 0.913 / 0.809 |
| | visual | 4-evaluator median, ties up / down | 23/30 (77%) / 18/30 (60%) | 30/30 (100%) / 30/30 (100%) | | | 0.831 / 0.727 |
| | mask | 4-evaluator median, ties up / down | 18/23 (78%) / 13/23 (57%) | 23/23 (100%) / 23/23 (100%) | | | 0.823 / 0.667 |

[Per-evaluator counts omitted from the public release.]

Table 2c. Leaves where all four evaluators disagree with the reference grade (all 60 form leaves; 'primary' marks the 53).

| Disease | Reference | Form item | Leaf ID | Reference grade | Reference severity used (%) | Abaxial / adaxial (%) | Panel grades (omitted) | Direction | Published SegFormer-B5 form grade | Primary |
|---|---|---|---|---|---|---|---|---|---|---|
| Coffee leaf rust | mask | roya07 | 20260819_Hvastatrix_GX_009 | G2 | 16.65 | 16.65 / 15.95 | omitted | all higher | G3 | yes |
| Coffee leaf rust | mask | roya15 | 20260819_Hvastatrix_GX_020 | G2 | 19.34 | 19.34 / 10.46 | omitted | all higher | G3 | yes |
| Coffee leaf rust | mask | roya21 | 20260819_Hvastatrix_GX_095 | G2 | 7.42 | 7.42 / 6.14 | omitted | all lower | G2 | yes |

**Rust G3 check.** Panel-unanimous G3 rust leaves: roya07, roya15. Rust reference (abaxial mask) grade counts, diseased leaves of the local rust pack, native resolution (G0/G1/G2/G3): train 0/66/31/2; val 0/16/5/0; test 1/11/9/0. G3 reference leaves: 20260819_Hvastatrix_GX_072 (train, 21.73%), 20260819_Hvastatrix_GX_119 (train, 23.41%).

## 3. Panel-selected vs unselected leaf-spot test leaves (original scans)

55 leaf-spot test leaves: 27 shown to the panel, 28 not shown. Values from `reanalysis/out/per_image.csv` (published models on the scans, leaf = mean of both faces). Difference = selected minus unselected. Permutation test: 10,000 random re-assignments of the 27 'selected' labels among the 55 leaves, seed 20260928, two-sided, p = (1 + count)/(10,001). CI: bootstrap within groups (10,000). All 21 diseased rust test leaves were shown to the panel, so rust has no unselected diseased test leaves to compare.

| Quantity | Selected (n = 27) mean; median | Unselected (n = 28) mean; median | Difference in means [95% CI] | p (perm., mean) | p (perm., median) |
|---|---|---|---|---|---|
| Reference severity (%) | 6.68; 4.16 | 4.01; 4.04 | +2.67 [0.01, 5.53] | 0.067 | 0.95 |
| SegFormer-B5 |error| (pp) | 0.31; 0.16 | 0.13; 0.07 | +0.18 [0.04, 0.33] | 0.018 (V2) | 0.20 |
| SegFormer-B5 signed error (pp) | 0.23; 0.09 | 0.10; 0.05 | +0.14 [−0.02, 0.31] | 0.12 | 0.55 |
| SegFormer-B5 leaf IoU | 0.878; 0.904 | 0.891; 0.903 | −0.012 [−0.044, 0.019] | 0.46 | 0.92 |
| SegFormer-B5 grade correct (mask reference) | 0.926; 1.000 | 0.964; 1.000 | −0.038 [−0.149, 0.071] | 0.61 | 1.00 |
| SegFormer-B5 grade correct (visual reference) | 0.815; 1.000 | 0.857; 1.000 | −0.042 [−0.228, 0.142] | 0.74 | 1.00 |

- Grade mix, visual grade (G0/G1/G2/G3): selected 0/14/11/2, unselected 0/16/12/0; permutation chi-square p = 0.52; mean grade difference +0.13, p = 0.48.
- Grade mix, mask-derived grade (G0/G1/G2/G3): selected 3/11/11/2, unselected 3/15/10/0; permutation chi-square p = 0.55; mean grade difference +0.19, p = 0.35.
- Reference severity range: selected 0.28–26.78%, unselected 0.13–10.81%. Collection date (06-21 / 06-28): selected 14/13, unselected 22/6; exact two-sided p = 0.050 (Fisher exact = exact permutation chi-square, 0.0496; the 10,000-permutation estimate was 0.051) (V2).
- Severity-range-matched sensitivity (selected leaves above the unselected maximum of 10.81% dropped: 20260628_Mcitricolor_GX_007, 20260628_Mcitricolor_GX_031, 20260628_Mcitricolor_GX_064, 20260628_Mcitricolor_GX_078, 20260628_Mcitricolor_GX_123; n = 22 vs 28): reference severity 4.15 vs 4.01% (p = 0.88); SegFormer-B5 |error| 0.21 vs 0.13 pp (p = 0.25); signed error +0.12 vs +0.10 pp (p = 0.81); leaf IoU 0.864 vs 0.891 (p = 0.11).
- SegFormer-B5 |error| relative to reference severity: median 0.041 vs 0.034 (permutation p, median 0.26; mean 0.52).
- Published DeepLabV3+ (secondary): |error| 0.34 vs 0.16 pp (p = 0.019); leaf IoU 0.810 vs 0.819 (p = 0.75); grade correct (visual) 0.852 vs 0.857 (p = 1.00).
- Published UNet++ (secondary): |error| 0.19 vs 0.11 pp (p = 0.12); leaf IoU 0.848 vs 0.857 (p = 0.70); grade correct (visual) 0.889 vs 0.857 (p = 1.00).

## 4. Model grade on the form image vs on the original scan

Panel leaves with a test scan in `per_image.csv` / `per_image_retrained.csv`: 48 (27 leaf spot, 21 rust). Not available: 12 healthy form leaves (7 rust-training leaves, roya02 and roya08, which are in no evaluated split, and the 3 healthy leaves of the leaf-spot form, which are rust-pack training leaves). Same grade rule on both images (rust abaxial, leaf spot mean of faces). Severity difference = form minus scan (pp); LoA = mean ± 1.96 SD. QWK CIs: 10,000 leaf resamples, seed 20260928 (percentile). Same-grade proportions: Wilson score 95% CI (V3), because the percentile bootstrap of a proportion with 0–1 discordant leaves is degenerate at 100% and too narrow below.

| Disease (n) | Model | Same grade [95% Wilson CI] (V3; percentile-bootstrap CI in parentheses) | Within one | Form grade higher / lower | QWK form vs scan [95% CI] | Severity difference mean (SD) [LoA] | Spearman |
|---|---|---|---|---|---|---|---|
| American leaf spot (27) | SegFormer-B5 published | 26/27 (96%) [82, 99%] (bootstrap 89, 100%) | 27/27 | 1 / 0 | 0.974 [0.91, 1.00] | −0.06 (0.27) [−0.59, 0.48] | 0.995 |
| American leaf spot (27) | DeepLabV3+ published | 26/27 (96%) [82, 99%] (bootstrap 89, 100%) | 27/27 | 1 / 0 | 0.972 [0.90, 1.00] | −0.27 (0.43) [−1.12, 0.57] | 0.992 |
| American leaf spot (27) | UNet++ published | 24/27 (89%) [72, 96%] (bootstrap 74, 100%) | 27/27 | 3 / 0 | 0.892 [0.77, 1.00] | −0.07 (0.31) [−0.68, 0.55] | 0.990 |
| American leaf spot (27) | DeepLabV3+ matched, seeds 0–2 | 25.3 ± 1.5 of 27 (24–27) | 27.0 of 27 | — | 0.946 ± 0.054 | −0.00 ± 0.36 (SD 0.34) | 0.994 |
| American leaf spot (27) | UNet++ matched, seeds 0–2 | 26.3 ± 0.6 of 27 (26–27) | 27.0 of 27 | — | 0.979 ± 0.018 | −0.04 ± 0.04 (SD 0.26) | 0.996 |
| Coffee leaf rust (21) | SegFormer-B5 published | 20/21 (95%) [77, 99%] (bootstrap 86, 100%) | 21/21 | 1 / 0 | 0.951 [0.84, 1.00] | +0.32 (1.49) [−2.60, 3.23] | 0.990 |
| Coffee leaf rust (21) | DeepLabV3+ published | 16/21 (76%) [55, 89%] (bootstrap 57, 90%) | 21/21 | 0 / 5 | 0.726 [0.47, 0.90] | −2.18 (2.13) [−6.36, 2.00] | 0.992 |
| Coffee leaf rust (21) | UNet++ published | 15/21 (71%) [50, 86%] (bootstrap 52, 90%) | 21/21 | 3 / 3 | 0.788 [0.55, 0.92] | +1.88 (6.62) [−11.10, 14.85] | 0.957 |
| Coffee leaf rust (21) | DeepLabV3+ matched, seeds 0–2 | 17.7 ± 0.6 of 21 (17–18) | 21.0 of 21 | — | 0.821 ± 0.030 | −0.25 ± 0.91 (SD 2.04) | 0.975 |
| Coffee leaf rust (21) | UNet++ matched, seeds 0–2 | 19.7 ± 0.6 of 21 (19–20) | 21.0 of 21 | — | 0.936 ± 0.022 | +0.49 ± 0.45 (SD 2.70) | 0.976 |

SegFormer-B5, both diseases: 46/48 diseased panel leaves same grade (96%; Wilson 95% CI 86–99%) (V3). Grade disagreements between form and scan, published SegFormer-B5: ojo17 form 0.57% (G1) vs scan 0.49% (G0); roya07 form 24.82% (G3) vs scan 19.47% (G2).

Grade counts on the 21 rust test leaves, form image vs scan (G0/G1/G2/G3; answers the reconciliation TODO(analysis) in 02_methods.tex): SegFormer-B5 form 1/10/8/2, scan 1/10/9/1; DeepLabV3+ form 2/12/7/0, scan 1/10/9/1; UNet++ form 2/9/6/4, scan 1/9/9/2. Reference (abaxial mask): 1/11/9/0. Leaf spot, 27 test leaves: SegFormer-B5 form 4/10/11/2, scan 5/9/11/2; DeepLabV3+ form 3/11/11/2, scan 4/10/11/2; UNet++ form 0/14/11/2, scan 3/11/11/2.

Other grade disagreements (published models): DeepLabV3+ leaf spot: ojo10 0.66→0.47% (G1/G0); UNet++ leaf spot: ojo01 0.73→0.34% (G1/G0), ojo05 0.57→0.39% (G1/G0), ojo22 0.53→0.29% (G1/G0); DeepLabV3+ rust: roya13 3.61→5.29% (G1/G2), roya15 17.14→22.65% (G2/G3), roya21 4.48→7.61% (G1/G2), roya27 3.13→5.23% (G1/G2), roya29 0.26→0.67% (G0/G1); UNet++ rust: roya09 38.73→10.35% (G3/G2), roya10 5.02→4.14% (G2/G1), roya13 4.77→5.39% (G1/G2), roya17 24.3→17.74% (G3/G2), roya27 4.07→5.46% (G1/G2), roya29 0.4→0.75% (G0/G1) (form→scan severity; grades form/scan).

## 5. Honest reading

1. **Matched models vs the panel.** No matched configuration differed detectably from the evaluators after correction. Leaf spot: matched UNet++ 0.842 ± 0.015, contrast +0.077 [−0.004, 0.160] (Holm-8 p = 0.43); matched DeepLabV3+ 0.808 ± 0.112 (seeds 0.884, 0.860, 0.680), contrast +0.043 [−0.023, 0.118]. Rust: matched UNet++ 0.792 ± 0.023, +0.064 [−0.068, 0.222]; matched DeepLabV3+ 0.763 ± 0.055, +0.035 [−0.086, 0.217]. The primary SegFormer-B5 result is unchanged (leaf spot +0.119 [0.061, 0.187], p <0.001; rust +0.087 [−0.019, 0.220], p = 0.100 with seed 20260928 vs 0.084 with the tiefree seed 7: Monte Carlo variation, same conclusion). Under the 8-contrast secondary family no secondary contrast is significant (smallest Holm p = 0.15, published DeepLabV3+ on leaf spot).
2. **Seeds matter as much as architecture on the panel.** On leaf spot, matched DeepLabV3+ spans 0.680–0.884 across seeds; seed 0 gives exactly the published SegFormer-B5 grades and seed 2 grades all 7 panel-G0 leaves as G1. The nine model grade vectors differ on only 7 of 30 leaf-spot leaves, all near-healthy leaves at the G0/G1 boundary (3 of them healthy), so every leaf-spot model–panel contrast is decided by those 7 leaves. On rust the models differ on 10 of 23 leaves. The matched-vs-published CIs (e.g., matched DeepLabV3+ − SegFormer-B5 on leaf spot −0.076 [−0.129, −0.036]) resample leaves with the three seeds fixed; they do not include seed-to-seed variability and SegFormer-B5 has a single run, so they must not be read as an architecture ranking.
3. **Matched UNet++ no longer over-grades near-healthy leaf-spot leaves** as the published UNet++ did (panel-G0 leaves graded G1, ties up: published 5/7; matched seeds 2, 1, 2/7), and its seed-mean LOO QWK is higher than the published UNet++ (+0.095 [0.012, 0.199]; exploratory, unadjusted). Matched DeepLabV3+ seed 2 shows the same over-grading (7/7).
4. **The reference as a rater.** For leaf spot, the mask-derived grade had a higher LOO QWK with the panel than the annotator's visual grade (0.860 vs 0.778; mask-derived minus evaluators +0.095 [0.018, 0.172], visual +0.013 [−0.097, 0.115]), but the paired difference mask-derived minus visual, +0.082 [0.000, 0.183] (p = 0.088), is not significant (V2); this is consistent with, but does not by itself establish, the author's decision to make the mask-derived grade primary. The visual grade over-grades panel-G0 leaves (4/7 graded G1). For rust, the reference grade's LOO QWK was close to the evaluators' mean (0.743 vs 0.728, +0.015 [−0.140, 0.237]); the interval is wide and no equivalence test was done, so this is not evidence of equivalence (V2). SegFormer-B5 minus the reference: leaf spot +0.025 [0.000, 0.088] (mask), +0.107 [0.025, 0.220] (visual); rust +0.072 [−0.063, 0.182]. Every evaluator was within one grade of the reference on every leaf; exact agreement per evaluator was 57–77% against the primary references (leaf-spot mask-derived 60–77%, rust 57–70%) and 50–77% against the leaf-spot visual grade (V2).
5. **Unanimous disagreements are all rust and all near a cut-point.** All four evaluators disagreed with the rust reference on 3 of 23 primary leaves: roya07 (16.65%, reference G2) and roya15 (19.34%, G2), both graded G3 by all four, and roya21 (7.42%, G2), graded G1 by all four. The claim to verify holds in this form: no rust *test* (panel) reference leaf is G3 (test 1/11/9/0), but the training split has 2 G3 reference leaves (21.73% and 23.41%), so 'no rust reference leaf is G3' is true only of the held-out/panel leaves. On the original scans all three published models predict more lesion than the reference on roya07 (18.6–20.6% vs 16.65%) and roya15 (22.6–25.1% vs 19.34%), i.e., on the side of the panel; on roya21 they predict 7.6–9.6% (reference 7.42%). No leaf-spot leaf had all four evaluators disagreeing with either reference.
6. **Selection.** The 27 panel leaf-spot leaves did not differ detectably from the 28 unselected test leaves in grade mix (visual p = 0.52, mask-derived p = 0.55) or median severity (4.16 vs 4.04%, p = 0.95), but the panel set holds all 5 test leaves above 10.81% (both visual-G3 leaves included), so its mean severity is higher (6.68 vs 4.01%, p = 0.067) and more of it comes from 28 June (13/27 vs 6/28, exact p = 0.050, borderline). SegFormer-B5 absolute error was larger on the panel leaves (0.31 vs 0.13 pp, permutation p = 0.018), but not after removing the 5 high-severity leaves (p = 0.25) or relative to severity (p = 0.26), consistent with the proportional error found earlier; leaf IoU and grade accuracy did not differ (p ≥ 0.46). The item-selection rule was not retained, so it cannot be stated how the 27 were chosen.
7. **Form image vs scan.** SegFormer-B5 gave the same grade on the form image and the scan for 26/27 leaf-spot and 20/21 rust leaves; both exceptions straddle a cut-point (ojo17: 0.57% on the form vs 0.49% on the scan; roya07: 24.82% vs 19.47%). The CNNs were less stable on rust (published DeepLabV3+ 16/21, lower on the form by 2.18 pp on average; published UNet++ 15/21, LoA −11.1 to 14.9 pp). Model–panel agreement on the forms is therefore a close but not exact proxy for model behavior on the scans; for SegFormer-B5 the proxy is good (2 of 48 grades differ).

## 6. Suggested text (LaTeX-ready; numbers from this folder)

```latex
% Methods, expert panel (after the Holm sentence)
The matched-protocol DeepLabV3+ and UNet++ (three seeds each) were run on the same form images with the same pipeline; their seed-mean contrasts were added to the secondary family, which was Holm-corrected over eight contrasts (four published, four matched). The reference grade (rust: abaxial mask; American leaf spot: mask-derived, with the visual grade as a secondary reference) was scored against the panel in the same way as a model.

% Results, expert panel
Under matched training, neither convolutional model differed detectably from the evaluators (American leaf spot: UNet++ +0.077 [$-$0.004, 0.160], DeepLabV3+ +0.043 [$-$0.023, 0.118]; rust: UNet++ +0.064 [$-$0.068, 0.222], DeepLabV3+ +0.035 [$-$0.086, 0.217]; all Holm $p \geq$ 0.43), and the spread across seeds (American leaf spot DeepLabV3+ LOO QWK 0.68--0.88) was as large as any difference between architectures; all nine model runs graded 23 of the 30 American leaf spot leaves identically, so these contrasts are decided by 7 near-healthy leaves.
Scored as a rater, the mask-derived American leaf spot grade had a higher leave-one-out QWK with the panel than the annotator's visual grade (0.86 vs 0.78; paired difference +0.08 [0.00, 0.18], $p$ = 0.088), and the rust reference was close to the evaluators' mean (0.74 vs 0.73; difference +0.02 [$-$0.14, 0.24]); all four evaluators disagreed with the reference on three rust leaves only, two graded G3 by all four evaluators although their reference severity (16.65\% and 19.34\%) lay below the 20\% cut-point.
The 27 American leaf spot test leaves shown to the panel did not differ detectably from the 28 not shown in grade mix (permutation $p$ = 0.52) or median severity ($p$ = 0.95), although they included all five test leaves above 11\% severity; SegFormer-B5 gave the same grade on the form image and on the original scan for 46 of the 48 diseased panel leaves.
```

## Notes and limits

- Matched models: 12 ONNX files (lrB, seeds 0–2, 2 architectures × 2 diseases), SHA-256 in `run_forms.json`. SegFormer-B5 exists as a single published run; its seed variability on the panel is unknown.
- The bootstrap treats evaluators and seeds as fixed. The seed SD over 3 seeds is itself very imprecise.
- Per-evaluator values are omitted from this public copy.
- Healthy form leaves: their reference G0 rests on an empty lesion mask present locally for all 12 (checked in `panel_analysis.py`).
- Not determinable from the files: how the 27 leaf-spot items were selected (rule not retained); why form-image and scan severities differ on individual leaves (the form pages are 1422 × 1516 px JPEG composites, about 711 px wide per face; the resizing and compression chain used to build the form is not recorded).
- The published SegFormer-B5 rust p-value depends on the bootstrap seed at the second decimal (0.084 with seed 7, 0.093 with 20260927, 0.100 with 20260928); the manuscript should keep one seed consistently (the existing text uses seed 7, which reproduces `tiefree_results.json` exactly).


## Verifier corrections

Independent verification (verifier 2, 28 Sep 2026; scripts and outputs in `verifier2/`; the earlier incomplete `verifier/` folder was not used or changed). Items changed in the text above are marked "(V2)".

**What was re-derived independently and confirmed.**
- Own re-implementation of the `run_models.py` / `infer.py` protocol (`verifier2/vv_infer.py`) ran all 12 matched lrB ONNX files (seeds 0–2) and published DeepLabV3+ (both diseases) on the 60 form images on CPU: every severity equals `form_predictions.csv` (matched) and `validation/model_predictions.csv` (published) exactly (max |Δ| = 0.00 pp, 30 leaves × 2 faces per run). The 6 served ONNX files are byte-identical (SHA-256) to `coffee-app/models/`. Matched kernels use `pg26038-ojo-real-pack` / `pg26038-roya-real-pack` (kernel-metadata.json), so the 53-leaf primary set (30 leaf spot, 23 rust) is unseen by them.
- Own statistics (`verifier2/vv_stats.py`, `vv_paired.py`; own QWK, own LOO medians, own bootstrap loop with seed 20260928; reference grades recomputed from the pack masks at native resolution and checked against the letterbox `ref_sev` grades): every Table 1 value (LOO QWK, contrasts, CIs, p, Holm-8, per-seed values, exact-vs-LOO, inside-range, panel-G0 over-grading), the model-disagreement leaves (7 leaf spot, 10 rust), Table 2a/2b values, the three unanimous rust disagreements (roya07, roya15, roya21), the rust pack grade counts (train 0/66/31/2, val 0/16/5/0, test 1/11/9/0; G3 training leaves 21.73% and 23.41%), the Section 3 means/medians/grade mixes, and every Section 4 count, QWK, disagreement leaf and severity difference/LoA were reproduced to the printed precision. Matched-vs-published paired contrasts reproduced (p differs from the analyst's only in the 3rd decimal, e.g., 0.024 vs 0.025).
- Section 0 identity table re-read from the form export zip (columns 2, 4, 6, 7 = name, role, institution, years): names, roles, affiliations and years match exactly; median 30, mean 32.25 years. `validation/expert_grades_raw.csv` is byte-identical to the export (5,149 bytes) and the `expert_consensus.csv` column order equals the submission order: the privacy flag is correct. No name appears in `results.json`, `form_predictions.csv` or `run_forms.json`.

**Corrections.**
1. Section 3, SegFormer-B5 |error| selected vs unselected: permutation p 0.015 → **0.018** (200,000 permutations, `vv_perm_precise.py`; an independent 10,000-permutation run gave 0.019). Conclusion unchanged.
2. Section 3, collection date: the 10,000-permutation p = 0.051 is replaced by the exact two-sided value **p = 0.050** (Fisher exact, 0.0496); it is borderline and exploratory, so it must not be described as either "significant" or "no difference".
3. Sections 2 and 5 and the LaTeX: the claim that the mask-derived leaf-spot grade agrees with the panel "better" than the visual grade was not tested by the analyst. The paired difference (mask-derived minus visual LOO QWK) is **+0.082 [0.000, 0.183], p = 0.088**: descriptively higher, not significant. Wording changed to "higher LOO QWK … consistent with" the author's decision, not "supports".
4. Sections 5 and LaTeX: "the rust reference agreed like a typical evaluator" implied equivalence; no equivalence (TOST) test was done and the CI is wide (+0.015 [−0.140, 0.237]). Reworded to "close to the evaluators' mean".
5. LaTeX: "did not differ from the 28 not shown" → "did not differ detectably"; the causal "because all models graded 23 of the 30 … identically" → a statement that the leaf-spot contrasts are decided by 7 near-healthy leaves.
6. Per-evaluator exact agreement range clarified: 57–77% holds for the primary references only; against the secondary (visual) leaf-spot grade it is 50–77%.

**Not changed, noted.** The SegFormer-B5 minus mask-derived reference CI [0.000, 0.088] has a two-sided p = 0.74 because most resamples give exactly 0 (the grade vectors differ on few leaves); quote the CI, not a p. The Section 4 "same grade" CIs are percentile-bootstrap intervals of a proportion (a Wilson interval would be more standard). The p for the rust SegFormer-B5 primary contrast is seed-sensitive (0.084 seed 7; 0.100 seed 20260928); the manuscript must use one seed throughout.
