# Round 4, fifth expert: five-evaluator expert-panel reanalysis

> **Public-release copy.** Statements that linked an evaluator label to an identity or affiliation were removed or reworded without the label; per-evaluator values and all per-leaf panel grades were removed (leaf-level panel detail, compared with the per-leaf four-evaluator grades of `r4_panel`, would reveal the added evaluator's grades); the Table 3c subpanels carry rank labels A–E assigned for this copy; the manuscript drafting notes (sections 7 and 7b) are omitted; results.json of this folder is not released. Every number shown is unchanged; local paths are written as `<ROOT>` (the working tree) and `<USER_HOME>`.

PG-26038 major revision, 28 Sep 2026. Scripts in this folder: `panel5_analysis.py` (all statistics, `results.json`), `crosscheck.py` (independent loop implementation with the `validation/final_analysis.py` functions, `crosscheck.json`), `make_md.py` (generator of this report; withheld from the public release). Logs: `panel5_analysis.log`, `crosscheck.log`. No model was re-run: grades come from `validation/model_predictions.csv` (original models) and `results/r4_panel/form_predictions.csv` (matched-protocol lrB runs, seeds 0–2). No training of any kind was run.

**Privacy.** The raw 5-row form export (`validation/private/form_export_5experts_raw.csv`) was read in place and never copied. Only the item names in the header of columns 8–67 and the 60 grade cells of each row were used; the identity columns (name, specialty, role, institution, years) and the timestamps were not read into any output. No name, timestamp or affiliation appears in this folder. Evaluator labels used inside the analysis follow the row order of the export; the first four are the evaluators of `results/r4_panel`, and sensitivity (c) uses those four. Per-evaluator values and per-leaf panel grades are not part of this public copy, and no table of individual grades is released.

**Protocol (unchanged).** Rust grade = abaxial face; American leaf spot grade = mean of both faces; cut-points 0.5, 5 and 20%. Primary set = the 53 form leaves unseen in training (30 leaf spot, 23 rust; the 7 rust-training healthy leaves are excluded). Reference grades: rust = grade of the abaxial reference mask (semi-automatic, detector + SAM 2); leaf spot = grade of the mean reference-mask severity of both faces (primary) and the annotator's visual grade (secondary); the 12 healthy form leaves have empty masks (G0). Reference grades were recomputed and equal those of `results/r4_panel`.

**Design.** PRIMARY (generalizes the published tie-free design): for each evaluator i and each 3-subset S of the other four evaluators, the reference is median(S), which is always an integer. Evaluator value = mean over the 20 (i, S) of QWK(evaluator i, median S). Model value = mean over the same 20 references of QWK(model, median S). Contrast = model minus evaluators. With four evaluators this reduces exactly to the published design (checked: sensitivity (c)). Sensitivities: (a) reference = median of the other four evaluators, ties (two-against-two middle values) resolved up and, separately, down (5 references); (b) the model against the 5-evaluator median, which is tie-free; here each evaluator's value includes its own grade in the median, which favours the evaluators; (c) the four-evaluator panel of the earlier analysis (= the four-evaluator design of `results/r4_panel`). Extra: each four-evaluator subpanel (dropping each evaluator in turn); pooled 53 leaves; all 60 leaves; rust graded from both faces.

**Statistics.** Bootstrap: 10,000 resamples of leaves, numpy `default_rng(20260928)`. The resample vectors are drawn exactly as in `results/r4_panel` (fresh generator per leaf set, one draw of n indices per resample), shared by every rater and design, so (c) reproduces the r4_panel intervals to the last digit. Evaluators and seeds are held fixed, so inference is conditional on these five readers. Two-sided p = min(1, 2 min(P*(d ≤ 0), P*(d ≥ 0))); p < 0.001 is written where the bootstrap gives fewer than 10 of 10,000 resamples on one side. Pre-specified primary: original SegFormer-B5 per disease (unadjusted). Secondary family: original DeepLabV3+ and UNet++ and the seed-mean contrasts of matched DeepLabV3+ and UNet++, per disease: 8 contrasts, Holm. A matched contrast = mean over seeds 0–2 of the model value minus the evaluator value, bootstrapped as one statistic; per-seed values are descriptive; seed SD = sample SD (ddof = 1). Differences are computed before rounding.

**Timing of the design (disclosure).** The five-evaluator generalization was specified in `round4_plan.md` (section 'FIFTH EXPERT'; file last modified 13:36 on 28 Sep 2026) after the four-evaluator results were known. A preliminary, unverified check (`validation/private/quick5.py`, output at 13:42) had already computed the primary SegFormer-B5 contrasts. 'Pre-specified' therefore means specified before this analysis, not before any look at five-evaluator data.

## 0. Structure and reproduction checks

- Export structure: 5 rows, 60 grade columns ('G0'–'G3' only). Header identical to the four-evaluator export: True; rows 1–4 identical to the four-evaluator export: True; rows 1–4 equal the E1–E4 grades used by `results/r4_panel`: True.
- Original-model form predictions in `results/r4_panel/form_predictions.csv` equal `validation/model_predictions.csv` (max |Δ| = 0.00 pp).
- Sensitivity (c) reproduces `results/r4_panel` exactly (estimate, 95% CI, p and Holm-8 p; 17/17 quantities identical), e.g. American leaf spot SegFormer-B5 +0.119 [0.061, 0.187], p <0.001; rust +0.087 [−0.019, 0.220], p = 0.100.
- The four-evaluator inter-rater values reproduce `validation/final_results.json` (Supplement Table S20: α 0.74/0.71/0.74, ICC 0.73/0.79/0.76, Fleiss 0.34/0.45/0.40, 16/60 unanimous, 59/60 within one step), and the four-evaluator pooled/all-60/both-faces values reproduce `final_results.json` (asserted in the script). E1–E4 per-evaluator agreement with the reference equals r4_panel Table 2b (asserted).
- Independent cross-check (`crosscheck.py`: loops, `final_analysis.qwk/kripp/icc21/fleiss`, separate parsing): primary, (a) up and (a) down SegFormer-B5 contrasts with CI and p, original DeepLabV3+/UNet++ point contrasts, (b) SegFormer-B5 QWK, the all-five-disagree leaves, per-evaluator exact agreement, and α/ICC/Fleiss with the α and ICC CIs: all match = True.

## 1. Inter-rater agreement

Table 1. Agreement among the five evaluators on all 60 form leaves (as Supplement Table S20), with the four-evaluator panel of the earlier analysis for comparison. 95% CIs: 10,000 leaf resamples, seed 20260928 (pooled: resampling of the 60 leaves). Pairwise values are means over the 10 (five evaluators) or 6 (four) evaluator pairs.

| Measure | Leaf spot, 5 evaluators (n = 30) | Leaf spot, 4 (earlier panel) | Rust, 5 (n = 30) | Rust, 4 | Pooled, 5 (n = 60) | Pooled, 4 |
|---|---|---|---|---|---|---|
| Krippendorff's α (ordinal) | 0.780 [0.664, 0.831] | 0.738 [0.600, 0.802] | 0.745 [0.582, 0.836] | 0.715 [0.518, 0.829] | 0.772 [0.685, 0.825] | 0.738 [0.632, 0.804] |
| ICC(2,1) | 0.771 [0.680, 0.823] | 0.728 [0.626, 0.789] | 0.792 [0.638, 0.869] | 0.791 [0.603, 0.875] | 0.782 [0.712, 0.831] | 0.759 [0.676, 0.817] |
| Fleiss' κ (nominal) | 0.421 [0.311, 0.517] | 0.339 [0.216, 0.445] | 0.502 [0.349, 0.627] | 0.453 [0.275, 0.600] | 0.466 [0.378, 0.545] | 0.402 [0.300, 0.496] |
| Mean pairwise QWK | 0.759 [0.670, 0.812] | 0.721 [0.621, 0.781] | 0.779 [0.606, 0.861] | 0.775 [0.567, 0.866] | 0.774 [0.700, 0.825] | 0.751 [0.661, 0.811] |
| Pairwise QWK, range | 0.572–0.927 | 0.572–0.812 | 0.683–0.885 | 0.683–0.885 | 0.654–0.883 | 0.654–0.848 |
| Mean pairwise exact agreement | 58.7% [51.3, 66.7] | 53.3% [45.6, 61.7] | 66.7% [57.3, 76.0] | 63.3% [53.3, 73.9] | 62.7% [56.7, 68.7] | 58.3% [51.9, 65.3] |
| Mean pairwise within-one agreement | 99.3% [98.0, 100.0] | 99.4% [98.3, 100.0] | 99.3% [98.0, 100.0] | 100.0% [100.0, 100.0] | 99.3% [98.3, 100.0] | 99.7% [99.2, 100.0] |
| Leaves graded identically by all | 5/30 | 5/30 | 10/30 | 11/30 | 15/60 | 16/60 |
| Leaves with all grades within one step | 29/30 | 29/30 | 29/30 | 30/30 | 58/60 | 59/60 |
| Largest agreeing group (5/4/3/2 evaluators; 4-panel: 4/3/2) | 5/14/10/1 | 5/17/8 | 10/11/8/1 | 11/10/9 | 15/25/18/2 | 16/27/17 |
| Mean grade per evaluator, range | 0.90–1.57 | 0.90–1.57 | 0.77–1.10 | 0.77–1.10 | 0.83–1.28 | 0.83–1.28 |

- American leaf spot (5 evaluators): leaves not within one step: 1 [leaf and grades withheld in the public copy]. Five-evaluator median grade counts G0/G1/G2/G3: 8/11/9/2. Four-evaluator two-against-two splits: 7/30.
- Coffee leaf rust (5 evaluators): leaves not within one step: 1 [leaf and grades withheld in the public copy]. Five-evaluator median grade counts G0/G1/G2/G3: 13/10/5/2. Four-evaluator two-against-two splits: 9/30.
- Pooled (5 evaluators): leaves not within one step: 2 [leaves and grades withheld in the public copy]. Five-evaluator median grade counts G0/G1/G2/G3: 21/21/14/4. Four-evaluator two-against-two splits: 16/60.

Primary 53-leaf set (secondary view; leaf spot identical to the table above because all 30 leaf-spot leaves are primary):

| Set | Evaluators | α | ICC(2,1) | Fleiss' κ | Mean pairwise QWK | Unanimous | All within one |
|---|---|---|---|---|---|---|---|
| Rust (23) | 5 | 0.737 [0.562, 0.839] | 0.772 [0.595, 0.864] | 0.475 [0.306, 0.618] | 0.759 [0.560, 0.857] | 8/23 | 22/23 |
| Rust (23) | 4 (earlier panel) | 0.728 [0.510, 0.846] | 0.783 [0.570, 0.876] | 0.442 [0.242, 0.612] | 0.769 [0.527, 0.869] | 9/23 | 23/23 |
| Pooled (53) | 5 | 0.766 [0.677, 0.820] | 0.767 [0.691, 0.821] | 0.448 [0.354, 0.532] | 0.759 [0.679, 0.815] | 13/53 | 51/53 |
| Pooled (53) | 4 (earlier panel) | 0.737 [0.630, 0.802] | 0.747 [0.657, 0.809] | 0.388 [0.279, 0.487] | 0.739 [0.645, 0.803] | 14/53 | 52/53 |

## 2. Primary model–panel comparison (five evaluators, 20 paired reference sets; 53 unseen leaves)

Table 2. Mean QWK over the 20 (evaluator, 3-evaluator consensus) references and contrast with the typical evaluator. Matched rows: mean ± SD over seeds 0–2 (range); contrast, CI and p for the seed-mean contrast. 'Inside range' compares the rater's value with the five per-evaluator values (Table 2b); 'evaluators below' = number of the five evaluators with a lower value. Supporting columns (point estimates): mean QWK against each of the five evaluators' grades; QWK against the five-evaluator median (tie-free); grades inside the range of the five panel grades.

| Disease (n) | Rater | QWK (20 references) [95% CI of the value] | Difference vs evaluators [95% CI] | p (two-sided) | Holm, 8 secondary | Exact vs references | Inside evaluators' range (evaluators below) | Mean QWK vs each evaluator | QWK vs 5-evaluator median | Grade within panel range |
|---|---|---|---|---|---|---|---|---|---|---|
| American leaf spot (30) | Evaluators (mean; range across the five evaluators) | 0.798 (0.70–0.87) [0.702, 0.857] | — | — | — | 65% | — | 0.759 (pairwise among evaluators) | — | — |
| | SegFormer-B5 original (primary) | 0.883 [0.787, 0.941] | +0.086 [0.033, 0.140] | 0.005 | — (primary) | 80% | above max (5/5) | 0.822 | 0.897 | 30/30 |
| | DeepLabV3+ original (secondary) | 0.860 [0.751, 0.927] | +0.062 [−0.011, 0.128] | 0.091 | 0.73 | 77% | yes (4/5) | 0.805 | 0.872 | 30/30 |
| | UNet++ original (secondary) | 0.752 [0.614, 0.848] | −0.046 [−0.160, 0.053] | 0.38 | 1.00 | 64% | yes (1/5) | 0.730 | 0.760 | 30/30 |
| | DeepLabV3+ matched, seeds 0–2 (secondary) | 0.810 ± 0.107 (0.687–0.883) | +0.013 [−0.052, 0.074] | 0.68 | 1.00 | 71% (seed mean) | yes (3/5) | 0.767 ± 0.081 | 0.820 ± 0.112 | 29–30/30 |
| | UNet++ matched, seeds 0–2 (secondary) | 0.843 ± 0.014 (0.835–0.860) | +0.046 [−0.032, 0.116] | 0.22 | 1.00 | 75% (seed mean) | yes (3/5) | 0.794 ± 0.010 | 0.855 ± 0.015 | 30/30 (all seeds) |
| Coffee leaf rust (23) | Evaluators (mean; range across the five evaluators) | 0.785 (0.74–0.81) [0.589, 0.881] | — | — | — | 67% | — | 0.759 (pairwise among evaluators) | — | — |
| | SegFormer-B5 original (primary) | 0.822 [0.617, 0.922] | +0.037 [−0.065, 0.133] | 0.40 | — (primary) | 73% | above max (5/5) | 0.774 | 0.832 | 21/23 |
| | DeepLabV3+ original (secondary) | 0.757 [0.596, 0.857] | −0.028 [−0.138, 0.122] | 0.70 | 1.00 | 69% | yes (1/5) | 0.723 | 0.726 | 22/23 |
| | UNet++ original (secondary) | 0.773 [0.573, 0.886] | −0.012 [−0.097, 0.062] | 0.85 | 1.00 | 63% | yes (1/5) | 0.726 | 0.741 | 20/23 |
| | DeepLabV3+ matched, seeds 0–2 (secondary) | 0.746 ± 0.046 (0.696–0.787) | −0.039 [−0.136, 0.069] | 0.45 | 1.00 | 66% (seed mean) | yes (1/5) | 0.712 ± 0.042 | 0.720 ± 0.032 | 21–22/23 |
| | UNet++ matched, seeds 0–2 (secondary) | 0.792 ± 0.026 (0.777–0.822) | +0.007 [−0.109, 0.120] | 0.84 | 1.00 | 69% (seed mean) | yes (3/5) | 0.750 ± 0.021 | 0.792 ± 0.038 | 21/23 (all seeds) |

Grade differences against the 20 references (rater minus reference, all 20 × n comparisons; counts for −2/−1/0/+1/+2): American leaf spot: evaluators 2/88/388/122/0, SegFormer-B5 original 0/32/482/86/0 (max |difference| 1), DeepLabV3+ original 0/32/462/106/0 (max |difference| 1), UNet++ original 0/32/382/186/0 (max |difference| 1); Coffee leaf rust: evaluators 2/72/308/78/0, SegFormer-B5 original 0/12/334/114/0 (max |difference| 1), DeepLabV3+ original 0/80/318/62/0 (max |difference| 1), UNet++ original 0/28/288/138/6 (max |difference| 2).

Table 2b (per-evaluator QWK) is omitted from the public release; Table 2 gives the range across the five evaluators.

Per-seed matched values (descriptive, unadjusted):

| Disease | Model | Seed | QWK (20 references) | Difference [95% CI] | p | Exact vs references | QWK vs 5-median | Within panel range | Panel-G0 leaves graded ≥ G1 |
|---|---|---|---|---|---|---|---|---|---|
| American leaf spot | DeepLabV3+ | 0 | 0.883 | +0.086 [0.033, 0.140] | 0.005 | 80% | 0.897 | 30/30 | 1/8 |
| American leaf spot | DeepLabV3+ | 1 | 0.860 | +0.062 [−0.011, 0.128] | 0.091 | 77% | 0.872 | 30/30 | 2/8 |
| American leaf spot | DeepLabV3+ | 2 | 0.687 | −0.110 [−0.234, 0.000] | 0.050 | 57% | 0.692 | 29/30 | 8/8 |
| American leaf spot | UNet++ | 0 | 0.835 | +0.038 [−0.050, 0.113] | 0.36 | 74% | 0.847 | 30/30 | 3/8 |
| American leaf spot | UNet++ | 1 | 0.860 | +0.062 [−0.011, 0.128] | 0.091 | 77% | 0.872 | 30/30 | 2/8 |
| American leaf spot | UNet++ | 2 | 0.835 | +0.038 [−0.050, 0.113] | 0.36 | 74% | 0.847 | 30/30 | 3/8 |
| Coffee leaf rust | DeepLabV3+ | 0 | 0.756 | −0.029 [−0.161, 0.083] | 0.66 | 65% | 0.731 | 21/23 | 3/6 |
| Coffee leaf rust | DeepLabV3+ | 1 | 0.787 | +0.001 [−0.114, 0.111] | 0.91 | 67% | 0.746 | 22/23 | 3/6 |
| Coffee leaf rust | DeepLabV3+ | 2 | 0.696 | −0.089 [−0.216, 0.074] | 0.23 | 66% | 0.684 | 22/23 | 3/6 |
| Coffee leaf rust | UNet++ | 0 | 0.777 | −0.008 [−0.149, 0.121] | 0.96 | 68% | 0.787 | 21/23 | 3/6 |
| Coffee leaf rust | UNet++ | 1 | 0.822 | +0.037 [−0.065, 0.133] | 0.40 | 73% | 0.832 | 21/23 | 3/6 |
| Coffee leaf rust | UNet++ | 2 | 0.778 | −0.007 [−0.139, 0.116] | 0.97 | 65% | 0.756 | 21/23 | 3/6 |

Panel-G0 leaves (five-evaluator median G0) graded ≥ G1, original models and references: American leaf spot: SegFormer-B5 original 1/8, DeepLabV3+ original 2/8, UNet++ original 6/8, Reference, mask-derived 2/8, Reference, visual 5/8; Coffee leaf rust: SegFormer-B5 original 3/6, DeepLabV3+ original 3/6, UNet++ original 3/6, Reference, mask-derived 3/6.

Matched vs original (paired on the same resamples; exploratory, unadjusted; seeds fixed, SegFormer-B5 a single run, so not an architecture ranking):

| Disease | Comparison | Difference [95% CI] | p |
|---|---|---|---|
| American leaf spot | matched DeepLabV3+ (seed mean) − original DeepLabV3+ | −0.050 [−0.094, −0.006] | 0.023 |
| American leaf spot | matched DeepLabV3+ (seed mean) − SegFormer-B5 | −0.073 [−0.125, −0.035] | <0.001 |
| American leaf spot | matched UNet++ (seed mean) − original UNet++ | +0.091 [0.011, 0.191] | 0.022 |
| American leaf spot | matched UNet++ (seed mean) − SegFormer-B5 | −0.040 [−0.110, 0.000] | 0.21 |
| Coffee leaf rust | matched DeepLabV3+ (seed mean) − original DeepLabV3+ | −0.011 [−0.106, 0.065] | 0.75 |
| Coffee leaf rust | matched DeepLabV3+ (seed mean) − SegFormer-B5 | −0.076 [−0.173, 0.031] | 0.14 |
| Coffee leaf rust | matched UNet++ (seed mean) − original UNet++ | +0.019 [−0.086, 0.127] | 0.71 |
| Coffee leaf rust | matched UNet++ (seed mean) − SegFormer-B5 | −0.030 [−0.091, 0.008] | 0.35 |

**Where the nine model runs differ (American leaf spot):** 7 of 30 leaves: ojo01; ojo05; ojo10; ojo11; ojo22; ojo24; ojo27 (per-leaf panel grades withheld in the public copy).

**Where the nine model runs differ (Coffee leaf rust):** 10 of 23 leaves: roya07; roya09; roya10; roya13; roya15; roya17; roya21; roya23; roya27; roya29 (per-leaf panel grades withheld in the public copy).

## 3. Sensitivity analyses

Table 3. SegFormer-B5 (primary contrast) under each design, and the smallest Holm-adjusted p of the 8 secondary contrasts within that design.

| Design | Disease (n) | Evaluators | SegFormer-B5 | Difference [95% CI] | p | Smallest Holm p, secondary (contrast) |
|---|---|---|---|---|---|---|
| PRIMARY: 5 evaluators, 3-evaluator consensus (20 references) | American leaf spot (30) | 0.798 | 0.883 | +0.086 [0.033, 0.140] | 0.005 | 0.73 (DeepLabV3+ original; +0.062) |
| PRIMARY: 5 evaluators, 3-evaluator consensus (20 references) | Coffee leaf rust (23) | 0.785 | 0.822 | +0.037 [−0.065, 0.133] | 0.40 | 1.00 (DeepLabV3+ original; −0.028) |
| (a) median of the other 4, ties up | American leaf spot (30) | 0.800 | 0.907 | +0.107 [0.039, 0.180] | 0.006 | 0.38 (DeepLabV3+ original; +0.085) |
| (a) median of the other 4, ties up | Coffee leaf rust (23) | 0.756 | 0.860 | +0.103 [−0.011, 0.223] | 0.070 | 1.00 (DeepLabV3+ original; +0.013) |
| (a) median of the other 4, ties down | American leaf spot (30) | 0.798 | 0.858 | +0.060 [−0.009, 0.125] | 0.082 | 1.00 (DeepLabV3+ original; +0.036) |
| (a) median of the other 4, ties down | Coffee leaf rust (23) | 0.814 | 0.786 | −0.028 [−0.149, 0.084] | 0.64 | 0.98 (UNet++ original; −0.082) |
| (b) 5-evaluator median (evaluators include their own grade) | American leaf spot (30) | 0.850 | 0.897 | +0.047 [−0.037, 0.132] | 0.26 | 0.61 (UNet++ original; −0.090) |
| (b) 5-evaluator median (evaluators include their own grade) | Coffee leaf rust (23) | 0.864 | 0.832 | −0.032 [−0.164, 0.083] | 0.59 | 0.010 (UNet++ original; −0.124) |
| (c) four-evaluator panel of the earlier analysis (= r4_panel) | American leaf spot (30) | 0.765 | 0.884 | +0.119 [0.061, 0.187] | <0.001 | 0.15 (DeepLabV3+ original; +0.095) |
| (c) four-evaluator panel of the earlier analysis (= r4_panel) | Coffee leaf rust (23) | 0.728 | 0.815 | +0.087 [−0.019, 0.220] | 0.100 | 1.00 (DeepLabV3+ original; +0.049) |

Table 3b. Secondary contrasts under each sensitivity design (difference [95% CI]; p; Holm over the 8 contrasts of that design).

| Design | Disease | DeepLabV3+ original | UNet++ original | DeepLabV3+ matched (seed mean) | UNet++ matched (seed mean) |
|---|---|---|---|---|---|
| PRIMARY | American leaf spot | 0.860: +0.062 [−0.011, 0.128]; p 0.091; Holm 0.73 | 0.752: −0.046 [−0.160, 0.053]; p 0.38; Holm 1.00 | 0.810: +0.013 [−0.052, 0.074]; p 0.68; Holm 1.00 | 0.843: +0.046 [−0.032, 0.116]; p 0.22; Holm 1.00 |
| PRIMARY | Coffee leaf rust | 0.757: −0.028 [−0.138, 0.122]; p 0.70; Holm 1.00 | 0.773: −0.012 [−0.097, 0.062]; p 0.85; Holm 1.00 | 0.746: −0.039 [−0.136, 0.069]; p 0.45; Holm 1.00 | 0.792: +0.007 [−0.109, 0.120]; p 0.84; Holm 1.00 |
| (a) median of the other 4, ties up | American leaf spot | 0.885: +0.085 [0.002, 0.169]; p 0.047; Holm 0.38 | 0.781: −0.019 [−0.140, 0.096]; p 0.76; Holm 1.00 | 0.837: +0.036 [−0.043, 0.118]; p 0.34; Holm 1.00 | 0.870: +0.069 [−0.019, 0.155]; p 0.12; Holm 0.82 |
| (a) median of the other 4, ties up | Coffee leaf rust | 0.770: +0.013 [−0.117, 0.198]; p 0.84; Holm 1.00 | 0.818: +0.061 [−0.063, 0.193]; p 0.29; Holm 1.00 | 0.755: −0.002 [−0.119, 0.142]; p 0.98; Holm 1.00 | 0.830: +0.074 [−0.058, 0.213]; p 0.24; Holm 1.00 |
| (a) median of the other 4, ties down | American leaf spot | 0.834: +0.036 [−0.051, 0.109]; p 0.38; Holm 1.00 | 0.722: −0.076 [−0.196, 0.029]; p 0.16; Holm 1.00 | 0.783: −0.015 [−0.088, 0.052]; p 0.66; Holm 1.00 | 0.817: +0.019 [−0.072, 0.097]; p 0.66; Holm 1.00 |
| (a) median of the other 4, ties down | Coffee leaf rust | 0.745: −0.069 [−0.210, 0.107]; p 0.41; Holm 1.00 | 0.732: −0.082 [−0.189, 0.023]; p 0.12; Holm 0.98 | 0.739: −0.075 [−0.204, 0.065]; p 0.26; Holm 1.00 | 0.756: −0.058 [−0.188, 0.067]; p 0.36; Holm 1.00 |
| (b) 5-evaluator median (evaluators include their own grade) | American leaf spot | 0.872: +0.022 [−0.074, 0.113]; p 0.63; Holm 1.00 | 0.760: −0.090 [−0.214, 0.026]; p 0.12; Holm 0.61 | 0.820: −0.030 [−0.113, 0.055]; p 0.46; Holm 1.00 | 0.855: +0.005 [−0.094, 0.099]; p 0.91; Holm 1.00 |
| (b) 5-evaluator median (evaluators include their own grade) | Coffee leaf rust | 0.726: −0.138 [−0.289, 0.013]; p 0.073; Holm 0.44 | 0.741: −0.124 [−0.256, −0.040]; p 0.001; Holm 0.010 | 0.720: −0.144 [−0.292, −0.026]; p 0.016; Holm 0.11 | 0.792: −0.072 [−0.207, 0.047]; p 0.23; Holm 0.92 |
| (c) four-evaluator panel of the earlier analysis (= r4_panel) | American leaf spot | 0.860: +0.095 [0.018, 0.172]; p 0.018; Holm 0.15 | 0.747: −0.018 [−0.132, 0.091]; p 0.75; Holm 1.00 | 0.808: +0.043 [−0.023, 0.118]; p 0.20; Holm 1.00 | 0.842: +0.077 [−0.004, 0.160]; p 0.061; Holm 0.43 |
| (c) four-evaluator panel of the earlier analysis (= r4_panel) | Coffee leaf rust | 0.777: +0.049 [−0.078, 0.263]; p 0.47; Holm 1.00 | 0.794: +0.066 [−0.037, 0.206]; p 0.18; Holm 1.00 | 0.763: +0.035 [−0.086, 0.217]; p 0.57; Holm 1.00 | 0.792: +0.064 [−0.068, 0.222]; p 0.30; Holm 1.00 |

Table 3c. Four-evaluator subpanels (the published design applied to each set of four of the five evaluators). Public copy: subpanels are labelled A–E by decreasing American leaf spot difference (labels assigned for this copy only); one of them is the four-evaluator panel of sensitivity (c).

| Subpanel | Disease | Evaluators (range) | SegFormer-B5 | Difference [95% CI] | p | Smallest Holm p, secondary |
|---|---|---|---|---|---|---|
| A | American leaf spot | 0.765 (0.67–0.82) | 0.884 | +0.119 [0.061, 0.187] | <0.001 | 0.15 |
| A | Coffee leaf rust | 0.728 (0.69–0.75) | 0.815 | +0.087 [−0.019, 0.220] | 0.100 | 1.00 |
| B | American leaf spot | 0.766 (0.67–0.84) | 0.867 | +0.101 [0.033, 0.168] | 0.008 | 0.60 |
| B | Coffee leaf rust | 0.796 (0.77–0.81) | 0.823 | +0.026 [−0.108, 0.166] | 0.63 | 1.00 |
| C | American leaf spot | 0.819 (0.73–0.90) | 0.906 | +0.087 [0.019, 0.157] | 0.016 | 0.92 |
| C | Coffee leaf rust | 0.776 (0.74–0.80) | 0.840 | +0.064 [−0.062, 0.176] | 0.27 | 1.00 |
| D | American leaf spot | 0.815 (0.72–0.90) | 0.889 | +0.074 [−0.008, 0.151] | 0.075 | 1.00 |
| D | Coffee leaf rust | 0.762 (0.70–0.82) | 0.796 | +0.034 [−0.096, 0.158] | 0.57 | 1.00 |
| E | American leaf spot | 0.822 (0.80–0.85) | 0.870 | +0.048 [−0.031, 0.112] | 0.21 | 0.97 |
| E | Coffee leaf rust | 0.863 (0.83–0.89) | 0.836 | −0.027 [−0.163, 0.092] | 0.67 | 0.66 |

Table 3d. Other leaf sets and grade rules (primary five-evaluator design; four-evaluator values in parentheses, which reproduce `validation/final_results.json`).

| Set | n | Evaluators | SegFormer-B5 | Difference [95% CI] | p | DeepLabV3+ orig. | UNet++ orig. | DeepLabV3+ matched | UNet++ matched |
|---|---|---|---|---|---|---|---|---|---|
| Pooled, 53 unseen leaves | 53 | 0.793 (0.749) | 0.858 (0.856) | +0.066 [0.016, 0.113] (+0.107 [0.053, 0.169]) | 0.013 (<0.001) | 0.822 (0.827) | 0.762 (0.771) | 0.785 (0.788) | 0.822 (0.821) |
| American leaf spot, all 30 | 30 | 0.798 (0.765) | 0.883 (0.884) | +0.086 [0.033, 0.140] (+0.119 [0.061, 0.187]) | 0.005 (<0.001) | 0.860 (0.860) | 0.752 (0.747) | 0.810 (0.808) | 0.843 (0.842) |
| Rust, all 30 (incl. 7 training healthy leaves) | 30 | 0.822 (0.776) | 0.881 (0.877) | +0.058 [−0.004, 0.137] (+0.102 [0.028, 0.222]) | 0.065 (0.008) | 0.836 (0.852) | 0.841 (0.857) | 0.827 (0.841) | 0.860 (0.862) |
| Pooled, all 60 | 60 | 0.814 (0.774) | 0.884 (0.882) | +0.070 [0.030, 0.112] (+0.108 [0.061, 0.166]) | 0.001 (<0.001) | 0.855 (0.860) | 0.809 (0.816) | 0.827 (0.831) | 0.856 (0.855) |
| Rust (23), grade from the mean of both faces | 23 | 0.785 (0.728) | 0.787 (0.793) | +0.002 [−0.140, 0.123] (+0.065 [−0.073, 0.228]) | 0.91 (0.32) | 0.725 (0.734) | 0.783 (0.806) | 0.713 (0.731) | 0.675 (0.687) |

## 4. The reference grade as a rater

Table 4a. Reference grades scored against the 20 three-evaluator references exactly like a model (primary set). Reference rows are descriptive and outside both test families.

| Disease (n) | Rater | QWK (20 references) | Difference vs evaluators [95% CI] | p | Exact vs references | Inside evaluators' range (below) | QWK vs 5-median | Within panel range | SegFormer-B5 minus reference [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| American leaf spot (30) | Evaluators | 0.798 | — | — | 65% | — | — | — | — |
| | Reference, mask-derived | 0.860 | +0.062 [−0.011, 0.128] | 0.091 | 77% | yes (4/5) | 0.872 | 30/30 | +0.023 [0.000, 0.084] |
| | Reference, visual (leaf spot, secondary reference) | 0.781 | −0.016 [−0.125, 0.076] | 0.76 | 67% | yes (2/5) | 0.791 | 30/30 | +0.102 [0.024, 0.211] |
| Coffee leaf rust (23) | Evaluators | 0.785 | — | — | 67% | — | — | — | — |
| | Reference, mask-derived | 0.751 | −0.034 [−0.178, 0.140] | 0.70 | 68% | yes (1/5) | 0.762 | 21/23 | +0.071 [−0.062, 0.178] |

Paired, American leaf spot, mask-derived minus visual reference: +0.079 [0.000, 0.175], p = 0.088 (four-evaluator design: +0.082 [0.000, 0.183], p = 0.088). Descriptively higher, not significant. The SegFormer-B5 minus mask-derived reference interval has one bound at exactly 0 because the two grade vectors differ on few leaves; its percentile p (0.74) is not meaningful, so quote the CI only.

Table 4b. Per-evaluator agreement with the reference grade (primary set; QWK CI from 10,000 leaf resamples, seed 20260928). 'Higher' = evaluator grade above the reference.

| Disease (n) | Reference | Evaluator | Exact | Within one | Higher / lower | Mean (evaluator − reference) | QWK [95% CI] |
|---|---|---|---|---|---|---|---|
| | mask | 5-evaluator median | 24/30 (80%) | 30/30 | 1 / 5 | | 0.872 |
| | visual | 5-evaluator median | 21/30 (70%) | 30/30 | 1 / 8 | | 0.791 |
| | mask | 5-evaluator median | 16/23 (70%) | 23/23 | 2 / 5 | | 0.762 |

Summary: every evaluator was within one grade of the reference on every primary leaf: True. Exact agreement per evaluator against the primary references: 57–77% (leaf-spot mask 60–77%, rust 57–70%); against the visual leaf-spot grade 50–77%.

Table 4c. Leaves where all five evaluators disagree with the reference grade (all 60 form leaves checked; all are primary leaves), and how the models graded them on the original scans (`reanalysis/out/per_image.csv` for the original models, `per_image_retrained.csv` for the matched lrB runs; rust = abaxial face). Public copy: the rows for leaves where four of five disagree (7 rows) and the sorted panel grades are omitted.

| Disease | Form item | Leaf ID | Reference grade (severity, %) | Evaluators disagreeing | Direction | SegFormer-B5 form grade | Original models on the scans, % (grade) | Matched runs on the scans, % (grade) |
|---|---|---|---|---|---|---|---|---|
| Coffee leaf rust (mask ref.) | roya15 | 20260819_Hvastatrix_GX_020 | G2 (19.34) | 5/5 | all higher | G3 | S 23.73 (G3), D 22.65 (G3), U 25.11 (G3) | D0 21.30 (G3), D1 21.77 (G3), D2 19.07 (G2), U0 20.01 (G3), U1 21.42 (G3), U2 21.10 (G3) |
| Coffee leaf rust (mask ref.) | roya21 | 20260819_Hvastatrix_GX_095 | G2 (7.42) | 5/5 | all lower | G2 | S 9.10 (G2), D 7.61 (G2), U 9.65 (G2) | D0 9.80 (G2), D1 8.06 (G2), D2 6.90 (G2), U0 8.91 (G2), U1 11.84 (G2), U2 9.41 (G2) |

All five disagree (rust mask reference): roya15, roya21. [Comparison with the four-evaluator list omitted from the public copy.] Leaf spot: no leaf had all five evaluators disagreeing with either reference. Rust reference grade counts on the 23 primary leaves (G0/G1/G2/G3): 3/11/9/0; no held-out rust leaf has a G3 reference (r4_panel: the training split has 2).

## 5. Analyses that do not depend on the evaluators (not redone)

The panel-selected vs unselected leaf-spot comparison and the form-image vs scan comparison use no evaluator grades; they are unchanged and are cited from `results/r4_panel/results.md` (sections 3 and 4, with the verifier-2/3 corrections): grade mix selected vs unselected, visual p = 0.52, mask-derived p = 0.55; median severity 4.16 vs 4.04%, p = 0.95; mean 6.68 vs 4.01%, p = 0.067; collection date exact p = 0.050 (borderline, exploratory); SegFormer-B5 |error| 0.31 vs 0.13 pp, p = 0.018; SegFormer-B5 same grade on form image and scan for 46/48 diseased panel leaves (Wilson 95% CI 86–99%). The 'nine runs differ on 7 of 30 leaf-spot leaves / 10 of 23 rust leaves' statement is also evaluator-independent and unchanged.

## 6. Does adding the fifth evaluator change any conclusion? (honest reading)

1. **Primary, American leaf spot: still significant, smaller margin.** SegFormer-B5 0.883 vs the typical evaluator 0.798, difference +0.086 [0.033, 0.140], p = 0.005 (n = 30; four-evaluator design +0.119 [0.061, 0.187], p <0.001). The margin shrinks mainly because the typical evaluator's value rises (0.765 → 0.798); the model's value is almost unchanged (0.884 → 0.883). SegFormer-B5 lies above all five per-evaluator values (0.698–0.871).
2. **The leaf-spot result is not robust to every reference definition.** Ties up +0.107 [0.039, 0.180], p = 0.006; ties down +0.060 [−0.009, 0.125], p = 0.082 (not significant). Against the five-evaluator median, where each evaluator's own grade is in the reference, +0.047 [−0.037, 0.132], p = 0.26. Across the five four-evaluator subpanels the contrast has p < 0.05 in 3 of 5 (Table 3c); it does not when either of two of the evaluators is dropped (p = 0.075 and p = 0.21). The result therefore depends on panel composition and on the reference rule; the tie-free primary design was fixed before this analysis, but after the four-evaluator results were known.
3. **Primary, rust: still not significant, and smaller.** 0.822 vs 0.785, +0.037 [−0.065, 0.133], p = 0.40 (n = 23; four-evaluator +0.087 [−0.019, 0.220], p = 0.100). No five-evaluator rust design gives p < 0.05 on the 23 primary leaves (smallest p = 0.070, ties up; the all-30 rust set, which adds the 7 training healthy leaves, gives +0.058 [−0.004, 0.137], p = 0.065) [verifier-corrected wording]. SegFormer-B5's value lies above all five per-evaluator values (0.745–0.811), but the interval includes 0.
4. **Secondary family: none significant after Holm** (smallest: DeepLabV3+ original, American leaf spot, +0.062 [−0.011, 0.128], p = 0.091, Holm p = 0.73). Matched models, American leaf spot: UNet++ 0.843 ± 0.014, +0.046 [−0.032, 0.116]; DeepLabV3+ 0.810 ± 0.107 (seeds 0.883, 0.860, 0.687), +0.013 [−0.052, 0.074]. Rust: UNet++ 0.792 ± 0.026, +0.007 [−0.109, 0.120]; DeepLabV3+ 0.746 ± 0.046, −0.039 [−0.136, 0.069]. Holm p below 1: DeepLabV3+ original American leaf spot 0.73; all others 1.00. Under (b), where the evaluators are favoured, original UNet++ on rust is significantly below the evaluators after Holm (−0.124 [−0.256, −0.040], Holm p = 0.010); this design is biased toward the evaluators and is not a test of the models.
5. **Inter-rater agreement is slightly higher with five evaluators** (all 60 leaves, pooled: α 0.772 [0.685, 0.825] vs four-evaluator 0.738 [0.632, 0.804]; leaf spot 0.780, rust 0.745); 15/60 leaves were graded identically by all five (16/60 by the four) and 58/60 had all grades within one step (59/60 by the four).
6. **Reference as a rater: mostly unchanged, one change.** Leaf spot mask-derived 0.860 (+0.062 [−0.011, 0.128]), visual 0.781 (−0.016 [−0.125, 0.076]); paired mask minus visual +0.079 [0.000, 0.175], p = 0.088: consistent with, not evidence for, the mask-derived primary reference. Rust reference 0.751 vs evaluators 0.785 (−0.034 [−0.178, 0.140]): close to the evaluators' mean; equivalence not tested. Every evaluator stayed within one grade of the reference on every leaf; exact agreement 57–77% against the primary references (unchanged range). **Change:** with four evaluators the mask-derived leaf-spot grade was above the typical evaluator (+0.095 [0.018, 0.172], p = 0.018); with five it is not (p = 0.091), so the Discussion sentence 'the mask-derived leaf-spot grade agreed with the panel more closely than the average evaluator did, whereas the annotator's visual grade did not' no longer holds.
7. **Unanimous disagreements: two leaves.** All five evaluators disagreed with the rust reference on roya15 (reference G2, 19.34%; all G3) and roya21 (reference G2, 7.42%; all G1). [Per-leaf comparison with the four-evaluator panel omitted from the public copy.] On the original scans, all three original models put roya15 above the reference (22.65–25.11%, G3; on the panel's side) and 5 of 6 matched runs graded it G3 (19.07–21.77%). On roya21 every model predicted G2 on the scan (original 7.61–9.65%, matched 6.90–11.84%), i.e., with the reference and not with the panel.
8. **Direction of the fifth evaluator's effect.** Adding the fifth evaluator raised the typical evaluator's agreement (0.765 → 0.798 leaf spot; 0.728 → 0.785 rust) and made every model–panel contrast smaller (all 5 model rows and the reference, both diseases); it did not favour the models. No model conclusion about significance changes: leaf spot primary significant, rust primary not significant, no secondary contrast significant after Holm; the descriptive mask-derived reference row no longer differs detectably from the evaluators (point 6). What changes: the leaf-spot margin (+0.119 → +0.086), its robustness (p = 0.082 with ties resolved down; p ≥ 0.05 when either of two of the evaluators is dropped).

## 7. [Sections 7 and 7b omitted from the public release]

Drafting notes for the manuscript (suggested LaTeX text and a list of statements to update).

## 8. Notes and limits

- Inference is conditional on these five evaluators (leaves resampled, evaluators and seeds fixed); a two-way (leaf and evaluator) bootstrap or an Obuchowski–Rockette MRMC analysis was not done.
- The 20 references of the primary design are 10 distinct three-evaluator medians, each used twice (once for each of the two evaluators not in the subset), so the model value is the mean QWK over the 10 distinct consensus grades.
- Seed SD over three seeds is imprecise; SegFormer-B5 is a single run, so its seed variability on the panel is unknown.
- Model grades come from the form images (lower resolution than the scans); reference grades from the original scans. The form-vs-scan agreement (r4_panel section 4) applies unchanged.
- Bootstrap percentile intervals with one bound at exactly 0 (paired differences of nearly identical grade vectors) should be quoted as CIs only.

## Verifier corrections (final independent verifier, 28 Sep 2026)

Independent scripts and outputs: `verifier_final/vf_panel5.py` (own QWK from 4×4 counts, own ordinal Krippendorff α from the coincidence matrix, own ICC(2,1) from two-way ANOVA, own Fleiss κ, own 3-subset medians, own Holm; 10,000 leaf resamples, `default_rng(20260928)`, one (B, n) index matrix per leaf set), `verifier_final/vf_extra.py`, `vf_results.json`, `vf_results_seed7.json` (same analysis with generator seed 7 as a Monte Carlo check), `vf_extra.json`. No code from `panel5_analysis.py`, `crosscheck.py` or `validation/final_analysis.py` was used. The raw export was read in place (item names and grade cells only); nothing from it was copied.

**Reproduced exactly (to the printed digits):** all Table 1 values (α, ICC(2,1), Fleiss κ, mean pairwise QWK and exact agreement with 95% CIs, pairwise QWK range, unanimous and within-one counts, leaves not within one step, largest agreeing group, mean grade per evaluator, pairwise QWK per pair, median grade counts, 2–2 splits) for 5 and 4 evaluators and for the 53-leaf view; Table 2 (all rater values, differences, CIs, p, Holm-8, exact vs references, evaluators below, mean QWK vs each evaluator, QWK vs 5-median, within-range counts, grade-difference counts); Table 2b (per-evaluator QWK with CIs, exact agreement, SegFormer-B5 on the same references); the per-seed table; panel-G0 counts; the matched-vs-original paired table; Tables 3, 3b, 3c, 3d; Tables 4a, 4b, 4c; rust reference grade counts 3/11/9/0. Sensitivity (c) reproduces `results/r4_panel/results.json` for all 114 compared quantities (estimates, CI bounds, p, Holm-8; max |Δ| 5e-5, i.e. 4-dp rounding). The quick check (`validation/private/quick5_results.txt`, 2,000 resamples) agrees within Monte Carlo error (leaf spot +0.086 [0.032, 0.140], p = 0.004; rust +0.037 [−0.067, 0.134], p = 0.381).

**Not independently re-derived:** the CI of the evaluators' own value in Table 2, the CI of mean pairwise within-one agreement in Table 1, and Section 5 (cited from r4_panel, already verified there).

**Monte Carlo robustness (generator seed 7 vs 20260928):** no significance statement changes. Leaf-spot primary p 0.004 vs 0.005; ties down 0.083 vs 0.082; one four-evaluator subpanel 0.074 vs 0.075; rust ties up 0.062 vs 0.070; original DeepLabV3+ leaf spot p 0.076 vs 0.091 (Holm 0.61 vs 0.73); mask minus visual reference p 0.083 vs 0.088; (b) original UNet++ rust Holm 0.019 vs 0.010. Quote p to two significant digits at most and name the seed.

**Corrections made in this file:**
1. Section 6 point 3: "No five-evaluator rust design gives p < 0.05" now states that this holds on the 23 primary leaves; the all-30 rust set (incl. 7 training healthy leaves) gives p = 0.065, below the quoted 0.070. Conclusion unchanged.
2. Section 7b: removed an evaluator label from one statement (privacy).
3. Per-seed table, American leaf spot, matched DeepLabV3+ seed 2: −0.110 [−0.234, 0.000], p = 0.050 sits exactly at the Monte Carlo resolution boundary (seed 7: [−0.233, −0.003], p = 0.047). It is descriptive and unadjusted; do not describe it as significant or non-significant, only as "lower than the evaluators".

**Privacy:** [Paragraph omitted from the public release; it concerned evaluator labels.]

**Data note:** the `*_grade` columns of `validation/model_predictions.csv` use the mean of both faces for rust (8 rust cells differ from the abaxial rule, e.g. roya15 SegFormer-B5 G2 vs abaxial G3). This analysis correctly recomputes grades from the severities with the protocol rule; the grade columns must not be used, or released, as rust grades.

Re-running `make_md.py` would overwrite this section and the two in-text corrections.
