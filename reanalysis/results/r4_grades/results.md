# Round 4, analysis A4: leaf-spot mask-derived primary grade, and the deployment-only rust rescaling

> **Public-release copy.** The expert-identity section was removed. All other text and every number are unchanged; local paths are written as `<ROOT>` (the working tree) and `<USER_HOME>`.

Created 2026-09-28 by `grades_analysis.py` (part 1), `rescale_analysis.py` (part 2), `verify_experts.py` (expert table) and `make_md.py` (this file). All numbers are in `results.json` (which merges `grades_results.json`, `rescale_results.json` and `experts_verified.json`); per-leaf grades in `leafspot_per_leaf_grades.csv`. No model inference was run: every input is an existing per-leaf output.

**Protocol (unchanged, round4_plan.md).** Leaf = alpha > 8; lesion = mask > 127 AND leaf; 1024 letterbox, ImageNet normalisation, raw argmax. Rust = abaxial face; American leaf spot = mean of both faces. Grades: G0 < 0.5 ≤ G1 < 5 ≤ G2 < 20 ≤ G3 (%). Bootstrap: 10,000 resamples of leaves, numpy `default_rng(20260928)`, percentile intervals. Accuracy and recall: Wilson 95% intervals. Paired grade comparisons: exact McNemar, Tango score 95% interval for Δaccuracy, Holm within family; helper functions imported unchanged from `results/paired/paired_analysis.py`.

## 0. Key numbers

**Part 1: American leaf spot, test partition, n = 55 leaves; mask-derived grade = PRIMARY reference.**

- SegFormer-B5 (published): accuracy 0.95 (52/55) [0.85, 0.98], QWK 0.95 [0.88, 1.00]; errors over/under 1/2. (Visual reference: 0.84 (46/55) [0.72, 0.91], QWK 0.82 [0.72, 0.91].)
- UNet++ (published): accuracy 0.98 (54/55) [0.90, 1.00], QWK 0.98 [0.93, 1.00]; errors over/under 1/0. (Visual reference: 0.87 (48/55) [0.76, 0.94], QWK 0.85 [0.75, 0.94].)
- DeepLabV3+ (published): accuracy 0.96 (53/55) [0.88, 0.99], QWK 0.97 [0.90, 1.00]; errors over/under 1/1. (Visual reference: 0.85 (47/55) [0.74, 0.92], QWK 0.84 [0.74, 0.92].)
- Matched UNet++ (lrB, seeds 0–2): accuracy 0.976 ± 0.010 (range 0.964–0.982; 54, 53, 54 of 55), QWK 0.977 ± 0.010 (0.965–0.983). (Visual: 0.879 ± 0.010, QWK 0.859 ± 0.012.)
- Matched DeepLabV3+ (lrB, seeds 0–2): accuracy 0.958 ± 0.010 (range 0.945–0.964; 53, 52, 53 of 55), QWK 0.961 ± 0.009 (0.950–0.966). (Visual: 0.861 ± 0.028, QWK 0.845 ± 0.025.)
- Colour threshold (redness index ≥ 0.026): accuracy 0.98 (54/55) [0.90, 1.00], QWK 0.98 [0.93, 1.00]; it assigns the same grade as the published UNet++ to all 55 leaves. (Visual: 0.87 (48/55) [0.76, 0.94], QWK 0.85 [0.75, 0.94].)
- The two references agree on 47/55 leaves (QWK 0.83 [0.72, 0.92]).
- **No pairwise conclusion changes** between references: under both, no pair of methods (published pairs, network vs colour threshold, matched seed vs SegFormer-B5; 12 comparisons) differs detectably in per-leaf grade correctness (at most 3 discordant leaves; every Holm-adjusted exact McNemar p = 1.00; every Tango interval includes 0; this is not a demonstration of equivalence, since with 55 leaves the Tango intervals reach from −0.12 to +0.15). **What changes** is the level (accuracy 0.95–0.98 instead of 0.84–0.89), the error direction (mask: 1–2 over-grades and 0–2 under-grades per method; visual: under-grades only), and the location of the weakest grade (G1 recall 0.88–0.96 instead of 0.73–0.80). See §1.7.
- Cross-check against `results/paired` (mask-derived grades, published models): accuracies, QWK point estimates, confusion matrices, Wilson intervals and McNemar counts identical; QWK CIs differ only by the bootstrap seed (all checks passed: True).

**Part 2: deployment-only rust rescaling.**

- What it is: in the deployed application only, for rust only, the severity of the post-processed ('refined') lesion mask is rescaled as s_cal = 1.1915 s + 0.4329 (clipped to 0–100); the manuscript states the two coefficients were fitted on the 21 rust test leaves (Supplement `note_methods.tex` line 22; main text `02_methods.tex` lines 42 and 75; `04_discussion.tex` line 41).
- **It cannot be reproduced.** No code that applies or fits it, and no per-leaf refined severities, exist in any searched location (§2.1). The deployed refinement pipeline is not in `<ROOT>` nor in the local app folders, and which model(s) it was fitted for is not recorded. It therefore cannot be refitted on validation leaves on its own input.
- Validation-only refit that the local data permit (published SegFormer-B5, RAW-argmax severities, 21 diseased rust validation leaves, OLS reference = a·predicted + b): a = 0.730 [0.642, 0.818], b = +0.487 pp [0.082, 0.891] (t-based 95% CI). On the 21 test leaves: MAE 0.99 → 0.59 pp (paired Δ −0.40, [-0.84, -0.04]), bias +0.99 → −0.27 pp, grade accuracy 19/21 → 19/21, QWK 0.88 → 0.83; but its positive intercept lifts 15/21 healthy test leaves to ≥ 0.5% (abaxial; 18/21 on either face; 0/21 before).
- Fitting on the 21 diseased + 21 healthy validation leaves gives a = 0.816, b = +0.014 pp: test MAE 0.99 → 0.40 pp (Δ −0.59, [-1.13, -0.15]), bias → −0.19 pp, accuracy 20/21, QWK 0.93, healthy ≥ 0.5% 0/21. A through-origin fit (a = 0.820) gives nearly the same result. Neither changes grade correctness detectably (McNemar p = 1.00).
- Recommendation: **remove the test-fitted rescaling (and its coefficients) from the paper**; if the application keeps a rescaling, refit it on validation leaves with the deployed pipeline's own output, include healthy leaves or fix the intercept at 0, and do not report it as evaluated. Suggested text in §2.6.

**Expert panel identities (author request):** verified against the form export (columns 1–6 only); table, LaTeX and two discrepancies to fix in §3.

## 1. American leaf spot grade agreement, mask-derived grade as the primary reference

**Design.** Test partition, 55 leaves, leaf severity = mean of both faces. Primary reference: the cut-points applied to the leaf-level reference severity (the manually corrected CVAT masks). Secondary reference: the annotator's visual grade. Methods: the three published models (`out/per_image.csv`); the matched-protocol CNNs, LR 6e-5/6e-4 (lrB), seeds 0 (`pg26038-mb-ojo-*-lrB`), 1 and 2 (`pg26038-mbs-ojo-*-lrB-seed1/2`) (`out/per_image_retrained.csv`); the colour-threshold baseline (redness index ≥ 0.026, per-leaf values from `results/baseline/per_leaf_test.csv`, not recomputed). Reference severities are identical across all inputs (max difference 4.0e-06 pp, the 6-decimal rounding of the baseline file).

### 1.1 The two reference standards

| Reference | G0 | G1 | G2 | G3 |
|---|---|---|---|---|
| Mask-derived (primary) | 6 | 26 | 21 | 2 |
| Visual (secondary) | 0 | 30 | 23 | 2 |

Agreement 47/55 (QWK 0.831 [0.721, 0.923]). All 8 disagreements are one grade, with the visual grade higher: 6 leaves visual G1 / mask G0 (reference severity 0.13–0.40%) and 2 visual G2 / mask G1 (4.80%, 4.92%). Confusion (rows mask G0–G3, columns visual G0–G3): [[0, 6, 0, 0], [0, 24, 2, 0], [0, 0, 21, 0], [0, 0, 0, 2]].

### 1.2 Grade agreement per method (both references)

| Method | Mask-derived (PRIMARY): accuracy (k/n) [Wilson 95% CI] | QWK [bootstrap 95% CI] | Over / under | Visual (secondary): accuracy [Wilson] | QWK [bootstrap] | Over / under |
|---|---|---|---|---|---|---|
| SegFormer-B5 (published) | 0.95 (52/55) [0.85, 0.98] | 0.95 [0.88, 1.00] | 1 / 2 | 0.84 (46/55) [0.72, 0.91] | 0.82 [0.72, 0.91] | 0 / 9 |
| UNet++ (published) | 0.98 (54/55) [0.90, 1.00] | 0.98 [0.93, 1.00] | 1 / 0 | 0.87 (48/55) [0.76, 0.94] | 0.85 [0.75, 0.94] | 0 / 7 |
| DeepLabV3+ (published) | 0.96 (53/55) [0.88, 0.99] | 0.97 [0.90, 1.00] | 1 / 1 | 0.85 (47/55) [0.74, 0.92] | 0.84 [0.74, 0.92] | 0 / 8 |
| UNet++ matched, seed 0 | 0.98 (54/55) [0.90, 1.00] | 0.98 [0.93, 1.00] | 1 / 0 | 0.87 (48/55) [0.76, 0.94] | 0.85 [0.75, 0.94] | 0 / 7 |
| UNet++ matched, seed 1 | 0.96 (53/55) [0.88, 0.99] | 0.97 [0.90, 1.00] | 2 / 0 | 0.89 (49/55) [0.78, 0.95] | 0.87 [0.78, 0.95] | 0 / 6 |
| UNet++ matched, seed 2 | 0.98 (54/55) [0.90, 1.00] | 0.98 [0.93, 1.00] | 1 / 0 | 0.87 (48/55) [0.76, 0.94] | 0.85 [0.75, 0.94] | 0 / 7 |
| DeepLabV3+ matched, seed 0 | 0.96 (53/55) [0.88, 0.99] | 0.97 [0.90, 1.00] | 1 / 1 | 0.85 (47/55) [0.74, 0.92] | 0.84 [0.74, 0.92] | 0 / 8 |
| DeepLabV3+ matched, seed 1 | 0.95 (52/55) [0.85, 0.98] | 0.95 [0.88, 1.00] | 1 / 2 | 0.84 (46/55) [0.72, 0.91] | 0.82 [0.72, 0.91] | 0 / 9 |
| DeepLabV3+ matched, seed 2 | 0.96 (53/55) [0.88, 0.99] | 0.97 [0.90, 1.00] | 2 / 0 | 0.89 (49/55) [0.78, 0.95] | 0.87 [0.78, 0.95] | 0 / 6 |
| Colour threshold (redness index >= 0.026) | 0.98 (54/55) [0.90, 1.00] | 0.98 [0.93, 1.00] | 1 / 0 | 0.87 (48/55) [0.76, 0.94] | 0.85 [0.75, 0.94] | 0 / 7 |

All errors are between adjacent grades for every method and both references. No QWK bootstrap resample was undefined.

**Every error against the mask-derived grade** (leaf, reference severity, predicted severity):

- `20260621_Mcitricolor_GX_030`: reference 4.92% (G1) → predicted G2 (5.06–5.31%) by all 10 methods.
- `20260621_Mcitricolor_GX_076`: reference 0.545% (G1) → predicted G0 (0.490–0.496%) by SegFormer-B5 (published), DeepLabV3+ matched, seed 1.
- `20260621_Mcitricolor_GX_110`: reference 4.80% (G1) → predicted G2 (5.04–5.33%) by UNet++ matched, seed 1, DeepLabV3+ matched, seed 2.
- `20260621_Mcitricolor_GX_155`: reference 0.60% (G1) → predicted G0 (0.40–0.47%) by SegFormer-B5 (published), DeepLabV3+ (published), DeepLabV3+ matched, seed 0, DeepLabV3+ matched, seed 1.

Every error against the mask-derived grade falls on a leaf whose reference severity lies 0.045–0.20 pp from a cut-point (distances 0.045, 0.080, 0.10 and 0.20 pp). The one leaf over-graded by every method (`20260621_Mcitricolor_GX_030`, 4.92%) is visual G2, so it is correct against the visual reference.

### 1.3 Matched CNNs, mean ± SD (range) over seeds 0–2

| Configuration | Reference | Accuracy per seed (k/55) | Accuracy mean ± SD (range) | QWK mean ± SD (range) | G1 recall mean ± SD (range) |
|---|---|---|---|---|---|
| UNet++ matched | mask-derived (primary) | 54, 53, 54 | 0.976 ± 0.010 (0.964–0.982) | 0.977 ± 0.010 (0.965–0.983) | 0.949 ± 0.022 (0.923–0.962) |
| UNet++ matched | visual (secondary) | 48, 49, 48 | 0.879 ± 0.010 (0.873–0.891) | 0.859 ± 0.012 (0.852–0.873) | 0.800 ± 0.000 (0.800–0.800) |
| DeepLabV3+ matched | mask-derived (primary) | 53, 52, 53 | 0.958 ± 0.010 (0.945–0.964) | 0.961 ± 0.009 (0.950–0.966) | 0.910 ± 0.022 (0.885–0.923) |
| DeepLabV3+ matched | visual (secondary) | 47, 46, 49 | 0.861 ± 0.028 (0.836–0.891) | 0.845 ± 0.025 (0.825–0.873) | 0.767 ± 0.033 (0.733–0.800) |

Per-seed Wilson and bootstrap intervals are in §1.2. The visual-reference values reproduce `pass3/seed_summary.json` exactly (accuracy, QWK and leaf IoU per seed).

### 1.4 Per-grade recall with Wilson 95% intervals

**Mask-derived reference (PRIMARY)**

| Method | G0 | G1 | G2 | G3 |
|---|---|---|---|---|
| SegFormer-B5 (published) | 6/6 = 1.00 [0.61, 1.00] | 23/26 = 0.88 [0.71, 0.96] | 21/21 = 1.00 [0.85, 1.00] | 2/2 = 1.00 [0.34, 1.00] |
| UNet++ (published) | 6/6 = 1.00 [0.61, 1.00] | 25/26 = 0.96 [0.81, 0.99] | 21/21 = 1.00 [0.85, 1.00] | 2/2 = 1.00 [0.34, 1.00] |
| DeepLabV3+ (published) | 6/6 = 1.00 [0.61, 1.00] | 24/26 = 0.92 [0.76, 0.98] | 21/21 = 1.00 [0.85, 1.00] | 2/2 = 1.00 [0.34, 1.00] |
| UNet++ matched, seed 0 | 6/6 = 1.00 [0.61, 1.00] | 25/26 = 0.96 [0.81, 0.99] | 21/21 = 1.00 [0.85, 1.00] | 2/2 = 1.00 [0.34, 1.00] |
| UNet++ matched, seed 1 | 6/6 = 1.00 [0.61, 1.00] | 24/26 = 0.92 [0.76, 0.98] | 21/21 = 1.00 [0.85, 1.00] | 2/2 = 1.00 [0.34, 1.00] |
| UNet++ matched, seed 2 | 6/6 = 1.00 [0.61, 1.00] | 25/26 = 0.96 [0.81, 0.99] | 21/21 = 1.00 [0.85, 1.00] | 2/2 = 1.00 [0.34, 1.00] |
| DeepLabV3+ matched, seed 0 | 6/6 = 1.00 [0.61, 1.00] | 24/26 = 0.92 [0.76, 0.98] | 21/21 = 1.00 [0.85, 1.00] | 2/2 = 1.00 [0.34, 1.00] |
| DeepLabV3+ matched, seed 1 | 6/6 = 1.00 [0.61, 1.00] | 23/26 = 0.88 [0.71, 0.96] | 21/21 = 1.00 [0.85, 1.00] | 2/2 = 1.00 [0.34, 1.00] |
| DeepLabV3+ matched, seed 2 | 6/6 = 1.00 [0.61, 1.00] | 24/26 = 0.92 [0.76, 0.98] | 21/21 = 1.00 [0.85, 1.00] | 2/2 = 1.00 [0.34, 1.00] |
| Colour threshold (redness index >= 0.026) | 6/6 = 1.00 [0.61, 1.00] | 25/26 = 0.96 [0.81, 0.99] | 21/21 = 1.00 [0.85, 1.00] | 2/2 = 1.00 [0.34, 1.00] |

**Visual reference (secondary)**

| Method | G0 | G1 | G2 | G3 |
|---|---|---|---|---|
| SegFormer-B5 (published) | no leaves | 22/30 = 0.73 [0.56, 0.86] | 22/23 = 0.96 [0.79, 0.99] | 2/2 = 1.00 [0.34, 1.00] |
| UNet++ (published) | no leaves | 24/30 = 0.80 [0.63, 0.90] | 22/23 = 0.96 [0.79, 0.99] | 2/2 = 1.00 [0.34, 1.00] |
| DeepLabV3+ (published) | no leaves | 23/30 = 0.77 [0.59, 0.88] | 22/23 = 0.96 [0.79, 0.99] | 2/2 = 1.00 [0.34, 1.00] |
| UNet++ matched, seed 0 | no leaves | 24/30 = 0.80 [0.63, 0.90] | 22/23 = 0.96 [0.79, 0.99] | 2/2 = 1.00 [0.34, 1.00] |
| UNet++ matched, seed 1 | no leaves | 24/30 = 0.80 [0.63, 0.90] | 23/23 = 1.00 [0.86, 1.00] | 2/2 = 1.00 [0.34, 1.00] |
| UNet++ matched, seed 2 | no leaves | 24/30 = 0.80 [0.63, 0.90] | 22/23 = 0.96 [0.79, 0.99] | 2/2 = 1.00 [0.34, 1.00] |
| DeepLabV3+ matched, seed 0 | no leaves | 23/30 = 0.77 [0.59, 0.88] | 22/23 = 0.96 [0.79, 0.99] | 2/2 = 1.00 [0.34, 1.00] |
| DeepLabV3+ matched, seed 1 | no leaves | 22/30 = 0.73 [0.56, 0.86] | 22/23 = 0.96 [0.79, 0.99] | 2/2 = 1.00 [0.34, 1.00] |
| DeepLabV3+ matched, seed 2 | no leaves | 24/30 = 0.80 [0.63, 0.90] | 23/23 = 1.00 [0.86, 1.00] | 2/2 = 1.00 [0.34, 1.00] |
| Colour threshold (redness index >= 0.026) | no leaves | 24/30 = 0.80 [0.63, 0.90] | 22/23 = 0.96 [0.79, 0.99] | 2/2 = 1.00 [0.34, 1.00] |

Against the mask-derived grade, G0 (6 leaves), G2 (21) and G3 (2) are recalled perfectly by every method; all errors are in mask-G1 (26 leaves): over-grades to G2 at 4.80–4.92% and under-grades to G0 at 0.545–0.60%. Against the visual grade, the 6 visual-G1/mask-G0 leaves are graded G0 by every method, which is why G1 recall is 0.73–0.80 there.

### 1.5 Confusion matrices (rows reference G0–G3, columns predicted G0–G3)

| Method | Mask-derived (primary) | Visual (secondary) |
|---|---|---|
| SegFormer-B5 (published) | [[6, 0, 0, 0], [2, 23, 1, 0], [0, 0, 21, 0], [0, 0, 0, 2]] | [[0, 0, 0, 0], [8, 22, 0, 0], [0, 1, 22, 0], [0, 0, 0, 2]] |
| UNet++ (published) | [[6, 0, 0, 0], [0, 25, 1, 0], [0, 0, 21, 0], [0, 0, 0, 2]] | [[0, 0, 0, 0], [6, 24, 0, 0], [0, 1, 22, 0], [0, 0, 0, 2]] |
| DeepLabV3+ (published) | [[6, 0, 0, 0], [1, 24, 1, 0], [0, 0, 21, 0], [0, 0, 0, 2]] | [[0, 0, 0, 0], [7, 23, 0, 0], [0, 1, 22, 0], [0, 0, 0, 2]] |
| UNet++ matched, seed 0 | [[6, 0, 0, 0], [0, 25, 1, 0], [0, 0, 21, 0], [0, 0, 0, 2]] | [[0, 0, 0, 0], [6, 24, 0, 0], [0, 1, 22, 0], [0, 0, 0, 2]] |
| UNet++ matched, seed 1 | [[6, 0, 0, 0], [0, 24, 2, 0], [0, 0, 21, 0], [0, 0, 0, 2]] | [[0, 0, 0, 0], [6, 24, 0, 0], [0, 0, 23, 0], [0, 0, 0, 2]] |
| UNet++ matched, seed 2 | [[6, 0, 0, 0], [0, 25, 1, 0], [0, 0, 21, 0], [0, 0, 0, 2]] | [[0, 0, 0, 0], [6, 24, 0, 0], [0, 1, 22, 0], [0, 0, 0, 2]] |
| DeepLabV3+ matched, seed 0 | [[6, 0, 0, 0], [1, 24, 1, 0], [0, 0, 21, 0], [0, 0, 0, 2]] | [[0, 0, 0, 0], [7, 23, 0, 0], [0, 1, 22, 0], [0, 0, 0, 2]] |
| DeepLabV3+ matched, seed 1 | [[6, 0, 0, 0], [2, 23, 1, 0], [0, 0, 21, 0], [0, 0, 0, 2]] | [[0, 0, 0, 0], [8, 22, 0, 0], [0, 1, 22, 0], [0, 0, 0, 2]] |
| DeepLabV3+ matched, seed 2 | [[6, 0, 0, 0], [0, 24, 2, 0], [0, 0, 21, 0], [0, 0, 0, 2]] | [[0, 0, 0, 0], [6, 24, 0, 0], [0, 0, 23, 0], [0, 0, 0, 2]] |
| Colour threshold (redness index >= 0.026) | [[6, 0, 0, 0], [0, 25, 1, 0], [0, 0, 21, 0], [0, 0, 0, 2]] | [[0, 0, 0, 0], [6, 24, 0, 0], [0, 1, 22, 0], [0, 0, 0, 2]] |

### 1.6 Paired comparisons (first minus second), both references

Family P = published pairs (as in `results/paired`, Table 3 order); B = each published network vs the colour threshold; M = each matched seed vs the published SegFormer-B5. Holm within family and reference. b / c = leaves graded correctly only by the first / only by the second method.

| Family | Comparison | Reference | Δaccuracy [Tango 95% CI] | McNemar b / c; exact p (Holm) | ΔQWK [bootstrap 95% CI] | Same grade |
|---|---|---|---|---|---|---|
| P | SegFormer-B5 (published) vs DeepLabV3+ (published) | mask | −0.018 [-0.096, 0.048] | 0 / 1; 1.00 (1.00) | −0.016 [-0.053, 0.000] | 54/55 |
| P | SegFormer-B5 (published) vs DeepLabV3+ (published) | visual | −0.018 [-0.096, 0.048] | 0 / 1; 1.00 (1.00) | −0.013 [-0.046, 0.000] | 54/55 |
| P | SegFormer-B5 (published) vs UNet++ (published) | mask | −0.036 [-0.123, 0.031] | 0 / 2; 0.50 (1.00) | −0.032 [-0.084, 0.000] | 53/55 |
| P | SegFormer-B5 (published) vs UNet++ (published) | visual | −0.036 [-0.123, 0.031] | 0 / 2; 0.50 (1.00) | −0.028 [-0.073, 0.000] | 53/55 |
| P | UNet++ (published) vs DeepLabV3+ (published) | mask | +0.018 [-0.048, 0.096] | 1 / 0; 1.00 (1.00) | +0.016 [0.000, 0.056] | 54/55 |
| P | UNet++ (published) vs DeepLabV3+ (published) | visual | +0.018 [-0.048, 0.096] | 1 / 0; 1.00 (1.00) | +0.014 [0.000, 0.049] | 54/55 |
| B | SegFormer-B5 (published) vs Colour threshold (redness index >= 0.026) | mask | −0.036 [-0.123, 0.031] | 0 / 2; 0.50 (1.00) | −0.032 [-0.084, 0.000] | 53/55 |
| B | SegFormer-B5 (published) vs Colour threshold (redness index >= 0.026) | visual | −0.036 [-0.123, 0.031] | 0 / 2; 0.50 (1.00) | −0.028 [-0.073, 0.000] | 53/55 |
| B | UNet++ (published) vs Colour threshold (redness index >= 0.026) | mask | +0.000 [-0.065, 0.065] | 0 / 0; 1.00 (1.00) | +0.000 [0.000, 0.000] | 55/55 |
| B | UNet++ (published) vs Colour threshold (redness index >= 0.026) | visual | +0.000 [-0.065, 0.065] | 0 / 0; 1.00 (1.00) | +0.000 [0.000, 0.000] | 55/55 |
| B | DeepLabV3+ (published) vs Colour threshold (redness index >= 0.026) | mask | −0.018 [-0.096, 0.048] | 0 / 1; 1.00 (1.00) | −0.016 [-0.056, 0.000] | 54/55 |
| B | DeepLabV3+ (published) vs Colour threshold (redness index >= 0.026) | visual | −0.018 [-0.096, 0.048] | 0 / 1; 1.00 (1.00) | −0.014 [-0.049, 0.000] | 54/55 |
| M | UNet++ matched, seed 0 vs SegFormer-B5 (published) | mask | +0.036 [-0.031, 0.123] | 2 / 0; 0.50 (1.00) | +0.032 [0.000, 0.084] | 53/55 |
| M | UNet++ matched, seed 0 vs SegFormer-B5 (published) | visual | +0.036 [-0.031, 0.123] | 2 / 0; 0.50 (1.00) | +0.028 [0.000, 0.073] | 53/55 |
| M | UNet++ matched, seed 1 vs SegFormer-B5 (published) | mask | +0.018 [-0.065, 0.108] | 2 / 1; 1.00 (1.00) | +0.015 [-0.043, 0.072] | 52/55 |
| M | UNet++ matched, seed 1 vs SegFormer-B5 (published) | visual | +0.055 [-0.014, 0.149] | 3 / 0; 0.25 (1.00) | +0.049 [0.000, 0.116] | 52/55 |
| M | UNet++ matched, seed 2 vs SegFormer-B5 (published) | mask | +0.036 [-0.031, 0.123] | 2 / 0; 0.50 (1.00) | +0.032 [0.000, 0.084] | 53/55 |
| M | UNet++ matched, seed 2 vs SegFormer-B5 (published) | visual | +0.036 [-0.031, 0.123] | 2 / 0; 0.50 (1.00) | +0.028 [0.000, 0.073] | 53/55 |
| M | DeepLabV3+ matched, seed 0 vs SegFormer-B5 (published) | mask | +0.018 [-0.048, 0.096] | 1 / 0; 1.00 (1.00) | +0.016 [0.000, 0.053] | 54/55 |
| M | DeepLabV3+ matched, seed 0 vs SegFormer-B5 (published) | visual | +0.018 [-0.048, 0.096] | 1 / 0; 1.00 (1.00) | +0.013 [0.000, 0.046] | 54/55 |
| M | DeepLabV3+ matched, seed 1 vs SegFormer-B5 (published) | mask | +0.000 [-0.065, 0.065] | 0 / 0; 1.00 (1.00) | +0.000 [0.000, 0.000] | 55/55 |
| M | DeepLabV3+ matched, seed 1 vs SegFormer-B5 (published) | visual | +0.000 [-0.065, 0.065] | 0 / 0; 1.00 (1.00) | +0.000 [0.000, 0.000] | 55/55 |
| M | DeepLabV3+ matched, seed 2 vs SegFormer-B5 (published) | mask | +0.018 [-0.065, 0.108] | 2 / 1; 1.00 (1.00) | +0.015 [-0.043, 0.072] | 52/55 |
| M | DeepLabV3+ matched, seed 2 vs SegFormer-B5 (published) | visual | +0.055 [-0.014, 0.149] | 3 / 0; 0.25 (1.00) | +0.049 [0.000, 0.116] | 52/55 |

Largest number of discordant leaves in any comparison: 3; smallest Holm-adjusted McNemar p: 1.00. ΔQWK bootstrap intervals with a bound at exactly 0 arise when all discordant leaves favour one method (degenerate, not evidence of equivalence; see `results/paired` §6). The published UNet++ and the colour threshold, and the matched DeepLabV3+ seed 1 and the published SegFormer-B5, assign identical grades to all 55 leaves (Δ = 0, Tango [−0.065, +0.065]).

### 1.7 Where conclusions change versus the visual-grade reference

Automated check (`pairs_whose_reading_changes` in results.json): 0 of 12 comparisons change sign of Δaccuracy, McNemar significance, Tango-interval exclusion of 0 or ΔQWK-interval exclusion of 0 between references.

1. **Unchanged: no detectable between-method difference in grading.** Published models, matched seeds and the colour threshold show no detectable difference in grade correctness under either reference (not a demonstration of equivalence: the Tango 95% intervals for Δaccuracy span up to −0.12 to +0.15). The McNemar counts b / c are identical under both references for 10 of 12 comparisons, because on the 8 reference-disagreement leaves all methods but two matched seeds give the same grade.
2. **Changed: level.** Accuracy rises from 0.84–0.89 (visual) to 0.95–0.98 (mask) and QWK from 0.82–0.87 to 0.95–0.98 for every method. This is expected: the networks and the threshold estimate the same annotated area that defines the mask-derived grade. It is agreement with the annotator's own masks through the cut-points, not an independent check.
3. **Changed: error direction.** Against the visual grade every leaf-spot error is an under-grade (6–9 per method). Against the mask-derived grade each method has 1–2 over-grades (the 4.92% leaf, plus the 4.80% leaf for two matched seeds) and 0–2 under-grades (0.545% and 0.60% leaves). The manuscript sentence 'under-grades for American leaf spot' (03a_results.tex line 68) and 'Leaf-spot errors were under-calls' (04_discussion.tex line 13) hold only for the visual reference.
4. **Changed: weakest grade.** Visual reference: G1 recall 0.73–0.80 (lowest). Mask reference: G1 recall 0.88–0.96; G0 becomes testable (6 leaves, 6/6 for every method, Wilson lower bound 0.61).
5. **Slight change in ordering (not significant).** Mask: published UNet++, matched UNet++ seeds 0 and 2 and the colour threshold 54/55; published DeepLabV3+, matched UNet++ seed 1 and matched DeepLabV3+ seeds 0 and 2 53/55; published SegFormer-B5 and matched DeepLabV3+ seed 1 52/55. Visual: matched UNet++ seed 1 and matched DeepLabV3+ seed 2 top at 49/55 (they grade the 4.80% leaf G2, which the visual grade calls G2). The published SegFormer-B5 is lowest under both references (tied with matched DeepLabV3+ seed 1).
6. **Colour threshold.** Competitive under both references (54/55 mask, 48/55 visual), identical grades to the published UNet++ on every leaf. The brief's framing (fact 12) is unchanged by the switch of reference.

### 1.8 Cross-check against `results/paired` and other verified folders

| Check | This analysis | Earlier folder | Match |
|---|---|---|---|
| SegFormer-B5, mask-derived: correct, QWK, confusion | 52/55, 0.9505 | 52/55, 0.9505 (paired) | yes |
| SegFormer-B5, mask-derived QWK 95% CI (seed 20260928 vs 20260927) | [0.881, 1.000] | [0.882, 1.000] (paired) | Monte Carlo only |
| SegFormer-B5, visual: correct, QWK, confusion | 46/55, 0.8247 | 46/55, 0.8247 (paired) | yes |
| UNet++, mask-derived: correct, QWK, confusion | 54/55, 0.9825 | 54/55, 0.9825 (paired) | yes |
| UNet++, mask-derived QWK 95% CI (seed 20260928 vs 20260927) | [0.934, 1.000] | [0.936, 1.000] (paired) | Monte Carlo only |
| UNet++, visual: correct, QWK, confusion | 48/55, 0.8522 | 48/55, 0.8522 (paired) | yes |
| DeepLabV3+, mask-derived: correct, QWK, confusion | 53/55, 0.9660 | 53/55, 0.9660 (paired) | yes |
| DeepLabV3+, mask-derived QWK 95% CI (seed 20260928 vs 20260927) | [0.905, 1.000] | [0.906, 1.000] (paired) | Monte Carlo only |
| DeepLabV3+, visual: correct, QWK, confusion | 47/55, 0.8379 | 47/55, 0.8379 (paired) | yes |
| McNemar b / c segformer_vs_deeplab, mask / visual | [0, 1] / [0, 1] | [0, 1] / [0, 1] (paired) | yes |
| McNemar b / c segformer_vs_unetpp, mask / visual | [0, 2] / [0, 2] | [0, 2] / [0, 2] (paired) | yes |
| McNemar b / c unetpp_vs_deeplab, mask / visual | [1, 0] / [1, 0] | [1, 0] / [1, 0] (paired) | yes |
| Visual = mask-derived grade | 47/55, QWK 0.831 | 47/55, QWK 0.831 (paired) | yes |
| Colour threshold, visual: correct, QWK, MAE, IoU | 48/55, 0.85, 0.23, 0.807 | 48/55, 0.85, 0.23, 0.807 (baseline) | yes |
| Matched UNet++, visual accuracy / QWK / IoU per seed | [0.873, 0.891, 0.873] / [0.852, 0.873, 0.852] / [0.874, 0.882, 0.878] | [0.873, 0.891, 0.873] / [0.852, 0.873, 0.852] / [0.874, 0.882, 0.878] (seed_summary.json) | yes |
| Matched DeepLabV3+, visual accuracy / QWK / IoU per seed | [0.855, 0.836, 0.891] / [0.838, 0.825, 0.873] / [0.855, 0.856, 0.83] | [0.855, 0.836, 0.891] / [0.838, 0.825, 0.873] / [0.855, 0.856, 0.83] (seed_summary.json) | yes |

All exact checks passed: **True**. The only differences are the bootstrap QWK bounds (different seed, ≤ 0.003).

### 1.9 Suggested manuscript text (LaTeX)

New Table 5 leaf-spot rows (rust rows unchanged). The leaf-spot QWK intervals below use this folder's bootstrap (seed 20260928); if Table 5 keeps the older intervals for the visual rows, keep them consistently.

```latex
Leaf spot & UNet++ & 0.98 (54/55) [0.90, 1.00] & 0.98 [0.93, 1.00] & 0.87 / 0.85 \\
 & DeepLabV3+ & 0.96 (53/55) [0.88, 0.99] & 0.97 [0.90, 1.00] & 0.85 / 0.84 \\
 & SegFormer-B5 & 0.95 (52/55) [0.85, 0.98] & 0.95 [0.88, 1.00] & 0.84 / 0.82 \\
```

Column headers: `Accuracy (k/n) [95% CI]` and `QWK [95% CI]` now refer to the mask-derived grade for leaf spot; the last column becomes `Visual grade: accuracy / QWK` (secondary). Caption: rust and leaf spot reference = grade derived from the reference mask; last column = leaf-spot agreement with the annotator's visual grade.

```latex
For American leaf spot the primary reference grade was derived from the reference masks through the cut-points; the annotator's visual grade is a secondary reference. Against the mask-derived grade, accuracy was 0.95 (52/55), 0.98 (54/55) and 0.96 (53/55) and quadratic-weighted kappa 0.95, 0.98 and 0.97 for SegFormer-B5, UNet++ and DeepLabV3+ (Table~\ref{tab5}); the matched UNet++ and DeepLabV3+ reached 0.976 $\pm$ 0.010 and 0.958 $\pm$ 0.010 (three seeds), and the color threshold 0.98 (54/55). All errors were one grade and lay on leaves whose reference severity was 0.045--0.20 percentage points from a cut-point; one leaf at 4.92\% was over-graded by every method. Because the models were trained to reproduce these masks, this agreement is not an independent check. The visual grade matched the mask-derived grade for 47 of 55 leaves; all eight disagreements were one grade higher in the visual grade, six at G0--G1 (mask severity 0.13--0.40\%) and two at G1--G2 (4.80 and 4.92\%). Against the visual grade, accuracy was 0.84--0.89 and kappa 0.82--0.87, and every error was an under-grade (6--9 per method), six of them on the six leaves graded G1 visually but G0 by their masks, which every method graded G0. No pair of models, and no model and the color threshold, differed detectably in per-leaf grade correctness under either reference (at most 3 discordant leaves; Holm-adjusted exact McNemar $p$ = 1.00).
```

Wording to change elsewhere if this becomes primary: `03a_results.tex` line 68 ('under-grades for American leaf spot, within which recall was lowest for G1 ... 0.73--0.80') should say that against the mask-derived grade leaf-spot errors were 1--2 over-grades and 0--2 under-grades per model, on leaves 0.045--0.20 pp from a cut-point (G1 recall 0.88--0.96), and that the under-grades and the low G1 recall (0.73--0.80) apply to the visual reference. `04_discussion.tex` line 13 ('Leaf-spot errors were under-calls') needs the same qualification. `02_methods.tex` must define the mask-derived grade as the leaf-spot reference and the visual grade as secondary (revision brief fact 2 still names the visual grade as the reference).

### 1.10 Caveats (part 1)

- The mask-derived grade comes from the same masks the networks were trained to reproduce; its agreement measures reproduction of the annotated area at the cut-points, not correctness against an independent judgement. The visual grade, from the same single annotator, is the only other leaf-spot reference; the expert panel is the independent human check.
- G3 rests on 2 leaves (Wilson lower bound 0.34) and mask-G0 on 6. All intervals reflect test-leaf sampling only; the published models were trained once, the matched ones three times.
- The matched-vs-SegFormer comparisons (family M) are per seed; they are not a test of architecture and SegFormer-B5's own seed variability is unknown.

## 2. The deployment-only rust rescaling

### 2.1 What it is and where it is documented

Search terms: `rescal`, `calibrat`, `scale factor`, `scale_factor`, `s_cal`, `sev_cal`, the literal coefficients `1.1915` and `0.4329`, and fitting calls (`polyfit`, `linregress`, `lstsq`, `LinearRegression`).

| Location searched | Result |
|---|---|
| `MDPI_rebuild/manuscript/sections/*.tex`, `manuscript.tex` | `02_methods.tex` line 42: the test partition's 'only other uses were fitting the deployment-only rust rescaling (Section 2.x) and a supporting per-face comparison'; line 75: the deployed application adds '... and, for rust only, a linear severity rescaling fitted on the 21 rust test leaves (Supplementary Note S1); none enters any reported figure'. `04_discussion.tex` line 41: 'a rust rescaling fitted on the 21 rust test leaves' was not benchmarked; line 13: 'a recalibration fitted on independent leaves could reduce this'. |
| `manuscript/supplementary/note_methods.tex` | Line 22: the full definition, s_cal = 1.1915 s + 0.4329 (clipped to 0–100), applied to the **refined-lesion severity** of the deployed pipeline, 'corrects a systematic under-report of that refined pipeline relative to the reference severity; the two coefficients were fitted on the 21 rust test leaves'; leaf spot uses no rescaling. Line 23: `% TODO(author): refit the deployment-only rust rescaling on validation leaves ...`. |
| `manuscript/supplementary/supplementary.tex` | Line 87: the deployed post-processing pipeline (thresholding, refinement and rust rescaling) was not evaluated against the reference. |
| Older drafts (`MDPI_rebuild/main.tex`, `body_draft.tex`, `raw_lines.txt`, `build_main.py`) and the thesis (`26038_Thesis_FINAL_UPDATED.docx`, section 3.6.5) | Same sentence, worded 'fitted on the 21 rust test leaves whose lesions were delineated with Segment Anything Model 2'; this wording predates brief fact 1 (all 142 rust leaves are SAM 2 masks). |
| `<ROOT>` (all code: `modal-recovery/kernels`, `modal-recovery/serving` (ONNX only), `notebooks/*`, `kernel-outputs/*`, `validation/*`, `reanalysis/*`) | **No match** for the coefficients or any rescaling/fitting code. `validation/run_models.py` (expert-panel inference) calls `coffee-app/backend/infer.py`: raw argmax, no rescaling. |
| Local app folders `C:/dev/coffee-app` (FastAPI backend `infer.py`), `C:/dev/coffee-cpp` (`analyzer.cpp`), `C:/dev/desktop-app`, `C:/dev/website` | **No match.** These apps compute severity = lesion / leaf from the raw argmax and grade it; none applies a BiRefNet erosion, a 0.50 probability threshold, region growing or a rescaling. |
| Code files under `<USER_HOME>` (py, js, ts, dart, kt, java, swift, cpp, h, json, ipynb, md, txt; excluding AppData, virtual environments, node_modules) | Only the manuscript-derived text files above and earlier Claude session transcripts (text copies of the same sentence). |

**What it is (from the text only).** A two-parameter linear map applied, in the deployed web/mobile application only and for rust only, to the severity of the post-processed lesion mask (BiRefNet leaf mask eroded by 12 iterations, lesion probability ≥ 0.50, components < 20 px removed, closing, redness-guided region grow with a per-leaf gate at the 60th percentile of leaf redness). Its stated purpose is to correct an **under-report** of that refined pipeline; the raw argmax outputs used for every reported result **over-report** (bias +0.99 to +1.75 pp). Stated fitting set: the 21 diseased rust test leaves.

**What cannot be determined from the files:** the code that applies it (the serverless worker behind https://pg.carlychery.com and the Capacitor mobile app are not in any local folder); the script and the per-leaf refined severities it was fitted on; the fitting method (OLS of reference on refined severity is the natural reading of the form, but unconfirmed); which rust model(s) it serves (all three? SegFormer-B5 only runs on the web tier); whether the 54 healthy test leaves were included; its fit statistics. An implication that can be computed from the coefficients: the deployed application reports ≥ 0.5% (G1) for any rust leaf whose refined severity is ≥ 0.056% (because the intercept alone is 0.43 pp), so its specificity on healthy leaves depends on the refined pipeline predicting almost no lesion; this was never evaluated.

### 2.2 Verdict

**The published rescaling cannot be reproduced**, and it cannot be refitted on validation leaves on its own input, because neither the deployed refinement pipeline nor its outputs exist locally. (Re-implementing the refinement from the Supplement's prose would be a guess: the region-grow bound, the fixed leaf-spot gate and the model are not specified.)

### 2.3 Validation-only refit on the raw-argmax outputs (what the local data permit)

Input: published models' raw-argmax rust severities (`out/per_image.csv`), abaxial face. Fit: OLS of reference severity on predicted severity, s_cal = clip(a·s + b, 0, 100), on validation leaves only; evaluated on the 21 diseased test leaves and the 21 locally available healthy test leaves (the paper used 54). Validation raw bias of SegFormer-B5: +0.57 pp (MAE 0.58; calibration slope of predicted on reference 1.29, test 1.22). Raw test MAE and bias reproduce Table 4 for all three models (0.99/+0.99, 1.77/+1.75, 1.47/+1.41). This is **not** the deployed transform (different input); it answers what a validation-fitted linear recalibration of the reported outputs would do.

| Model | Fit | Parameters (t-based 95% CI; bootstrap CI over validation leaves) | Test MAE, pp: raw → rescaled [95% CI] | ΔMAE [paired 95% CI] | Test bias, pp: raw → rescaled [95% CI] | Grade accuracy | QWK [95% CI] | Healthy test ≥ 0.5% after: abaxial (either face) |
|---|---|---|---|---|---|---|---|---|
| SegFormer-B5 | **Validation OLS, 21 diseased leaves (primary; same set type as the published fit)** | a = 0.730 [0.642, 0.818] (boot [0.623, 0.908]); b = +0.487 [0.082, 0.891] (boot [-0.016, 0.864]); R² 0.941 | 0.99 → 0.59 [0.37, 0.83] | −0.40 [-0.84, -0.04] | +0.99 → −0.27 [-0.59, 0.04] | 19/21 → 19/21 (over/under 1/1); McNemar p 1.00 | 0.88 → 0.83 [0.61, 1.00] | 15/21 (18/21) |
| SegFormer-B5 | Validation OLS, 21 diseased + 21 healthy leaves | a = 0.816 [0.769, 0.863] (boot [0.733, 0.932]); b = +0.014 [-0.140, 0.168] (boot [-0.125, 0.135]); R² 0.968 | 0.99 → 0.40 [0.23, 0.61] | −0.59 [-1.13, -0.15] | +0.99 → −0.19 [-0.39, 0.09] | 19/21 → 20/21 (over/under 0/1); McNemar p 1.00 | 0.88 → 0.93 [0.71, 1.00] | 0/21 (0/21) |
| SegFormer-B5 | Validation, through origin, 21 diseased leaves | a = 0.820 (boot [0.750, 0.907]); b = 0 | 0.99 → 0.40 [0.23, 0.61] | −0.59 [-1.12, -0.15] | +0.99 → −0.17 [-0.38, 0.11] | 19/21 → 20/21 (over/under 0/1); McNemar p 1.00 | 0.88 → 0.93 [0.71, 1.00] | 0/21 (0/21) |
| SegFormer-B5 | Illustration only: published 1.1915 s + 0.4329 applied to raw argmax | a = 1.1915; b = +0.4329 | 0.99 → 2.66 [1.68, 3.78] | +1.67 [1.19, 2.21] | +0.99 → +2.66 [1.68, 3.78] | 19/21 → 16/21 (over/under 5/0); McNemar p 0.25 | 0.88 → 0.71 [0.53, 0.86] | 14/21 (17/21) |
| SegFormer-B5 | Illustration only: OLS fitted on the 21 test leaves (in-sample) | a = 0.807 [0.764, 0.851]; b = +0.256 [-0.136, 0.649]; R² 0.987 | 0.99 → 0.35 [0.20, 0.57] | −0.64 [-1.15, -0.21] | +0.99 → −0.00 [-0.21, 0.27] | 19/21 → 19/21 (over/under 1/1); McNemar p 1.00 | 0.88 → 0.83 [0.61, 1.00] | 1/21 (2/21) |
| UNet++ | **Validation OLS, 21 diseased leaves (primary; same set type as the published fit)** | a = 0.625 [0.534, 0.716] (boot [0.513, 0.770]); b = +0.628 [0.161, 1.096] (boot [0.114, 1.133]); R² 0.916 | 1.77 → 0.89 [0.55, 1.30] | −0.88 [-1.67, -0.17] | +1.75 → −0.33 [-0.87, 0.16] | 17/21 → 19/21 (over/under 1/1); McNemar p 0.69 | 0.79 → 0.83 [0.61, 1.00] | 21/21 (21/21) |
| UNet++ | Validation OLS, 21 diseased + 21 healthy leaves | a = 0.708 [0.661, 0.755] (boot [0.628, 0.806]); b = +0.121 [-0.051, 0.293] (boot [-0.004, 0.266]); R² 0.958 | 1.77 → 0.78 [0.49, 1.10] | −0.99 [-1.75, -0.28] | +1.75 → −0.24 [-0.66, 0.21] | 17/21 → 20/21 (over/under 0/1); McNemar p 0.38 | 0.79 → 0.93 [0.71, 1.00] | 0/21 (0/21) |
| UNet++ | Validation, through origin, 21 diseased leaves | a = 0.728 (boot [0.658, 0.811]); b = 0 | 1.77 → 0.78 [0.50, 1.10] | −0.99 [-1.76, -0.26] | +1.75 → −0.22 [-0.62, 0.24] | 17/21 → 20/21 (over/under 0/1); McNemar p 0.38 | 0.79 → 0.93 [0.71, 1.00] | 0/21 (0/21) |
| UNet++ | Illustration only: published 1.1915 s + 0.4329 applied to raw argmax | a = 1.1915; b = +0.4329 | 1.77 → 3.56 [2.16, 5.09] | +1.80 [1.25, 2.40] | +1.75 → +3.56 [2.16, 5.09] | 17/21 → 14/21 (over/under 7/0); McNemar p 0.25 | 0.79 → 0.64 [0.47, 0.79] | 4/21 (6/21) |
| UNet++ | Illustration only: OLS fitted on the 21 test leaves (in-sample) | a = 0.718 [0.647, 0.789]; b = +0.286 [-0.426, 0.998]; R² 0.959 | 1.77 → 0.73 [0.45, 1.06] | −1.04 [-1.78, -0.34] | +1.75 → +0.00 [-0.41, 0.45] | 17/21 → 19/21 (over/under 1/1); McNemar p 0.69 | 0.79 → 0.83 [0.61, 1.00] | 0/21 (0/21) |
| DeepLabV3+ | **Validation OLS, 21 diseased leaves (primary; same set type as the published fit)** | a = 0.613 [0.513, 0.714] (boot [0.513, 0.762]); b = +0.754 [0.245, 1.263] (boot [0.265, 1.239]); R² 0.895 | 1.47 → 1.19 [0.65, 1.82] | −0.28 [-1.17, 0.50] | +1.41 → −0.50 [-1.25, 0.23] | 19/21 → 19/21 (over/under 1/1); McNemar p 1.00 | 0.88 → 0.83 [0.61, 1.00] | 21/21 (21/21) |
| DeepLabV3+ | Validation OLS, 21 diseased + 21 healthy leaves | a = 0.710 [0.656, 0.765] (boot [0.638, 0.825]); b = +0.166 [-0.028, 0.360] (boot [0.036, 0.309]); R² 0.946 | 1.47 → 1.02 [0.53, 1.59] | −0.46 [-1.20, 0.24] | +1.41 → −0.42 [-1.03, 0.28] | 19/21 → 20/21 (over/under 0/1); McNemar p 1.00 | 0.88 → 0.93 [0.71, 1.00] | 0/21 (0/21) |
| DeepLabV3+ | Validation, through origin, 21 diseased leaves | a = 0.738 (boot [0.671, 0.841]); b = 0 | 1.47 → 1.02 [0.55, 1.58] | −0.45 [-1.18, 0.23] | +1.41 → −0.39 [-0.98, 0.30] | 19/21 → 19/21 (over/under 0/2); McNemar p 1.00 | 0.88 → 0.87 [0.67, 1.00] | 0/21 (0/21) |
| DeepLabV3+ | Illustration only: published 1.1915 s + 0.4329 applied to raw argmax | a = 1.1915; b = +0.4329 | 1.47 → 3.16 [1.83, 4.62] | +1.69 [1.15, 2.27] | +1.41 → +3.16 [1.83, 4.62] | 19/21 → 14/21 (over/under 7/0); McNemar p 0.06 | 0.88 → 0.64 [0.47, 0.79] | 1/21 (3/21) |
| DeepLabV3+ | Illustration only: OLS fitted on the 21 test leaves (in-sample) | a = 0.748 [0.635, 0.861]; b = +0.321 [-0.746, 1.389]; R² 0.910 | 1.47 → 0.95 [0.50, 1.52] | −0.53 [-1.14, 0.06] | +1.41 → −0.00 [-0.58, 0.69] | 19/21 → 19/21 (over/under 1/1); McNemar p 1.00 | 0.88 → 0.83 [0.61, 1.00] | 0/21 (0/21) |

Before rescaling, 0/21 healthy test leaves reach 0.5% for every model (abaxial and either face). Bootstrap CIs: 10,000 resamples of test leaves (paired: the same resamples for raw and rescaled); parameter bootstrap: 10,000 resamples of validation leaves.

### 2.4 Reading

- **SegFormer-B5, primary validation refit** (a = 0.730, b = +0.487 pp): test MAE falls from 0.99 to 0.59 pp (Δ −0.40, 95% CI [-0.84, -0.04]) and the bias from +0.99 to −0.27 pp, but grade accuracy stays 19/21 (2 errors fixed, 2 created; QWK 0.88 → 0.83) and the intercept lifts every healthy leaf to at least 0.49%: 15/21 healthy test leaves become G1 (abaxial rule; Wilson 95% CI of the FP rate [0.50, 0.86]). The same happens, worse, for the CNNs (21/21). A validation fit on diseased leaves only is therefore unusable for grading.
- **With healthy validation leaves in the fit, or without an intercept**, the intercept is ≈ 0 (SegFormer-B5 b = +0.014 pp, [-0.140, 0.168]), healthy leaves stay G0 (0/21), test MAE falls to 0.40 pp (Δ −0.59, [-1.13, -0.15]) with a small negative bias (−0.19 pp), and one more leaf is graded correctly (20/21, QWK 0.93; McNemar p = 1.00). For UNet++ the MAE reduction is also clear (1.77 → 0.78 pp); for DeepLabV3+ its CI includes 0.
- **The published coefficients do not transfer to the raw outputs** (they were fitted for a pipeline that under-reports): applied to raw SegFormer-B5 they raise test MAE from 0.99 to 2.66 pp and create 3 more grade errors, and they lift 14/21 healthy leaves to ≥ 0.5% (the intercept is 0.43 pp). This is an illustration only, not an evaluation of the deployed application.
- **Optimism of test fitting**: the in-sample test fit (a = 0.807, b = +0.256) gives test MAE 0.35 pp and bias exactly 0 by construction, better than any validation fit (0.59, 0.40); a test-fitted rescaling cannot be reported as an out-of-sample result.
- The validation and test calibration slopes of SegFormer-B5 differ (1.29 vs 1.22), with 21 leaves each, so the rescaling parameters are themselves uncertain (bootstrap CI of a: [0.62, 0.91] diseased-only; [0.73, 0.93] with healthy leaves).

### 2.5 Recommendation

1. **Remove the deployment-only rust rescaling and its coefficients from the paper.** It was fitted on the test leaves, cannot be reproduced from any retained code or data, applies to a post-processing pipeline that was never evaluated, and none of the reported results uses it. Keep one sentence saying that the deployed application's post-processing (including any severity adjustment) was not evaluated.
2. If the application keeps a rust rescaling: refit it on validation leaves only, on the deployed pipeline's own refined severities, with the healthy validation leaves included or the intercept fixed at 0 (a positive intercept grades healthy leaves G1), and record the script, model and coefficients in the code release.
3. The Discussion sentence 'a recalibration fitted on independent leaves could reduce this' (over-grading) is only weakly supported: a validation-fitted linear recalibration of the raw SegFormer-B5 output reduced test MAE (0.99 → 0.40 pp with healthy leaves in the fit) but gave a net gain of only one correctly graded leaf of 21 (19 to 20: two over-grades corrected, one new under-grade; McNemar p = 1.00). Suggested wording below.

### 2.6 Suggested LaTeX replacements

`02_methods.tex` line 42, replace the last clause:
```latex
The test partition was sealed before final training and informed no modeling, threshold, or calibration decision; its only other use was a supporting per-face comparison of test IoU for the abaxial-face rule.
```
`02_methods.tex` line 75, replace the last sentence:
```latex
All reported results use each model's raw argmax segmentation. The deployed application adds a BiRefNet-derived leaf mask~\cite{birefnet}, a lesion-probability threshold (0.50, fixed a priori), and lesion refinement (Supplementary Note~\sref{note:methods}); this post-processing was not evaluated and enters no reported figure.
```
`04_discussion.tex` line 41, replace the parenthesis:
```latex
The deployed pipeline (post-processing, on-device performance) was not benchmarked, and per-surface test IoU was consulted in support of the abaxial rust rule.
```
`04_discussion.tex` line 13, replace 'a recalibration fitted on independent leaves could reduce this':
```latex
a linear recalibration fitted on the diseased and healthy validation leaves reduced the SegFormer-B5 rust MAE on the test leaves (0.99 to 0.40 percentage points) but raised grade accuracy by only one leaf (19 to 20 of 21), so its benefit for grading remains to be shown
```
Supplement `note_methods.tex` line 22: delete the sentence beginning 'For rust only, a linear rescaling ...' through '... fitted on the 21 rust test leaves.' and keep 'American leaf spot used no severity rescaling' only if a rust rescaling is still described; delete the TODO on line 23. `supplementary.tex` line 87: '(thresholding, refinement, and rust rescaling)' → '(thresholding and refinement)'.

If the author prefers to keep a statement about the rescaling instead of removing it:
```latex
For rust only, the deployed application also rescales the refined-lesion severity linearly; its coefficients were fitted on the rust test leaves, the fitting script and data were not retained, and the rescaling was not evaluated.
```

## 3. [Section omitted from the public release]

This section listed the expert-panel identities (names, occupations, affiliations, years of experience). Identities are not part of the code release.

## 4. Files

- `grades_analysis.py` → `grades_results.json`, `leafspot_per_leaf_grades.csv` (part 1)
- `rescale_analysis.py` → `rescale_results.json` (part 2)
- `verify_experts.py` → `experts_verified.json` (part 3; names, no grades)
- `make_md.py` → `results.md`, `results.json` (merged)
- Run: `python grades_analysis.py && python rescale_analysis.py && python verify_experts.py && python make_md.py` (Python 3.13.15, numpy 2.5.3, scipy 1.18.1).


## Verifier corrections (independent verification, 28 Sep 2026)

Verifier script: `verifier_check.py` → `verifier_check.json`, written from scratch (own grade, weighted-kappa via confusion/expected matrices, Wilson, exact McNemar, Holm, a numerically solved Tango score interval, OLS with t intervals); it does not import `grades_analysis.py`, `rescale_analysis.py` or `results/paired/paired_analysis.py`.

**Re-derived and confirmed (no change):**
- Leaf sets and protocol: 55 leaf-spot test leaves, both faces present for every leaf and method; leaf severity = mean of face severities; reference severities identical across `per_image.csv`, `per_image_retrained.csv` (lrB, seeds 0–2) and `results/baseline/per_leaf_test.csv` (≤ 1e-5 pp); visual grade = `grade_label`; cut-points 0.5/5/20 with G = number of cut-points ≤ severity.
- Reference counts (mask 6/26/21/2; visual 0/30/23/2), 47/55 agreement, QWK 0.831 [0.721, 0.923], the 8 disagreement leaves and their severities.
- For all 10 methods and both references: correct counts, Wilson CIs, QWK point estimates, QWK bootstrap CIs (identical to 3 decimals), over/under counts, confusion matrices, per-grade recall, no undefined bootstrap resamples; colour threshold = published UNet++ on 55/55 leaves; seed means ± SD (ddof 1) for accuracy and QWK.
- All 24 paired rows (families P/B/M × 2 references): b/c, exact p, Holm-adjusted p (all 1.00), Tango intervals (identical to 3 decimals), same-grade counts; 10 of 12 comparisons have identical b/c under both references.
- Part 2: n = 21/21/21/21 (val diseased, val healthy, test diseased, test healthy; abaxial); raw test MAE/bias/accuracy/QWK for all three models; all fitted parameters and t-based CIs; test MAE, bias, grade accuracy, over/under, McNemar p, QWK and healthy ≥ 0.5% counts (abaxial and either face) for every fit, including the published coefficients and the in-sample test fit; validation raw bias +0.57, MAE 0.58, slopes 1.29/1.22; Wilson [0.50, 0.86] for 15/21. Paired ΔMAE bootstrap CIs agree within Monte Carlo error (e.g., SegFormer-B5 diseased-only fit: verifier [−0.85, −0.04] vs [−0.84, −0.04]; with healthy leaves [−1.14, −0.16] vs [−1.13, −0.15]); the 2-decimal values in this file are kept.
- Sources re-opened: `03a_results.tex` l. 68, `04_discussion.tex` l. 13 and 41, `02_methods.tex` l. 42, 75 and 91, `note_methods.tex` l. 22, 23 and 32, `supplementary.tex` l. 87 (all quoted content confirmed); a fresh grep of `C:/dev` for `1.1915`/`0.4329` finds only manuscript copies, this folder and a coincidental training-log loss value (`loss=0.4329`), and `C:/dev/coffee-app/backend/infer.py` uses raw argmax with no rescaling; the form export (columns 1–6 only) confirms the four names, roles, institutions and years, and that the years column is 'Years of experience in plant pathology'.

**Corrections made in this file (wording only; no number in results.json changed):**
1. Leaf `20260621_Mcitricolor_GX_076` has reference severity 0.545%, not 0.55%, so it lies 0.045 pp (not 0.05 pp) above the 0.5% cut-point, and its two under-grading predictions are 0.490% and 0.496% (previously '0.49–0.50%', which read as if one prediction were at the cut-point). Changed in §1.2, §1.4, §1.7 item 3, the §1.9 LaTeX paragraph and the §1.9 wording note: '0.05–0.20 pp' → '0.045–0.20 pp'.
2. §1.9 LaTeX: 'every error was an under-grade, because the six G0--G1 leaves received the grade implied by their masks' overstated the explanation (the six leaves account for 6 of 6–9 visual-reference errors per method). Reworded to state that six of the 6–9 errors per method are those six leaves.
3. Note: `make_md.py` was not changed; re-running it would regenerate results.md without these corrections and without this section.

### Second independent verification (28 Sep 2026)

Script: `verifier2_check.py` → `verifier2_check.json`, written from scratch (imports nothing from `grades_analysis.py`, `rescale_analysis.py`, `verifier_check.py` or `results/paired`). QWK uses the pairwise squared-difference form (observed / expected under independence); Tango intervals use the closed-form constrained MLE on a 1e-5 grid; OLS uses `numpy.polyfit` with t intervals.

**Re-derived and confirmed:** leaf sets (55 leaf-spot test leaves, both faces for every leaf and input, 0 overlap with the 55 validation leaves; rust 21/21/21/21 with 0 val–test overlap); leaf-spot severity = mean of the two faces (the baseline file's own SegFormer column equals the collapsed per-face mean to 1e-15); cut-points 0.5/5/20; reference counts, 47/55 agreement and QWK 0.831 [0.721, 0.923]; for all 10 methods and both references the correct counts, Wilson intervals, QWK and its bootstrap interval (seed 20260928, identical to 3 decimals), over/under counts, per-grade recall; the four mask-reference error leaves (0.545, 0.601, 4.799, 4.920%; 0.045–0.20 pp from a cut-point; the 4.92% leaf wrong for all 10 methods); colour threshold = published UNet++ on 55/55 leaves; seed means ± SD (ddof 1); all 24 paired rows (b/c, exact p, Holm p = 1.00 throughout, Tango intervals to 3 decimals); all part-2 fits, test MAE/bias/accuracy/QWK/healthy counts and McNemar p. Paired ΔMAE bootstrap intervals differ from this file by ≤ 0.04 pp only because `rescale_analysis.py` draws its test index matrix third from the seeded generator (documented, Monte Carlo only). `results/paired` (mask-derived grades) and `pass3/seed_summary.json` agree. All ten cited manuscript/Supplement lines were re-opened and their quoted content confirmed; a fresh grep of code files under `C:/dev` for the coefficients, `s_cal`, `rescal` and `calibrat` finds no code that fits or applies the rescaling; the form export (columns 2–7 of the CSV: name, specialty, role, other, institution, years; timestamps and grades not read) matches the §3 table.

**Corrections made in this pass (wording only; no number changed):**
4. "Indistinguishable" / "differs in per-leaf grade correctness" overstated a non-significant result as equivalence (no TOST was run for grade accuracy here). Key numbers (§0), §1.7 item 1 and the §1.9 LaTeX now say "no detectable difference" / "differed detectably", and §0 and §1.7 note that the Tango intervals reach −0.12 to +0.15.
5. §2.5 item 3 and the §2.6 Discussion replacement said the validation recalibration "changed the grade of only one of 21 leaves". For SegFormer-B5 (fit on diseased + healthy validation leaves) it changes the grade of 3 test leaves (two over-grades corrected, one new under-grade: `GX_021`, 5.23% reference, 5.76 → 4.71%), a net gain of one (19 → 20/21). Reworded to "net gain of one" / "raised grade accuracy by only one leaf", and the §2.6 sentence now states that the fit used the diseased and healthy validation leaves (the diseased-only fit gives MAE 0.59, not 0.40).
6. Note: `make_md.py` does not contain corrections 1–6; re-running it would drop them.
