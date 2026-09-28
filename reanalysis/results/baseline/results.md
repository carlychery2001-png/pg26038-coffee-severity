# Training-free color-threshold baseline (PG-26038 major revision)

Created 2026-09-27 23:45 by `baseline.py` (numbers), `make_figures.py` (figures) and `make_md.py` (this file). Python 3.13.15, numpy 2.5.3, scipy 1.18.1. All numbers are in `results.json`. Section 10 lists what changed from the previous version of this folder.

## 1. Reproduction of the published network values (done first)

All published values reproduced from `per_image.csv` at the published precision: **yes**. The rebuilt reference labels also match `per_image.csv` pixel for pixel (388 images: rebuilt reference leaf and lesion pixel counts equal per_image.csv exactly).

| Disease | Model | n | Leaf IoU | MAE | Bias | QWK | Bland–Altman LoA | Match |
|---|---|---|---|---|---|---|---|---|
| Coffee leaf rust | SegFormer-B5 | 21 | 0.767 | 0.99 | +0.99 | 0.88 | −1.68 to +3.66 | all |
| Coffee leaf rust | UNet++ | 21 | 0.647 | 1.77 | +1.75 | 0.79 | −2.68 to +6.17 | all |
| Coffee leaf rust | DeepLabV3+ | 21 | 0.632 | 1.47 | +1.41 | 0.88 | −3.07 to +5.89 | all |
| American leaf spot | SegFormer-B5 | 55 | 0.884 | 0.22 | +0.16 | 0.82 | −0.47 to +0.79 | all |
| American leaf spot | UNet++ | 55 | 0.853 | 0.15 | −0.03 | 0.85 | −0.50 to +0.44 | all |
| American leaf spot | DeepLabV3+ | 55 | 0.815 | 0.25 | +0.08 | 0.84 | −0.66 to +0.82 | all |

The healthy-leaf counts of revision brief item 10 are also reproduced (rust networks, 21 local healthy test leaves, maximum over both faces): any predicted lesion pixel SegFormer-B5 19/21, UNet++ 19/21, DeepLabV3+ 13/21; maximum predicted severity 0.38%, 0.27%, 0.09%.

## 2. Baseline definition

- **Space and leaf mask.** The indices are thresholded inside the **reference leaf mask** (alpha > 8), in the same 1024 × 1024 letterbox space (`run_inference.py load()`: RGB bilinear, labels nearest) in which the networks were evaluated. The networks do not use this mask: they segment the leaf themselves, and their severity denominator is their predicted leaf. The baseline's severity denominator is therefore the exact reference leaf area, which favors it slightly on severity.
- **Indices** (lesion = index above threshold; lesions are redder/browner than green tissue for both diseases): (i) redness index (r − g)/(r + g + b); (ii) CIELAB a* (sRGB, D65); (iii) excess red 1.4r′ − g′ on chromatic coordinates.
- **Threshold rules:** (a) **per-leaf** Otsu: one threshold per leaf from the pooled leaf pixels of the faces that the leaf rule uses (rust: abaxial; leaf spot: both faces), 256 bins over the pooled range (as `skimage.filters.threshold_otsu`); a per-face variant is reported as a secondary check. (b) One global threshold per disease, grid search on the **validation diseased leaves** to maximize mean leaf lesion IoU (grids: ri −0.3 to 0.6 step 0.001, astar −40 to 60 step 0.1, exr −0.6 to 1.2 step 0.002).
- **Tie handling.** The letterboxed RGB values are 8-bit integers, so the redness index and excess red are ratios of integers and many pixels sit exactly on a grid threshold. Global thresholds for these two indices are therefore compared in exact integer arithmetic, and pixels exactly on the threshold count as lesion (e.g. excess red ≥ 0.12 ⇔ 32r − 28g − 3b ≥ 0). Ties at the selected thresholds: rust (exr ≥ 0.12): 1791 leaf pixels exactly on the threshold over 168 validation and test images (10.4 per abaxial and 10.9 per adaxial image on average), 851 of which a float64 evaluation of the index put below the threshold; leaf spot (ri ≥ 0.026): 1 leaf pixel exactly on the threshold over 220 validation and test images, 0 of them put below the threshold by a float64 evaluation. CIELAB a* is compared in float64.
- **No post-processing** (no smoothing, morphology or speck removal), matching the raw-argmax evaluation of the networks.
- **Selection:** per disease, the configuration with the highest validation mean leaf IoU; evaluated **once** on test with the paper's leaf-level rules (rust: abaxial face, reference grade from reference severity with cut-points 0.5/5/20%; leaf spot: mean of the two faces, reference grade = annotator's visual grade).
- **Healthy leaves: two face rules, both reported.** *Either face* (primary; the rule of revision brief item 10): a healthy leaf is flagged when either face meets the criterion (leaf value = maximum over the two faces). *Leaf rule* (secondary): the paper's diseased-leaf rule of the detector's disease (rust: abaxial face only; leaf spot: mean of the two faces). Flag criteria: any lesion pixel (> 0%), ≥ 0.5% (predicted G1 or higher) and ≥ 5%.
- **Uncertainty:** 10000 resamples of leaves, percentile 95% CI, numpy default_rng(seed=20260927) re-seeded per disease; one index matrix per disease shared by all methods (paired differences); undefined resamples dropped and counted. Grade accuracy also has a Wilson interval; healthy-leaf proportions use Wilson intervals.

### Validation selection (mean leaf IoU on the validation diseased leaves)

| Disease | Index | Rule | Threshold | Validation IoU | Healthy validation leaves ≥ 0.5%: either face (leaf rule) |
|---|---|---|---|---|---|
| Coffee leaf rust | redness index (r − g)/(r + g + b) | per-leaf Otsu | per leaf | 0.266 |  |
| Coffee leaf rust | redness index (r − g)/(r + g + b) | global (validation-tuned) | −0.021 | 0.578 | 21/21 (20/21) |
| Coffee leaf rust | CIELAB a* | per-leaf Otsu | per leaf | 0.316 |  |
| Coffee leaf rust | CIELAB a* | global (validation-tuned) | −7.6 | 0.474 | 21/21 (21/21) |
| Coffee leaf rust | excess red 1.4r′ − g′ | per-leaf Otsu | per leaf | 0.412 |  |
| Coffee leaf rust | excess red 1.4r′ − g′ | global (validation-tuned) **(selected)** | 0.12 | 0.609 | 20/21 (18/21) |
| American leaf spot | redness index (r − g)/(r + g + b) | per-leaf Otsu | per leaf | 0.552 |  |
| American leaf spot | redness index (r − g)/(r + g + b) | global (validation-tuned) **(selected)** | 0.026 | 0.789 | 0/21 (0/21) |
| American leaf spot | CIELAB a* | per-leaf Otsu | per leaf | 0.519 |  |
| American leaf spot | CIELAB a* | global (validation-tuned) | 0 | 0.780 | 0/21 (0/21) |
| American leaf spot | excess red 1.4r′ − g′ | per-leaf Otsu | per leaf | 0.562 |  |
| American leaf spot | excess red 1.4r′ − g′ | global (validation-tuned) | 0.176 | 0.783 | 0/21 (0/21) |

Selected: rust = excess red 1.4r′ − g′ ≥ 0.12 (validation IoU plateau within 0.005 of the maximum: 0.118 to 0.124); American leaf spot = redness index (r − g)/(r + g + b) ≥ 0.026 (plateau 0.018 to 0.032). Otsu was the weakest rule for every index: it always splits the leaf into two classes, so on leaves with few lesions it labels a large share of healthy tissue as lesion. Per-face Otsu (secondary) gave validation IoU rust ri 0.266, astar 0.316, exr 0.412; leaf spot ri 0.648, astar 0.616, exr 0.655 (identical to per-leaf Otsu for rust, where only the abaxial face is used).

## 3. Test-set comparison (selected baseline vs the three networks)

Diseased test leaves: rust n = 21, American leaf spot n = 55. Healthy leaves: the 21 healthy rust-pack TEST leaves available locally (EARTH site; **the paper used 54, the other 33 are not in the local pack**) and the 21 healthy VALIDATION leaves (EARTH). Healthy columns: leaves with predicted severity ≥ 0.5% (predicted G1 or higher) on **either face** (primary; revision brief item 10), with the count under the paper's leaf rule (rust: abaxial; leaf spot: mean of faces) in parentheses. Brackets: 95% CI.

| Disease | Method | Leaf IoU | MAE (pp) | Bias (pp) | Pearson r | CCC | Grade accuracy | QWK | Healthy test ≥ 0.5% | Healthy val ≥ 0.5% |
|---|---|---|---|---|---|---|---|---|---|---|
| Coffee leaf rust | **Color threshold** (exr ≥ 0.12) | 0.600 [0.554–0.645] | 1.02 [0.44–1.85] | +0.17 [−0.55 to 1.11] | 0.922 | 0.921 [0.747–0.986] | 20/21 = 0.952 | 0.93 [0.71–1.00] | 21/21 (20/21) | 20/21 (18/21) |
| Coffee leaf rust | SegFormer-B5 | 0.767 [0.743–0.790] | 0.99 [0.48–1.60] | +0.99 [0.48–1.60] | 0.994 | 0.958 [0.910–0.979] | 19/21 = 0.905 | 0.88 [0.68–1.00] | 0/21 (0/21) | 0/21 (0/21) |
| Coffee leaf rust | UNet++ | 0.647 [0.607–0.684] | 1.77 [0.92–2.76] | +1.75 [0.88–2.74] | 0.979 | 0.897 [0.786–0.949] | 17/21 = 0.810 | 0.79 [0.57–0.94] | 0/21 (0/21) | 0/21 (0/21) |
| Coffee leaf rust | DeepLabV3+ | 0.632 [0.590–0.669] | 1.47 [0.65–2.48] | +1.41 [0.57–2.44] | 0.954 | 0.900 [0.731–0.977] | 19/21 = 0.905 | 0.88 [0.68–1.00] | 0/21 (0/21) | 0/21 (0/21) |
| American leaf spot | **Color threshold** (ri ≥ 0.026) | 0.807 [0.777–0.836] | 0.23 [0.15–0.32] | −0.11 [−0.22 to −0.01] | 0.997 | 0.997 [0.992–0.999] | 48/55 = 0.873 | 0.85 [0.75–0.94] | 0/21 (0/21) | 0/21 (0/21) |
| American leaf spot | SegFormer-B5 | 0.884 [0.868–0.900] | 0.22 [0.15–0.30] | +0.16 [0.08–0.25] | 0.999 | 0.998 [0.996–0.999] | 46/55 = 0.836 | 0.82 [0.72–0.91] | 0/21 (0/21) | not run |
| American leaf spot | UNet++ | 0.853 [0.830–0.874] | 0.15 [0.10–0.20] | −0.03 [−0.09 to 0.03] | 0.999 | 0.999 [0.997–1.000] | 48/55 = 0.873 | 0.85 [0.75–0.94] | 1/21 (0/21) | not run |
| American leaf spot | DeepLabV3+ | 0.815 [0.790–0.837] | 0.25 [0.18–0.33] | +0.08 [−0.02 to 0.18] | 0.998 | 0.998 [0.996–0.999] | 47/55 = 0.855 | 0.84 [0.74–0.93] | 0/21 (0/21) | not run |

Bland–Altman limits of agreement of the baseline (bias ± 1.96 SD): rust −3.79 to +4.14 pp; American leaf spot −0.88 to +0.66 pp (networks: rust −1.68 to +3.66, −2.68 to +6.17, −3.07 to +5.89; leaf spot −0.47 to +0.79, −0.50 to +0.44, −0.66 to +0.82, for SegFormer-B5, UNet++, DeepLabV3+). 'not run' = no network predictions exist for the healthy validation leaves with the leaf-spot models (inference was not re-run).

### Paired differences, network minus baseline (same bootstrap resamples; 95% CI)

| Disease | Network | Δ leaf IoU | Leaves with higher network / baseline IoU | Δ MAE (pp) | Δ |bias| (pp) | Δ CCC | Δ QWK |
|---|---|---|---|---|---|---|---|
| Coffee leaf rust | SegFormer-B5 | +0.167 [0.127–0.206]* | 21 / 0 of 21 | −0.03 [−0.84 to 0.59] | +0.82 [−0.25 to 1.39] | +0.037 [−0.043 to 0.204] | −0.05 [−0.27 to 0.20] |
| Coffee leaf rust | UNet++ | +0.047 [0.007–0.085]* | 12 / 9 of 21 | +0.75 [−0.01 to 1.64] | +1.58 [0.48–2.42]* | −0.024 [−0.132 to 0.092] | −0.14 [−0.38 to 0.11] |
| Coffee leaf rust | DeepLabV3+ | +0.032 [−0.002 to 0.066] | 12 / 9 of 21 | +0.45 [−0.13 to 1.22] | +1.24 [0.25–1.94]* | −0.021 [−0.099 to 0.023] | −0.05 [−0.27 to 0.20] |
| American leaf spot | SegFormer-B5 | +0.077 [0.062–0.094]* | 55 / 0 of 55 | −0.01 [−0.09 to 0.07] | +0.06 [−0.09 to 0.19] | +0.001 [−0.001 to 0.005] | −0.03 [−0.07 to 0.00] |
| American leaf spot | UNet++ | +0.046 [0.036–0.056]* | 53 / 2 of 55 | −0.08 [−0.14 to −0.03]* | −0.08 [−0.15 to 0.01] | +0.002 [0.000–0.006]* | +0.00 [0.00–0.00] |
| American leaf spot | DeepLabV3+ | +0.008 [−0.002 to 0.017] | 24 / 31 of 55 | +0.02 [−0.05 to 0.09] | −0.03 [−0.19 to 0.15] | +0.000 [−0.001 to 0.004] | −0.01 [−0.05 to 0.00] |

\* = 95% CI excludes zero. Negative Δ MAE or Δ |bias| = the network is closer to the reference. A QWK CI of exactly [0, 0] means the two methods assigned the same grade to every test leaf. Coffee leaf rust: the IoU difference is clearly positive for SegFormer-B5 and UNet++; for DeepLabV3+ the CI includes zero (DeepLabV3+ +0.032 [−0.002 to 0.066], network higher on 12/21 and baseline higher on 9/21 leaves), so the threshold is not distinguishable from it in IoU. American leaf spot: the IoU difference is clearly positive for SegFormer-B5 and UNet++; for DeepLabV3+ the CI includes zero (DeepLabV3+ +0.008 [−0.002 to 0.017], network higher on 24/55 and baseline higher on 31/55 leaves), so the threshold is not distinguishable from it in IoU.

### Leaf IoU by reference grade (descriptive)

| Disease | Grade | n | Baseline | SegFormer-B5 | UNet++ | DeepLabV3+ |
|---|---|---|---|---|---|---|
| Coffee leaf rust | G0 | 1 | 0.478 | 0.745 | 0.714 | 0.589 |
| Coffee leaf rust | G1 | 11 | 0.568 | 0.776 | 0.636 | 0.625 |
| Coffee leaf rust | G2 | 9 | 0.652 | 0.759 | 0.652 | 0.645 |
| American leaf spot | G1 | 30 | 0.753 | 0.858 | 0.814 | 0.776 |
| American leaf spot | G2 | 23 | 0.867 | 0.913 | 0.896 | 0.857 |
| American leaf spot | G3 | 2 | 0.935 | 0.953 | 0.948 | 0.909 |

Baseline grade confusion (rows reference G0–G3, columns predicted G0–G3): rust [[1, 0, 0, 0], [0, 11, 0, 0], [0, 1, 8, 0], [0, 0, 0, 0]]; American leaf spot [[0, 0, 0, 0], [6, 24, 0, 0], [0, 1, 22, 0], [0, 0, 0, 2]].

## 4. Specificity on healthy leaves (both face rules, all flag criteria)

Either face = primary (revision brief item 10: maximum over the two faces). Leaf rule = the paper's diseased-leaf rule of the detector's disease (rust: abaxial face only; leaf spot: mean of faces).

| Split (n) | Detector | Either face: any pixel | Either face: ≥ 0.5% (G1+) [Wilson 95% CI of FP rate] | Either face: ≥ 5% | Either face: median / max severity (%) | Leaf rule: any pixel | Leaf rule: ≥ 0.5% |
|---|---|---|---|---|---|---|---|
| test (21) | Coffee leaf rust Color threshold | 21/21 | 21/21 [84.5–100.0%] | 0/21 | 1.173 / 4.908 | 21/21 | 20/21 |
| test (21) | Coffee leaf rust Color threshold, specificity-constrained (secondary) | 21/21 | 0/21 [0.0–15.5%] | 0/21 | 0.171 / 0.478 | 21/21 | 0/21 |
| test (21) | Coffee leaf rust SegFormer-B5 | 19/21 | 0/21 [0.0–15.5%] | 0/21 | 0.143 / 0.382 | 17/21 | 0/21 |
| test (21) | Coffee leaf rust UNet++ | 19/21 | 0/21 [0.0–15.5%] | 0/21 | 0.011 / 0.267 | 15/21 | 0/21 |
| test (21) | Coffee leaf rust DeepLabV3+ | 13/21 | 0/21 [0.0–15.5%] | 0/21 | 0.016 / 0.090 | 7/21 | 0/21 |
| test (21) | American leaf spot Color threshold | 21/21 | 0/21 [0.0–15.5%] | 0/21 | 0.118 / 0.326 | 21/21 | 0/21 |
| test (21) | American leaf spot SegFormer-B5 | 19/21 | 0/21 [0.0–15.5%] | 0/21 | 0.062 / 0.226 | 19/21 | 0/21 |
| test (21) | American leaf spot UNet++ | 21/21 | 1/21 [0.8–22.7%] | 0/21 | 0.104 / 0.543 | 21/21 | 0/21 |
| test (21) | American leaf spot DeepLabV3+ | 21/21 | 0/21 [0.0–15.5%] | 0/21 | 0.101 / 0.409 | 21/21 | 0/21 |
| val (21) | Coffee leaf rust Color threshold | 21/21 | 20/21 [77.3–99.2%] | 1/21 | 0.907 / 6.269 | 21/21 | 18/21 |
| val (21) | Coffee leaf rust Color threshold, specificity-constrained (secondary) | 21/21 | 0/21 [0.0–15.5%] | 0/21 | 0.139 / 0.452 | 21/21 | 0/21 |
| val (21) | Coffee leaf rust SegFormer-B5 | 21/21 | 0/21 [0.0–15.5%] | 0/21 | 0.163 / 0.392 | 19/21 | 0/21 |
| val (21) | Coffee leaf rust UNet++ | 21/21 | 0/21 [0.0–15.5%] | 0/21 | 0.037 / 0.497 | 20/21 | 0/21 |
| val (21) | Coffee leaf rust DeepLabV3+ | 14/21 | 0/21 [0.0–15.5%] | 0/21 | 0.010 / 0.058 | 13/21 | 0/21 |
| val (21) | American leaf spot Color threshold | 21/21 | 0/21 [0.0–15.5%] | 0/21 | 0.103 / 0.217 | 21/21 | 0/21 |

Where the baseline's lesion pixels fall on the 21 healthy test leaves: 65% (rust detector) and 84% (leaf-spot detector) of the flagged pixels on both faces lie in the 3-px leaf-margin band, which is 4.6% of the leaf area (rust detector, abaxial faces only: 58%). The location of the remaining pixels was not quantified.

**Specificity definition (consistent with revision brief item 10; not a new finding of this analysis).** The brief already established two points: the published raw-argmax models predict a few lesion pixels on most healthy leaves, so the manuscript's statement that the zero-false-positive models left every healthy leaf without any predicted lesion is wrong; and the published 0/54 and 1/54 counts cannot mean 'any lesion pixel' (`% TODO(author)`: confirm the threshold used). This file uses the brief's either-face rule as the primary healthy-leaf rule for every detector: rust networks mark lesion pixels on 19/21, 19/21, 13/21 local healthy test leaves and leaf-spot networks on 19/21, 21/21, 21/21 (SegFormer-B5, UNet++, DeepLabV3+). No rust network reaches 0.5% on either face. Among the leaf-spot networks, UNet++ reaches 0.5% on one face of 1/21 leaves (maximum 0.54%; 0/21 under the mean-of-faces rule); the other leaf-spot networks stay below 0.5% on both faces. **Face rule matters for the counts:** under the paper's diseased-leaf rule (rust abaxial face only) the rust networks' any-pixel counts are 17/21, 15/21, 7/21, and the rust color threshold reaches ≥ 0.5% on 20/21 test and 18/21 validation leaves instead of 21/21 and 20/21. Any count quoted next to brief-based manuscript text must use the either-face rule, or state the face rule explicitly.

## 5. Secondary and sensitivity analyses (not the primary baseline)

### 5a. Specificity-constrained threshold (secondary; added after the primary test evaluation, threshold chosen on validation only)

- Coffee leaf rust: highest validation IoU with 0/21 healthy validation leaves ≥ 0.5% on either face = excess red 1.4r′ − g′ ≥ 0.154 (validation IoU 0.516). Test: IoU 0.518 [0.470–0.560], MAE 2.12 [1.42–2.96], bias −2.12 [−2.96 to −1.42], CCC 0.814, grade accuracy 15/21, QWK 0.64 [0.36–0.85]; healthy test ≥ 0.5% on either face: 0/21 (leaf rule 0/21). SegFormer-B5 minus constrained baseline: Δ IoU +0.250 [0.205–0.292], Δ MAE −1.13 [−1.61 to −0.66].
- American leaf spot: the IoU-optimal threshold already leaves all 21 healthy validation leaves below 0.5% on both faces, so the constrained threshold is identical to the primary one.

### 5b. POST HOC: leaf-margin band excluded from the lesion prediction

POST HOC (added after seeing where the primary baseline's healthy-leaf false positives fell); secondary only. Band = reference leaf minus its binary erosion (3x3 cross, scipy default) by 3 or 12 letterbox px; lesion pixels predicted inside the band are removed; severity denominator unchanged. 12 px is the leaf erosion of the deployed application at a 1024-px canvas. Threshold re-tuned on validation for each band. Healthy columns: either face (leaf rule).

| Disease | Band | Index ≥ threshold | Validation IoU | Test IoU [95% CI] | MAE | Bias | QWK | Healthy test ≥ 0.5% | Healthy val ≥ 0.5% |
|---|---|---|---|---|---|---|---|---|---|
| Coffee leaf rust | 3 px | exr ≥ 0.116 | 0.662 | 0.633 [0.590–0.670] | 1.04 | +0.10 | 0.87 | 15/21 (14/21) | 13/21 (13/21) |
| Coffee leaf rust | 12 px | exr ≥ 0.116 | 0.578 | 0.547 [0.500–0.590] | 1.40 | −0.62 | 0.80 | 11/21 (10/21) | 11/21 (11/21) |
| American leaf spot | 3 px | astar ≥ −1.4 | 0.789 | 0.791 [0.751–0.827] | 0.32 | −0.26 | 0.85 | 0/21 (0/21) | 0/21 (0/21) |
| American leaf spot | 12 px | astar ≥ −1.4 | 0.723 | 0.733 [0.692–0.771] | 0.88 | −0.84 | 0.73 | 0/21 (0/21) | 0/21 (0/21) |

### 5c. All configurations on test (transparency only; selection used validation alone)

| Disease | Configuration | Validation IoU | Test IoU | Test MAE | Test bias | Test QWK | Healthy test ≥ 0.5%: either face (leaf rule) | Healthy test median severity (%): either face (leaf rule) |
|---|---|---|---|---|---|---|---|---|
| Coffee leaf rust | ri_otsu | 0.266 | 0.350 | 31.84 | +29.94 | −0.29 | 21/21 (21/21) | 83.22 (66.30) |
| Coffee leaf rust | ri_global | 0.578 | 0.579 | 0.89 | −0.12 | 0.93 | 21/21 (21/21) | 1.40 (0.92) |
| Coffee leaf rust | ri_otsu_face | 0.266 | 0.350 | 31.84 | +29.94 | −0.29 | 21/21 (21/21) | 69.58 (66.30) |
| Coffee leaf rust | astar_otsu | 0.316 | 0.352 | 21.93 | +18.97 | −0.25 | 21/21 (21/21) | 87.82 (38.27) |
| Coffee leaf rust | astar_global | 0.474 | 0.487 | 1.11 | −0.36 | 0.59 | 21/21 (21/21) | 57.15 (2.02) |
| Coffee leaf rust | astar_otsu_face | 0.316 | 0.352 | 21.93 | +18.97 | −0.25 | 21/21 (21/21) | 60.53 (38.27) |
| Coffee leaf rust | exr_otsu | 0.412 | 0.433 | 23.10 | +20.79 | −0.21 | 21/21 (21/21) | 88.13 (72.76) |
| Coffee leaf rust | exr_global | 0.609 | 0.600 | 1.02 | +0.17 | 0.93 | 21/21 (20/21) | 1.17 (0.96) |
| Coffee leaf rust | exr_otsu_face | 0.412 | 0.433 | 23.10 | +20.79 | −0.21 | 21/21 (21/21) | 72.76 (72.76) |
| American leaf spot | ri_otsu | 0.552 | 0.623 | 14.85 | +14.71 | 0.08 | 21/21 (21/21) | 83.22 (64.47) |
| American leaf spot | ri_global | 0.789 | 0.807 | 0.23 | −0.11 | 0.85 | 0/21 (0/21) | 0.12 (0.11) |
| American leaf spot | ri_otsu_face | 0.648 | 0.650 | 12.11 | +11.96 | 0.03 | 21/21 (21/21) | 69.58 (60.99) |
| American leaf spot | astar_otsu | 0.519 | 0.543 | 18.83 | +18.76 | −0.13 | 21/21 (21/21) | 87.82 (47.12) |
| American leaf spot | astar_global | 0.780 | 0.795 | 0.28 | −0.24 | 0.84 | 0/21 (0/21) | 0.21 (0.18) |
| American leaf spot | astar_otsu_face | 0.616 | 0.612 | 14.40 | +14.34 | −0.02 | 21/21 (21/21) | 60.53 (52.49) |
| American leaf spot | exr_otsu | 0.562 | 0.659 | 12.08 | +11.97 | 0.18 | 21/21 (21/21) | 88.13 (66.53) |
| American leaf spot | exr_global | 0.783 | 0.801 | 0.23 | −0.08 | 0.85 | 0/21 (0/21) | 0.11 (0.11) |
| American leaf spot | exr_otsu_face | 0.655 | 0.680 | 9.41 | +9.24 | 0.13 | 21/21 (21/21) | 72.76 (61.28) |

`otsu` = per-leaf Otsu (the specified rule); `otsu_face` = per-face Otsu (secondary). The either-face healthy values of the Otsu rows are identical for the two detectors because Otsu has no disease-specific parameter and, under the either-face rule, is applied to both faces for both detectors. **Abaxial tuning caveat:** the rust thresholds were tuned on abaxial faces (the rust leaf rule), so the either-face rule also applies them to adaxial faces, which they were not tuned for; this raises the rust detectors' healthy-leaf values, markedly for a* (a* ≥ −7.6: median 57.15% on either face vs 2.02% abaxial) and modestly for the selected excess-red threshold (1.17% vs 0.96%).

## 6. Where the networks add value (2–3 sentences)

The networks' advantage is in lesion delineation and, for rust, in keeping healthy leaves below the G1 cut-point. SegFormer-B5 and UNet++ had a higher leaf IoU than the validation-tuned color threshold in both diseases, with paired 95% CIs excluding zero (rust +0.167 and +0.047; leaf spot +0.077 and +0.046; SegFormer-B5 higher on 21/21 and 55/55 leaves), whereas DeepLabV3+ was not distinguishable from it (rust +0.032 [−0.002 to 0.066], higher on 12/21 leaves; leaf spot +0.008 [−0.002 to 0.017], with the threshold higher on 31/55 leaves); on the 21 local healthy test leaves no rust network reached 0.5% predicted severity on either face, while the IoU-tuned rust threshold did on 21/21 (20/21 on the abaxial face alone), and no color-threshold variant evaluated combined convolutional-model-level rust IoU with zero such leaves (specificity-constrained: IoU 0.518, 0/21; margin-excluded, post hoc: IoU 0.633, 15/21). For severity and grade alone the tuned threshold was competitive (rust MAE 1.02 vs 0.99–1.77 pp, QWK 0.93 vs 0.79–0.88; leaf spot MAE 0.23 vs 0.15–0.25 pp, QWK 0.85 vs 0.82–0.85), which is partly expected because the rust reference masks were seeded by a color detector before SAM 2 refinement and because the baseline's severity denominator is the exact reference leaf area.

## 7. Suggested manuscript sentences (LaTeX-ready, precise, non-overclaiming)

1. *Methods (Section \ref{sec:evaluation}).* As a training-free reference, three per-pixel color indices (the redness index $(r-g)/(r+g+b)$, CIELAB $a^*$ and excess red $1.4r'-g'$ on chromatic coordinates) were thresholded inside the reference leaf mask (alpha $>$ 8), in the same 1024 $\times$ 1024 letterbox space used to evaluate the networks, with either a per-leaf Otsu threshold or one global threshold per disease chosen on the validation diseased leaves to maximize mean leaf lesion IoU; the configuration with the highest validation IoU (rust: excess red $\geq$ 0.12; American leaf spot: redness index $\geq$ 0.026) was applied once to the test leaves, without post-processing, and scored with the same leaf-level rules and bootstrap procedure. Because it used the reference leaf mask, the baseline's severity denominator was the exact leaf area, whereas the networks segment the leaf themselves.

2. *Results.* The color threshold reached a test leaf IoU of 0.600 (95\% CI 0.554--0.645) for rust and 0.807 (0.777--0.836) for American leaf spot; SegFormer-B5 and UNet++ exceeded it for both diseases (paired differences +0.167 and +0.047 for rust and +0.077 and +0.046 for American leaf spot, all 95\% CIs excluding zero), whereas DeepLabV3+ was not distinguishable from it (rust +0.032, 95\% CI $-$0.002 to 0.066; American leaf spot +0.008, $-$0.002 to 0.017) (Table~X).

3. *Results.* Its severity and grade agreement were similar to those of the networks (rust: MAE 1.02 vs 0.99--1.77 percentage points, QWK 0.93 vs 0.79--0.88; American leaf spot: MAE 0.23 vs 0.15--0.25, QWK 0.85 vs 0.82--0.85), with paired bootstrap intervals for the MAE and QWK differences that included zero in every comparison but one (American leaf spot UNet++, MAE lower by 0.08 percentage points), although with 21 rust test leaves these intervals are wide and do not establish equivalence.

4. *Results.* On the 21 healthy rust-pack test leaves available locally (the full test partition held 54), the rust color threshold predicted a severity of at least 0.5\% (grade G1 or higher) on at least one face of all 21 leaves (median of the per-leaf maximum 1.17\%), whereas none of the three rust networks did on either face; in a secondary analysis, raising the threshold on validation until no healthy validation leaf reached 0.5\% on either face removed these flags (0/21) but lowered rust IoU to 0.518 with a severity bias of $-$2.12 percentage points, while the leaf-spot color threshold kept all 21 healthy leaves below 0.5\% on both faces (0/21).

5. *Discussion/Limitations.* The networks' advantage over a validation-tuned color threshold thus lies mainly in lesion delineation, clearly so for SegFormer-B5 and UNet++, and, for rust, in keeping healthy leaves below the G1 cut-point, rather than in severity or grade agreement; the comparison favors the color baseline, because the rust reference masks were seeded by a color detector before SAM~2 refinement and the baseline used the reference leaf mask, and the rust healthy-leaf contrast is confounded with site (EARTH vs Sarch\'i).

Face rule for sentence 4: counts use the either-face rule of revision brief item 10 (a leaf is flagged if either face reaches the criterion), so they can sit next to the brief-based specificity text; under the abaxial-only rule the rust threshold count is 20/21.

## 8. Suggested table and figure captions

```latex
% Table: training-free color threshold vs networks (test leaves). Healthy = 21 local healthy test leaves (paper: 54); >= 0.5% predicted severity on either face.
\begin{tabular}{llccccccc}
\toprule
Disease & Method & Leaf IoU & MAE (pp) & Bias (pp) & CCC & Grade acc. & QWK & Healthy $\geq$G1 \\
\midrule
Coffee leaf rust & Color threshold & 0.600 (0.554--0.645) & 1.02 (0.44--1.85) & +0.17 & 0.921 & 20/21 & 0.93 (0.71--1.00) & 21/21 \\
 & SegFormer-B5 & 0.767 (0.743--0.790) & 0.99 (0.48--1.60) & +0.99 & 0.958 & 19/21 & 0.88 (0.68--1.00) & 0/21 \\
 & UNet++ & 0.647 (0.607--0.684) & 1.77 (0.92--2.76) & +1.75 & 0.897 & 17/21 & 0.79 (0.57--0.94) & 0/21 \\
 & DeepLabV3+ & 0.632 (0.590--0.669) & 1.47 (0.65--2.48) & +1.41 & 0.900 & 19/21 & 0.88 (0.68--1.00) & 0/21 \\
\midrule
American leaf spot & Color threshold & 0.807 (0.777--0.836) & 0.23 (0.15--0.32) & $-$0.11 & 0.997 & 48/55 & 0.85 (0.75--0.94) & 0/21 \\
 & SegFormer-B5 & 0.884 (0.868--0.900) & 0.22 (0.15--0.30) & +0.16 & 0.998 & 46/55 & 0.82 (0.72--0.91) & 0/21 \\
 & UNet++ & 0.853 (0.830--0.874) & 0.15 (0.10--0.20) & $-$0.03 & 0.999 & 48/55 & 0.85 (0.75--0.94) & 1/21 \\
 & DeepLabV3+ & 0.815 (0.790--0.837) & 0.25 (0.18--0.33) & +0.08 & 0.998 & 47/55 & 0.84 (0.74--0.93) & 0/21 \\
\bottomrule
\end{tabular}
```

**Figure (figure_baseline_comparison).** Training-free color threshold versus the three networks on the test leaves. Points are the leaf-level mean lesion IoU (top), severity MAE in percentage points (middle) and the percentage of the 21 healthy rust-pack test leaves available locally with a predicted severity of at least 0.5\% (grade G1 or higher) on either face (bottom). Bars are 95\% percentile bootstrap intervals over leaves (10,000 resamples) for IoU and MAE and 95\% Wilson intervals for the healthy-leaf rate. Filled grey diamonds: color threshold selected on validation (rust: excess red $\geq$ 0.12; American leaf spot: redness index $\geq$ 0.026). Open diamond (rust only): threshold raised on validation until no healthy validation leaf reached 0.5\% on either face (excess red $\geq$ 0.154); for American leaf spot this constraint did not change the threshold. Rust: abaxial face; American leaf spot: mean of both faces. Created by the authors.

**Figure (figure_baseline_overlays).** Abaxial faces of three test leaves per disease: the leaf with the median color-threshold IoU, the leaf with the largest absolute IoU difference between the color threshold and SegFormer-B5, and the healthy leaf with the median color-threshold predicted severity (paper's leaf rule). Top: scan with the reference lesion outline (magenta). Middle and bottom: color-threshold and SegFormer-B5 predictions on the grey leaf (yellow, overlap with the reference; pink, predicted only; blue, reference only). Healthy leaves: the inset enlarges (about 5--6$\times$) the boxed 32 $\times$ 32-pixel window that contains the most color-threshold lesion pixels on that face. Values are face-level IoU and predicted severity (the leaf-spot leaf-level values average both faces). Created by the authors.

## 9. Notes and caveats

- **Healthy coverage:** 21 of the paper's 54 healthy rust-test leaves are present locally; all healthy-leaf numbers here refer to those 21 (plus 21 validation leaves). All healthy leaves are from EARTH (collected 12, 15 and 22 June 2026); diseased rust leaves are from Sarchí, so rust healthy-vs-diseased contrasts are confounded with site and scan session.
- **Healthy-leaf face rule:** the primary rule is the brief's either-face rule (item 10); the paper's diseased-leaf rule (rust abaxial only) gives lower counts for the rust detectors (Section 4). Quote counts with their face rule. The rust color thresholds were tuned on abaxial faces only, so the either-face rule also applies them to adaxial faces (Section 5c).
- **Circularity (rust):** every rust reference mask was produced by a color/redness detector followed by SAM 2 box refinement, unedited. A color-threshold baseline therefore shares part of the reference-generation rule; its rust IoU, MAE and QWK are likely optimistic relative to an independent hand-drawn reference. Leaf-spot masks were detector-seeded but manually corrected in CVAT.
- **Tuning set:** thresholds were tuned on validation diseased leaves only (leaf IoU is undefined on healthy leaves). The rust CNNs, by contrast, were selected with a healthy-leaf false-positive penalty on validation.
- **Leaf mask:** the baseline used the reference leaf mask (exact denominator); the networks segment the leaf themselves. This favors the baseline slightly on severity. The manuscript sentences state this once in Methods (sentence 1) and once in Limitations (sentence 5).
- **Ties:** redness index and excess red thresholds are compared exactly in integer arithmetic (Section 2); a float64 evaluation of 1.4r/s − g/s puts some exact ties on either side of the threshold depending on rounding.
- **Otsu:** per-leaf Otsu (as specified) and per-face Otsu both failed for both diseases (Otsu always splits the leaf), with gross over-estimation of severity on test and on every healthy leaf (Section 5c).
- **Post-hoc analysis:** the margin-exclusion variant (5b) was added after seeing where the primary baseline's healthy-leaf false positives fell and is labeled as post hoc.

## 10. Changes from the previous version of this folder

1. **DeepLabV3+ wording (Sections 6 and 7).** The text no longer says that both convolutional models exceeded the threshold: only SegFormer-B5 and UNet++ have paired IoU CIs above zero; DeepLabV3+ is not distinguishable from the threshold (CIs include zero for both diseases).
2. **Healthy-leaf face rule reconciled with revision brief item 10.** Primary rule is now either face (maximum over both faces) for every detector; the paper's leaf rule is reported alongside. Rust color threshold ≥ 0.5%: 21/21 test and 20/21 validation (previously 20/21 and 18/21, abaxial). Rust networks, any pixel: 19/21, 19/21, 13/21 (previously 17/21, 15/21, 7/21, abaxial). The specificity note now cites the brief instead of presenting the any-pixel result as new. Under the either-face rule, leaf-spot UNet++ reaches 0.5% on one face of 1/21 healthy leaves (0/21 with the mean of faces, as before).
3. **Specificity-constrained threshold** re-chosen on validation under the either-face rule: the result is the same threshold as before (excess red ≥ 0.154), so its test results are unchanged (IoU 0.518, bias −2.12, 0/21 healthy test leaves ≥ 0.5% on either face).
4. **Sentence 5 / Section 6 wording:** 'rejecting healthy tissue' replaced by 'keeping healthy leaves below the G1 cut-point' (under the any-pixel rule the rust networks also mark lesion pixels on most healthy leaves).
5. **Methods sentence 1:** now says the indices were thresholded inside the reference leaf mask, in the same letterbox space, and states the exact-denominator advantage; sentence 5 repeats it once as a limitation.
6. **Exact tie handling** for the redness index and excess red (Section 2). Selected thresholds, validation IoUs of the global rules, test IoU, grade accuracy, QWK and the confusion matrices are unchanged. Values that changed at the reported precision (all rust; the leaf-spot threshold has a single tie pixel): baseline bias +0.17 [upper CI 1.11] (previously +0.16 [1.10]), upper LoA +4.14 (previously +4.12), Pearson r 0.922 (previously 0.923), CCC 0.921 [lower CI 0.747] (previously 0.922 [0.749]), MAE upper CI 1.85 (previously 1.84); paired Δ IoU UNet++ +0.047 (previously +0.046) and DeepLabV3+ +0.032 [−0.002 to 0.066] (previously +0.031 [upper 0.065]), with DeepLabV3+ higher on 12/21 and the threshold on 9/21 leaves (previously 13/21 and 8/21; the leaf that changed side, GX_009, is a near-tie: DeepLabV3+ 0.7584 vs threshold 0.7588, previously 0.7583); second-decimal shifts in the rust Δ MAE, Δ |bias| and Δ CCC cells of the paired table; baseline IoU of the single G0 rust leaf 0.478 (previously 0.480); post-hoc 12-px rust bias −0.62 (previously −0.63); rust healthy-test median / max severity under the abaxial rule 0.962 / 2.344 (previously 0.959 / 2.325).
7. **Per-leaf Otsu** (as specified) replaces per-face Otsu, which is kept as a secondary row; leaf-spot per-leaf Otsu validation IoU: ri 0.552, astar 0.519, exr 0.562 (per-face, previous version: ri 0.648, a* 0.616, exr 0.655). Otsu remains by far the weakest rule.
8. **Figures.** Comparison: marker edges now match the marker fill (no white edge cutting the CI bars), the color threshold is drawn in neutral dark grey (no longer close to SegFormer-B5 blue), and the bottom row uses the either-face rule. Overlays: the healthy-leaf panels carry a zoom inset so the sparse false-positive pixels are visible at print size.
9. **Margin-band location sentence:** now computed on both faces (65% rust detector, 84% leaf-spot detector; previously 58% rust abaxial-only and 84%); the unquantified 'cut petiole and midrib' claim and the '(overlay figure)' reference were removed.

## Files

- `baseline.py`
- `make_figures.py`
- `make_md.py`
- `results.json`
- `results.md`
- `per_leaf_test.csv`
- `per_leaf_healthy.csv`
- `validation_curves.csv`
- `overlay_examples.json`
- `figure_baseline_comparison.pdf`
- `figure_baseline_comparison.png`
- `figure_baseline_overlays.pdf`
- `figure_baseline_overlays.png`
