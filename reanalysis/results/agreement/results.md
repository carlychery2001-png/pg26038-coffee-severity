# Severity agreement reanalysis (Bland–Altman done properly)

PG-26038 major revision, 27 September 2026. Scripts: `pixel_metrics.py` (reference rebuilt in letterbox space, FP/FN split, boundary F1), `agreement.py` (all statistics → `results.json`), `make_figures.py`, `make_md.py`. No model inference was re-run; predictions are the saved class maps in `out/preds`.

**Scope.** Diseased test leaves at leaf level: rust = abaxial face of the 21 leaves; American leaf spot = mean of both faces of the 55 leaves. D = predicted − reference severity (percentage points, pp); A = mean of the two. Bootstrap = 10,000 resamples of leaves, percentile 95% CIs, seed 20260927 (paired across models within a disease). Rust reference grades are derived from the reference severity (cut-points 0.5/5/20%); leaf-spot reference grades are the annotator's visual grades.

## 1. Reproduction of the published values

All published values reproduce exactly (`reproduction_all_match` = true); all reference and predicted pixel counts rebuilt from the images/masks/prediction maps match `per_image.csv` for all 393 model × image pairs.

| Disease | Model | Leaf IoU | MAE | Bias | QWK | Constant LoA (published) | Reproduced |
|---|---|---|---|---|---|---|---|
| Coffee leaf rust | SegFormer-B5 | 0.767 | 0.99 | +0.99 | 0.88 | −1.68 to +3.66 | −1.68 to +3.66 ✓ |
| Coffee leaf rust | UNet++ | 0.647 | 1.77 | +1.75 | 0.79 | −2.68 to +6.17 | −2.68 to +6.17 ✓ |
| Coffee leaf rust | DeepLabV3+ | 0.632 | 1.47 | +1.41 | 0.88 | −3.07 to +5.89 | −3.07 to +5.89 ✓ |
| American leaf spot | SegFormer-B5 | 0.884 | 0.22 | +0.16 | 0.82 | −0.47 to +0.79 | −0.47 to +0.79 ✓ |
| American leaf spot | UNet++ | 0.853 | 0.15 | −0.03 | 0.85 | −0.50 to +0.44 | −0.50 to +0.44 ✓ |
| American leaf spot | DeepLabV3+ | 0.815 | 0.25 | +0.08 | 0.84 | −0.66 to +0.82 | −0.66 to +0.82 ✓ |

## 2. Reference severity (test diseased leaves)

| Disease | n | Mean (%) | Median (%) | SD | Range (%) | IQR (%) | Leaves > 10% | Reference grades |
|---|---|---|---|---|---|---|---|---|
| Coffee leaf rust | 21 | 5.47 | 3.83 | 5.20 | 0.41–19.34 | 1.89–7.46 | 3 | G0 1, G1 11, G2 9, G3 0; mask-derived (cut-points 0.5/5/20%) |
| American leaf spot | 55 | 5.32 | 4.15 | 5.43 | 0.13–26.78 | 0.87–7.45 | 8 | G0 0, G1 30, G2 23, G3 2; annotator's visual grade |

No rust test leaf reaches 20% reference severity (maximum 19.34%); only 3 rust leaves exceed 10%, so every rust statement about the 20% cut-point is anchored by those three leaves.

## 3. Bias and constant limits of agreement (95% CIs)

| Disease | Model | Bias (pp) [bootstrap CI] | SD of D | Lower LoA [bootstrap CI] | Upper LoA [bootstrap CI] | Lower / upper LoA, parametric CI (B&A 1986) |
|---|---|---|---|---|---|---|
| Coffee leaf rust | SegFormer-B5 | +0.99 [0.48, 1.60] | 1.36 | −1.68 [−2.16, −0.63] | +3.66 [1.72, 5.05] | [−2.76, −0.61] / [2.59, 4.74] |
| Coffee leaf rust | UNet++ | +1.75 [0.88, 2.74] | 2.26 | −2.68 [−3.33, −1.62] | +6.17 [3.59, 8.04] | [−4.46, −0.90] / [4.39, 7.95] |
| Coffee leaf rust | DeepLabV3+ | +1.41 [0.57, 2.44] | 2.28 | −3.07 [−3.99, −1.34] | +5.89 [2.62, 8.37] | [−4.87, −1.26] / [4.09, 7.69] |
| American leaf spot | SegFormer-B5 | +0.16 [0.09, 0.25] | 0.32 | −0.47 [−0.59, −0.32] | +0.79 [0.54, 1.03] | [−0.62, −0.32] / [0.64, 0.94] |
| American leaf spot | UNet++ | −0.03 [−0.09, 0.03] | 0.24 | −0.50 [−0.72, −0.28] | +0.44 [0.26, 0.60] | [−0.61, −0.38] / [0.33, 0.55] |
| American leaf spot | DeepLabV3+ | +0.08 [−0.02, 0.18] | 0.38 | −0.66 [−0.89, −0.41] | +0.82 [0.56, 1.07] | [−0.84, −0.48] / [0.64, 1.00] |

Constant LoA assume that D does not depend on severity; Section 4 shows that this assumption fails for rust (all models) and, to a small degree, for leaf spot, so the constant LoA are reported only to reproduce the published values and should be replaced by the regression-based LoA.

## 4. Proportional bias: regression of differences on means

OLS of D on A; 95% CI and p from the t distribution (df = n − 2). Because the residual spread grows with A, the slope is also given with a bootstrap percentile CI and an HC3 heteroscedasticity-robust p value. 'Abs-residual slope' = slope of the absolute residuals on A (tests whether the SD of D grows with A).

| Disease | Model | Slope [t CI] | p | Slope bootstrap CI | p (HC3) | Intercept (pp) [t CI] | p | R² | Abs-residual slope; p |
|---|---|---|---|---|---|---|---|---|---|
| Coffee leaf rust | SegFormer-B5 | 0.208 [0.154, 0.261] | <0.001 | [0.152, 0.288] | <0.001 | −0.25 [−0.69, 0.19] | 0.247 | 0.78 | 0.031; 0.118 |
| Coffee leaf rust | UNet++ | 0.311 [0.215, 0.406] | <0.001 | [0.238, 0.491] | <0.001 | −0.22 [−1.06, 0.61] | 0.580 | 0.71 | 0.081; 0.008 |
| Coffee leaf rust | DeepLabV3+ | 0.247 [0.102, 0.392] | 0.002 | [0.146, 0.510] | 0.016 | −0.12 [−1.34, 1.11] | 0.845 | 0.40 | 0.137; 0.007 |
| American leaf spot | SegFormer-B5 | 0.045 [0.035, 0.055] | <0.001 | [0.034, 0.057] | <0.001 | −0.08 [−0.16, 0.00] | 0.053 | 0.60 | 0.008; 0.049 |
| American leaf spot | UNet++ | 0.014 [0.002, 0.025] | 0.021 | [−0.011, 0.023] | 0.098 | −0.10 [−0.19, −0.01] | 0.027 | 0.10 | 0.010; 0.019 |
| American leaf spot | DeepLabV3+ | 0.040 [0.025, 0.056] | <0.001 | [0.014, 0.055] | <0.001 | −0.14 [−0.25, −0.02] | 0.021 | 0.35 | 0.011; 0.063 |

Calibration regression (secondary): predicted = α + β × reference.

| Disease | Model | β [bootstrap CI] | α (pp) | Log-ratio: geometric mean ratio pred/ref [CI] | Ratio LoA | Leaves over-estimated |
|---|---|---|---|---|---|---|
| Coffee leaf rust | SegFormer-B5 | 1.223 [1.162, 1.306] | −0.23 | 1.143 [1.102, 1.189] | 0.95–1.37 | 20/21 |
| Coffee leaf rust | UNet++ | 1.335 [1.253, 1.575] | −0.09 | 1.244 [1.149, 1.351] | 0.85–1.82 | 18/21 |
| Coffee leaf rust | DeepLabV3+ | 1.216 [1.127, 1.512] | +0.23 | 1.174 [1.071, 1.304] | 0.74–1.87 | 16/21 |
| American leaf spot | SegFormer-B5 | 1.045 [1.034, 1.058] | −0.08 | 0.993 [0.972, 1.012] | 0.85–1.16 | 35/55 |
| American leaf spot | UNet++ | 1.013 [0.987, 1.023] | −0.09 | 0.979 [0.959, 0.998] | 0.84–1.14 | 25/55 |
| American leaf spot | DeepLabV3+ | 1.040 [1.010, 1.056] | −0.13 | 0.987 [0.960, 1.012] | 0.81–1.20 | 32/55 |

## 5. Regression-based limits of agreement (Bland & Altman 1999)

D̂(A) = b0 + b1·A; |residual| = c0 + c1·A; SD(A) = √(π/2)·(c0 + c1·A); LoA(A) = D̂(A) ± 1.96·SD(A). c0 > 0 for every model, so the modelled SD is positive over the whole observed range.

| Disease | Model | D̂(A) | SD(A) |
|---|---|---|---|
| Coffee leaf rust | SegFormer-B5 | −0.25011 + 0.20766 * A | sqrt(pi/2) * (0.20160 + 0.03079 * A) |
| Coffee leaf rust | UNet++ | −0.22414 + 0.31071 * A | sqrt(pi/2) * (0.30476 + 0.08139 * A) |
| Coffee leaf rust | DeepLabV3+ | −0.11593 + 0.24713 * A | sqrt(pi/2) * (0.20309 + 0.13654 * A) |
| American leaf spot | SegFormer-B5 | −0.07732 + 0.04467 * A | sqrt(pi/2) * (0.06320 + 0.00832 * A) |
| American leaf spot | UNet++ | −0.09874 + 0.01359 * A | sqrt(pi/2) * (0.09351 + 0.00998 * A) |
| American leaf spot | DeepLabV3+ | −0.13711 + 0.04047 * A | sqrt(pi/2) * (0.13169 + 0.01076 * A) |

### 5a. At the 5% and 20% cut-points (reference scale)

Values are read off the fitted line at the point whose reference severity equals the cut-point (R = A − D/2, i.e. D = (b0 + b1·R)/(1 − b1/2)); 'relative' = fitted D as % of the cut-point. 'Fitted prediction reaches the cut at' = reference severity above which the fitted mean prediction R + D̂ is at or above the cut-point. Leaves with reference severity between that value and the cut-point are therefore prone to over-grading, but the band only marks where the fitted mean crosses the cut: at its lower edge the fitted prediction just reaches the cut, so an over-grade there is roughly as likely as not (the chance rises toward the cut-point), and the edge itself is uncertain (bootstrap CI). The last column gives the number of test leaves in the band and, in parentheses, how many of them were actually over-graded.

| Disease | Model | Cut | Fitted D (pp) [bootstrap CI] | Relative (%) [CI] | Regression-based 95% LoA | Fitted prediction reaches the cut at reference = [CI] | Test leaves in band (over-graded) |
|---|---|---|---|---|---|---|---|
| Coffee leaf rust | SegFormer-B5 | 5% | +0.88 [0.64, 1.22] | +17.6 [12.7, 24.3] | −0.03 to +1.79 | 4.29% [4.07, 4.46] | 1 (1) |
| Coffee leaf rust | SegFormer-B5 | 20% | +4.36 [3.14, 6.21] | +21.8 [15.7, 31.0] | +2.18 to +6.53 | 16.46% [15.35, 17.31] | 2 (1) |
| Coffee leaf rust | UNet++ | 5% | +1.57 [1.05, 2.27] | +31.5 [21.1, 45.3] | −0.33 to +3.48 | 3.85% [3.50, 4.19] | 1 (1) |
| Coffee leaf rust | UNet++ | 20% | +7.09 [5.22, 11.88] | +35.5 [26.1, 59.4] | +1.64 to +12.55 | 14.82% [12.81, 15.90] | 2 (2) |
| Coffee leaf rust | DeepLabV3+ | 5% | +1.28 [0.60, 2.36] | +25.6 [12.0, 47.2] | −1.11 to +3.67 | 4.00% [3.44, 4.49] | 1 (1) |
| Coffee leaf rust | DeepLabV3+ | 20% | +5.51 [3.02, 12.38] | +27.5 [15.1, 61.9] | −2.62 to +13.64 | 15.70% [12.61, 17.39] | 2 (1) |
| American leaf spot | SegFormer-B5 | 5% | +0.15 [0.10, 0.21] | +3.0 [1.9, 4.1] | −0.11 to +0.41 | 4.86% [4.80, 4.91] | 1 (0) |
| American leaf spot | SegFormer-B5 | 20% | +0.83 [0.63, 1.08] | +4.2 [3.1, 5.4] | +0.26 to +1.41 | 19.20% [18.98, 19.39] | 0 (0) |
| American leaf spot | UNet++ | 5% | −0.03 [−0.10, 0.02] | −0.6 [−2.0, 0.4] | −0.38 to +0.32 | 5.03% [4.98, 5.10] | 0 (0) |
| American leaf spot | UNet++ | 20% | +0.17 [−0.25, 0.34] | +0.9 [−1.2, 1.7] | −0.55 to +0.90 | 19.83% [19.67, 20.25] | 0 (0) |
| American leaf spot | DeepLabV3+ | 5% | +0.07 [−0.02, 0.14] | +1.3 [−0.5, 2.8] | −0.39 to +0.52 | 4.94% [4.87, 5.02] | 0 (0) |
| American leaf spot | DeepLabV3+ | 20% | +0.69 [0.21, 0.95] | +3.4 [1.1, 4.8] | −0.18 to +1.55 | 19.34% [19.10, 19.79] | 0 (0) |

### 5b. At fixed mean severities A (as plotted)

| Disease | Model | A = 1% | A = 5% | A = 10% | A = 20% |
|---|---|---|---|---|---|
| Coffee leaf rust | SegFormer-B5 | −0.04 (−0.61 to +0.53) | +0.79 (−0.09 to +1.66) | +1.83 (+0.57 to +3.08) | +3.90 (+1.90 to +5.91) |
| Coffee leaf rust | UNet++ | +0.09 (−0.86 to +1.04) | +1.33 (−0.42 to +3.08) | +2.88 (+0.14 to +5.63) | +5.99 (+1.24 to +10.74) |
| Coffee leaf rust | DeepLabV3+ | +0.13 (−0.70 to +0.97) | +1.12 (−1.06 to +3.30) | +2.36 (−1.50 to +6.21) | +4.83 (−2.38 to +12.03) |
| American leaf spot | SegFormer-B5 | −0.03 (−0.21 to +0.14) | +0.15 (−0.11 to +0.40) | +0.37 (+0.01 to +0.73) | +0.82 (+0.25 to +1.38) |
| American leaf spot | UNet++ | −0.09 (−0.34 to +0.17) | −0.03 (−0.38 to +0.32) | +0.04 (−0.44 to +0.51) | +0.17 (−0.55 to +0.89) |
| American leaf spot | DeepLabV3+ | −0.10 (−0.45 to +0.25) | +0.07 (−0.39 to +0.52) | +0.27 (−0.32 to +0.86) | +0.67 (−0.18 to +1.52) |

Cells: fitted D (regression-based 95% LoA), pp. Observed A range: rust 0.39–22.2%, leaf spot 0.12–27.3%; only 2 rust and 3 leaf-spot leaves have A > 15%.

### 5c. Grade errors against the cut-points (observed)

Leaves sorted by reference severity. For leaf spot the reference grade is the annotator's visual grade, so each error is also labelled with the grade derived from the reference mask severity (same cut-points): *visual/mask gap* = the predicted grade equals the mask-derived grade (an exact reproduction of the reference mask would give the same error); *severity error* = the mask-derived grade equals the visual grade and the predicted severity crossed a cut-point. For rust the reference grade is mask-derived, so every rust error is a severity error.

- Coffee leaf rust, SegFormer-B5: QWK 0.88, accuracy 0.90; 2 over-grades, 0 under-grades. Reference → predicted severity: 4.71% → 5.17% (G1→G2); 19.34% → 23.73% (G2→G3)
- Coffee leaf rust, UNet++: QWK 0.79, accuracy 0.81; 4 over-grades, 0 under-grades. Reference → predicted severity: 3.83% → 5.01% (G1→G2); 4.71% → 5.39% (G1→G2); 16.65% → 20.59% (G2→G3); 19.34% → 25.11% (G2→G3)
- Coffee leaf rust, DeepLabV3+: QWK 0.88, accuracy 0.90; 2 over-grades, 0 under-grades. Reference → predicted severity: 4.71% → 5.29% (G1→G2); 19.34% → 22.65% (G2→G3)
- American leaf spot, SegFormer-B5: QWK 0.82, accuracy 0.84; 0 over-grades, 9 under-grades. Reference → predicted severity: 0.132% → 0.107% (visual G1→G0; mask-derived G0: visual/mask gap); 0.282% → 0.302% (visual G1→G0; mask-derived G0: visual/mask gap); 0.344% → 0.291% (visual G1→G0; mask-derived G0: visual/mask gap); 0.358% → 0.305% (visual G1→G0; mask-derived G0: visual/mask gap); 0.387% → 0.352% (visual G1→G0; mask-derived G0: visual/mask gap); 0.399% → 0.338% (visual G1→G0; mask-derived G0: visual/mask gap); 0.545% → 0.490% (visual G1→G0; mask-derived G1: severity error); 0.601% → 0.448% (visual G1→G0; mask-derived G1: severity error); 4.799% → 4.823% (visual G2→G1; mask-derived G1: visual/mask gap)
- American leaf spot, UNet++: QWK 0.85, accuracy 0.87; 0 over-grades, 7 under-grades. Reference → predicted severity: 0.132% → 0.131% (visual G1→G0; mask-derived G0: visual/mask gap); 0.282% → 0.339% (visual G1→G0; mask-derived G0: visual/mask gap); 0.344% → 0.286% (visual G1→G0; mask-derived G0: visual/mask gap); 0.358% → 0.263% (visual G1→G0; mask-derived G0: visual/mask gap); 0.387% → 0.393% (visual G1→G0; mask-derived G0: visual/mask gap); 0.399% → 0.376% (visual G1→G0; mask-derived G0: visual/mask gap); 4.799% → 4.885% (visual G2→G1; mask-derived G1: visual/mask gap)
- American leaf spot, DeepLabV3+: QWK 0.84, accuracy 0.85; 0 over-grades, 8 under-grades. Reference → predicted severity: 0.132% → 0.106% (visual G1→G0; mask-derived G0: visual/mask gap); 0.282% → 0.324% (visual G1→G0; mask-derived G0: visual/mask gap); 0.344% → 0.297% (visual G1→G0; mask-derived G0: visual/mask gap); 0.358% → 0.297% (visual G1→G0; mask-derived G0: visual/mask gap); 0.387% → 0.408% (visual G1→G0; mask-derived G0: visual/mask gap); 0.399% → 0.369% (visual G1→G0; mask-derived G0: visual/mask gap); 0.601% → 0.465% (visual G1→G0; mask-derived G1: severity error); 4.799% → 4.924% (visual G2→G1; mask-derived G1: visual/mask gap)

For rust every grade error is a one-grade over-call (no model under-graded a rust leaf), and all 8 over-calls (all models) lie inside or at the lower edge of the band just below a cut-point in which the fitted mean prediction is at or above the cut (Section 5a): 7 lie inside it, and the UNet++ over-call 3.83% → 5.01% lies 0.02 pp below the lower edge of the UNet++ 5% band (3.85%; reference 3.828% vs 3.849%). Conversely, not every leaf in a band was over-graded: the leaf at 16.65% lies in the 20% band of all three models but was graded correctly by SegFormer-B5 (predicted 19.47%) and DeepLabV3+ (predicted 18.63%). The bands therefore mark where leaves are prone to over-grading, not where it is certain.

For leaf spot all grade errors are under-grades, of two kinds. (i) Visual/mask gap: six leaves visually graded G1 have mask severity 0.13–0.40% (mask-derived G0) and one leaf visually graded G2 has mask severity 4.80% (mask-derived G1). For these seven leaves the predicted grade equals the mask-derived grade, so even an exact reproduction of the reference mask would have produced the same error; they reflect the known gap between visual grades and mask severity, not model error, and account for all 7 UNet++ errors, 7 of the 9 SegFormer-B5 errors and 7 of the 8 DeepLabV3+ errors. (ii) Severity errors across the 0.5% cut-point: SegFormer-B5 0.545% → 0.490%; SegFormer-B5 0.601% → 0.448%; DeepLabV3+ 0.601% → 0.465%. Here the mask-derived grade agrees with the visual G1, and the model under-estimated severity by 0.06–0.15 pp, which took the prediction below 0.5%. These small under-estimates at low severity are in the direction of the negative leaf-spot intercepts (SegFormer-B5 −0.08 pp, p = 0.053, fitted D at A = 0.5% −0.05 pp; DeepLabV3+ −0.14 pp, p = 0.021, fitted D at A = 0.5% −0.12 pp); they are not caused by the proportional over-estimation, which is negligible at these severities.

## 6. Concordance, relative error

| Disease | Model | Lin's CCC [bootstrap CI] | CCC z-transform CI | Pearson r (precision) | Cb (accuracy) [CI] | MAE (pp) [CI] | Relative error: MAE / mean reference (%) [CI] | Relative bias (%) | RMSE (pp) |
|---|---|---|---|---|---|---|---|---|---|
| Coffee leaf rust | SegFormer-B5 | 0.958 [0.910, 0.979] | [0.926, 0.976] | 0.994 | 0.964 [0.929, 0.981] | 0.99 [0.48, 1.60] | 18.1 [12.4, 24.2] | +18.1 | 1.66 |
| Coffee leaf rust | UNet++ | 0.897 [0.786, 0.949] | [0.819, 0.942] | 0.979 | 0.916 [0.819, 0.953] | 1.77 [0.92, 2.76] | 32.3 [21.9, 43.9] | +31.9 | 2.81 |
| Coffee leaf rust | DeepLabV3+ | 0.900 [0.731, 0.977] | [0.805, 0.950] | 0.954 | 0.943 [0.821, 0.982] | 1.47 [0.65, 2.48] | 26.9 [13.8, 44.9] | +25.8 | 2.64 |
| American leaf spot | SegFormer-B5 | 0.998 [0.996, 0.999] | [0.997, 0.999] | 0.999 | 0.999 [0.997, 0.999] | 0.22 [0.15, 0.30] | 4.1 [3.4, 5.0] | +3.1 | 0.36 |
| American leaf spot | UNet++ | 0.999 [0.997, 1.000] | [0.998, 0.999] | 0.999 | 1.000 [1.000, 1.000] | 0.15 [0.10, 0.20] | 2.7 [2.1, 3.7] | −0.5 | 0.24 |
| American leaf spot | DeepLabV3+ | 0.998 [0.996, 0.999] | [0.996, 0.998] | 0.998 | 0.999 [0.998, 1.000] | 0.25 [0.18, 0.33] | 4.6 [3.8, 5.6] | +1.5 | 0.38 |

CCC = r × Cb, so the shortfall of CCC from 1 splits additively on the log scale, −log CCC = −log r − log Cb. Share of the log-shortfall due to accuracy (Cb) / precision (r): rust SegFormer-B5 85% / 15%, UNet++ 81% / 19%, DeepLabV3+ 55% / 45%; leaf spot SegFormer-B5 68% / 32%, UNet++ 11% / 89%, DeepLabV3+ 38% / 62%. For rust SegFormer-B5 the shortfall is therefore mainly, not entirely, a matter of accuracy (systematic over-estimation); for leaf spot every CCC is at least 0.997, so the split is immaterial there.

## 7. Where the over-estimation comes from: FP/FN decomposition of the per-leaf severity error

D = FP% − FN% + L exactly (per face, hence per leaf and on average), where FP% and FN% are false-positive and false-negative lesion pixels as % of the reference leaf and L is the leaf-mask (denominator) term. FP pixels are split into: *band* = within 2 px of a reference lesion, in a predicted component that overlaps it (predicted lesion slightly wider than the reference lesion); *extension* = same components, > 2 px from the reference lesion; *spurious* = predicted components that touch no reference lesion. Slopes are OLS slopes on A; they add up exactly to the Bland–Altman slope (slope FP − slope FN + slope L).

| Disease | Model | Bias | FP% [CI] | FN% [CI] | L | FP band | FP extension | FP spurious | FN: missed lesions | Slope D | Slope FP | Slope FP band | Slope FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Coffee leaf rust | SegFormer-B5 | +0.99 | 1.29 [0.73, 1.95] | 0.29 [0.21, 0.38] | −0.006 | 0.88 | 0.29 | 0.12 | 0.01 | 0.208 | 0.235 | 0.168 | 0.028 |
| Coffee leaf rust | UNet++ | +1.75 | 2.29 [1.34, 3.36] | 0.41 [0.30, 0.52] | −0.132 | 1.31 | 0.88 | 0.10 | 0.06 | 0.311 | 0.363 | 0.205 | 0.028 |
| Coffee leaf rust | DeepLabV3+ | +1.41 | 2.10 [1.20, 3.16] | 0.58 [0.40, 0.78] | −0.110 | 1.17 | 0.81 | 0.12 | 0.09 | 0.247 | 0.318 | 0.180 | 0.052 |
| American leaf spot | SegFormer-B5 | +0.16 | 0.34 [0.25, 0.44] | 0.17 [0.13, 0.23] | −0.006 | 0.29 | 0.03 | 0.03 | 0.03 | 0.045 | 0.053 | 0.048 | 0.009 |
| American leaf spot | UNet++ | −0.03 | 0.31 [0.24, 0.38] | 0.28 [0.21, 0.37] | −0.052 | 0.27 | 0.02 | 0.02 | 0.09 | 0.014 | 0.044 | 0.043 | 0.020 |
| American leaf spot | DeepLabV3+ | +0.08 | 0.48 [0.37, 0.60] | 0.36 [0.27, 0.48] | −0.034 | 0.42 | 0.04 | 0.02 | 0.10 | 0.040 | 0.079 | 0.071 | 0.028 |

Reading (rust SegFormer-B5): FP area 1.29% of the leaf against FN 0.29%; 68% of the FP area lies in the 2-px band around reference lesions, 23% is extension beyond 2 px and 9% is spurious blobs; the band term alone has slope 0.168 on A, 81% of the Bland–Altman slope. Missed lesions contribute almost nothing (0.014%). So the proportional over-estimation arises because predicted lesions extend slightly beyond the reference lesion borders; more lesion border means more such pixels, which makes the error scale with severity. The same pattern holds for leaf spot (85% of SegFormer-B5 FP area in the band) and for the rust CNNs, which additionally show more extension beyond 2 px. The leaf-mask term L is negligible. Note that the rust reference is semi-automatic (detector + SAM 2, spot-checked, not edited), so this describes disagreement with the reference border placement; it does not establish which border is correct.

## 8. IoU sensitivity analyses

| Disease | Model | Mean leaf IoU (published) [CI] | Median leaf IoU [CI] | Pooled (micro) IoU [CI] | Minimum leaf IoU |
|---|---|---|---|---|---|
| Coffee leaf rust | SegFormer-B5 | 0.767 [0.743, 0.790] | 0.767 [0.745, 0.804] | 0.769 [0.732, 0.797] | 0.638 |
| Coffee leaf rust | UNet++ | 0.647 [0.607, 0.684] | 0.648 [0.607, 0.714] | 0.662 [0.613, 0.701] | 0.481 |
| Coffee leaf rust | DeepLabV3+ | 0.632 [0.590, 0.669] | 0.646 [0.591, 0.695] | 0.656 [0.594, 0.704] | 0.384 |
| American leaf spot | SegFormer-B5 | 0.884 [0.868, 0.900] | 0.904 [0.887, 0.917] | 0.908 [0.890, 0.923] | 0.732 |
| American leaf spot | UNet++ | 0.853 [0.830, 0.874] | 0.880 [0.862, 0.891] | 0.891 [0.870, 0.909] | 0.587 |
| American leaf spot | DeepLabV3+ | 0.815 [0.790, 0.837] | 0.850 [0.815, 0.860] | 0.852 [0.825, 0.873] | 0.519 |

The architecture ordering is the same under all three summaries (rust: SegFormer-B5 > UNet++ ≥ DeepLabV3+; leaf spot: SegFormer-B5 > UNet++ > DeepLabV3+). For leaf spot the mean leaf IoU is 0.02–0.04 below the median and pooled values because a few low-severity leaves with small lesions have low IoU; for rust the three summaries nearly coincide.

## 9. Boundary F1 (lesion class, 2-px tolerance, letterbox space)

Boundary = lesion mask minus its 3 × 3 erosion; a boundary pixel is matched if a boundary pixel of the other mask lies within 2 px (Euclidean). Rust images are 791 × 1024 (letterbox scale 1.0, so 2 px = 2 native px); leaf-spot images are 1583 × 2048 (scale 0.5, so 2 px = 4 native px). The physical pixel size was not recorded, so no tolerance in mm is given. Leaf mean = mean over leaves of per-leaf F1 (leaf spot: mean of the two faces); pooled = F1 of the pooled matched-pixel counts.

| Disease | Model | BF1 @2 px, leaf mean [CI] | BF1 @2 px, pooled [CI] | Pooled boundary precision | Pooled boundary recall | BF1 @1 / @3 / @5 px (leaf mean) |
|---|---|---|---|---|---|---|
| Coffee leaf rust | SegFormer-B5 | 0.890 [0.857, 0.919] | 0.865 [0.830, 0.900] | 0.847 | 0.883 | 0.793 / 0.923 / 0.947 |
| Coffee leaf rust | UNet++ | 0.749 [0.695, 0.801] | 0.714 [0.659, 0.771] | 0.708 | 0.719 | 0.548 / 0.857 / 0.918 |
| Coffee leaf rust | DeepLabV3+ | 0.710 [0.649, 0.766] | 0.688 [0.614, 0.756] | 0.662 | 0.717 | 0.522 / 0.838 / 0.922 |
| American leaf spot | SegFormer-B5 | 0.948 [0.940, 0.957] | 0.945 [0.930, 0.957] | 0.958 | 0.931 | 0.912 / 0.962 / 0.969 |
| American leaf spot | UNet++ | 0.906 [0.889, 0.921] | 0.911 [0.890, 0.930] | 0.962 | 0.866 | 0.857 / 0.920 / 0.931 |
| American leaf spot | DeepLabV3+ | 0.881 [0.862, 0.898] | 0.876 [0.847, 0.898] | 0.926 | 0.830 | 0.772 / 0.912 / 0.928 |

## 10. Healthy rust-pack test leaves (context only)

Only 21 of the 54 healthy rust-test leaves used in the paper are present locally (all from the EARTH site); the numbers below cover those 21 (rust models) and do not replace the published 54-leaf counts. Two leaf-level definitions are shown because they give different counts for the same leaves: **abaxial face only** (the rust leaf-level rule used for the diseased leaves throughout this analysis) and **maximum over both faces** (a leaf counts if either face carries a predicted lesion; the definition used in the revision brief, fact 10).

| Model | Definition | Mean predicted severity (%) | Max (%) | Leaves with ≥ 1 predicted lesion px | Leaves ≥ 0.5% | Leaves ≥ 5% |
|---|---|---|---|---|---|---|
| SegFormer-B5 | abaxial face only | 0.121 | 0.355 | 17/21 | 0/21 | 0/21 |
| SegFormer-B5 | max over both faces | 0.157 | 0.382 | 19/21 | 0/21 | 0/21 |
| UNet++ | abaxial face only | 0.031 | 0.155 | 15/21 | 0/21 | 0/21 |
| UNet++ | max over both faces | 0.051 | 0.267 | 19/21 | 0/21 | 0/21 |
| DeepLabV3+ | abaxial face only | 0.011 | 0.085 | 7/21 | 0/21 | 0/21 |
| DeepLabV3+ | max over both faces | 0.026 | 0.090 | 13/21 | 0/21 | 0/21 |

**Use one definition in the manuscript and state it.** Any-pixel counts: abaxial face only 17/21, 15/21, 7/21 (maximum 0.36/0.16/0.09%); maximum over both faces 19/21, 19/21, 13/21 (maximum 0.38/0.27/0.09%), for SegFormer-B5, UNet++ and DeepLabV3+. The both-face numbers are the ones in the revision brief; if the manuscript follows the brief, it should say 'on either face'. The two definitions agree on the grading conclusion: no healthy leaf reaches 0.5% on either face, so every healthy leaf is graded G0 by all three models.

These false alarms are small in area (at most 0.38% of the leaf on any face). They do not, however, cleanly show that spurious detections are unimportant on the diseased rust leaves, because the healthy leaves come from the EARTH site while the diseased rust leaves come from Sarchí (site and disease status are confounded). The better evidence is on the diseased leaves themselves (Section 7): for rust SegFormer-B5, spurious predicted components that touch no reference lesion make up 9% of the false-positive area (0.12 of 1.29% of the leaf), against 68% in the 2-px band around reference lesions.

**Inconsistency to resolve (outside this task):** the manuscript says the zero-false-positive models left every healthy leaf without any predicted lesion pixel and reports 0/54 for rust SegFormer-B5, but with the raw argmax maps most of the 21 local healthy leaves already carry ≥ 1 predicted lesion pixel (counts above), and none reaches 0.5%. The published counts must therefore rest on a threshold; the revision brief (fact 10) notes that the training kernels defined a false positive as predicted severity ≥ 5%. The specificity definition in Methods/Results and in Figure 7 (label fig7, false-positive rate on healthy test leaves) should be checked against the code that produced the 0/54 counts (`% TODO(author)`).

## 11. Answer: is the over-estimation proportional to severity, and how large is it at the cut-points?

**Rust: yes, for all three models.** The difference grows by 0.208 pp per 1-pp increase in mean severity for SegFormer-B5 (95% CI 0.154 to 0.261, p < 0.001; bootstrap CI 0.152 to 0.288), 0.311 for UNet++ and 0.247 for DeepLabV3+, and the intercepts do not differ from zero (SegFormer-B5 −0.25 pp, p = 0.25). The over-estimation is thus essentially multiplicative: the calibration line is predicted = 1.22 × reference − 0.23 pp for SegFormer-B5 (UNet++ slope 1.34, DeepLabV3+ 1.22), i.e. SegFormer-B5 predicts roughly one-fifth more lesion area than the reference between 5% and 20% severity. For SegFormer-B5 the fitted over-estimate is +0.88 pp at a reference severity of 5% (95% CI 0.64 to 1.22; 18% of the value; regression-based LoA −0.03 to +1.79) and +4.36 pp at 20% (95% CI 3.14 to 6.21; 22%; LoA +2.18 to +6.53). The fitted mean prediction therefore crosses the 5% cut-point for reference severities of about 4.3–5% (lower edge 95% CI [4.07, 4.46]%) and the 20% cut-point for about 16.5–20% ([15.35, 17.31]%), so leaves in these bands are prone to over-grading; at the lower edge of a band the fitted prediction only just reaches the cut, so an over-grade there is roughly as likely as not. Both SegFormer-B5 rust grade errors (reference 4.71% and 19.34%) lay in these bands, and the third test leaf in them (16.65%, predicted 19.47%) was graded correctly. The 20% estimates rest on the three rust test leaves above 10% (maximum reference 19.3%) and have wide intervals; they indicate direction and order of magnitude, not a precise value.

**American leaf spot: a small proportional component.** Slope 0.045 (0.035 to 0.055) for SegFormer-B5 and 0.040 for DeepLabV3+ (both robust to bootstrap and HC3); for UNet++ the slope 0.014 (t p = 0.021) is not robust (bootstrap CI [−0.011, 0.023], HC3 p = 0.10). SegFormer-B5 over-estimates by +0.15 pp at 5% (3%) and +0.83 pp at 20% (4%); this can move a leaf up a grade only if its reference severity lies in 4.86–5% or 19.20–20% (one test leaf, which the annotator graded G2 visually, so no over-grade results). Leaf-spot grade errors are all under-grades instead. For SegFormer-B5, seven of its nine errors reflect the visual-grade/mask gap: six G1 leaves with mask severity 0.13–0.40% and one G2 leaf at 4.80%, whose predicted grade equals the grade derived from the reference mask. The other two (0.545% → 0.490% and 0.601% → 0.448%) are small under-estimates across the 0.5% cut-point, consistent with the slightly negative leaf-spot intercept rather than with the proportional over-estimation (Section 5c).

## 12. Suggested manuscript sentences

*Methods (Section sec:evaluation; replaces the Bland–Altman sentence):*

> Agreement between predicted and reference per-leaf severity was assessed by Bland–Altman analysis: the difference (predicted − reference) was plotted against the mean of the two, and the bias (mean difference) and 95% limits of agreement (bias ± 1.96 SD of the differences) were estimated with 95% bootstrap confidence intervals (10,000 leaf resamples). Because the differences increased with severity, proportional bias was tested by least-squares regression of the differences on the means, and regression-based limits of agreement were obtained by modelling the mean difference and its standard deviation as linear functions of the mean (Bland and Altman, 1999); Lin's concordance correlation coefficient (CCC) and its bias-correction factor C_b are also reported.

*Results / Discussion (2–5 sentences):*

> For rust, the over-estimation was proportional to severity for all three models: the difference increased by 0.21 percentage points per percentage point of mean severity for SegFormer-B5 (95% CI 0.15 to 0.26; p < 0.001) and by 0.25 and 0.31 for DeepLabV3+ and UNet++, with intercepts compatible with zero. For SegFormer-B5 this corresponds to a fitted over-estimate of 0.88 points at a reference severity of 5% (95% CI 0.64 to 1.22) and 4.4 points at 20% (3.1 to 6.2), the latter resting on the three test leaves above 10% severity. The fitted prediction crosses the 5% and 20% cut-points for reference severities of about 4.3–5% and 16.5–20%, respectively, so such leaves are prone to over-grading; both SegFormer-B5 rust grade errors (4.71% and 19.34%) lay in these bands, and the third leaf in them (16.65%) was graded correctly. For American leaf spot the proportional component was small (SegFormer-B5 slope 0.045, 95% CI 0.035 to 0.055; +0.15 points at 5% and +0.83 at 20%) and was not robust for UNet++. Lin's CCC was 0.958 (95% CI 0.910 to 0.979) for rust SegFormer-B5 and 0.998 for leaf spot; the rust shortfall mainly reflected accuracy (C_b = 0.964) rather than precision (r = 0.994), and the mean absolute error was 18% of the mean reference severity for rust (4% for leaf spot). Decomposing the per-leaf error showed that the rust over-estimate came from false-positive lesion pixels (1.29% of leaf area against 0.29% false-negative for SegFormer-B5), 68% of which lay within two pixels of a reference lesion, indicating that predicted lesions extended slightly beyond the borders of the semi-automatic reference lesions; separate spurious detections (9% of the false-positive area) and missed lesions contributed little.

*Figure 5 caption (new):*

> Per-leaf agreement between predicted and reference severity for SegFormer-B5: (a) coffee leaf rust, abaxial face, n = 21 test leaves; (b) American leaf spot, mean of both faces, n = 55. Each point is a leaf. The dashed line is the least-squares regression of the differences on the means, the dotted lines are the regression-based 95% limits of agreement (Bland and Altman, 1999), and the solid grey line marks zero difference; the slope of the regression line is given with its 95% confidence interval. pp = percentage points.

*Supplementary figure caption:*

> Per-leaf Bland–Altman agreement for all six models, (a) coffee leaf rust (n = 21) and (b) American leaf spot (n = 55), with the regression of the differences on the means (dashed), regression-based 95% limits of agreement (dotted), and zero difference (solid grey). The y axis is shared within each disease. Slopes are least-squares estimates with t-based 95% CIs; for leaf-spot UNet++ the bootstrap CI includes zero.

## 13. Caveats and notes for the authors

- New references needed: Bland JM, Altman DG (1999) Measuring agreement in method comparison studies. *Stat Methods Med Res* 8:135–160; Lin LI-K (1989) A concordance correlation coefficient to evaluate reproducibility. *Biometrics* 45:255–268 (and the 2000 correction, *Biometrics* 56:324–325); for boundary F1, e.g. Csurka G, Larlus D, Perronnin F (2013) What is a good evaluation measure for semantic segmentation? *BMVC*. The existing ref44 is Bland & Altman 1986.
- n = 21 rust leaves, 3 above 10% severity, none above 19.3%: the rust 20% cut-point estimates are imprecise and partly extrapolated; the 5% estimates are well supported.
- OLS p values assume constant residual variance, which does not hold; the rust conclusions are unchanged under bootstrap and HC3 inference. For rust SegFormer-B5 the absolute-residual slope is not significant (p = 0.12), so its LoA width is compatible with a constant SD; the proportional finding concerns the mean difference.
- The rust reference is semi-automatic (detector + SAM 2, spot-checked, not edited): 'over-estimation' means relative to that reference. Severity agreement is not an independent check because the reference masks are also the training targets (already stated in the manuscript).
- Each model was trained once; the CIs capture test-leaf sampling only.
- Bootstrap CIs here use seed 20260927 and differ from the published Table 2/4 CIs by at most 0.02 pp (MAE) and 0.001 (IoU) (e.g., rust SegFormer-B5 MAE [0.48, 1.60] here vs [0.48, 1.61] published; leaf IoU [0.743, 0.790] vs [0.743, 0.791]). Point estimates are identical. Keep the published CIs unless the tables are regenerated in full.
- Suggested abbreviations: CCC = concordance correlation coefficient; LoA = limits of agreement.

## Files

- `results.json`: every number above (and more: parametric CIs, HC3, calibration, log-ratio, tolerance sweep; per grade error the mask-derived grade and its category; per cut-point band the leaves in it and whether each was over-graded; CCC log-shortfall shares; healthy leaves under both face definitions).
- `figure5.pdf` / `figure5.png`: new Figure 5 (SegFormer-B5). `figureS_agreement_all.pdf` / `.png`: all six models.
- `pixel_metrics.py`, `agreement.py`, `make_figures.py`, `make_md.py`; `work/pixel_metrics.csv` (per image), `work/leaf_level.csv` (per leaf).
