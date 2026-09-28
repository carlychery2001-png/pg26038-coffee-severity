# A5: Rust operating point (is the rust over-estimation an operating-point artifact?)

> **Public-release copy.** The expert-identity section was removed. All other text and every number are unchanged; local paths are written as `<ROOT>` (the working tree) and `<USER_HOME>`.

PG-26038 major revision, round 4, 28 September 2026. Scripts: `infer_prob.py` (CPU inference with onnxruntime that keeps the softmax lesion probability), `oppoint.py` (all analyses, writes `results.json` and `work/*`), `make_md.py` (this file). **No model was trained or fine-tuned.** Everything below re-uses the existing ONNX models and changes only the decision rule applied to their output.

## Method

- **Models.** Published rust SegFormer-B5, UNet++ and DeepLabV3+ (`modal-recovery/serving/*roya*.onnx`) and the matched-protocol rust CNNs (UNet++ and DeepLabV3+, lrB = 6e-5/6e-4, seeds 0, 1, 2; `reanalysis/kaggle_runs/*roya*lrB*/output/*.onnx`): 9 models.
- **Images.** Rust pack, abaxial face only (the rust leaf-level rule): 21 diseased validation leaves, 21 diseased test leaves, and the 21 locally available healthy test leaves (specificity check only).
- **Protocol.** Identical to `run_inference.py`: leaf = alpha > 8; lesion = mask > 127 AND leaf; aspect-preserving letterbox to 1024 × 1024 with black padding; ImageNet normalization; SegFormer logits upsampled bilinearly (PIL) to 1024 before any decision; metrics in letterbox space. Rust scans are 791 × 1024 px, so the letterbox scale is 1.0 and 1 px = 1 native scan pixel (the physical pixel size was not recorded, so no distance in mm is given). The softmax over the three classes (background, leaf, lesion) was stored per pixel (lesion probability quantized to 1/65535).
- **Operating points.** (i) *argmax*: the published rule. (ii) *Validation-calibrated threshold*: a pixel is lesion if p(lesion) ≥ t, otherwise it takes the argmax of background vs leaf; t was searched on the grid 0.01–0.99 (step 0.01), extended by 0.991–0.999 (step 0.001), 0.9995 and 0.9999, on the 21 **validation** leaves (the extension was added after a preview run put the validation optimum of matched UNet++ seed 1 on the former upper edge 0.99 with validation bias still positive; the reason is the validation curve, but the preview's test results for the eight CNNs had already been computed, which we disclose), once to minimize |bias| and once to minimize MAE, and each chosen t was then applied **once** to the test leaves. (iii) *Erosion*: the argmax lesion mask eroded by 1 px or 2 px (3 × 3 structuring element, 1 or 2 iterations); eroded pixels become leaf, so the leaf denominator is unchanged. The erosion depth was fixed a priori (not tuned); validation values are reported for reference.
- **Endpoints (21 test leaves).** Mean leaf lesion IoU, MAE, bias (D = predicted − reference severity, percentage points, pp), grade accuracy and QWK (mask-derived rust grades, cut-points 0.5/5/20%), and the proportional-bias slope: OLS regression of D on A = (predicted + reference)/2, as in `results/agreement` Section 4, with t-based and bootstrap 95% CIs. Paired bootstrap differences (operating point − argmax) test whether a change is larger than resampling noise.
- **Signed boundary distance (test leaves).** For every predicted lesion contour pixel (lesion mask minus its 3 × 3 erosion): the Euclidean distance to the reference lesion contour, **positive when the pixel lies outside the reference lesion** (sd = EDT to the reference lesion if outside; sd = 1 − EDT to the non-lesion region if inside). A pixel on the reference contour has sd = 0, the first pixel ring outside +1, the second ring inside −1. Distances are pooled over all contour pixels of the 21 leaves ("all"); also given for predicted components that touch a reference lesion only ("attached", excludes spurious blobs), and in the reverse direction (each reference contour pixel relative to the prediction; + = the prediction extends beyond the reference there). Distances take discrete values (0, 1, 1.41, 2, …), so medians and their CIs are coarse; the mean is also reported.
- **Bootstrap.** 10,000 resamples of the 21 test leaves, seed 20260928, percentile 95% CIs; the same resamples for every model and operating point (paired). Bootstrap CIs here therefore differ slightly from `results/agreement` (seed 20260927); point estimates are identical.

## 1. Reproduction of the baseline (argmax)

All argmax pixel counts (leaf, lesion, intersection, union for every model × image, 567 pairs) were checked against `out/per_image.csv` / `out/per_image_retrained.csv` during inference and matched exactly (the script stops on any mismatch). Leaf-level reproduction: all match = **True**.

| Model | Bias here | Bias `results/agreement` | Slope here [t CI] | Slope `results/agreement` [t CI] |
|---|---|---|---|---|
| SegFormer-B5 (published) | +0.990 | +0.990 | 0.208 [0.154, 0.261] | 0.208 [0.154, 0.261] |
| UNet++ (published) | +1.749 | +1.749 | 0.311 [0.215, 0.406] | 0.311 [0.215, 0.406] |
| DeepLabV3+ (published) | +1.411 | +1.411 | 0.247 [0.102, 0.392] | 0.247 [0.102, 0.392] |

| Matched model | Test IoU here / `retrained_summary.json` | Test bias here / summary | Val IoU here / summary |
|---|---|---|---|
| UNet++ matched lrB seed 0 | 0.776 / 0.776 | +0.47 / +0.47 | 0.795 / 0.795 |
| UNet++ matched lrB seed 1 | 0.755 / 0.755 | +0.98 / +0.98 | 0.782 / 0.782 |
| UNet++ matched lrB seed 2 | 0.746 / 0.746 | +1.01 / +1.01 | 0.777 / 0.777 |
| DeepLabV3+ matched lrB seed 0 | 0.721 / 0.721 | +0.77 / +0.77 | 0.746 / 0.746 |
| DeepLabV3+ matched lrB seed 1 | 0.718 / 0.718 | +0.52 / +0.52 | 0.745 / 0.745 |
| DeepLabV3+ matched lrB seed 2 | 0.736 / 0.736 | +0.13 / +0.13 | 0.758 / 0.758 |

## 2. Thresholds chosen on validation (21 leaves)

| Model | t (min \|bias\|) | t (min MAE) | Val bias: argmax → t_bias | Val MAE: argmax → t_MAE | Val slope: argmax → t_bias | Val bias at t = 0.50 | Val bias: 1-px / 2-px erosion |
|---|---|---|---|---|---|---|---|
| SegFormer-B5 (published) | 0.97 | 0.96 | +0.57 → +0.01 | 0.58 → 0.25 | 0.287 → 0.112 | +0.57 | −0.66 / −1.61 |
| UNet++ (published) | 0.89 | 0.89 | +1.00 → −0.01 | 1.08 → 0.32 | 0.428 → 0.141 | +0.97 | −0.22 / −1.20 |
| DeepLabV3+ (published) | 0.93 | 0.96 | +0.88 → −0.02 | 1.00 → 0.45 | 0.438 → 0.174 | +0.86 | −0.30 / −1.26 |
| UNet++ matched lrB seed 0 | 0.89 | 0.82 | +0.27 → −0.01 | 0.33 → 0.25 | 0.181 → 0.098 | +0.26 | −0.87 / −1.78 |
| UNet++ matched lrB seed 1 | 0.996 | 0.98 | +0.60 → +0.01 | 0.61 → 0.37 | 0.331 → 0.169 | +0.59 | −0.59 / −1.54 |
| UNet++ matched lrB seed 2 | 0.98 | 0.96 | +0.55 → +0.00 | 0.58 → 0.31 | 0.320 → 0.159 | +0.54 | −0.61 / −1.55 |
| DeepLabV3+ matched lrB seed 0 | 0.78 | 0.81 | +0.27 → +0.01 | 0.36 → 0.25 | 0.145 → 0.061 | +0.26 | −0.84 / −1.71 |
| DeepLabV3+ matched lrB seed 1 | 0.8 | 0.85 | +0.28 → −0.00 | 0.35 → 0.20 | 0.103 → 0.022 | +0.27 | −0.84 / −1.72 |
| DeepLabV3+ matched lrB seed 2 | 0.74 | 0.83 | +0.21 → −0.00 | 0.31 → 0.23 | 0.106 → 0.039 | +0.20 | −0.90 / −1.77 |

t = 0.50 is not the argmax rule (argmax calls lesion when p(lesion) exceeds both other classes), so it is listed separately. Full validation curves (IoU, MAE, bias, slope for every t): `work/val_curves.csv`.

## 3. Test endpoints (21 test leaves, abaxial face)

Bias, MAE and IoU with bootstrap 95% CIs; slope with t-based CI and bootstrap CI. "Over / under-graded" = leaves whose predicted grade is above / below the reference grade; "Leaves over-estimated" = leaves with D > 0. [Verifier: wording corrected.]

| Model | Operating point | t | Leaf IoU [CI] | MAE (pp) [CI] | Bias (pp) [CI] | Slope [t CI]; bootstrap CI | p (slope) | Grade acc | QWK | Over / under-graded | Leaves over-estimated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SegFormer-B5 (published) | argmax (paper rule) | — | 0.767 [0.743, 0.791] | 0.99 [0.48, 1.62] | +0.99 [+0.48, +1.62] | 0.208 [0.154, 0.261]; [0.152, 0.291] | <0.001 | 0.905 | 0.878 | 2 / 0 | 20/21 |
| SegFormer-B5 (published) | threshold, val min abs(bias) | 0.97 | 0.789 [0.769, 0.810] | 0.34 [0.17, 0.58] | +0.12 [−0.09, +0.41] | 0.075 [0.036, 0.114]; [0.028, 0.126] | <0.001 | 0.952 | 0.938 | 1 / 0 | 7/21 |
| SegFormer-B5 (published) | threshold, val min MAE | 0.96 | 0.790 [0.769, 0.811] | 0.36 [0.16, 0.64] | +0.19 [−0.05, +0.50] | 0.086 [0.045, 0.127]; [0.039, 0.142] | <0.001 | 0.952 | 0.938 | 1 / 0 | 10/21 |
| SegFormer-B5 (published) | argmax + 1-px erosion | — | 0.692 [0.660, 0.725] | 0.79 [0.59, 1.00] | −0.62 [−0.88, −0.30] | −0.003 [−0.068, 0.063]; [−0.068, 0.090] | 0.934 | 0.857 | 0.803 | 0 / 3 | 1/21 |
| SegFormer-B5 (published) | argmax + 2-px erosion | — | 0.481 [0.421, 0.541] | 1.94 [1.38, 2.54] | −1.94 [−2.54, −1.38] | −0.240 [−0.332, −0.147]; [−0.318, −0.114] | <0.001 | 0.714 | 0.707 | 0 / 6 | 0/21 |
| UNet++ (published) | argmax (paper rule) | — | 0.647 [0.608, 0.684] | 1.77 [0.91, 2.76] | +1.75 [+0.88, +2.75] | 0.311 [0.215, 0.406]; [0.239, 0.493] | <0.001 | 0.810 | 0.788 | 4 / 0 | 18/21 |
| UNet++ (published) | threshold, val min abs(bias) | 0.89 | 0.685 [0.656, 0.713] | 0.77 [0.43, 1.20] | +0.30 [−0.15, +0.83] | 0.134 [0.053, 0.215]; [0.071, 0.277] | 0.003 | 0.905 | 0.874 | 1 / 1 | 10/21 |
| UNet++ (published) | threshold, val min MAE | 0.89 | 0.685 [0.656, 0.713] | 0.77 [0.43, 1.20] | +0.30 [−0.15, +0.83] | 0.134 [0.053, 0.215]; [0.071, 0.277] | 0.003 | 0.905 | 0.874 | 1 / 1 | 10/21 |
| UNet++ (published) | argmax + 1-px erosion | — | 0.669 [0.644, 0.695] | 0.83 [0.49, 1.27] | +0.14 [−0.33, +0.72] | 0.118 [0.023, 0.213]; [0.048, 0.307] | 0.018 | 0.905 | 0.870 | 0 / 2 | 7/21 |
| UNet++ (published) | argmax + 2-px erosion | — | 0.540 [0.497, 0.583] | 1.37 [0.98, 1.80] | −1.19 [−1.70, −0.70] | −0.086 [−0.196, 0.024]; [−0.174, 0.129] | 0.116 | 0.762 | 0.701 | 0 / 5 | 2/21 |
| DeepLabV3+ (published) | argmax (paper rule) | — | 0.632 [0.590, 0.669] | 1.47 [0.63, 2.51] | +1.41 [+0.56, +2.46] | 0.247 [0.102, 0.392]; [0.147, 0.516] | 0.002 | 0.905 | 0.878 | 2 / 0 | 16/21 |
| DeepLabV3+ (published) | threshold, val min abs(bias) | 0.93 | 0.636 [0.601, 0.671] | 0.78 [0.39, 1.29] | +0.00 [−0.49, +0.62] | 0.051 [−0.068, 0.171]; [−0.025, 0.253] | 0.382 | 0.905 | 0.870 | 0 / 2 | 4/21 |
| DeepLabV3+ (published) | threshold, val min MAE | 0.96 | 0.628 [0.593, 0.663] | 0.89 [0.55, 1.30] | −0.27 [−0.75, +0.31] | 0.008 [−0.109, 0.126]; [−0.067, 0.201] | 0.883 | 0.905 | 0.870 | 0 / 2 | 3/21 |
| DeepLabV3+ (published) | argmax + 1-px erosion | — | 0.625 [0.589, 0.658] | 1.04 [0.64, 1.51] | −0.25 [−0.80, +0.41] | 0.014 [−0.122, 0.150]; [−0.079, 0.268] | 0.833 | 0.905 | 0.870 | 0 / 2 | 4/21 |
| DeepLabV3+ (published) | argmax + 2-px erosion | — | 0.507 [0.460, 0.553] | 1.77 [1.19, 2.46] | −1.66 [−2.40, −0.99] | −0.249 [−0.379, −0.119]; [−0.343, 0.005] | <0.001 | 0.714 | 0.675 | 0 / 6 | 2/21 |
| UNet++ matched lrB seed 0 | argmax (paper rule) | — | 0.776 [0.751, 0.800] | 0.50 [0.22, 0.85] | +0.47 [+0.17, +0.82] | 0.079 [0.021, 0.138]; [0.039, 0.211] | 0.010 | 0.952 | 0.938 | 1 / 0 | 16/21 |
| UNet++ matched lrB seed 0 | threshold, val min abs(bias) | 0.89 | 0.780 [0.759, 0.801] | 0.39 [0.23, 0.59] | +0.07 [−0.14, +0.33] | 0.023 [−0.029, 0.075]; [−0.013, 0.141] | 0.373 | 0.952 | 0.927 | 0 / 1 | 6/21 |
| UNet++ matched lrB seed 0 | threshold, val min MAE | 0.82 | 0.780 [0.758, 0.801] | 0.38 [0.19, 0.62] | +0.18 [−0.05, +0.47] | 0.040 [−0.014, 0.094]; [0.003, 0.162] | 0.141 | 0.952 | 0.927 | 0 / 1 | 8/21 |
| UNet++ matched lrB seed 0 | argmax + 1-px erosion | — | 0.661 [0.628, 0.695] | 1.04 [0.68, 1.47] | −1.04 [−1.47, −0.68] | −0.150 [−0.212, −0.087]; [−0.198, −0.013] | <0.001 | 0.857 | 0.803 | 0 / 3 | 0/21 |
| UNet++ matched lrB seed 0 | argmax + 2-px erosion | — | 0.452 [0.396, 0.509] | 2.30 [1.57, 3.16] | −2.30 [−3.16, −1.57] | −0.409 [−0.495, −0.324]; [−0.479, −0.239] | <0.001 | 0.667 | 0.654 | 0 / 7 | 0/21 |
| UNet++ matched lrB seed 1 | argmax (paper rule) | — | 0.755 [0.720, 0.787] | 0.98 [0.50, 1.57] | +0.98 [+0.50, +1.57] | 0.141 [0.053, 0.229]; [0.085, 0.327] | 0.003 | 0.905 | 0.878 | 2 / 0 | 21/21 |
| UNet++ matched lrB seed 1 | threshold, val min abs(bias) | 0.996 | 0.776 [0.750, 0.801] | 0.51 [0.25, 0.84] | +0.14 [−0.18, +0.54] | 0.028 [−0.050, 0.107]; [−0.021, 0.191] | 0.462 | 0.952 | 0.927 | 0 / 1 | 5/21 |
| UNet++ matched lrB seed 1 | threshold, val min MAE | 0.98 | 0.776 [0.747, 0.803] | 0.51 [0.18, 0.93] | +0.38 [+0.03, +0.83] | 0.064 [−0.018, 0.145]; [0.013, 0.235] | 0.118 | 1.000 | 1.000 | 0 / 0 | 9/21 |
| UNet++ matched lrB seed 1 | argmax + 1-px erosion | — | 0.687 [0.658, 0.716] | 0.98 [0.66, 1.33] | −0.60 [−1.05, −0.11] | −0.074 [−0.176, 0.028]; [−0.137, 0.141] | 0.146 | 0.905 | 0.870 | 0 / 2 | 2/21 |
| UNet++ matched lrB seed 1 | argmax + 2-px erosion | — | 0.483 [0.434, 0.532] | 2.06 [1.42, 2.82] | −1.92 [−2.73, −1.19] | −0.311 [−0.446, −0.177]; [−0.397, −0.035] | <0.001 | 0.667 | 0.654 | 0 / 7 | 1/21 |
| UNet++ matched lrB seed 2 | argmax (paper rule) | — | 0.746 [0.710, 0.780] | 1.02 [0.49, 1.64] | +1.01 [+0.48, +1.63] | 0.155 [0.062, 0.247]; [0.094, 0.345] | 0.002 | 0.905 | 0.878 | 2 / 0 | 18/21 |
| UNet++ matched lrB seed 2 | threshold, val min abs(bias) | 0.98 | 0.755 [0.726, 0.784] | 0.53 [0.27, 0.86] | +0.21 [−0.11, +0.61] | 0.047 [−0.029, 0.124]; [−0.003, 0.211] | 0.212 | 0.952 | 0.927 | 0 / 1 | 6/21 |
| UNet++ matched lrB seed 2 | threshold, val min MAE | 0.96 | 0.757 [0.726, 0.787] | 0.55 [0.23, 0.94] | +0.35 [+0.00, +0.79] | 0.069 [−0.011, 0.149]; [0.017, 0.238] | 0.087 | 0.952 | 0.927 | 0 / 1 | 9/21 |
| UNet++ matched lrB seed 2 | argmax + 1-px erosion | — | 0.666 [0.632, 0.701] | 0.93 [0.65, 1.24] | −0.56 [−1.00, −0.14] | −0.066 [−0.160, 0.028]; [−0.136, 0.136] | 0.157 | 0.905 | 0.870 | 0 / 2 | 5/21 |
| UNet++ matched lrB seed 2 | argmax + 2-px erosion | — | 0.474 [0.420, 0.529] | 1.89 [1.24, 2.69] | −1.89 [−2.69, −1.24] | −0.314 [−0.426, −0.203]; [−0.406, −0.072] | <0.001 | 0.714 | 0.675 | 0 / 6 | 0/21 |
| DeepLabV3+ matched lrB seed 0 | argmax (paper rule) | — | 0.721 [0.694, 0.748] | 0.87 [0.41, 1.44] | +0.77 [+0.28, +1.38] | 0.161 [0.081, 0.241]; [0.103, 0.306] | <0.001 | 0.952 | 0.938 | 1 / 0 | 15/21 |
| DeepLabV3+ matched lrB seed 0 | threshold, val min abs(bias) | 0.78 | 0.728 [0.704, 0.751] | 0.63 [0.31, 1.06] | +0.36 [−0.02, +0.86] | 0.107 [0.030, 0.184]; [0.050, 0.243] | 0.009 | 0.905 | 0.874 | 1 / 1 | 11/21 |
| DeepLabV3+ matched lrB seed 0 | threshold, val min MAE | 0.81 | 0.728 [0.705, 0.751] | 0.61 [0.30, 1.02] | +0.31 [−0.07, +0.79] | 0.100 [0.024, 0.176]; [0.044, 0.233] | 0.012 | 0.905 | 0.874 | 1 / 1 | 10/21 |
| DeepLabV3+ matched lrB seed 0 | argmax + 1-px erosion | — | 0.665 [0.644, 0.687] | 1.06 [0.77, 1.38] | −0.76 [−1.18, −0.31] | −0.055 [−0.152, 0.043]; [−0.137, 0.120] | 0.256 | 0.857 | 0.803 | 0 / 3 | 2/21 |
| DeepLabV3+ matched lrB seed 0 | argmax + 2-px erosion | — | 0.473 [0.429, 0.520] | 2.06 [1.38, 2.83] | −2.03 [−2.82, −1.34] | −0.295 [−0.424, −0.167]; [−0.411, −0.070] | <0.001 | 0.667 | 0.654 | 0 / 7 | 1/21 |
| DeepLabV3+ matched lrB seed 1 | argmax (paper rule) | — | 0.718 [0.693, 0.742] | 0.64 [0.36, 0.96] | +0.52 [+0.19, +0.88] | 0.106 [0.056, 0.156]; [0.058, 0.159] | <0.001 | 0.905 | 0.878 | 2 / 0 | 16/21 |
| DeepLabV3+ matched lrB seed 1 | threshold, val min abs(bias) | 0.8 | 0.723 [0.701, 0.745] | 0.42 [0.25, 0.60] | +0.10 [−0.15, +0.35] | 0.051 [0.004, 0.099]; [−0.003, 0.092] | 0.034 | 0.905 | 0.874 | 1 / 1 | 11/21 |
| DeepLabV3+ matched lrB seed 1 | threshold, val min MAE | 0.85 | 0.724 [0.702, 0.745] | 0.41 [0.26, 0.57] | +0.00 [−0.23, +0.24] | 0.038 [−0.009, 0.085]; [−0.019, 0.078] | 0.110 | 0.905 | 0.874 | 1 / 1 | 8/21 |
| DeepLabV3+ matched lrB seed 1 | argmax + 1-px erosion | — | 0.661 [0.636, 0.686] | 1.03 [0.71, 1.38] | −1.03 [−1.38, −0.71] | −0.125 [−0.177, −0.073]; [−0.183, −0.069] | <0.001 | 0.857 | 0.803 | 0 / 3 | 0/21 |
| DeepLabV3+ matched lrB seed 1 | argmax + 2-px erosion | — | 0.472 [0.429, 0.518] | 2.31 [1.61, 3.10] | −2.31 [−3.10, −1.61] | −0.387 [−0.457, −0.317]; [−0.456, −0.303] | <0.001 | 0.667 | 0.654 | 0 / 7 | 0/21 |
| DeepLabV3+ matched lrB seed 2 | argmax (paper rule) | — | 0.736 [0.711, 0.760] | 0.36 [0.23, 0.51] | +0.13 [−0.06, +0.33] | 0.025 [−0.017, 0.067]; [−0.015, 0.084] | 0.225 | 0.952 | 0.927 | 0 / 1 | 14/21 |
| DeepLabV3+ matched lrB seed 2 | threshold, val min abs(bias) | 0.74 | 0.738 [0.715, 0.761] | 0.32 [0.18, 0.49] | −0.19 [−0.39, −0.01] | −0.021 [−0.063, 0.021]; [−0.066, 0.030] | 0.300 | 0.952 | 0.927 | 0 / 1 | 9/21 |
| DeepLabV3+ matched lrB seed 2 | threshold, val min MAE | 0.83 | 0.736 [0.712, 0.758] | 0.39 [0.20, 0.60] | −0.35 [−0.58, −0.16] | −0.045 [−0.087, −0.003]; [−0.091, 0.003] | 0.039 | 0.952 | 0.927 | 0 / 1 | 3/21 |
| DeepLabV3+ matched lrB seed 2 | argmax + 1-px erosion | — | 0.648 [0.627, 0.669] | 1.38 [0.95, 1.88] | −1.38 [−1.88, −0.95] | −0.219 [−0.268, −0.170]; [−0.277, −0.149] | <0.001 | 0.857 | 0.803 | 0 / 3 | 0/21 |
| DeepLabV3+ matched lrB seed 2 | argmax + 2-px erosion | — | 0.438 [0.401, 0.478] | 2.61 [1.79, 3.58] | −2.61 [−3.58, −1.79] | −0.494 [−0.568, −0.420]; [−0.581, −0.386] | <0.001 | 0.619 | 0.600 | 0 / 8 | 0/21 |

## 4. Paired change versus argmax (test; operating point − argmax; bootstrap 95% CI)

| Model | Operating point | Δ bias (pp) [CI] | Δ MAE (pp) [CI] | Δ slope [CI] | Δ leaf IoU [CI] | Bias verdict | Slope verdict |
|---|---|---|---|---|---|---|---|
| SegFormer-B5 (published) | threshold, val min abs(bias) | −0.87 [−1.24, −0.55] | −0.65 [−1.07, −0.27] | −0.133 [−0.168, −0.120] | +0.022 [+0.011, +0.033] | disappears (CI includes 0) | shrinks (still > 0) |
| SegFormer-B5 (published) | threshold, val min MAE | −0.80 [−1.14, −0.51] | −0.63 [−1.02, −0.28] | −0.122 [−0.152, −0.110] | +0.023 [+0.013, +0.033] | disappears (CI includes 0) | shrinks (still > 0) |
| SegFormer-B5 (published) | argmax + 1-px erosion | −1.61 [−2.17, −1.12] | −0.20 [−0.78, +0.28] | −0.210 [−0.236, −0.187] | −0.075 [−0.103, −0.045] | reversed (now significant under-estimation) | disappears (CI includes 0) |
| SegFormer-B5 (published) | argmax + 2-px erosion | −2.93 [−4.00, −2.01] | +0.95 [+0.30, +1.46] | −0.447 [−0.496, −0.386] | −0.287 [−0.342, −0.226] | reversed (now significant under-estimation) | reversed (now significant under-estimation) |
| UNet++ (published) | threshold, val min abs(bias) | −1.45 [−1.98, −0.98] | −0.99 [−1.67, −0.38] | −0.177 [−0.229, −0.157] | +0.038 [+0.016, +0.060] | disappears (CI includes 0) | shrinks (still > 0) |
| UNet++ (published) | threshold, val min MAE | −1.45 [−1.98, −0.98] | −0.99 [−1.67, −0.38] | −0.177 [−0.229, −0.157] | +0.038 [+0.016, +0.060] | disappears (CI includes 0) | shrinks (still > 0) |
| UNet++ (published) | argmax + 1-px erosion | −1.61 [−2.19, −1.10] | −0.94 [−1.69, −0.26] | −0.193 [−0.221, −0.165] | +0.023 [−0.007, +0.051] | disappears (CI includes 0) | shrinks (still > 0) |
| UNet++ (published) | argmax + 2-px erosion | −2.94 [−4.02, −1.99] | −0.40 [−1.37, +0.43] | −0.397 [−0.451, −0.332] | −0.107 [−0.161, −0.052] | reversed (now significant under-estimation) | disappears (CI includes 0; point estimate reversed sign) |
| DeepLabV3+ (published) | threshold, val min abs(bias) | −1.41 [−1.95, −0.92] | −0.69 [−1.39, −0.05] | −0.196 [−0.282, −0.163] | +0.005 [−0.016, +0.027] | disappears (CI includes 0) | disappears (CI includes 0) |
| DeepLabV3+ (published) | threshold, val min MAE | −1.68 [−2.33, −1.10] | −0.58 [−1.34, +0.10] | −0.239 [−0.338, −0.201] | −0.004 [−0.031, +0.024] | disappears (CI includes 0) | disappears (CI includes 0) |
| DeepLabV3+ (published) | argmax + 1-px erosion | −1.66 [−2.28, −1.12] | −0.44 [−1.13, +0.19] | −0.233 [−0.275, −0.207] | −0.007 [−0.033, +0.020] | disappears (CI includes 0) | disappears (CI includes 0) |
| DeepLabV3+ (published) | argmax + 2-px erosion | −3.07 [−4.21, −2.05] | +0.30 [−0.90, +1.30] | −0.496 [−0.580, −0.442] | −0.124 [−0.176, −0.069] | reversed (now significant under-estimation) | disappears (CI includes 0; point estimate reversed sign) |
| UNet++ matched lrB seed 0 | threshold, val min abs(bias) | −0.39 [−0.54, −0.27] | −0.11 [−0.29, +0.04] | −0.057 [−0.072, −0.051] | +0.004 [−0.002, +0.010] | disappears (CI includes 0) | disappears (CI includes 0) |
| UNet++ matched lrB seed 0 | threshold, val min MAE | −0.29 [−0.39, −0.19] | −0.12 [−0.26, +0.00] | −0.040 [−0.050, −0.035] | +0.004 [−0.001, +0.008] | disappears (CI includes 0) | shrinks (still > 0) |
| UNet++ matched lrB seed 0 | argmax + 1-px erosion | −1.51 [−2.04, −1.04] | +0.53 [+0.01, +1.05] | −0.229 [−0.249, −0.200] | −0.114 [−0.144, −0.083] | reversed (now significant under-estimation) | reversed (now significant under-estimation) |
| UNet++ matched lrB seed 0 | argmax + 2-px erosion | −2.76 [−3.77, −1.89] | +1.79 [+1.10, +2.63] | −0.488 [−0.533, −0.413] | −0.324 [−0.375, −0.267] | reversed (now significant under-estimation) | reversed (now significant under-estimation) |
| UNet++ matched lrB seed 1 | threshold, val min abs(bias) | −0.84 [−1.13, −0.58] | −0.48 [−0.80, −0.20] | −0.113 [−0.139, −0.103] | +0.022 [+0.008, +0.035] | disappears (CI includes 0) | disappears (CI includes 0) |
| UNet++ matched lrB seed 1 | threshold, val min MAE | −0.60 [−0.80, −0.42] | −0.47 [−0.70, −0.26] | −0.077 [−0.096, −0.070] | +0.021 [+0.013, +0.030] | shrinks (still > 0) | shrinks (still > 0) |
| UNet++ matched lrB seed 1 | argmax + 1-px erosion | −1.58 [−2.12, −1.11] | −0.01 [−0.49, +0.41] | −0.215 [−0.243, −0.171] | −0.068 [−0.093, −0.041] | reversed (now significant under-estimation) | disappears (CI includes 0; point estimate reversed sign) |
| UNet++ matched lrB seed 1 | argmax + 2-px erosion | −2.90 [−3.92, −2.01] | +1.08 [+0.32, +1.84] | −0.452 [−0.511, −0.342] | −0.272 [−0.320, −0.218] | reversed (now significant under-estimation) | reversed (now significant under-estimation) |
| UNet++ matched lrB seed 2 | threshold, val min abs(bias) | −0.80 [−1.09, −0.54] | −0.48 [−0.82, −0.18] | −0.108 [−0.143, −0.093] | +0.010 [−0.003, +0.023] | disappears (CI includes 0) | disappears (CI includes 0) |
| UNet++ matched lrB seed 2 | threshold, val min MAE | −0.65 [−0.89, −0.44] | −0.47 [−0.76, −0.20] | −0.086 [−0.114, −0.074] | +0.011 [+0.001, +0.022] | shrinks (still > 0) | shrinks (still > 0) |
| UNet++ matched lrB seed 2 | argmax + 1-px erosion | −1.57 [−2.12, −1.09] | −0.09 [−0.68, +0.39] | −0.221 [−0.246, −0.186] | −0.079 [−0.109, −0.049] | reversed (now significant under-estimation) | disappears (CI includes 0; point estimate reversed sign) |
| UNet++ matched lrB seed 2 | argmax + 2-px erosion | −2.90 [−3.94, −1.98] | +0.87 [+0.02, +1.72] | −0.469 [−0.522, −0.382] | −0.271 [−0.329, −0.212] | reversed (now significant under-estimation) | reversed (now significant under-estimation) |
| DeepLabV3+ matched lrB seed 0 | threshold, val min abs(bias) | −0.41 [−0.55, −0.28] | −0.24 [−0.44, −0.05] | −0.054 [−0.067, −0.048] | +0.007 [−0.001, +0.014] | disappears (CI includes 0) | shrinks (still > 0) |
| DeepLabV3+ matched lrB seed 0 | threshold, val min MAE | −0.46 [−0.63, −0.32] | −0.26 [−0.48, −0.05] | −0.060 [−0.077, −0.054] | +0.007 [−0.002, +0.015] | disappears (CI includes 0) | shrinks (still > 0) |
| DeepLabV3+ matched lrB seed 0 | argmax + 1-px erosion | −1.53 [−2.08, −1.06] | +0.19 [−0.27, +0.60] | −0.215 [−0.245, −0.178] | −0.057 [−0.084, −0.030] | reversed (now significant under-estimation) | disappears (CI includes 0; point estimate reversed sign) |
| DeepLabV3+ matched lrB seed 0 | argmax + 2-px erosion | −2.80 [−3.84, −1.91] | +1.19 [+0.36, +1.96] | −0.456 [−0.522, −0.365] | −0.249 [−0.295, −0.197] | reversed (now significant under-estimation) | reversed (now significant under-estimation) |
| DeepLabV3+ matched lrB seed 1 | threshold, val min abs(bias) | −0.42 [−0.57, −0.29] | −0.23 [−0.41, −0.05] | −0.055 [−0.082, −0.048] | +0.005 [−0.001, +0.011] | disappears (CI includes 0) | disappears (CI includes 0) |
| DeepLabV3+ matched lrB seed 1 | threshold, val min MAE | −0.52 [−0.70, −0.36] | −0.23 [−0.47, −0.02] | −0.068 [−0.102, −0.060] | +0.006 [−0.002, +0.014] | disappears (CI includes 0) | disappears (CI includes 0) |
| DeepLabV3+ matched lrB seed 1 | argmax + 1-px erosion | −1.55 [−2.09, −1.08] | +0.39 [+0.01, +0.76] | −0.231 [−0.263, −0.212] | −0.057 [−0.082, −0.032] | reversed (now significant under-estimation) | reversed (now significant under-estimation) |
| DeepLabV3+ matched lrB seed 1 | argmax + 2-px erosion | −2.82 [−3.86, −1.94] | +1.66 [+1.10, +2.30] | −0.493 [−0.544, −0.441] | −0.246 [−0.286, −0.204] | reversed (now significant under-estimation) | reversed (now significant under-estimation) |
| DeepLabV3+ matched lrB seed 2 | threshold, val min abs(bias) | −0.32 [−0.44, −0.22] | −0.04 [−0.21, +0.14] | −0.046 [−0.067, −0.040] | +0.002 [−0.003, +0.008] | not significant at argmax already; now significantly negative | not significant at argmax already; still not significant |
| DeepLabV3+ matched lrB seed 2 | threshold, val min MAE | −0.48 [−0.65, −0.33] | +0.02 [−0.19, +0.25] | −0.070 [−0.102, −0.061] | −0.001 [−0.009, +0.007] | not significant at argmax already; now significantly negative | not significant at argmax already; still not significant |
| DeepLabV3+ matched lrB seed 2 | argmax + 1-px erosion | −1.51 [−2.04, −1.05] | +1.02 [+0.65, +1.49] | −0.244 [−0.273, −0.216] | −0.088 [−0.108, −0.067] | not significant at argmax already; now significantly negative | not significant at argmax already; now significantly negative |
| DeepLabV3+ matched lrB seed 2 | argmax + 2-px erosion | −2.74 [−3.74, −1.88] | +2.25 [+1.51, +3.15] | −0.519 [−0.576, −0.451] | −0.298 [−0.330, −0.263] | not significant at argmax already; now significantly negative | not significant at argmax already; now significantly negative |

Verdict rule (fixed before reading the numbers): *disappears* = the operating point's bootstrap CI includes 0; *shrinks* = its CI still excludes 0 but the paired change is negative with a CI that excludes 0; *persists* = otherwise; *reversed* = the CI excludes 0 on the negative side (significant under-estimation). Where the argmax CI already includes 0, the verdict says so (matched DeepLabV3+ seed 2 has no significant argmax bias or slope).

## 5. Matched CNNs: mean ± SD over 3 seeds (range)

| Model | Operating point | Thresholds (s0, s1, s2) | Leaf IoU | MAE (pp) | Bias (pp) | Slope | QWK |
|---|---|---|---|---|---|---|---|
| UNet++ matched | argmax (paper rule) | — | 0.759 ± 0.015 (0.746 to 0.776) | 0.83 ± 0.29 (0.50 to 1.02) | +0.82 ± 0.30 (+0.47 to +1.01) | 0.125 ± 0.040 (0.079 to 0.155) | 0.898 ± 0.035 (0.878 to 0.938) |
| UNet++ matched | threshold, val min abs(bias) | 0.89, 0.996, 0.98 | 0.770 ± 0.013 (0.755 to 0.780) | 0.48 ± 0.08 (0.39 to 0.53) | +0.14 ± 0.07 (+0.07 to +0.21) | 0.033 ± 0.013 (0.023 to 0.047) | 0.927 ± 0.000 (0.927 to 0.927) |
| UNet++ matched | threshold, val min MAE | 0.82, 0.98, 0.96 | 0.771 ± 0.012 (0.757 to 0.780) | 0.48 ± 0.09 (0.38 to 0.55) | +0.31 ± 0.11 (+0.18 to +0.38) | 0.057 ± 0.016 (0.040 to 0.069) | 0.951 ± 0.042 (0.927 to 1.000) |
| UNet++ matched | argmax + 1-px erosion | — | 0.672 ± 0.014 (0.661 to 0.687) | 0.98 ± 0.06 (0.93 to 1.04) | −0.74 ± 0.26 (−1.04 to −0.56) | −0.097 ± 0.046 (−0.150 to −0.066) | 0.848 ± 0.039 (0.803 to 0.870) |
| UNet++ matched | argmax + 2-px erosion | — | 0.470 ± 0.016 (0.452 to 0.483) | 2.08 ± 0.21 (1.89 to 2.30) | −2.03 ± 0.23 (−2.30 to −1.89) | −0.345 ± 0.056 (−0.409 to −0.311) | 0.661 ± 0.012 (0.654 to 0.675) |
| DeepLabV3+ matched | argmax (paper rule) | — | 0.725 ± 0.010 (0.718 to 0.736) | 0.62 ± 0.25 (0.36 to 0.87) | +0.47 ± 0.32 (+0.13 to +0.77) | 0.097 ± 0.068 (0.025 to 0.161) | 0.914 ± 0.032 (0.878 to 0.938) |
| DeepLabV3+ matched | threshold, val min abs(bias) | 0.78, 0.8, 0.74 | 0.730 ± 0.008 (0.723 to 0.738) | 0.46 ± 0.16 (0.32 to 0.63) | +0.09 ± 0.28 (−0.19 to +0.36) | 0.046 ± 0.064 (−0.021 to 0.107) | 0.892 ± 0.030 (0.874 to 0.927) |
| DeepLabV3+ matched | threshold, val min MAE | 0.81, 0.85, 0.83 | 0.729 ± 0.006 (0.724 to 0.736) | 0.47 ± 0.12 (0.39 to 0.61) | −0.01 ± 0.33 (−0.35 to +0.31) | 0.031 ± 0.073 (−0.045 to 0.100) | 0.892 ± 0.030 (0.874 to 0.927) |
| DeepLabV3+ matched | argmax + 1-px erosion | — | 0.658 ± 0.009 (0.648 to 0.665) | 1.16 ± 0.19 (1.03 to 1.38) | −1.06 ± 0.31 (−1.38 to −0.76) | −0.133 ± 0.082 (−0.219 to −0.055) | 0.803 ± 0.000 (0.803 to 0.803) |
| DeepLabV3+ matched | argmax + 2-px erosion | — | 0.461 ± 0.020 (0.438 to 0.473) | 2.33 ± 0.28 (2.06 to 2.61) | −2.32 ± 0.29 (−2.61 to −2.03) | −0.392 ± 0.100 (−0.494 to −0.295) | 0.636 ± 0.031 (0.600 to 0.654) |

## 6. Healthy test leaves (21 local, abaxial face)

| Model | Operating point | Leaves with ≥ 1 predicted lesion px | Leaves ≥ 0.5% | Leaves ≥ 5% | Max (%) |
|---|---|---|---|---|---|
| SegFormer-B5 (published) | argmax (paper rule) | 17/21 | 0/21 | 0/21 | 0.355 |
| SegFormer-B5 (published) | threshold, val min abs(bias) | 16/21 | 0/21 | 0/21 | 0.258 |
| SegFormer-B5 (published) | threshold, val min MAE | 16/21 | 0/21 | 0/21 | 0.264 |
| SegFormer-B5 (published) | argmax + 1-px erosion | 15/21 | 0/21 | 0/21 | 0.163 |
| SegFormer-B5 (published) | argmax + 2-px erosion | 14/21 | 0/21 | 0/21 | 0.087 |
| UNet++ (published) | argmax (paper rule) | 15/21 | 0/21 | 0/21 | 0.155 |
| UNet++ (published) | threshold, val min abs(bias) | 6/21 | 0/21 | 0/21 | 0.089 |
| UNet++ (published) | threshold, val min MAE | 6/21 | 0/21 | 0/21 | 0.089 |
| UNet++ (published) | argmax + 1-px erosion | 7/21 | 0/21 | 0/21 | 0.096 |
| UNet++ (published) | argmax + 2-px erosion | 5/21 | 0/21 | 0/21 | 0.055 |
| DeepLabV3+ (published) | argmax (paper rule) | 7/21 | 0/21 | 0/21 | 0.085 |
| DeepLabV3+ (published) | threshold, val min abs(bias) | 4/21 | 0/21 | 0/21 | 0.023 |
| DeepLabV3+ (published) | threshold, val min MAE | 3/21 | 0/21 | 0/21 | 0.018 |
| DeepLabV3+ (published) | argmax + 1-px erosion | 5/21 | 0/21 | 0/21 | 0.035 |
| DeepLabV3+ (published) | argmax + 2-px erosion | 2/21 | 0/21 | 0/21 | 0.011 |
| UNet++ matched lrB seed 0 | argmax (paper rule) | 7/21 | 0/21 | 0/21 | 0.122 |
| UNet++ matched lrB seed 0 | threshold, val min abs(bias) | 6/21 | 0/21 | 0/21 | 0.096 |
| UNet++ matched lrB seed 0 | threshold, val min MAE | 6/21 | 0/21 | 0/21 | 0.102 |
| UNet++ matched lrB seed 0 | argmax + 1-px erosion | 6/21 | 0/21 | 0/21 | 0.035 |
| UNet++ matched lrB seed 0 | argmax + 2-px erosion | 1/21 | 0/21 | 0/21 | 0.005 |
| UNet++ matched lrB seed 1 | argmax (paper rule) | 5/21 | 0/21 | 0/21 | 0.032 |
| UNet++ matched lrB seed 1 | threshold, val min abs(bias) | 2/21 | 0/21 | 0/21 | 0.021 |
| UNet++ matched lrB seed 1 | threshold, val min MAE | 2/21 | 0/21 | 0/21 | 0.022 |
| UNet++ matched lrB seed 1 | argmax + 1-px erosion | 3/21 | 0/21 | 0/21 | 0.012 |
| UNet++ matched lrB seed 1 | argmax + 2-px erosion | 1/21 | 0/21 | 0/21 | 0.001 |
| UNet++ matched lrB seed 2 | argmax (paper rule) | 5/21 | 0/21 | 0/21 | 0.036 |
| UNet++ matched lrB seed 2 | threshold, val min abs(bias) | 4/21 | 0/21 | 0/21 | 0.021 |
| UNet++ matched lrB seed 2 | threshold, val min MAE | 5/21 | 0/21 | 0/21 | 0.027 |
| UNet++ matched lrB seed 2 | argmax + 1-px erosion | 4/21 | 0/21 | 0/21 | 0.012 |
| UNet++ matched lrB seed 2 | argmax + 2-px erosion | 0/21 | 0/21 | 0/21 | 0.000 |
| DeepLabV3+ matched lrB seed 0 | argmax (paper rule) | 8/21 | 0/21 | 0/21 | 0.051 |
| DeepLabV3+ matched lrB seed 0 | threshold, val min abs(bias) | 6/21 | 0/21 | 0/21 | 0.036 |
| DeepLabV3+ matched lrB seed 0 | threshold, val min MAE | 5/21 | 0/21 | 0/21 | 0.036 |
| DeepLabV3+ matched lrB seed 0 | argmax + 1-px erosion | 3/21 | 0/21 | 0/21 | 0.023 |
| DeepLabV3+ matched lrB seed 0 | argmax + 2-px erosion | 1/21 | 0/21 | 0/21 | 0.005 |
| DeepLabV3+ matched lrB seed 1 | argmax (paper rule) | 7/21 | 0/21 | 0/21 | 0.013 |
| DeepLabV3+ matched lrB seed 1 | threshold, val min abs(bias) | 4/21 | 0/21 | 0/21 | 0.007 |
| DeepLabV3+ matched lrB seed 1 | threshold, val min MAE | 4/21 | 0/21 | 0/21 | 0.005 |
| DeepLabV3+ matched lrB seed 1 | argmax + 1-px erosion | 1/21 | 0/21 | 0/21 | 0.001 |
| DeepLabV3+ matched lrB seed 1 | argmax + 2-px erosion | 0/21 | 0/21 | 0/21 | 0.000 |
| DeepLabV3+ matched lrB seed 2 | argmax (paper rule) | 3/21 | 0/21 | 0/21 | 0.024 |
| DeepLabV3+ matched lrB seed 2 | threshold, val min abs(bias) | 3/21 | 0/21 | 0/21 | 0.014 |
| DeepLabV3+ matched lrB seed 2 | threshold, val min MAE | 3/21 | 0/21 | 0/21 | 0.011 |
| DeepLabV3+ matched lrB seed 2 | argmax + 1-px erosion | 2/21 | 0/21 | 0/21 | 0.005 |
| DeepLabV3+ matched lrB seed 2 | argmax + 2-px erosion | 0/21 | 0/21 | 0/21 | 0.000 |

Only 21 of the 54 healthy rust-test leaves of the paper are available locally (revision brief, fact 8); counts are abaxial face only (the rust leaf-level rule), which is why the argmax any-pixel counts equal `results/agreement` Section 10 (abaxial) rather than the both-face numbers of fact 10.

## 7. Signed boundary distance, predicted vs reference lesion contours (test, 21 leaves; px; + = prediction outside reference)

| Model | Operating point | Contour px | Median [CI] | IQR (Q1 to Q3) | Mean [CI] | Share > 0 | Share ≥ 2 px outside | Share ≥ 2 px inside | Attached only: median; IQR; mean | Reverse (ref contour): median; IQR; mean |
|---|---|---|---|---|---|---|---|---|---|---|
| SegFormer-B5 (published) | argmax (paper rule) | 25044 | 1.00 [0.00, 1.00] | 0.00 to 1.00 | +2.93 [+1.95, +4.23] | 0.513 | 0.183 | 0.028 | 0.00; 0.00 to 1.00; +0.64 | 0.41; 0.00 to 1.24; +0.51 |
| SegFormer-B5 (published) | threshold, val min abs(bias) | 22896 | 0.00 [0.00, 0.00] | −0.41 to 1.00 | +1.46 [+0.78, +2.36] | 0.303 | 0.102 | 0.049 | 0.00; −0.41 to 1.00; +0.05 | 0.00; −1.00 to 0.41; −0.17 |
| SegFormer-B5 (published) | threshold, val min MAE | 23067 | 0.00 [0.00, 0.00] | −0.41 to 1.00 | +1.58 [+0.87, +2.51] | 0.320 | 0.108 | 0.047 | 0.00; −0.41 to 1.00; +0.10 | 0.00; −1.00 to 0.41; −0.12 |
| SegFormer-B5 (published) | argmax + 1-px erosion | 20446 | −0.41 [−1.00, −0.41] | −1.24 to 0.00 | +0.60 [+0.03, +1.30] | 0.173 | 0.087 | 0.090 | −0.41; −1.24 to 0.00; −0.48 | −1.00; −1.00 to 0.00; −0.73 |
| SegFormer-B5 (published) | argmax + 2-px erosion | 15991 | −1.83 [−2.00, −1.83] | −2.61 to −1.00 | −1.28 [−1.68, −0.81] | 0.085 | 0.051 | 0.473 | −1.83; −2.61 to −1.00; −1.63 | −2.00; −2.83 to −1.00; −3.43 |
| UNet++ (published) | argmax (paper rule) | 25098 | 1.00 [1.00, 1.41] | 0.00 to 2.24 | +2.91 [+1.91, +4.17] | 0.685 | 0.373 | 0.034 | 1.00; 0.00 to 2.00; +1.46 | 1.00; 0.00 to 2.00; +0.27 |
| UNet++ (published) | threshold, val min abs(bias) | 21742 | 0.00 [0.00, 1.00] | −0.41 to 1.00 | +0.93 [+0.39, +1.56] | 0.437 | 0.187 | 0.081 | 0.00; −0.41 to 1.00; +0.43 | 0.00; −1.00 to 1.00; −2.02 |
| UNet++ (published) | threshold, val min MAE | 21742 | 0.00 [0.00, 1.00] | −0.41 to 1.00 | +0.93 [+0.39, +1.56] | 0.437 | 0.187 | 0.081 | 0.00; −0.41 to 1.00; +0.43 | 0.00; −1.00 to 1.00; −2.02 |
| UNet++ (published) | argmax + 1-px erosion | 20717 | 0.00 [0.00, 0.00] | −1.00 to 1.00 | +0.91 [+0.23, +1.73] | 0.358 | 0.175 | 0.101 | 0.00; −1.00 to 1.00; +0.23 | 0.00; −1.41 to 1.00; −1.71 |
| UNet++ (published) | argmax + 2-px erosion | 16902 | −1.00 [−1.24, −1.00] | −2.00 to 0.00 | −0.60 [−1.11, −0.01] | 0.178 | 0.095 | 0.305 | −1.00; −2.00 to 0.00; −0.92 | −1.41; −2.83 to 0.00; −4.92 |
| DeepLabV3+ (published) | argmax (paper rule) | 25812 | 1.00 [1.00, 1.41] | 0.00 to 2.24 | +2.37 [+1.37, +3.52] | 0.630 | 0.382 | 0.060 | 1.00; 0.00 to 2.24; +1.16 | 1.00; −1.00 to 2.00; −1.25 |
| DeepLabV3+ (published) | threshold, val min abs(bias) | 22323 | 0.00 [0.00, 1.00] | −1.00 to 1.41 | +0.44 [−0.13, +1.09] | 0.433 | 0.207 | 0.133 | 0.00; −1.00 to 1.00; −0.03 | 0.00; −1.41 to 1.00; −3.03 |
| DeepLabV3+ (published) | threshold, val min MAE | 21700 | 0.00 [0.00, 0.00] | −1.00 to 1.00 | +0.14 [−0.36, +0.73] | 0.391 | 0.178 | 0.154 | 0.00; −1.24 to 1.00; −0.23 | 0.00; −2.00 to 1.00; −3.47 |
| DeepLabV3+ (published) | argmax + 1-px erosion | 21747 | 0.00 [−0.41, 0.00] | −1.24 to 1.00 | +0.55 [−0.23, +1.47] | 0.374 | 0.188 | 0.154 | 0.00; −1.24 to 1.00; +0.03 | 0.00; −2.00 to 1.00; −3.36 |
| DeepLabV3+ (published) | argmax + 2-px erosion | 17802 | −1.24 [−1.83, −1.00] | −2.61 to 0.00 | −0.98 [−1.57, −0.28] | 0.185 | 0.092 | 0.358 | −1.24; −2.61 to 0.00; −1.28 | −2.00; −3.00 to 0.00; −5.97 |
| UNet++ matched lrB seed 0 | argmax (paper rule) | 23582 | 0.00 [0.00, 0.00] | −0.41 to 1.00 | +1.56 [+0.79, +2.61] | 0.401 | 0.132 | 0.040 | 0.00; −0.41 to 1.00; +0.32 | 0.00; 0.00 to 1.00; +0.07 |
| UNet++ matched lrB seed 0 | threshold, val min abs(bias) | 22742 | 0.00 [0.00, 0.00] | −0.41 to 1.00 | +0.98 [+0.35, +1.84] | 0.305 | 0.102 | 0.054 | 0.00; −0.41 to 1.00; +0.03 | 0.00; −1.00 to 0.41; −0.48 |
| UNet++ matched lrB seed 0 | threshold, val min MAE | 22966 | 0.00 [0.00, 0.00] | −0.41 to 1.00 | +1.13 [+0.47, +2.03] | 0.331 | 0.109 | 0.049 | 0.00; −0.41 to 1.00; +0.11 | 0.00; −1.00 to 1.00; −0.31 |
| UNet++ matched lrB seed 0 | argmax + 1-px erosion | 19634 | −1.00 [−1.00, −1.00] | −1.24 to 0.00 | −0.13 [−0.65, +0.55] | 0.130 | 0.066 | 0.142 | −1.00; −1.83 to 0.00; −0.82 | −1.00; −1.41 to 0.00; −1.62 |
| UNet++ matched lrB seed 0 | argmax + 2-px erosion | 15180 | −2.00 [−2.16, −2.00] | −2.61 to −1.24 | −1.80 [−2.09, −1.42] | 0.065 | 0.035 | 0.590 | −2.00; −3.00 to −1.24; −1.99 | −2.24; −3.00 to −1.41; −3.81 |
| UNet++ matched lrB seed 1 | argmax (paper rule) | 24735 | 0.00 [0.00, 1.00] | 0.00 to 1.00 | +2.68 [+1.64, +4.15] | 0.481 | 0.173 | 0.027 | 0.00; 0.00 to 1.00; +0.60 | 0.41; 0.00 to 1.00; +0.54 |
| UNet++ matched lrB seed 1 | threshold, val min abs(bias) | 22500 | 0.00 [0.00, 0.00] | −0.41 to 1.00 | +1.17 [+0.47, +2.18] | 0.280 | 0.101 | 0.051 | 0.00; −0.41 to 1.00; +0.06 | 0.00; −1.00 to 0.41; −0.45 |
| UNet++ matched lrB seed 1 | threshold, val min MAE | 23109 | 0.00 [0.00, 0.00] | −0.41 to 1.00 | +1.54 [+0.76, +2.69] | 0.332 | 0.117 | 0.041 | 0.00; −0.41 to 1.00; +0.22 | 0.00; 0.00 to 1.00; −0.04 |
| UNet++ matched lrB seed 1 | argmax + 1-px erosion | 20489 | −1.00 [−1.00, −0.41] | −1.24 to 0.00 | +0.62 [−0.12, +1.67] | 0.161 | 0.087 | 0.102 | −1.00; −1.24 to 0.00; −0.53 | −1.00; −1.41 to 0.00; −0.81 |
| UNet++ matched lrB seed 1 | argmax + 2-px erosion | 15966 | −2.00 [−2.00, −1.83] | −2.61 to −1.00 | −1.30 [−1.77, −0.66] | 0.086 | 0.055 | 0.517 | −2.00; −2.61 to −1.00; −1.67 | −2.00; −2.83 to −1.00; −3.44 |
| UNet++ matched lrB seed 2 | argmax (paper rule) | 24427 | 0.00 [0.00, 1.00] | 0.00 to 1.00 | +2.41 [+1.54, +3.65] | 0.481 | 0.182 | 0.029 | 0.00; 0.00 to 1.00; +0.68 | 0.41; 0.00 to 1.00; +0.34 |
| UNet++ matched lrB seed 2 | threshold, val min abs(bias) | 22652 | 0.00 [0.00, 0.00] | −0.41 to 1.00 | +1.29 [+0.54, +2.37] | 0.304 | 0.120 | 0.054 | 0.00; −0.41 to 1.00; +0.10 | 0.00; −1.00 to 0.41; −0.48 |
| UNet++ matched lrB seed 2 | threshold, val min MAE | 22991 | 0.00 [0.00, 0.00] | −0.41 to 1.00 | +1.51 [+0.73, +2.65] | 0.334 | 0.130 | 0.047 | 0.00; −0.41 to 1.00; +0.21 | 0.00; −1.00 to 1.00; −0.14 |
| UNet++ matched lrB seed 2 | argmax + 1-px erosion | 20497 | −1.00 [−1.00, −0.41] | −1.24 to 0.00 | +0.57 [−0.13, +1.56] | 0.180 | 0.104 | 0.107 | −1.00; −1.24 to 0.00; −0.44 | −1.00; −1.41 to 0.00; −0.87 |
| UNet++ matched lrB seed 2 | argmax + 2-px erosion | 16135 | −2.00 [−2.00, −1.83] | −2.61 to −1.00 | −1.16 [−1.65, −0.47] | 0.105 | 0.069 | 0.506 | −2.00; −2.61 to −1.00; −1.58 | −2.00; −2.83 to −1.00; −3.69 |
| DeepLabV3+ matched lrB seed 0 | argmax (paper rule) | 23931 | 1.00 [0.00, 1.00] | 0.00 to 1.41 | +1.98 [+1.20, +3.07] | 0.511 | 0.211 | 0.050 | 0.00; −0.41 to 1.00; +0.61 | 0.41; 0.00 to 1.24; +0.02 |
| DeepLabV3+ matched lrB seed 0 | threshold, val min abs(bias) | 22832 | 0.00 [0.00, 0.00] | −0.41 to 1.00 | +1.23 [+0.64, +1.97] | 0.432 | 0.162 | 0.067 | 0.00; −0.41 to 1.00; +0.32 | 0.00; −1.00 to 1.00; −0.41 |
| DeepLabV3+ matched lrB seed 0 | threshold, val min MAE | 22714 | 0.00 [0.00, 0.00] | −0.41 to 1.00 | +1.11 [+0.54, +1.80] | 0.422 | 0.156 | 0.069 | 0.00; −0.41 to 1.00; +0.26 | 0.00; −1.00 to 1.00; −0.45 |
| DeepLabV3+ matched lrB seed 0 | argmax + 1-px erosion | 19778 | −0.41 [−1.00, −0.41] | −1.24 to 0.00 | −0.07 [−0.53, +0.48] | 0.200 | 0.088 | 0.161 | −0.41; −1.24 to 0.00; −0.53 | −1.00; −2.00 to 0.00; −1.83 |
| DeepLabV3+ matched lrB seed 0 | argmax + 2-px erosion | 15895 | −1.83 [−2.00, −1.83] | −2.61 to −1.00 | −1.53 [−1.97, −0.97] | 0.092 | 0.054 | 0.477 | −1.83; −2.61 to −1.00; −1.69 | −2.00; −3.00 to −1.00; −4.89 |
| DeepLabV3+ matched lrB seed 1 | argmax (paper rule) | 24311 | 0.00 [0.00, 1.00] | −0.41 to 1.41 | +1.91 [+0.97, +3.18] | 0.486 | 0.201 | 0.063 | 0.00; −0.41 to 1.00; +0.39 | 0.00; −1.00 to 1.00; −0.01 |
| DeepLabV3+ matched lrB seed 1 | threshold, val min abs(bias) | 23110 | 0.00 [0.00, 0.00] | −1.00 to 1.00 | +1.11 [+0.41, +2.05] | 0.404 | 0.151 | 0.083 | 0.00; −1.00 to 1.00; +0.09 | 0.00; −1.00 to 1.00; −0.78 |
| DeepLabV3+ matched lrB seed 1 | threshold, val min MAE | 22832 | 0.00 [0.00, 0.00] | −1.00 to 1.00 | +0.91 [+0.28, +1.79] | 0.385 | 0.140 | 0.089 | 0.00; −1.00 to 1.00; +0.01 | 0.00; −1.00 to 1.00; −0.86 |
| DeepLabV3+ matched lrB seed 1 | argmax + 1-px erosion | 19935 | −1.00 [−1.00, −0.41] | −1.83 to 0.00 | −0.14 [−0.66, +0.55] | 0.187 | 0.078 | 0.189 | −1.00; −1.83 to 0.00; −0.76 | −1.00; −2.00 to 0.00; −2.10 |
| DeepLabV3+ matched lrB seed 1 | argmax + 2-px erosion | 15814 | −2.00 [−2.00, −1.83] | −3.00 to −1.00 | −1.71 [−2.05, −1.27] | 0.077 | 0.037 | 0.502 | −2.00; −3.00 to −1.00; −1.94 | −2.24; −3.16 to −1.00; −4.90 |
| DeepLabV3+ matched lrB seed 2 | argmax (paper rule) | 23650 | 0.00 [0.00, 0.00] | −0.41 to 1.00 | +1.24 [+0.59, +2.17] | 0.401 | 0.136 | 0.079 | 0.00; −0.41 to 1.00; +0.07 | 0.00; −1.00 to 1.00; −0.64 |
| DeepLabV3+ matched lrB seed 2 | threshold, val min abs(bias) | 22854 | 0.00 [0.00, 0.00] | −1.00 to 1.00 | +0.71 [+0.19, +1.45] | 0.334 | 0.107 | 0.095 | 0.00; −1.00 to 1.00; −0.16 | 0.00; −1.00 to 1.00; −1.13 |
| DeepLabV3+ matched lrB seed 2 | threshold, val min MAE | 22455 | 0.00 [0.00, 0.00] | −1.00 to 1.00 | +0.52 [+0.02, +1.23] | 0.301 | 0.095 | 0.105 | 0.00; −1.00 to 1.00; −0.28 | 0.00; −1.00 to 0.41; −1.25 |
| DeepLabV3+ matched lrB seed 2 | argmax + 1-px erosion | 19412 | −1.00 [−1.00, −1.00] | −1.83 to 0.00 | −0.65 [−1.02, −0.15] | 0.125 | 0.055 | 0.217 | −1.00; −1.83 to 0.00; −1.09 | −1.00; −2.00 to 0.00; −2.46 |
| DeepLabV3+ matched lrB seed 2 | argmax + 2-px erosion | 15207 | −2.00 [−2.16, −2.00] | −3.00 to −1.24 | −2.12 [−2.36, −1.81] | 0.055 | 0.029 | 0.596 | −2.00; −3.00 to −1.24; −2.32 | −2.24; −3.16 to −1.41; −5.57 |

The mean is pulled up by spurious predicted blobs far from any reference lesion (compare "all" with "attached"), and the reverse mean is pulled down by missed reference lesions far from any prediction; the median is the robust summary. Per-leaf medians and the share of distances beyond ±10 px are in `results.json`.

## 8. Context only: thresholds chosen on the test set (NOT a valid estimate)

| Model | Test-optimal t (min \|bias\|); bias; slope | Test-optimal t (min MAE); MAE; slope | Slope over the whole grid (min to max) | Slope at t = 0.5 / 0.7 / 0.9 / 0.95 / 0.99 / 0.999 |
|---|---|---|---|---|
| SegFormer-B5 (published) | 0.98; +0.03; 0.060 | 0.99; 0.33; 0.032 | −0.148 to 0.375 | 0.207 / 0.176 / 0.124 / 0.095 / 0.032 / −0.055 |
| UNet++ (published) | 0.92; +0.04; 0.098 | 0.94; 0.65; 0.061 | −2.000 to 0.669 | 0.307 / 0.241 / 0.123 / 0.038 / −0.270 / −1.686 |
| DeepLabV3+ (published) | 0.93; +0.00; 0.051 | 0.92; 0.77; 0.063 | −0.572 to 0.672 | 0.243 / 0.179 / 0.080 / 0.025 / −0.099 / −0.311 |
| UNet++ matched lrB seed 0 | 0.92; +0.01; 0.014 | 0.82; 0.38; 0.040 | −0.316 to 0.211 | 0.078 / 0.058 / 0.020 / 0.001 / −0.051 / −0.153 |
| UNet++ matched lrB seed 1 | 0.998; +0.03; 0.012 | 0.993; 0.49; 0.041 | −0.085 to 0.241 | 0.140 / 0.123 / 0.099 / 0.083 / 0.048 / −0.008 |
| UNet++ matched lrB seed 2 | 0.992; −0.00; 0.018 | 0.97; 0.53; 0.060 | −0.199 to 0.281 | 0.152 / 0.131 / 0.096 / 0.076 / 0.025 / −0.067 |
| DeepLabV3+ matched lrB seed 0 | 0.93; −0.02; 0.054 | 0.93; 0.52; 0.054 | −0.244 to 0.369 | 0.159 / 0.124 / 0.070 / 0.039 / −0.029 / −0.133 |
| DeepLabV3+ matched lrB seed 1 | 0.85; +0.00; 0.038 | 0.85; 0.41; 0.038 | −0.276 to 0.305 | 0.105 / 0.071 / 0.020 / −0.010 / −0.077 / −0.174 |
| DeepLabV3+ matched lrB seed 2 | 0.59; +0.00; 0.007 | 0.74; 0.32; −0.021 | −0.422 to 0.229 | 0.023 / −0.013 / −0.073 / −0.107 / −0.178 / −0.293 |

This shows how close the validation-chosen thresholds came to the best possible single threshold on these 21 test leaves; it is optimistic by construction and must not be reported as performance.

## 9. Answer: does the rust over-estimation disappear, shrink or persist?

### SegFormer-B5 (primary; published model, 21 test leaves)

- **Validation-calibrated threshold.** Validation chose t = 0.97 (min |bias|) and t = 0.96 (min MAE) on p(lesion); on validation the bias fell from +0.57 to +0.01 pp. Applied once to test (t = 0.97): bias +0.99 [+0.48, +1.62] → **+0.12 [−0.09, +0.41] pp** (paired change −0.87 [−1.24, −0.55]); MAE 0.99 → 0.34 [0.17, 0.58] pp; leaf IoU 0.767 → 0.789 [0.769, 0.810] (paired +0.022 [+0.011, +0.033]); proportional-bias slope 0.208 [0.154, 0.261] → **0.075 [0.036, 0.114]** (bootstrap [0.028, 0.126]; p < 0.001; paired change −0.133 [−0.168, −0.120], a 64% reduction); grade accuracy 0.905 → 0.952, QWK 0.878 → 0.938, over-graded leaves 2 → 1 (none under-graded), leaves over-estimated 20/21 → 7/21. The MAE-chosen threshold (t = 0.96) gives the same picture: bias +0.19 [−0.05, +0.50], MAE 0.36, IoU 0.790, slope 0.086 [0.045, 0.127].
  **Verdict: the constant bias disappears; the proportional over-estimation shrinks by 64% but persists** (slope CI still excludes 0).
- **Erosion.** 1-px erosion: bias −0.62 [−0.88, −0.30] pp (now significant under-estimation), slope −0.003 [−0.068, 0.063] (proportional component gone), leaf IoU 0.692 (paired −0.075 [−0.103, −0.045]), MAE 0.79, 3 leaves under-graded. 2-px erosion over-corrects: bias −1.94 [−2.54, −1.38], slope −0.240 [−0.332, −0.147], IoU 0.481. Removing one boundary pixel ring removes the proportional component entirely but also a fixed amount of area from every lesion, so small lesions become under-estimated; the excess predicted area is therefore less than one pixel ring wide on average, which a probability threshold can trim more finely.
- **Signed boundary distance (argmax).** Of the predicted lesion contour pixels (n = 25044, 21 leaves), 51% lie outside the reference lesion, 32% on its contour (convention of this file; see Verifier corrections: with a plain distance-to-contour definition 39% lie on the contour) and only 3% ≥ 2 px inside; 18% are ≥ 2 px outside. Because just over half are outside, the pooled median is 1.00 px, but it sits on the 0/1 step of a discrete distribution (bootstrap CI [0.00, 1.00]; IQR 0.00 to 1.00 px; median of per-leaf medians 0.00), so the shares and the mean are the more informative summaries. The mean (+2.93 px [+1.95, +4.23]) is inflated by spurious blobs; for contours of components that touch a reference lesion it is +0.64 px [+0.35, +0.97] (median 0.00, IQR 0.00 to 1.00). Reverse direction: reference contour pixels lie a median 0.41 px inside the prediction (IQR 0.00 to 1.24). So the argmax prediction is displaced outward by a fraction of a pixel to one pixel, systematically. After the validation-calibrated threshold the median is 0.00 [0.00, 0.00] (IQR −0.41 to 1.00), 30% of contour pixels lie outside, the attached-contour mean is +0.05 px [−0.17, +0.31] and the reverse median is 0.00 (IQR −1.00 to 0.41): the systematic outward shift is removed.
- **Healthy test leaves (21, abaxial).** Leaves with ≥ 1 predicted lesion pixel: argmax 17/21 → threshold 16/21 (max 0.36% → 0.26%). No healthy leaf reaches 0.5% for any of the 9 models under any operating point, so the calibrated threshold costs no specificity on these leaves.

### Published CNNs and matched-protocol CNNs

- **UNet++ (published)** (t = 0.89): bias +1.75 [+0.88, +2.75] → +0.30 [−0.15, +0.83]; slope 0.311 → 0.134 [0.053, 0.215] (bootstrap [0.071, 0.277]); MAE 1.77 → 0.77; IoU 0.647 → 0.685; QWK 0.788 → 0.874. Bias: disappears (CI includes 0); slope: shrinks (still > 0).
- **DeepLabV3+ (published)** (t = 0.93): bias +1.41 [+0.56, +2.46] → +0.00 [−0.49, +0.62]; slope 0.247 → 0.051 [−0.068, 0.171] (bootstrap [−0.025, 0.253]); MAE 1.47 → 0.78; IoU 0.632 → 0.636; QWK 0.878 → 0.870. Bias: disappears (CI includes 0); slope: disappears (CI includes 0).
- **Matched UNet++, 3 seeds** (thresholds 0.89, 0.996, 0.98): bias +0.82 ± 0.30 (+0.47 to +1.01) → +0.14 ± 0.07 (+0.07 to +0.21) pp; slope 0.125 ± 0.040 (0.079 to 0.155) → 0.033 ± 0.013 (0.023 to 0.047); MAE 0.83 ± 0.29 (0.50 to 1.02) → 0.48 ± 0.08 (0.39 to 0.53); leaf IoU 0.759 ± 0.015 (0.746 to 0.776) → 0.770 ± 0.013 (0.755 to 0.780). Per-seed verdicts (bias; slope): seed 0: disappears (CI includes 0) / disappears (CI includes 0); seed 1: disappears (CI includes 0) / disappears (CI includes 0); seed 2: disappears (CI includes 0) / disappears (CI includes 0).
- **Matched DeepLabV3+, 3 seeds** (thresholds 0.78, 0.8, 0.74): bias +0.47 ± 0.32 (+0.13 to +0.77) → +0.09 ± 0.28 (−0.19 to +0.36) pp; slope 0.097 ± 0.068 (0.025 to 0.161) → 0.046 ± 0.064 (−0.021 to 0.107); MAE 0.62 ± 0.25 (0.36 to 0.87) → 0.46 ± 0.16 (0.32 to 0.63); leaf IoU 0.725 ± 0.010 (0.718 to 0.736) → 0.730 ± 0.008 (0.723 to 0.738). Per-seed verdicts (bias; slope): seed 0: disappears (CI includes 0) / shrinks (still > 0); seed 1: disappears (CI includes 0) / disappears (CI includes 0); seed 2: not significant at argmax already; now significantly negative / not significant at argmax already; still not significant.

### Overall

At argmax, 8 of the 9 rust models have a bias whose bootstrap CI excludes 0 and 8 of 9 a positive proportional slope whose CI excludes 0. With the validation-calibrated threshold (min |bias|), the test bias CI includes 0 for 8 of 9 models (6 of 9 with the MAE-chosen threshold) and the slope bootstrap CI includes 0 for 6 of 9 (3 of 9) [Verifier: CI type made explicit; with the t-based slope CI the counts are 5 of 9 for both threshold rules, because matched DeepLabV3+ seed 1 (t CI 0.004 to 0.099) and seed 2 under the MAE rule (−0.087 to −0.003, negative) exclude 0 under the t CI while matched UNet++ seeds 0–2 under the MAE rule include 0]. The chosen thresholds are all far above 0.5 (range 0.74–0.996).

**Conclusion.** The rust over-estimation is to a large extent an operating-point artifact, not a fixed property of the networks: the argmax lesion border lies systematically slightly outside the reference border (mean offset of attached contours +0.07 to +0.68 px for SegFormer-B5 and the six matched CNNs, i.e. less than one pixel; +1.16 px for published DeepLabV3+ and +1.46 px for published UNet++) [Verifier: qualified; the earlier text said "on average less than one pixel" for all models], and raising the lesion-probability threshold, calibrated once on validation leaves, brings the test bias to a value compatible with 0 for 8 of 9 models (exception: DeepLabV3+ matched lrB seed 2: −0.19 [−0.39, −0.01] pp (argmax +0.13 [−0.06, +0.33]), i.e. a model without significant argmax bias that the threshold slightly over-corrected) and reduces the proportional (severity-dependent) over-estimation substantially. It does not remove the proportional component for every model: for SegFormer-B5 the slope falls from 0.208 to 0.075 but stays above 0 (CI [0.036, 0.114]), and for published UNet++ from 0.311 to 0.134 (CI [0.053, 0.215]). The honest summary for the primary model is therefore *bias disappears, proportional over-estimation shrinks by 64% but persists*. A fixed 1-px erosion removes the slope but over-corrects into under-estimation and costs IoU, so it is not a better operating point.

**Why the thresholds are so high (mechanism, consistent with but not proven by these data).** Both the published SegFormer-B5 kernel (`modal-recovery/kernels/lesion_roya_segformer_realpack_kernel.py`, lines 142–158) and the matched rust CNN kernels (`reanalysis/kaggle_runs/pg26038-mb*-roya-*-lrB*/…py`, `focal_tversky` and `ce_w`) train with 0.7 × Focal Tversky (α = 0.3 on false positives, β = 0.7 on false negatives, γ = 0.75) + 0.3 × cross-entropy with class weights 0.25/0.5/3.0 (background/leaf/lesion). Both terms penalize missed lesion pixels more than extra ones, which pushes lesion probabilities up at lesion borders; a threshold above 0.5 undoes this. [Verifier:] the published rust CNN training scripts `notebooks/pg26038-roya-deeplab-realpack/pg26038-roya-deeplab-realpack.py` and `notebooks/pg26038-roya-unetpp-realpack/pg26038-roya-unetpp-realpack.py` (lines 218–228) define the same loss (`focal_tversky(a=0.3, b=0.7, gexp=0.75)`, `ce_w = [0.25, 0.5, 3.0]`, 0.7/0.3 mix); that these scripts produced the served ONNX files was not re-established by hash here. **The Tversky β ablation (retraining with α = β = 0.5, or with an unweighted cross-entropy) that would test this causally could not be run: round 4 permits no training of any kind.**

## 10. Caveats

- *Relative to a semi-automatic reference.* The rust reference masks are detector + SAM 2 output, spot-checked but not edited (revision brief, fact 1). "Over-estimation" and the boundary offset are measured against that reference's border convention; a calibrated threshold aligns the models to it and does not show which border is botanically correct.
- *Calibration data.* The threshold was chosen on the same 21 validation leaves that were used for checkpoint selection (SegFormer-B5: validation lesion IoU; rust CNNs: IoU minus healthy false-positive rate), and validation and test leaves come from one site and one collection day (Sarchí, 2026-08-19). How well a fixed threshold transfers to other scanners, sites or seasons is untested.
- *Grid extension (disclosed).* The threshold grid was widened above 0.99 after a preview run placed one model's validation optimum on the former edge (see Method); no chosen threshold lies on the final grid's upper edge (flag `at_grid_upper_edge` = False for all models).
- *Small n.* 21 test leaves; only 3 exceed 10% reference severity, so the slope estimates lean on those leaves.
- *Pixel scale.* Distances are in letterbox pixels = native scan pixels for rust (scale 1.0); the physical pixel size was not recorded.
- *Healthy leaves.* Only 21 of the 54 healthy rust-test leaves are available locally (fact 8); counts use the abaxial face only.
- *Seeds.* SegFormer-B5 exists as a single run; its seed-to-seed variability is unknown. Matched CNN results are given per seed and as mean ± SD.
- *Erosion depth* (1 and 2 px) was fixed a priori; its validation values are reported but it was not tuned.

## 11. Suggested manuscript text (Results or Discussion; numbers from this file)

> To test whether the rust over-estimation reflects the decision rule rather than the networks themselves, we kept each model's softmax lesion probability and, without retraining, chose a lesion-probability threshold on the 21 validation leaves that minimized the absolute bias, then applied it once to the test leaves. For SegFormer-B5 (threshold 0.97), the test bias fell from +0.99 to +0.12 percentage points (95% CI −0.09 to +0.41), the MAE from 0.99 to 0.34, and leaf-level IoU rose from 0.767 to 0.789; the proportional-bias slope decreased from 0.21 to 0.075 (95% CI 0.036 to 0.114) but did not vanish. At the default argmax rule, 51% of the predicted lesion contour pixels lay outside the reference lesions (mean offset of contours attached to a reference lesion +0.64 pixels, 95% CI +0.35 to +0.97), against 30% (+0.05 pixels, −0.17 to +0.31) after calibration. Similar reductions in bias and slope were seen for the published and matched-protocol CNNs. The over-estimation is therefore largely an operating-point effect, consistent with a training loss that penalized missed lesion pixels more than extra ones (Focal Tversky β = 0.7 > α = 0.3; lesion class weight 3.0); a loss ablation was not performed. Because the rust reference masks are semi-automatic, this calibration aligns the predictions with the reference border convention rather than establishing the true lesion border.

## 12. [Section omitted from the public release]

This section listed the expert-panel identities (names, occupations, affiliations, years of experience). Identities are not part of the code release.

## Verifier corrections (independent verification, 28 Sep 2026)

**Method of verification.** Fresh CPU inference (onnxruntime 1.30, float64 softmax, no quantisation, no reuse of `work/prob`) of all 9 models on the same 63 abaxial images, with an independently written pipeline (`verifier_r4/v_infer.py`: direct threshold counting, erosion by 3 × 3 minimum filter, two boundary-distance conventions) and statistics (`verifier_r4/v_stats.py`: `scipy.stats.linregress`, own QWK, bootstrap with `default_rng(20260928).integers(0, 21, (10000, 21))`). Output: `verifier_r4/v_results.json`, `verifier_r4/v_stats.log`.

**Confirmed.**
- Argmax reproduction: 567 model × image pairs (leaf, lesion, intersection, union, reference counts) equal `out/per_image.csv` / `out/per_image_retrained.csv`; 0 mismatches. Baseline bias and slope equal `results/agreement` (e.g. SegFormer-B5 +0.990, slope 0.208 [0.154, 0.261]).
- Protocol: rust abaxial face; leaf = alpha > 8; lesion = mask > 127 AND leaf; letterbox 1024; ImageNet normalisation; SegFormer logits upsampled (PIL bilinear) before the decision; grades < 0.5 / 5 / 20%; leaf sets 21 diseased validation, 21 diseased test, 21 healthy test (split.csv); bootstrap unit = leaf, B = 10,000, seed 20260928, percentile CIs, paired across operating points.
- All 18 validation-chosen thresholds identical (no ties); no choice on the grid edge.
- Every test endpoint in Sections 3–6 (IoU, MAE, bias, slope and its t CI, grade accuracy, QWK, over/under-graded and over-estimated counts, all bootstrap CIs, all paired Δ with CIs, healthy-leaf counts and maxima) and every boundary summary in Section 7 (n, median, IQR, mean with CI, shares; all/attached/reverse) agree within rounding. Only differences: matched UNet++ seeds 1 and 2 at t = 0.996 / 0.98 differ by one contour pixel and matched UNet++ seed 1's slope p by 0.001, from the analyst's 1/65535 quantisation of p(lesion); no reported number changes.
- Verdict counts (bias CI includes 0 for 8/9 models at the min-|bias| threshold, 6/9 at the min-MAE threshold; matched UNet++ seed 2 min-MAE lower bound +0.0008) and the 64% slope reduction for SegFormer-B5 (0.2077 → 0.0751) are correct. Loss settings in the SegFormer-B5 kernel (lines 142–158) and the matched rust CNN kernels (lines 223–233) are as stated.
- Grid-extension disclosure: under the original grid (upper edge 0.99) matched UNet++ seed 1 would have used t = 0.99, giving test bias +0.28 [−0.06, +0.71] and slope 0.048 (bootstrap [−0.002, 0.218]) (from the analyst's saved probabilities): the same verdicts as with t = 0.996, so the post-hoc extension changes no conclusion.

**Corrected in this file.**
1. Section 3 note: the description of the over/under-graded columns was wrong ("unless stated in the last column"); reworded.
2. Section 9 conclusion: "on average less than one pixel" held only for SegFormer-B5 and the matched CNNs (attached-contour mean offset +0.07 to +0.68 px); published UNet++ (+1.46 px) and DeepLabV3+ (+1.16 px) exceed one pixel. Qualified.
3. Section 9 overall: the slope counts (6/9, 3/9) use the bootstrap CI; this is now stated, with the t-CI counts (5/9, 5/9). The verdict rule in Section 4 uses bootstrap CIs throughout; the slope CIs quoted in the text are t-based. For the paper, quote one type per statement.
4. Section 7/9 boundary convention: under the file's convention (inside pixels: 1 − EDT to non-lesion), reference-contour pixels that touch the background only diagonally get −0.41 instead of 0, so "on its contour" is understated. With the plain definition (signed Euclidean distance to the reference contour pixel set), SegFormer-B5 argmax: 51% outside (unchanged), 39% on the contour, 3% ≥ 2 px inside, attached mean +0.68 px [+0.40, +1.01]; after the threshold (t = 0.97): 30% outside, 49% on the contour, IQR 0 to 1 px, attached mean +0.13 px [−0.08, +0.39]. The qualitative conclusion (outward shift at argmax, removed by the threshold) holds under both conventions.
5. Section 9 mechanism: the published rust CNN training scripts were checked and use the same loss (see text); the link from these scripts to the served ONNX was not re-established.
6. Section 12: records the manuscript addition of the expert identities (above).

**Not corrected, for the authors.**
- Verdicts across 9 models × 4 operating points are unadjusted descriptive CIs; no Holm family or equivalence (TOST) test was defined for A5, so "disappears" means "95% CI includes 0", not "shown to be zero".
- The matched UNet++ argmax bias SD is 0.305 (0.30 here; `pass3/seed_facts.md` prints 0.31). Use one value consistently.
- `verify/` in this folder is an earlier, partial verification pass by another agent (it stopped partway; not used here and not referenced). `verifier_r4/out/*.npz` holds per-pixel distance arrays (about 4.5 MB). Both can be left out of the public release.
- `results.md` was edited by hand after `make_md.py`; re-running `make_md.py` would drop these corrections.

## Files

- `infer_prob.py`: CPU inference (onnxruntime) keeping the softmax; `work/prob/<model>/<split>_<grade>/*.npz` (argmax map, background-vs-leaf map, lesion probability × 65535; about 87 MB, regenerable with `infer_prob.py`, so it can be left out of the public code release); `work/infer_prob.log` (567 argmax reproductions, 0 mismatches).
- `oppoint.py`: all analyses → `results.json`; `work/leaf_level.csv` (per test leaf × model × operating point); `work/val_curves.csv` (validation IoU/MAE/bias/slope for every threshold); `work/boundary_counts.npz` (signed-distance histograms per model × operating point); `work/verdicts.json`.
- `make_md.py` + `md_interpretation.py`: this file.
