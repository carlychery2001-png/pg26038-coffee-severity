# A3 (round 4): detector-independent colour baseline with leaf-edge erosion

Created 2026-09-28 08:20. Scripts: `r4_baseline.py` (all main numbers → `results.json`, `per_leaf_test.csv`, `per_leaf_healthy.csv`, `validation_grid.csv`; the validation-only pre-run that fixed the index before any test evaluation of a new index → `validation_only.json`, `valonly.log`), `detector_rule.py` (detector characterisation → `detector_rule.json`), `exclusion_sensitivity.py` (→ `exclusion_sensitivity.json`), `paired_extra.py` (→ `paired_extra.json`) and `make_md.py` (this file). Python 3.13.15, numpy 2.5.3, scipy 1.18.1. No training and no network inference were run; network values come from `reanalysis/out/per_image.csv`.

## 0. Summary

- **Detector rule:** the source of the rust weak-mask detector (`roya_lesion.py`) is not on this computer, so its index and thresholds cannot be documented from code. From its saved output masks, its colour criterion behaves like a hue boundary between orange/brown and green (best single rules: HSV hue < 68.8°, excess red > 0.1333 = 2/15, i.e. strictly above the neutral-grey value), combined with spatial clean-up; no single threshold reproduces its masks closely (best pooled IoU 0.65). The existing rust baseline (excess red ≥ 0.12) uses essentially the detector's colour criterion (Section 1).
- **New baseline:** CIELAB a* (least related usable index; Section 2), threshold, erosion and exclusion chosen on validation leaf IoU. Rust: a* ≥ −8.2, erosion 2 px, exclusion none; leaf spot: a* ≥ −0.9, erosion 1 px, exclusion none. No petiole/cut-end exclusion was selected for either disease.
- **Rust test (n = 21):** leaf IoU 0.564 [0.513, 0.610], MAE 1.20 [0.70, 1.81] pp, bias −0.59 [−1.31, 0.10], grade accuracy 16/21 = 0.76 [0.57, 0.90], QWK 0.68 [0.33, 0.90].
- **Leaf-spot test (n = 55):** leaf IoU 0.807 [0.769, 0.841], MAE 0.23 [0.16, 0.32] pp, bias −0.18 [−0.28, −0.10] pp; mask-derived grade (primary) accuracy 54/55 = 0.98 [0.95, 1.00], QWK 0.98 [0.93, 1.00]; visual grade (secondary) 48/55 = 0.87 [0.78, 0.95], QWK 0.85 [0.75, 0.94].
- **Healthy rust test leaves (21 available):** new rust baseline ≥ 0.5%: 12/21 abaxial [36.5–75.5%], 21/21 either face; > 0%: 21/21 and 21/21. Existing excess-red threshold with the same 2-px erosion: 9/21 either face [24.5–63.5%], 8/21 abaxial (21/21 and 20/21 without erosion). Published networks: 0/21 at ≥ 0.5% on either face.
- **Answer:** yes, a validation-tuned, edge-eroded, detector-independent threshold still flags healthy rust leaves at the G1 cut-point (Section 8).

## 1. The colour rule of the detector that seeded the rust SAM 2 masks

**Source code: not available.** The rust SAM 2 refinement script (`notebooks/pg26038-sam2-roya-refine/pg26038-sam2-roya-refine.py`) says that its seeds are weak 0/255 masks from `roya_lesion.py`, "el detector propio de roya". That file is not on this computer (searched by name and content under `C:/dev`, and by name under `<USER_HOME>/OneDrive`, Documents and Desktop; no other script in the project reproduces it). **Its index, thresholds and post-processing therefore cannot be documented from code.** What the refine script does document: SAM 2 (`sam2_l.pt`, Ultralytics) is prompted with the bounding box (3-px padding) of every weak-mask blob of ≥ 4 px (at most 300 per leaf, largest first); the refined blobs are unioned and clipped to the leaf alpha; a refined blob larger than 15% of the leaf (or empty) falls back to its weak pixels.

**Empirical characterisation from the detector's output.** The weak masks are present (`datasets/pg26038-sam2-roya/seeds/`, aligned with `images/`, 2048 px high, twice the pack resolution): 283 rust faces with an image (train 198, test 42, val 42, not in pack 1). On a uniform 1/16 sample of leaf pixels (5,457,738 pixels, 156,707 of them seed pixels), the best single global threshold per index reproduces the weak masks as follows (pooled pixel IoU with the weak masks; AUC = ROC AUC for weak-mask vs other leaf pixels; AUC < 0.5 means lower values are weak-mask pixels):

| Index | Best rule | Pooled IoU with weak masks | AUC |
|---|---|---|---|
| HSV hue signed (deg) | < 68.75 | 0.651 | 0.098 |
| excess red 1.4r'-g' | > 0.1333 (= 2/15; neutral-grey pixels excluded, see Verifier corrections) | 0.641 | 0.966 |
| chromatic r' | ≥ 0.3666 | 0.619 | 0.912 |
| CIELAB hue angle h_ab (deg) | < 113.447 | 0.614 | 0.104 |
| redness index (r-g)/(r+g+b) | ≥ 0.0017 | 0.582 | 0.968 |
| r-g | ≥ 1 | 0.582 | 0.948 |
| g/r | < 0.9953 | 0.544 | 0.034 |
| CIELAB a* | ≥ −3.1149 | 0.336 | 0.894 |
| R | ≥ 115 | 0.237 | 0.835 |

Best two-index AND rules (128 quantile bins per index): chromatic r' ≥ 0.357 AND CIELAB a* ≥ −8.304 (IoU 0.667); r-g ≥ −5 AND chromatic r' ≥ 0.354 (IoU 0.663); redness index (r-g)/(r+g+b) ≥ −0.019 AND chromatic r' ≥ 0.351 (IoU 0.661).
Per-face IoU of the best single rule (HSV hue signed (deg)): median 0.704 (10th–90th percentile 0.311–0.866, 283 faces). Per-face optimal thresholds: HSV hue signed (deg) median 69.6 (10th–90th percentile 63.107 to 78.321; per-face best IoU median 0.756); excess red 1.4r'-g' median 0.134 (10th–90th percentile 0.094 to 0.142; per-face best IoU median 0.724).
Weak-mask morphology: 5,786 blobs over 283 faces, smallest blob 10 px (2048-px scale), no blob < 4 px, 19,337 enclosed hole pixels (of 2,510,216 weak-mask pixels), 0 pixels outside the leaf alpha, median weak-mask area 2.19% of the leaf.

**Reading (inference, not documentation).** No colour threshold, single or paired, reproduces the weak masks closely, the masks are smooth blobs with a minimum size of 10 px, and the per-face optimal threshold varies, so the detector combined a colour criterion with spatial post-processing (smoothing, morphology and/or area filtering) and possibly a per-leaf adaptive element; its exact form cannot be recovered. Its colour criterion is best approximated by a hue-type boundary between orange/brown and green tissue (HSV hue below about 70°; excess red above about 0.133). The value 0.133 = 1.4/3 − 1/3 is the excess red of a neutral grey, so this excess-red rule is a line through the white point of the chromaticity plane, i.e. itself a hue boundary.

**Comparison with the existing baseline** (`results/baseline`). Its rust rule, excess red ≥ 0.12, is the same index as the best single-index approximation of the detector (> 0.1333) with a threshold 0.013 lower (pooled IoU of excess red ≥ 0.12 itself with the weak masks: 0.526, verifier's stride-4 sample); excess red, HSV hue, CIELAB h_ab and the redness index all reproduce the weak masks about equally (pooled IoU 0.58–0.65). The existing rust baseline therefore shares the detector's colour criterion, which is the circularity concern. The leaf-spot rule (redness index ≥ 0.026) belongs to a different disease whose masks were corrected by hand in CVAT; it is not related to the rust detector.

## 2. Choice of the detector-independent index

Relatedness to the detector was measured on the **training** faces only (198 faces, 3,741,692 sampled leaf pixels): (a) the best pooled IoU with which one threshold on the candidate reproduces the weak masks, (b) Spearman ρ with the three non-candidate indices that best reproduce the detector (excess red 1.4r'-g', chromatic r', redness index (r-g)/(r+g+b)). Usefulness as a lesion index is the best validation leaf IoU from the validation-only pre-run (`validation_only.json`), which ran before any test evaluation of a new index.

| Candidate | Weak-mask reproduction IoU (train) | AUC | ρ with excess red 1.4r'-g' | ρ with chromatic r' | ρ with redness index (r-g)/(r+g+b) | Best val IoU, rust | Best val IoU, leaf spot |
|---|---|---|---|---|---|---|---|
| CIELAB a* | 0.350 | 0.900 | 0.233 | −0.353 | 0.389 | 0.575 | 0.803 |
| CIELAB hue angle h_ab (deg) | 0.623 | 0.094 | −0.526 | −0.925 | −0.384 | 0.664 | 0.808 |
| HSV hue signed (deg) | 0.661 | 0.089 | −0.573 | −0.944 | −0.411 | 0.666 | 0.809 |
| HSV saturation | 0.092 | 0.677 | −0.156 | 0.518 | −0.334 | 0.049 / 0.034 | 0.092 / 0.059 |

**Choice: CIELAB a*, lesion = a* ≥ t.** HSV saturation is excluded because it is not a lesion index here (green tissue is as saturated as lesions): best validation IoU ≤ 0.09 in either direction for both diseases. The two hue indices are hue boundaries like the detector's criterion: they reproduce its weak masks as well as excess red does (train IoU 0.66 and 0.62; |ρ| up to 0.94), and their validation-chosen thresholds (h′ ≤ 72°, h_ab ≤ 116°) sit where the detector's best single rules sit. a* reproduces the weak masks poorly (train IoU 0.35) and has the weakest rank correlation with the detector-like indices (|ρ| 0.23–0.39), so it is the least related usable index. Honest caveat: an a* condition (a* ≥ −8.3) appears as the second term of the best two-index approximation of the detector, but it adds only 0.048 IoU to chromatic r′ alone, so a* is not independent of lesion colour (no colour index can be) but is clearly different from the detector's hue-type criterion.

## 3. Method

- **Evaluation protocol unchanged** (round4_plan.md): leaf = alpha > 8; lesion = mask > 127 AND leaf; 1024 × 1024 aspect-preserving letterbox (RGB bilinear, labels nearest; copy of `run_inference.py load()`); rebuilt reference leaf and lesion counts equal `per_image.csv` for all 388 validation/test images. Rust = abaxial face; American leaf spot = mean of both faces (IoU = mean of face IoUs, severity = mean of face severities). Grades < 0.5, 0.5–< 5, 5–< 20, ≥ 20%. Rust reference grade = mask-derived; leaf-spot reference grade = mask-derived (**primary**, author decision of 28 Sep 2026) and the annotator's visual grade (secondary).
- **Baseline**: threshold inside the reference leaf mask, as in `results/baseline`; the severity denominator is always the **full** reference leaf area (erosion and exclusion only remove predicted lesion pixels); no smoothing or speck removal. a* is sRGB → CIELAB (D65) in float64; excess red and the redness index are compared exactly in integer arithmetic, as in `results/baseline`.
- **Leaf-edge erosion** e ∈ {0, 1, 2, 4, 8} px on the 1024 canvas: predictions are kept only on leaf pixels whose taxicab distance to the background is > e, identical to scipy `binary_erosion` (default 3 × 3 cross) applied e times (checked on 12 images) and to the band definition of `results/baseline` 5b.
- **Petiole / cut-end exclusion** (one option or none): 'open r' drops predictions on leaf pixels removed by a morphological opening of the leaf mask with a Euclidean disk of radius r ∈ {4, 8, 12, 16} px, i.e. parts narrower than about 2r: the petiole stalk (the healthy scans include curled, often brownish petioles) and, unavoidably, the narrow drip tip; 'ends k' drops predictions within k ∈ {32, 64, 128} px of either end of the leaf's major axis (first principal axis of the leaf-pixel coordinates); both ends are excluded because which end is the cut petiole is not recorded. Rationale: petiole and cut end are non-lamina tissue included in the protocol's leaf mask and often brown or yellow on healthy leaves.
- **Selection**: threshold × erosion × exclusion (5 × 8 options × full threshold grid; a* grid −40 to 60 in steps of 0.1) chosen jointly by the **validation mean leaf IoU of the diseased validation leaves** (rust n = 21, leaf spot n = 55); ties → smaller erosion, then 'none'. Healthy leaves are not used in the primary selection. The chosen configuration was applied once to test.
- **Uncertainty**: 10000 resamples of leaves, percentile 95% CI, default_rng(20260928) re-seeded per disease, one index matrix per disease shared by all methods. Healthy-leaf proportions: Wilson 95% intervals.
- **Healthy leaves**: the 21 healthy rust-pack test leaves available locally (EARTH); images for the other 33 of the paper's 54 were not retained. Flags at > 0% (any predicted lesion pixel) and ≥ 0.5% (predicted G1 or higher), under the **abaxial** rule (the rust leaf rule) and the **either-face** rule (maximum over the two faces); the mean-of-faces rule is also listed. Healthy validation leaves (21, EARTH) are secondary.

## 4. Reproduction check (done first)

The existing baseline was re-implemented and reproduced at the published precision before any new result was computed: rust exr >= 0.12, no erosion: IoU 0.600, MAE 1.02, bias +0.17, QWK 0.93, healthy test ≥ 0.5% 21/21 either face and 20/21 abaxial; leaf spot ri >= 0.026, no erosion: IoU 0.807, MAE 0.23, bias −0.11, QWK against the visual grade (the reference used in `results/baseline`) 0.85. The erosion code reproduces `results/baseline` 5b (excess red ≥ 0.116, 3-px band: IoU 0.633, 15/21 healthy; 12-px band: 0.547, 11/21). All match: **yes**.

## 5. Validation selection

Best validation mean leaf IoU of every candidate over threshold × erosion × exclusion (diseased validation leaves; rust n = 21, leaf spot n = 55). The last columns give the validation IoU with no erosion and no exclusion (threshold re-optimised) and the number of the 40 erosion × exclusion options whose best IoU is within 0.005 of the maximum.

| Disease | Index | Best val IoU | Erosion (px) | Exclusion | Threshold | Val IoU, e = 0, none (threshold) | Options within 0.005 |
|---|---|---|---|---|---|---|---|
| Coffee leaf rust | HSV hue | 0.666 | 1 | open 8 | h′ ≤ 72° | 0.643 (h′ ≤ 72°) | 8 |
| Coffee leaf rust | CIELAB hue angle h_ab | 0.664 | 1 | open 8 | h_ab ≤ 116° | 0.639 (h_ab ≤ 116°) | 8 |
| Coffee leaf rust | CIELAB a* **(chosen)** | 0.575 | 2 | none | a* ≥ −8.2 | 0.474 (a* ≥ −7.6) | 7 |
| Coffee leaf rust | HSV saturation (lesion = high) | 0.049 | 8 | open 16 | S ≥ 0.456 | 0.048 (S ≥ 0.456) | 35 |
| Coffee leaf rust | HSV saturation (lesion = low) | 0.034 | 1 | none | S ≤ 0.607 | 0.033 (S ≤ 0.607) | 40 |
| Coffee leaf rust | excess red 1.4r′ − g′ (existing index) | 0.667 | 2 | none | exr ≥ 0.116 | 0.609 (exr ≥ 0.12) | 7 |
| American leaf spot | HSV hue | 0.809 | 1 | none | h′ ≤ 47.5° | 0.793 (h′ ≤ 47.5°) | 2 |
| American leaf spot | CIELAB hue angle h_ab | 0.808 | 1 | none | h_ab ≤ 95.7° | 0.787 (h_ab ≤ 95.7°) | 2 |
| American leaf spot | CIELAB a* **(chosen)** | 0.803 | 1 | none | a* ≥ −0.9 | 0.780 (a* ≥ 0) | 2 |
| American leaf spot | HSV saturation (lesion = high) | 0.092 | 8 | none | S ≥ 0.376 | 0.084 (S ≥ 0.373) | 14 |
| American leaf spot | HSV saturation (lesion = low) | 0.059 | 1 | none | S ≤ 0.671 | 0.059 (S ≤ 0.686) | 40 |
| American leaf spot | redness index (r − g)/(r + g + b) (existing index) | 0.798 | 1 | none | ri ≥ 0.023 | 0.789 (ri ≥ 0.026) | 2 |

Coffee leaf rust, CIELAB a*, validation IoU by erosion (no exclusion): 0 px 0.474, 1 px 0.558, 2 px 0.575, 4 px 0.571, 8 px 0.551. Options within 0.005 of the best: e 2, none, a* ≥ −8.2 (0.575); e 2, open 4, a* ≥ −8.2 (0.575); e 2, open 8, a* ≥ −8.2 (0.574); e 2, open 12, a* ≥ −8.2 (0.572); e 4, none, a* ≥ −8.2 (0.571); e 4, open 4, a* ≥ −8.2 (0.571); e 4, open 8, a* ≥ −8.2 (0.570).

American leaf spot, CIELAB a*, validation IoU by erosion (no exclusion): 0 px 0.780, 1 px 0.803, 2 px 0.798, 4 px 0.781, 8 px 0.752. Options within 0.005 of the best: e 1, none, a* ≥ −0.9 (0.803); e 1, open 4, a* ≥ −0.9 (0.799).

No petiole/cut-end exclusion improved validation IoU for the chosen index (nor for the existing indices); the 'ends' options never came within 0.005 of the best.

## 6. Test results (chosen configuration applied once)

### Coffee leaf rust (n = 21 test leaves; abaxial face)

Chosen: **a* ≥ −8.2, erosion 2 px, exclusion none** (validation IoU 0.575). Brackets: 95% bootstrap CI. Grade columns use the mask-derived reference.

| Method | Configuration | Leaf IoU | MAE (pp) | Bias (pp) | Grade acc. | QWK |
|---|---|---|---|---|---|---|
| **New baseline (primary)** | a* ≥ −8.2, erosion 2 px, exclusion none | 0.564 [0.513, 0.610] | 1.20 [0.70, 1.81] | −0.59 [−1.31, 0.10] | 16/21 = 0.762 [0.57, 0.90] | 0.68 [0.33, 0.90] |
| New index, specificity-constrained (secondary) | a* ≥ −1.2, erosion 2 px, exclusion none | 0.307 [0.264, 0.351] | 3.49 [2.31, 4.88] | −3.49 [−4.88, −2.31] | 9/21 = 0.429 [0.24, 0.67] | 0.48 [0.29, 0.65] |
| Existing baseline, no erosion | exr ≥ 0.12, erosion 0 px, exclusion none | 0.600 [0.555, 0.644] | 1.02 [0.45, 1.87] | +0.17 [−0.55, 1.14] | 20/21 = 0.952 [0.86, 1.00] | 0.93 [0.72, 1.00] |
| Existing threshold + chosen erosion (sensitivity) | exr ≥ 0.12, erosion 2 px, exclusion none | 0.645 [0.608, 0.679] | 1.00 [0.49, 1.70] | −0.31 [−0.96, 0.53] | 19/21 = 0.905 [0.76, 1.00] | 0.87 [0.66, 1.00] |
| Existing index re-tuned at chosen erosion (sensitivity) | exr ≥ 0.116, erosion 2 px, exclusion none | 0.639 [0.596, 0.676] | 1.06 [0.43, 2.07] | +0.20 [−0.56, 1.33] | 19/21 = 0.905 [0.76, 1.00] | 0.87 [0.66, 1.00] |
| SegFormer-B5 (published) | network (predicted leaf) | 0.767 [0.743, 0.791] | 0.99 [0.48, 1.62] | +0.99 [0.48, 1.62] | 19/21 = 0.905 [0.76, 1.00] | 0.88 [0.68, 1.00] |
| UNet++ (published) | network (predicted leaf) | 0.647 [0.608, 0.684] | 1.77 [0.91, 2.76] | +1.75 [0.88, 2.75] | 17/21 = 0.810 [0.62, 0.95] | 0.79 [0.55, 0.94] |
| DeepLabV3+ (published) | network (predicted leaf) | 0.632 [0.590, 0.669] | 1.47 [0.63, 2.51] | +1.41 [0.56, 2.46] | 19/21 = 0.905 [0.76, 1.00] | 0.88 [0.68, 1.00] |

Validation chose no exclusion, so the '+ exclusion' sensitivity rows equal the rows without it and are omitted.

Paired differences, network minus new baseline (same resamples): SegFormer-B5 Δ IoU +0.204 [0.152, 0.259] (network higher on 21/21 leaves), Δ MAE −0.21 [−0.73, 0.35]; UNet++ Δ IoU +0.083 [0.028, 0.139] (network higher on 16/21 leaves), Δ MAE +0.57 [−0.20, 1.43]; DeepLabV3+ Δ IoU +0.068 [0.011, 0.125] (network higher on 16/21 leaves), Δ MAE +0.28 [−0.59, 1.28].

New baseline minus existing baseline: vs existing baseline, no erosion: Δ IoU −0.036 [−0.077, 0.005], Δ MAE +0.17 [−0.69, 0.84]; vs existing threshold + chosen erosion (sensitivity): Δ IoU −0.082 [−0.131, −0.035], Δ MAE +0.19 [−0.51, 0.78].

Existing threshold with the chosen 2-px erosion (`paired_extra.json`): minus the same threshold without erosion Δ IoU +0.045 [0.029, 0.062]; networks minus it: SegFormer-B5 Δ IoU +0.122 [0.092, 0.153] (network higher on 20/21), Δ MAE −0.01 [−0.73, 0.63]; UNet++ Δ IoU +0.001 [−0.032, 0.036] (network higher on 10/21), Δ MAE +0.76 [0.05, 1.65]; DeepLabV3+ Δ IoU −0.014 [−0.041, 0.014] (network higher on 11/21), Δ MAE +0.47 [−0.13, 1.21].

New-baseline grade confusion (rows: mask-derived reference G0–G3; columns: predicted G0–G3): [[1, 0, 0, 0], [1, 8, 2, 0], [0, 2, 7, 0], [0, 0, 0, 0]].

### American leaf spot (n = 55 test leaves; mean of both faces)

Chosen: **a* ≥ −0.9, erosion 1 px, exclusion none** (validation IoU 0.803). Brackets: 95% bootstrap CI. Grade columns use the mask-derived reference (primary); the visual-grade columns are secondary.

| Method | Configuration | Leaf IoU | MAE (pp) | Bias (pp) | Grade acc. | QWK | Grade acc. (visual) | QWK (visual) |
|---|---|---|---|---|---|---|---|---|
| **New baseline (primary)** | a* ≥ −0.9, erosion 1 px, exclusion none | 0.807 [0.769, 0.841] | 0.23 [0.16, 0.32] | −0.18 [−0.28, −0.10] | 54/55 = 0.982 [0.95, 1.00] | 0.98 [0.93, 1.00] | 48/55 = 0.873 [0.78, 0.95] | 0.85 [0.75, 0.94] |
| Existing baseline, no erosion | ri ≥ 0.026, erosion 0 px, exclusion none | 0.807 [0.777, 0.836] | 0.23 [0.15, 0.33] | −0.11 [−0.22, −0.01] | 54/55 = 0.982 [0.95, 1.00] | 0.98 [0.93, 1.00] | 48/55 = 0.873 [0.78, 0.95] | 0.85 [0.75, 0.94] |
| Existing threshold + chosen erosion (sensitivity) | ri ≥ 0.026, erosion 1 px, exclusion none | 0.817 [0.789, 0.844] | 0.25 [0.17, 0.35] | −0.19 [−0.30, −0.10] | 54/55 = 0.982 [0.95, 1.00] | 0.98 [0.93, 1.00] | 48/55 = 0.873 [0.78, 0.95] | 0.85 [0.75, 0.94] |
| Existing index re-tuned at chosen erosion (sensitivity) | ri ≥ 0.023, erosion 1 px, exclusion none | 0.818 [0.789, 0.844] | 0.22 [0.15, 0.32] | −0.12 [−0.22, −0.02] | 54/55 = 0.982 [0.95, 1.00] | 0.98 [0.93, 1.00] | 48/55 = 0.873 [0.78, 0.95] | 0.85 [0.75, 0.94] |
| SegFormer-B5 (published) | network (predicted leaf) | 0.884 [0.868, 0.900] | 0.22 [0.15, 0.30] | +0.16 [0.08, 0.25] | 52/55 = 0.945 [0.87, 1.00] | 0.95 [0.88, 1.00] | 46/55 = 0.836 [0.73, 0.93] | 0.82 [0.72, 0.91] |
| UNet++ (published) | network (predicted leaf) | 0.853 [0.830, 0.874] | 0.15 [0.10, 0.20] | −0.03 [−0.09, 0.03] | 54/55 = 0.982 [0.95, 1.00] | 0.98 [0.93, 1.00] | 48/55 = 0.873 [0.78, 0.95] | 0.85 [0.75, 0.94] |
| DeepLabV3+ (published) | network (predicted leaf) | 0.815 [0.790, 0.837] | 0.25 [0.17, 0.33] | +0.08 [−0.02, 0.18] | 53/55 = 0.964 [0.91, 1.00] | 0.97 [0.90, 1.00] | 47/55 = 0.855 [0.76, 0.95] | 0.84 [0.74, 0.92] |

Validation chose no exclusion, so the '+ exclusion' sensitivity rows equal the rows without it and are omitted. For leaf spot the specificity-constrained threshold equals the primary one (no healthy validation leaf reached 0.5%) and is omitted.

Paired differences, network minus new baseline (same resamples): SegFormer-B5 Δ IoU +0.078 [0.056, 0.103] (network higher on 55/55 leaves), Δ MAE −0.01 [−0.10, 0.07]; UNet++ Δ IoU +0.046 [0.029, 0.065] (network higher on 45/55 leaves), Δ MAE −0.09 [−0.15, −0.03]; DeepLabV3+ Δ IoU +0.008 [−0.010, 0.029] (network higher on 14/55 leaves), Δ MAE +0.01 [−0.06, 0.10].

New baseline minus existing baseline: vs existing baseline, no erosion: Δ IoU −0.001 [−0.015, 0.013], Δ MAE +0.00 [−0.05, 0.05]; vs existing threshold + chosen erosion (sensitivity): Δ IoU −0.011 [−0.028, 0.005], Δ MAE −0.02 [−0.06, 0.02].

Existing threshold with the chosen 1-px erosion (`paired_extra.json`): minus the same threshold without erosion Δ IoU +0.010 [0.006, 0.015]; networks minus it: SegFormer-B5 Δ IoU +0.067 [0.054, 0.082] (network higher on 54/55), Δ MAE −0.03 [−0.12, 0.05]; UNet++ Δ IoU +0.036 [0.026, 0.045] (network higher on 50/55), Δ MAE −0.11 [−0.17, −0.05]; DeepLabV3+ Δ IoU −0.003 [−0.011, 0.006] (network higher on 23/55), Δ MAE −0.00 [−0.08, 0.08].

New-baseline grade confusion (rows: mask-derived reference G0–G3; columns: predicted G0–G3): [[6, 0, 0, 0], [0, 25, 1, 0], [0, 0, 21, 0], [0, 0, 0, 2]].

## 7. Healthy rust-pack test leaves (n = 21 available locally)

Leaves flagged, with Wilson 95% intervals for the ≥ 0.5% proportion. Abaxial = the rust leaf rule; either face = maximum over both faces; mean = mean of the two faces (the leaf-spot leaf rule). The last column gives the median / maximum severity on the abaxial face and on the either-face maximum.

| Detector | Configuration | Abaxial > 0% | Abaxial ≥ 0.5% | Either face > 0% | Either face ≥ 0.5% | Mean ≥ 0.5% | Median / max % (abaxial; either) |
|---|---|---|---|---|---|---|---|
| Coffee leaf rust | **New baseline (primary)**: a* ≥ −8.2, erosion 2 px, exclusion none | 21/21 | 12/21 [36.5–75.5%] | 21/21 | 21/21 [84.5–100.0%] | 21/21 | 0.63 / 5.81; 68.06 / 96.11 |
| Coffee leaf rust | New index, specificity-constrained (secondary): a* ≥ −1.2, erosion 2 px, exclusion none | 19/21 | 0/21 [0.0–15.5%] | 21/21 | 0/21 [0.0–15.5%] | 0/21 | 0.03 / 0.14; 0.08 / 0.41 |
| Coffee leaf rust | Existing baseline, no erosion: exr ≥ 0.12, erosion 0 px, exclusion none | 21/21 | 20/21 [77.3–99.2%] | 21/21 | 21/21 [84.5–100.0%] | 21/21 | 0.96 / 2.34; 1.17 / 4.91 |
| Coffee leaf rust | Existing threshold + chosen erosion (sensitivity): exr ≥ 0.12, erosion 2 px, exclusion none | 21/21 | 8/21 [20.8–59.1%] | 21/21 | 9/21 [24.5–63.5%] | 5/21 | 0.33 / 1.51; 0.38 / 3.70 |
| Coffee leaf rust | Existing index re-tuned at chosen erosion (sensitivity): exr ≥ 0.116, erosion 2 px, exclusion none | 21/21 | 14/21 [45.4–82.8%] | 21/21 | 15/21 [50.0–86.2%] | 10/21 | 0.64 / 2.28; 0.65 / 5.13 |
| Coffee leaf rust | SegFormer-B5 (published): network | 17/21 | 0/21 [0.0–15.5%] | 19/21 | 0/21 [0.0–15.5%] | 0/21 | 0.09 / 0.36; 0.14 / 0.38 |
| Coffee leaf rust | UNet++ (published): network | 15/21 | 0/21 [0.0–15.5%] | 19/21 | 0/21 [0.0–15.5%] | 0/21 | 0.01 / 0.16; 0.01 / 0.27 |
| Coffee leaf rust | DeepLabV3+ (published): network | 7/21 | 0/21 [0.0–15.5%] | 13/21 | 0/21 [0.0–15.5%] | 0/21 | 0.00 / 0.09; 0.02 / 0.09 |
| American leaf spot | **New baseline (primary)**: a* ≥ −0.9, erosion 1 px, exclusion none | 21/21 | 0/21 [0.0–15.5%] | 21/21 | 0/21 [0.0–15.5%] | 0/21 | 0.05 / 0.16; 0.09 / 0.33 |
| American leaf spot | Existing baseline, no erosion: ri ≥ 0.026, erosion 0 px, exclusion none | 21/21 | 0/21 [0.0–15.5%] | 21/21 | 0/21 [0.0–15.5%] | 0/21 | 0.10 / 0.33; 0.12 / 0.33 |
| American leaf spot | Existing threshold + chosen erosion (sensitivity): ri ≥ 0.026, erosion 1 px, exclusion none | 21/21 | 0/21 [0.0–15.5%] | 21/21 | 0/21 [0.0–15.5%] | 0/21 | 0.04 / 0.15; 0.05 / 0.19 |
| American leaf spot | Existing index re-tuned at chosen erosion (sensitivity): ri ≥ 0.023, erosion 1 px, exclusion none | 21/21 | 0/21 [0.0–15.5%] | 21/21 | 0/21 [0.0–15.5%] | 0/21 | 0.04 / 0.16; 0.05 / 0.20 |
| American leaf spot | SegFormer-B5 (published): network | 16/21 | 0/21 [0.0–15.5%] | 19/21 | 0/21 [0.0–15.5%] | 0/21 | 0.05 / 0.21; 0.06 / 0.23 |
| American leaf spot | UNet++ (published): network | 21/21 | 0/21 [0.0–15.5%] | 21/21 | 1/21 [0.8–22.7%] | 0/21 | 0.09 / 0.28; 0.10 / 0.54 |
| American leaf spot | DeepLabV3+ (published): network | 21/21 | 0/21 [0.0–15.5%] | 21/21 | 0/21 [0.0–15.5%] | 0/21 | 0.09 / 0.31; 0.10 / 0.41 |

Rows for 'network' use the published models' predictions from `per_image.csv` (leaf-spot networks: cross-disease run on these leaves).

Coffee leaf rust new baseline, effect of the erosion on its healthy-leaf flags (chosen threshold, both faces of the 21 healthy test leaves; for rust a* mostly adaxial pixels): 1,090,733 flagged pixels without erosion; the chosen 2-px erosion removes 70,021 (6%); 1,020,712 (94%) remain.

American leaf spot new baseline, effect of the erosion on its healthy-leaf flags (chosen threshold, both faces of the 21 healthy test leaves): 17,800 flagged pixels without erosion; the chosen 1-px erosion removes 14,727 (83%); 3,073 (17%) remain.

**Face dependence of a\*.** The rust a* threshold was tuned on abaxial faces. On the healthy adaxial faces (dark, glossy green) a* ≥ −8.2 covers most of the face (median 68.1%, maximum 96.1%), so the either-face count (21/21) mainly reflects applying an abaxial-tuned a* threshold to the other face; the abaxial rule (median 0.63%, maximum 5.81%) is the fair test for a rust detector. The same effect was reported for a* in `results/baseline` (Section 5c).

**Sensitivity: pre-specified petiole/cut-end exclusions** (`exclusion_sensitivity.json`; thresholds and erosion fixed; not a selection step). Healthy test leaves ≥ 0.5% (abaxial / either face) and diseased rust test IoU:

| Exclusion | New baseline: healthy abaxial / either | New baseline: test IoU | Existing exr ≥ 0.12, 2 px: healthy abaxial / either | Existing: test IoU |
|---|---|---|---|---|
| none | 12/21 / 21/21 | 0.564 | 8/21 / 9/21 | 0.645 |
| open 4 | 12/21 / 21/21 | 0.564 | 8/21 / 9/21 | 0.645 |
| open 8 | 11/21 / 21/21 | 0.559 | 7/21 / 9/21 | 0.638 |
| open 12 | 11/21 / 21/21 | 0.558 | 7/21 / 9/21 | 0.636 |
| open 16 | 11/21 / 21/21 | 0.555 | 7/21 / 9/21 | 0.632 |
| ends 32 | 11/21 / 21/21 | 0.559 | 7/21 / 9/21 | 0.638 |
| ends 64 | 11/21 / 21/21 | 0.536 | 7/21 / 9/21 | 0.611 |
| ends 128 | 11/21 / 21/21 | 0.476 | 3/21 / 5/21 | 0.541 |

Healthy VALIDATION leaves (21, EARTH; secondary), ≥ 0.5% abaxial / either face: Coffee leaf rust new baseline (primary): 10/21 / 21/21; Coffee leaf rust new index, specificity-constrained (secondary): 0/21 / 0/21; Coffee leaf rust existing baseline, no erosion: 18/21 / 20/21; Coffee leaf rust existing threshold + chosen erosion (sensitivity): 9/21 / 9/21; American leaf spot new baseline (primary): 0/21 / 0/21; American leaf spot existing baseline, no erosion: 0/21 / 0/21; American leaf spot existing threshold + chosen erosion (sensitivity): 0/21 / 0/21.

## 8. Which conclusion about rust specificity survives

1. **Survives: a validation-tuned colour threshold does not keep healthy rust leaves below G1.** The detector-independent a* threshold with the validation-chosen 2-px erosion flags 12/21 healthy test leaves at ≥ 0.5% on the abaxial face [36.5–75.5%] and 21/21 on either face; the existing excess-red threshold with the same erosion flags 8/21 abaxial and 9/21 either face (re-tuned at that erosion, exr ≥ 0.116: 14/21 and 15/21). The pre-specified petiole/cut-end exclusions change these counts by at most one leaf, except 'ends 128' for excess red (3/21 abaxial, 5/21 either face) at a test IoU of 0.541. All three published rust networks: 0/21 on either face. Among the variants evaluated, only thresholds constrained on the healthy validation leaves reach 0/21 (a* ≥ −1.2 with 2-px erosion: test IoU 0.307 [0.264, 0.351], bias −3.49 pp, grade accuracy 9/21; excess red ≥ 0.154 in `results/baseline`: IoU 0.518). So the manuscript's statement that the networks' main added value for rust is keeping healthy leaves below the G1 cut-point holds for a detector-independent index as well.
2. **Does not survive as stated: the "21/21" count depends strongly on the leaf margin.** A 2-px erosion alone lowers the existing threshold's count from 21/21 to 9/21 (either face) and raises its rust test IoU to 0.645 [0.608, 0.679], level with UNet++ (Δ +0.001 [−0.032, 0.036]) and DeepLabV3+ (Δ −0.014 [−0.041, 0.014]); only SegFormer-B5 stays clearly above it (Δ +0.122 [0.092, 0.153]). Quote healthy-leaf counts for colour thresholds together with the erosion used, or cite the range across the variants here.
3. **At the any-pixel level nobody is specific.** Every threshold variant marks lesion pixels on 21/21 healthy leaves, and the networks on 19/21, 19/21 and 13/21 (either face). The specificity contrast exists only at the ≥ 0.5% (G1) cut-point.
4. **Circularity is visible in IoU, not in specificity.** The detector-independent a* baseline reaches a lower rust IoU (0.564) than the detector-like excess-red threshold (no erosion 0.600; with erosion 0.645; paired Δ −0.082 [−0.131, −0.035]), consistent with a reference that favours the detector's own colour criterion; against the a* baseline all three networks have a higher rust IoU with CIs excluding zero (Section 6). For American leaf spot (hand-corrected masks) the a* baseline equals the existing one (IoU 0.807 vs 0.807) and keeps 0/21 healthy leaves at ≥ 0.5%; the leaf-spot conclusions of `results/baseline` are unchanged.
5. **Site confound remains.** All healthy leaves are from EARTH and all diseased rust leaves from Sarchí, so any healthy-vs-diseased contrast is confounded with site and scan session.

## 9. Suggested manuscript sentences (LaTeX-ready)

1. *Methods.* Because the rust reference masks were seeded by a color detector whose rule is approximated by a hue boundary (excess red $>$0.133, HSV hue $<$70$^{\circ}$), we added a second training-free baseline on an index that reproduces the detector's weak masks poorly, CIELAB $a^*$, with the threshold, a leaf-edge erosion (0--8 pixels) and an optional petiole exclusion chosen jointly on the validation leaves by leaf IoU; the existing threshold was also re-evaluated with the chosen erosion.
2. *Results.* With the validation-chosen 2-pixel erosion, the $a^*$ threshold reached a rust leaf IoU of 0.564 (95\% CI 0.513--0.610; MAE 1.20 percentage points, QWK 0.68) and flagged 12 of the 21 available healthy test leaves at $\geq$0.5\% on the abaxial face; the excess-red threshold with the same erosion flagged 8 (abaxial) and 9 (either face), versus none for the three networks.
3. *Limitations.* Thresholds constrained to leave every healthy validation leaf below 0.5\% did reach 0/21 but lost lesion delineation (rust IoU 0.307 for $a^*$), and these healthy-leaf counts are sensitive to how the leaf margin is handled.

## 10. Notes and caveats

- The detector's rule is **inferred** from its output masks; its code is not available. Do not state its index or thresholds as documented facts.
- Healthy coverage: 21 of the paper's 54 healthy rust-test leaves (images for 33 were not retained). Proportions over 21 leaves have wide intervals (Wilson, Section 7).
- The baseline uses the reference leaf mask (exact denominator); the networks segment the leaf themselves. This favours the baseline on severity.
- Thresholds were tuned on abaxial rust faces; the either-face rule applies them to adaxial faces they were not tuned for (large effect for a*, Section 7).
- The index choice used training-leaf relatedness and a validation-only pre-run; no new index was evaluated on test before the choice was fixed. Test values of the other candidate indices were not computed.
- The petiole/cut-end exclusion was feasible and pre-specified but was not selected on validation; its effect on healthy leaves is a sensitivity analysis only.
- Expert-panel identities (name, occupation, years of experience; round4_plan.md) are outside this analysis and are not recorded in this folder.

## Files

- `r4_baseline.py`
- `detector_rule.py`
- `exclusion_sensitivity.py`
- `paired_extra.py`
- `make_md.py`
- `results.json`
- `detector_rule.json`
- `exclusion_sensitivity.json`
- `paired_extra.json`
- `validation_only.json`
- `per_leaf_test.csv`
- `per_leaf_healthy.csv`
- `validation_grid.csv`
- `run.log`
- `valonly.log`
- `detector_rule.log`
- `exclusion_sensitivity.log`
- `results.md`

## Verifier corrections

Independent verification, 2026-09-28 (scripts and outputs in `verifier/`; the verifier's scripts were written from scratch and do not import `r4_baseline.py`). The copy of this file before these corrections is `verifier/results_md_before_verifier.md`.

**What was re-derived independently and matched (to the precision printed above):**
- Protocol: 388 val/test images rebuilt (leaf = alpha > 8, lesion = mask > 127 AND leaf, 1024 letterbox); reference leaf and lesion pixel counts equal `per_image.csv` for all 388. Leaf sets: rust val 21 / test 21 diseased, leaf spot val 55 / test 55, healthy rust-pack val 21 / test 21. Rust = abaxial face, leaf spot = mean of both face IoUs and severities; grades < 0.5, 0.5–< 5, 5–< 20, ≥ 20%; leaf-spot mask-derived grade from the mean reference severity of both faces.
- Validation selection (own a* code, erosion via `scipy.ndimage.binary_erosion`, own opening/axis-end exclusions): rust a* ≥ −8.2, 2 px, no exclusion (val IoU 0.5749; by erosion 0.474 / 0.558 / 0.575 / 0.571 / 0.551); leaf spot a* ≥ −0.9, 1 px, no exclusion (0.8034). No exclusion option beat 'none' (best: rust e2 open 4, 0.5748). Specificity-constrained a* ≥ −1.2; excess red re-tuned at 2 px = 0.116.
- Every test row of Section 6 (IoU, MAE, bias with 95% CIs, grade accuracy with bootstrap CI, QWK with CI, visual-grade columns), all paired differences in Section 6 and `paired_extra.json`, the confusion matrix, every row of the Section 7 healthy-leaf table (counts, Wilson intervals, medians and maxima), and the exclusion-sensitivity rows 'none', 'open 8' and 'ends 128' for both rules. Bootstrap unit = leaf, 10,000 resamples, default_rng(20260928) per disease, percentile intervals; healthy proportions Wilson.
- Detector characterisation (`verifier/verify_detector.py`, a different deterministic stride-4 1/16 sample of the 283 seed faces): HSV hue < 68.75° pooled IoU 0.652 (reported 0.651); best excess-red rule 0.641 (0.641); best a* rule ≥ −3.1, 0.335 (0.336). The source `roya_lesion.py` was also not found by the verifier (searched by name under C:/dev and <USER_HOME> to depth 4, and by content in .py files under C:/dev); the leaf-spot SAM 2 refine script (`notebooks/pg26038-sam2-lesion-refine`) also names only "the classical detector" without its code.

**Corrections made to the text:**
1. *Excess-red detector rule.* The best single excess-red rule is excess red > 2/15 ≈ 0.1333, i.e. strictly above the value of neutral-grey pixels (the stored threshold is 0.13333333333333336, which excludes pixels at the grey value). Written as "≥ 0.1333" it would include neutral-grey leaf pixels and the pooled IoU falls to 0.546 (verifier sample). Sections 0, 1 (table and comparison paragraph) and the Section 9 LaTeX sentence now say "> 0.1333" / "$>$0.133". Added: excess red ≥ 0.12 (the existing baseline) itself reproduces the weak masks with pooled IoU 0.526 (verifier sample).
2. *Section 7:* a copy-paste phrase "(for rust a* mostly adaxial pixels)" was removed from the American-leaf-spot erosion-decomposition sentence.
3. *Section 8.1:* "Only thresholds constrained on the healthy validation leaves reach 0/21" is now qualified as "Among the variants evaluated", because other indices or rules were not tested on this point.
`make_md.py` was patched for corrections 1 (summary, comparison paragraph, LaTeX sentence), 2 and 3; the table cell of correction 1 and this section are not generated by `make_md.py` and must be re-applied if it is rerun.

**Caveats added by the verifier (numbers unchanged):**
- *Pure-black leaf pixels.* 1,475 leaf pixels (alpha > 8) in 84 of the 168 rust-pack val/test faces are pure black (0,0,0) after letterboxing; none in the leaf-spot pack; all lie on the leaf margin and none survives a 1-px erosion (`verifier/check_black_all.json`). The excess-red and redness-index code treats them as index 0 (not lesion), whereas a* = 0 for black, so the a* rule counts them as lesion. This does not affect any primary or eroded configuration; it affects only the a* rows without erosion (e.g. rust validation IoU at e = 0, 0.474, and the "flagged pixels without erosion" decomposition in Section 7). The existing-baseline 20/21 abaxial count (no erosion) depends on the not-lesion convention: counting black pixels as lesion gives 21/21 (`verifier/check_black.py`).
- *Multiplicity.* The three network-minus-a* rust IoU contrasts are reported with unadjusted 95% CIs. On the same resamples, bootstrap two-sided p-values with Holm adjustment over the three networks: SegFormer-B5 < 0.001, UNet++ 0.005, DeepLabV3+ 0.017, so all three remain significant; Bonferroni 98.33% intervals: +0.204 [0.143, 0.273], +0.083 [0.017, 0.152], +0.068 [−0.001, 0.137] (`verifier/check_multiplicity.json`). Against excess red ≥ 0.12 with 2-px erosion: UNet++ Holm p = 0.95, DeepLabV3+ 0.68, SegFormer-B5 < 0.001. No TOST or equivalence claim is made in this folder; "level with UNet++ and DeepLabV3+" means only that the CIs include zero, not demonstrated equivalence.
- *Order of decisions.* `validation_only.json` (08:04) records `index_choice: "hab"`, the value of the script constant at the time of the validation-only run; the final a* choice was set afterwards and is justified by `detector_rule.json` (created 07:39, training leaves only) and the validation-only numbers. The files contain no test evaluation of any other candidate index, but whether the choice was fixed before any test evaluation can only be taken from the analyst's statement; file timestamps do not prove it.
- *Grade-accuracy intervals* in Section 6 are bootstrap percentile intervals (Wilson intervals are in `results.json`).
