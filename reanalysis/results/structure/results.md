# Validation-set results and data structure

PG-26038 major revision, reanalysis for the reviewers' questions on the rust reference and protocol, clustering by collection date, grade distributions and class weights. Scripts: `structure_analysis.py` (all numbers, `results.json`), `make_figures.py` (figures), `make_md.py` (this file). No model inference was re-run; everything is computed from `reanalysis/out/per_image.csv`, the saved class maps and the two packs.

**Leaf-level rules (as in the paper).** Rust = abaxial face; American leaf spot = mean of the two faces (IoU = mean of face IoUs, severity = mean of face severities). Reference grade: rust = cut-points 0.5/5/20% on the reference severity; leaf spot = annotator's visual grade. MAE and bias in percentage points (pp); LoA = bias ± 1.96 SD. 95% CIs: 10,000 leaf resamples, percentile, seed 20260927. Diseased leaves only unless stated.

## 0. Reproduction of the published test values

All published test values reproduced exactly: **yes** (test diseased leaves: rust 21, leaf spot 55). Integrity: reference labels rebuilt from the packs match `per_image.csv` for all 388 own-disease validation/test images (rows matched: 1164/1164, i.e. 388 images × 3 models); predicted counts recomputed from the saved class maps match for 1164/1164 rows.

| Disease | Model | IoU | MAE | Bias | QWK | LoA | Published (IoU / MAE / bias / QWK / LoA) |
|---|---|---|---|---|---|---|---|
| Rust | SegFormer-B5 | 0.767 | 0.99 | +0.99 | 0.88 | −1.68 to +3.66 | 0.767 / 0.99 / +0.99 / 0.88 / −1.68 to +3.66 |
| Rust | UNet++ | 0.647 | 1.77 | +1.75 | 0.79 | −2.68 to +6.17 | 0.647 / 1.77 / +1.75 / 0.79 / −2.68 to +6.17 |
| Rust | DeepLabV3+ | 0.632 | 1.47 | +1.41 | 0.88 | −3.07 to +5.89 | 0.632 / 1.47 / +1.41 / 0.88 / −3.07 to +5.89 |
| Leaf spot | SegFormer-B5 | 0.884 | 0.22 | +0.16 | 0.82 | −0.47 to +0.79 | 0.884 / 0.22 / +0.16 / 0.82 / −0.47 to +0.79 |
| Leaf spot | UNet++ | 0.853 | 0.15 | −0.03 | 0.85 | −0.50 to +0.44 | 0.853 / 0.15 / −0.03 / 0.85 / −0.50 to +0.44 |
| Leaf spot | DeepLabV3+ | 0.815 | 0.25 | +0.08 | 0.84 | −0.66 to +0.82 | 0.815 / 0.25 / +0.08 / 0.84 / −0.66 to +0.82 |

## 1. Validation vs test, six models

Diseased leaves: rust 21 validation / 21 test; leaf spot 55 / 55. The validation split was used for checkpoint selection (and, for the rust CNNs, for the healthy-leaf false-positive penalty), so it is not an independent replication; it is shown to document consistency. All rust masks (train, validation, test) come from the same semi-automatic detector + SAM 2 protocol, so rust validation and test share one reference protocol.

**Table 1a. Leaf-level metrics on validation and test (95% leaf-bootstrap CI).**

| Disease | Model | Split | IoU [95% CI] | MAE, pp [95% CI] | Bias, pp [95% CI] | QWK [95% CI] | LoA, pp | Mean ref. severity, % (max) | MAE / mean ref. severity |
|---|---|---|---|---|---|---|---|---|---|
| Rust | SegFormer-B5 | Validation | 0.776 [0.747, 0.802] | 0.58 [0.29, 0.95] | +0.57 [0.28, 0.95] | 1.00 [1.00, 1.00] | −1.00 to +2.14 | 3.35 (6.3) | 17% |
| Rust | SegFormer-B5 | Test | 0.767 [0.743, 0.791] | 0.99 [0.47, 1.60] | +0.99 [0.47, 1.60] | 0.88 [0.68, 1.00] | −1.68 to +3.66 | 5.47 (19.3) | 18% |
| Rust | UNet++ | Validation | 0.671 [0.637, 0.701] | 1.08 [0.66, 1.59] | +1.00 [0.55, 1.53] | 0.88 [0.55, 1.00] | −1.33 to +3.33 | 3.35 (6.3) | 32% |
| Rust | UNet++ | Test | 0.647 [0.608, 0.684] | 1.77 [0.89, 2.75] | +1.75 [0.86, 2.74] | 0.79 [0.55, 0.94] | −2.68 to +6.17 | 5.47 (19.3) | 32% |
| Rust | DeepLabV3+ | Validation | 0.653 [0.620, 0.681] | 1.00 [0.56, 1.52] | +0.88 [0.39, 1.44] | 0.67 [0.29, 1.00] | −1.58 to +3.34 | 3.35 (6.3) | 30% |
| Rust | DeepLabV3+ | Test | 0.632 [0.590, 0.668] | 1.47 [0.62, 2.52] | +1.41 [0.54, 2.47] | 0.88 [0.68, 1.00] | −3.07 to +5.89 | 5.47 (19.3) | 27% |
| Leaf spot | SegFormer-B5 | Validation | 0.859 [0.826, 0.886] | 0.45 [0.22, 0.79] | +0.18 [−0.08, 0.54] | 0.88 [0.79, 0.97] | −2.15 to +2.51 | 5.89 (49.7) | 8% |
| Leaf spot | SegFormer-B5 | Test | 0.884 [0.868, 0.900] | 0.22 [0.15, 0.30] | +0.16 [0.08, 0.25] | 0.82 [0.72, 0.91] | −0.47 to +0.79 | 5.32 (26.8) | 4% |
| Leaf spot | UNet++ | Validation | 0.833 [0.795, 0.864] | 0.31 [0.17, 0.50] | −0.05 [−0.22, 0.16] | 0.84 [0.74, 0.92] | −1.48 to +1.37 | 5.89 (49.7) | 5% |
| Leaf spot | UNet++ | Test | 0.853 [0.830, 0.874] | 0.15 [0.10, 0.20] | −0.03 [−0.09, 0.03] | 0.85 [0.75, 0.94] | −0.50 to +0.44 | 5.32 (26.8) | 3% |
| Leaf spot | DeepLabV3+ | Validation | 0.790 [0.752, 0.823] | 0.48 [0.25, 0.81] | +0.07 [−0.21, 0.43] | 0.85 [0.75, 0.94] | −2.30 to +2.44 | 5.89 (49.7) | 8% |
| Leaf spot | DeepLabV3+ | Test | 0.815 [0.790, 0.837] | 0.25 [0.18, 0.33] | +0.08 [−0.02, 0.18] | 0.84 [0.73, 0.92] | −0.66 to +0.82 | 5.32 (26.8) | 5% |

**Table 1b. Test minus validation (point difference, 95% CI from independent leaf resamples of the two sets).**

| Disease | Model | ΔIoU | ΔMAE, pp | ΔBias, pp | ΔQWK |
|---|---|---|---|---|---|
| Rust | SegFormer-B5 | −0.009 [−0.045, 0.029] | +0.41 [−0.23, 1.11] | +0.42 [−0.22, 1.12] | −0.12 [−0.32, 0.00] |
| Rust | UNet++ | −0.024 [−0.073, 0.026] | +0.69 [−0.32, 1.77] | +0.75 [−0.29, 1.84] | −0.09 [−0.38, 0.26] |
| Rust | DeepLabV3+ | −0.021 [−0.071, 0.028] | +0.47 [−0.53, 1.59] | +0.53 [−0.51, 1.68] | +0.20 [−0.14, 0.60] |
| Leaf spot | SegFormer-B5 | +0.026 [−0.007, 0.062] | −0.23 [−0.57, 0.02] | −0.02 [−0.38, 0.26] | −0.06 [−0.19, 0.07] |
| Leaf spot | UNet++ | +0.020 [−0.019, 0.063] | −0.16 [−0.36, −0.01] | +0.03 [−0.19, 0.21] | +0.01 [−0.12, 0.15] |
| Leaf spot | DeepLabV3+ | +0.024 [−0.017, 0.070] | −0.23 [−0.57, 0.02] | +0.01 [−0.36, 0.31] | −0.01 [−0.15, 0.12] |

Reading:
- Rust IoU was consistent between validation and test for all three models (differences −0.009, −0.024, −0.021; every 95% CI includes 0), with the same ordering of the published single-run models (SegFormer-B5 > UNet++ > DeepLabV3+). This ordering describes these trained models, not the architectures: the matched-protocol rust retraining (revision brief, items 9 and 11) suggests that it reflects the unequal training protocols of the published models. That retraining changed several settings at once (epochs and early-stopping patience, geometric augmentation only, learning rate and model selection on validation lesion IoU), so it does not isolate the training budget, and it is a single run with the seed repeats still pending.
- Rust validation MAE and bias were lower than on test (0.58, 1.08, 1.00 vs 0.99, 1.77, 1.47 pp), although the differences were within bootstrap uncertainty (test minus validation MAE: SegFormer-B5 +0.41 [−0.23, 1.11]; UNet++ +0.69 [−0.32, 1.77]; DeepLabV3+ +0.47 [−0.53, 1.59] pp; the CIs of the bias differences also include 0). A possible contributor, not an established effect, is that the validation leaves are less severe (mean reference severity 3.35% vs 5.47%; maximum 6.3% vs 19.3%) while the absolute error grows with severity (Figure 5); MAE relative to the mean reference severity is similar in the two splits (17%, 32%, 30% validation vs 18%, 32%, 27% test).
- Rust validation reference grades are only G1 (16) and G2 (5), so rust validation QWK (1.00, 0.88, 0.67) rests on one grade boundary. Its CIs are wide for the two CNNs (UNet++ [0.55, 1.00], DeepLabV3+ [0.29, 1.00]) and degenerate for SegFormer-B5 ([1.00, 1.00]): all 21 of its validation grades are correct (exact agreement 100%), so every resample with a defined kappa gives 1.00 and the interval conveys no sampling uncertainty.
- Leaf spot IoU was slightly lower on validation than on test (differences +0.026, +0.020, +0.024; CIs include 0). Leaf-spot validation MAE (0.45, 0.31, 0.48 pp) and LoA are wider than on test, driven in part by one validation leaf with 49.7% reference severity (SegFormer-B5 error +7.6 pp on that leaf; the test maximum is 26.8%). Without that leaf, validation MAE is 0.32, 0.23, 0.35 pp. Leaf-spot QWK was similar (0.88, 0.84, 0.85 validation vs 0.82, 0.85, 0.84 test).
- In both splits the largest grade error of any model was 1 grade (all errors adjacent-grade).

## 2. Grade distributions per split

Leaves in the local packs: rust pack 198/42/42 (train/validation/test; half diseased, half healthy), leaf-spot pack 259/55/55. Severities in letterbox space (as in the evaluation); grades recomputed at native resolution are identical for every leaf (rust images are already 1024 px on the long side; leaf-spot severities differ by at most 0.030 pp).

**Table 2a. Reference grade counts per split (G0/G1/G2/G3, leaves).**

| Disease | Grade source | Train | Validation | Test | All |
|---|---|---|---|---|---|
| Rust (diseased) | Mask-derived, abaxial (paper rule) | 0/66/31/2 | 0/16/5/0 | 1/11/9/0 | 1/93/45/2 |
| Rust (diseased) | Mask-derived, adaxial | 1/54/43/1 | 1/14/5/1 | 2/10/8/1 | 4/78/56/3 |
| Rust (diseased) | Mask-derived, mean of both faces | 0/55/43/1 | 0/14/7/0 | 1/11/9/0 | 1/80/59/1 |
| Rust (healthy controls) | Mask-derived (all G0) | 99/0/0/0 | 21/0/0/0 | 21/0/0/0 | 141/0/0/0 |
| Leaf spot | Visual (annotator; paper reference) | 0/140/110/9 | 0/30/23/2 | 0/30/23/2 | 0/200/156/13 |
| Leaf spot | Mask-derived, mean of both faces | 16/133/102/8 | 3/28/22/2 | 6/26/21/2 | 25/187/145/12 |

**Table 2b. Reference severity (%) of diseased leaves, median [IQR] (range).**

| Disease | Train | Validation | Test |
|---|---|---|---|
| Rust | 3.26 [1.68, 6.45] (0.53–23.4) | 3.34 [2.04, 4.00] (0.66–6.3) | 3.83 [1.89, 7.46] (0.41–19.3) |
| Leaf spot | 4.15 [1.34, 7.64] (0.19–49.7) | 4.49 [1.48, 7.34] (0.34–49.7) | 4.15 [0.87, 7.45] (0.13–26.8) |

**Table 2c. Leaf spot: visual grade vs mask-derived grade.**

| Split | Exact agreement | QWK | Mask grade below visual | Mask grade above visual |
|---|---|---|---|---|
| Train | 90.0% | 0.87 | 26 | 0 |
| Validation | 92.7% | 0.90 | 4 | 0 |
| Test | 85.5% | 0.83 | 8 | 0 |
| All | 89.7% | 0.87 | 38 | 0 |

All splits pooled (rows visual G1–G3, columns mask G0–G3): G1 → 25/175/0/0; G2 → 0/12/144/0; G3 → 0/0/1/12. The mask-derived grade never exceeded the visual grade; disagreements were always one grade lower (mostly visual G1 leaves whose mask severity is below 0.5%).

Grade mix did not differ detectably between splits (permutation chi-square tests, 10,000 permutations): rust mask-derived p = 0.22; leaf spot visual p = 1.00 (visual-grade counts are identical in validation and test and proportional in training, consistent with a grade-stratified split); leaf spot mask-derived p = 0.93. With 21 rust leaves per held-out split these tests have little power, and the rust validation leaves are milder than the test leaves (G2 5/21 = 24% vs 9/21 = 43%; maximum reference severity 6.3% vs 19.3%; neither held-out split has a G3 leaf, the training split has 2). Rust healthy leaves: 99/21/21 (no reference lesion pixels inside the leaf on either face; reference severity 0%). **Only 21 healthy rust-pack test leaves are present locally; the paper's specificity analysis used 54.**

Note: the local rust pack holds 141 diseased rust leaves (99 train + 21 validation + 21 test, both faces each); the revision brief says 142. The difference was not resolved here and should be checked against the manuscript's leaf counts.

## 3. Collection dates and date-clustered bootstrap

**Table 3a. Leaves per collection date (from the leaf_id prefix) and split.**

| Set | Site | Date | Train | Validation | Test |
|---|---|---|---|---|---|
| Rust (diseased) | Sarchí | 2026-08-19 | 99 | 21 | 21 |
| Rust pack, healthy controls | EARTH (Guácimo) | 2026-06-12 | 50 | 6 | 13 |
| Rust pack, healthy controls | EARTH (Guácimo) | 2026-06-15 | 46 | 15 | 6 |
| Rust pack, healthy controls | EARTH (Guácimo) | 2026-06-22 | 3 | 0 | 2 |
| Leaf spot | EARTH (Guácimo) | 2026-06-21 | 170 | 33 | 36 |
| Leaf spot | EARTH (Guácimo) | 2026-06-28 | 89 | 22 | 19 |

Leaf-spot grade mix differs strongly by date (visual G1/G2/G3): train 06-21: 127/43/0; train 06-28: 13/67/9; validation 06-21: 24/9/0; validation 06-28: 6/14/2; test 06-21: 26/10/0; test 06-28: 4/13/2. Date and severity are therefore confounded for leaf spot.

**Rust:** All rust leaves (train, val, test) were collected on a single date (2026-08-19); a date-clustered bootstrap is not possible for rust, and between-date variation cannot be estimated. Only the leaf bootstrap is reported.

**Table 3b. Leaf spot, test (55 leaves: 36 on 2026-06-21, 19 on 2026-06-28): 95% CIs under three bootstrap schemes.**

| Metric | Model | Estimate | Per date (06-21 / 06-28) | Leaf bootstrap | Date-stratified | Date-clustered (two-stage) | Width ratio, clustered / leaf |
|---|---|---|---|---|---|---|---|
| IoU | SegFormer-B5 | 0.884 | 0.872 / 0.909 | [0.868, 0.900] (width 0.031) | [0.869, 0.899] (width 0.031) | [0.862, 0.917] (width 0.055) | 1.8 |
| IoU | UNet++ | 0.853 | 0.833 / 0.891 | [0.830, 0.874] (width 0.043) | [0.831, 0.873] (width 0.042) | [0.820, 0.901] (width 0.081) | 1.9 |
| IoU | DeepLabV3+ | 0.815 | 0.796 / 0.850 | [0.790, 0.837] (width 0.047) | [0.791, 0.837] (width 0.046) | [0.781, 0.862] (width 0.081) | 1.7 |
| MAE, pp | SegFormer-B5 | 0.22 | 0.15 / 0.36 | [0.15, 0.30] (width 0.15) | [0.15, 0.29] (width 0.14) | [0.11, 0.43] (width 0.32) | 2.2 |
| MAE, pp | UNet++ | 0.15 | 0.10 / 0.23 | [0.10, 0.20] (width 0.10) | [0.10, 0.20] (width 0.09) | [0.08, 0.29] (width 0.20) | 2.1 |
| MAE, pp | DeepLabV3+ | 0.25 | 0.14 / 0.45 | [0.18, 0.33] (width 0.15) | [0.18, 0.32] (width 0.13) | [0.11, 0.52] (width 0.41) | 2.7 |

Schemes: leaf = resample the 55 leaves; date-stratified = keep both dates and resample leaves within each date; date-clustered = resample the two dates with replacement, then leaves within each drawn date. Only two collection dates exist for leaf spot; a cluster bootstrap with two clusters estimates between-date variance from two values and is not a reliable basis for inference. It is reported as a sensitivity illustration only. Part of the between-date difference reflects the different grade mix of the two dates (the 06-28 leaves are more severe, which raises MAE and IoU), not only a date effect.

**Table 3c. Rust, test (21 leaves, one date): leaf-bootstrap CIs only.**

| Model | IoU [95% CI] (width) | MAE, pp [95% CI] (width) |
|---|---|---|
| SegFormer-B5 | 0.767 [0.743, 0.791] (0.048) | 0.99 [0.47, 1.60] (1.13) |
| UNet++ | 0.647 [0.608, 0.684] (0.075) | 1.77 [0.89, 2.75] (1.87) |
| DeepLabV3+ | 0.632 [0.590, 0.668] (0.078) | 1.47 [0.62, 2.52] (1.90) |

## 4. Rust: abaxial, adaxial and both-face results (Supplement)

For each face rule the reference severity and reference grade use the same rule (adaxial face; or mean of the two faces, with IoU = mean of the face IoUs). The last column compares each rule's predicted grade with the paper's abaxial reference grade. 21 diseased leaves per split; 95% leaf-bootstrap CIs from the same resamples as Table 1a, so the abaxial rows repeat the Table 1a values and CIs exactly.

**Table 4a. Rust, test split.**

| Model | Face rule | IoU [95% CI] | MAE, pp [95% CI] | Bias, pp [95% CI] | QWK [95% CI] | LoA, pp | QWK vs abaxial reference grade |
|---|---|---|---|---|---|---|---|
| SegFormer-B5 | Abaxial (paper) | 0.767 [0.743, 0.791] | 0.99 [0.47, 1.60] | +0.99 [0.47, 1.60] | 0.88 [0.68, 1.00] | −1.68 to +3.66 | 0.88 |
| SegFormer-B5 | Adaxial | 0.613 [0.560, 0.662] | 1.36 [0.80, 1.98] | +0.59 [−0.20, 1.39] | 0.80 [0.63, 0.94] | −3.08 to +4.27 | 0.63 |
| SegFormer-B5 | Both faces (mean) | 0.690 [0.659, 0.719] | 1.00 [0.56, 1.52] | +0.79 [0.28, 1.37] | 0.88 [0.69, 1.00] | −1.77 to +3.35 | 0.88 |
| UNet++ | Abaxial (paper) | 0.647 [0.608, 0.684] | 1.77 [0.89, 2.75] | +1.75 [0.86, 2.74] | 0.79 [0.55, 0.94] | −2.68 to +6.17 | 0.79 |
| UNet++ | Adaxial | 0.441 [0.388, 0.492] | 2.16 [1.50, 2.90] | +0.79 [−0.38, 1.81] | 0.58 [0.31, 0.77] | −4.39 to +5.96 | 0.76 |
| UNet++ | Both faces (mean) | 0.544 [0.508, 0.576] | 1.58 [1.03, 2.24] | +1.27 [0.57, 2.02] | 0.82 [0.57, 1.00] | −2.20 to +4.74 | 0.82 |
| DeepLabV3+ | Abaxial (paper) | 0.632 [0.590, 0.668] | 1.47 [0.62, 2.52] | +1.41 [0.54, 2.47] | 0.88 [0.68, 1.00] | −3.07 to +5.89 | 0.88 |
| DeepLabV3+ | Adaxial | 0.499 [0.442, 0.551] | 1.60 [0.80, 2.55] | −0.71 [−1.84, 0.27] | 0.77 [0.52, 0.94] | −5.69 to +4.27 | 0.85 |
| DeepLabV3+ | Both faces (mean) | 0.565 [0.525, 0.601] | 0.96 [0.48, 1.52] | +0.35 [−0.30, 1.01] | 1.00 [1.00, 1.00] | −2.69 to +3.39 | 0.86 |

**Table 4b. Rust, validation split.**

| Model | Face rule | IoU [95% CI] | MAE, pp [95% CI] | Bias, pp [95% CI] | QWK [95% CI] | LoA, pp | QWK vs abaxial reference grade |
|---|---|---|---|---|---|---|---|
| SegFormer-B5 | Abaxial (paper) | 0.776 [0.747, 0.802] | 0.58 [0.29, 0.95] | +0.57 [0.28, 0.95] | 1.00 [1.00, 1.00] | −1.00 to +2.14 | 1.00 |
| SegFormer-B5 | Adaxial | 0.641 [0.593, 0.688] | 1.12 [0.56, 1.81] | −0.29 [−1.13, 0.41] | 0.77 [0.55, 1.00] | −3.95 to +3.38 | 0.54 |
| SegFormer-B5 | Both faces (mean) | 0.709 [0.678, 0.739] | 0.77 [0.47, 1.10] | +0.14 [−0.34, 0.57] | 1.00 [1.00, 1.00] | −1.98 to +2.26 | 0.54 |
| UNet++ | Abaxial (paper) | 0.671 [0.637, 0.701] | 1.08 [0.66, 1.59] | +1.00 [0.55, 1.53] | 0.88 [0.55, 1.00] | −1.33 to +3.33 | 0.88 |
| UNet++ | Adaxial | 0.425 [0.374, 0.470] | 3.47 [1.92, 5.61] | −1.01 [−3.58, 1.05] | 0.40 [−0.09, 0.81] | −12.03 to +10.01 | 0.44 |
| UNet++ | Both faces (mean) | 0.548 [0.520, 0.575] | 1.93 [1.17, 2.97] | −0.00 [−1.35, 1.08] | 0.40 [−0.03, 0.79] | −5.80 to +5.80 | 0.59 |
| DeepLabV3+ | Abaxial (paper) | 0.653 [0.620, 0.681] | 1.00 [0.56, 1.52] | +0.88 [0.39, 1.44] | 0.67 [0.29, 1.00] | −1.58 to +3.34 | 0.67 |
| DeepLabV3+ | Adaxial | 0.457 [0.368, 0.538] | 3.57 [1.45, 6.20] | −2.62 [−5.46, −0.30] | 0.42 [−0.05, 0.85] | −14.85 to +9.61 | 0.67 |
| DeepLabV3+ | Both faces (mean) | 0.555 [0.506, 0.601] | 1.90 [0.89, 3.19] | −0.87 [−2.35, 0.35] | 0.44 [−0.04, 0.81] | −7.30 to +5.55 | 0.88 |

Reference grade mix (test): abaxial 1/11/9/0, adaxial 2/10/8/1, both faces 1/11/9/0 (G0/G1/G2/G3). The adaxial rust masks were produced by the same detector + SAM 2 protocol as the abaxial ones; the manuscript describes the adaxial face as carrying chlorosis rather than sporulation, so adaxial agreement measures a different visual target. On validation, adaxial and both-face QWK vary widely between models (0.40–1.00) and most of their CIs are wide. A QWK CI of [1.00, 1.00] (e.g. SegFormer-B5 both faces on validation, DeepLabV3+ both faces on test) means that every grade was correct; the percentile interval is then degenerate and conveys no sampling uncertainty.

Reading: for all three models the adaxial face gives the lowest IoU and the widest limits of agreement on test and validation; the mean of the two faces is intermediate in IoU. The abaxial rule used in the paper gives the highest IoU for every model and split.

## 5. Class pixel frequencies in the training split and implied inverse-frequency weights

Pixel counts summed over training images after letterboxing to 1024 × 1024 (the training input; the letterbox padding is background, class 0). Flip/rotation augmentation does not change class counts.

**Table 5. Class frequencies (%) and cross-entropy weights (background / leaf / lesion).**

| Set | Images | Background | Leaf | Lesion | Lesion share of leaf tissue | Inverse-frequency weights (scaled to bg = 0.25) | Ratio leaf:bg, lesion:bg | Paper weights (ratios) |
|---|---|---|---|---|---|---|---|---|
| Rust, train (all images) | 396 | 92.3 | 7.53 | 0.173 | 2.25% | 0.25 / 3.06 / 133 | 12.3, 533 | 0.25 / 0.5 / 3.0 (2, 12) |
| Rust, train (diseased leaves only) | 198 | 92.8 | 6.82 | 0.346 | 4.83% | 0.25 / 3.40 / 67 | 13.6, 268 | 0.25 / 0.5 / 3.0 (2, 12) |
| Leaf spot, train | 518 | 92.6 | 6.96 | 0.391 | 5.32% | 0.25 / 3.33 / 59 | 13.3, 237 | 0.25 / 0.5 / 3.0 (2, 12) |
| Both diseases, train | 914 | 92.5 | 7.21 | 0.297 | 3.95% | 0.25 / 3.21 / 78 | 12.8, 312 | 0.25 / 0.5 / 3.0 (2, 12) |
| Rust, all splits | 564 | 92.3 | 7.56 | 0.175 | 2.26% | 0.25 / 3.05 / 132 | 12.2, 527 | 0.25 / 0.5 / 3.0 (2, 12) |
| Leaf spot, all splits | 738 | 92.6 | 7.02 | 0.394 | 5.31% | 0.25 / 3.30 / 59 | 13.2, 235 | 0.25 / 0.5 / 3.0 (2, 12) |

Alternative weightings, training split, scaled to background = 0.25 (rust; leaf spot): square-root inverse frequency 0.25 / 0.88 / 5.77; 0.25 / 0.91 / 3.85. ENet log weighting 1/ln(1.02 + f) 0.25 / 1.82 / 7.72; 0.25 / 1.94 / 7.05. Excluding the letterbox padding, background/leaf/lesion = 90.03 / 9.75 / 0.22% (rust) and 90.49 / 9.00 / 0.51% (leaf spot).

Reading: the paper's weights 0.25/0.5/3.0 increase with class rarity, but they do **not** follow inverse class frequency. Inverse frequency implies leaf:background ≈ 12:1 and lesion:background ≈ 533:1 for rust (13:1 and 237:1 for leaf spot), against 2:1 and 12:1 in the paper; the leaf/background ratio (2) is even below the square-root inverse-frequency ratio. No single power w ∝ f^(−p) reproduces all three paper weights (the three pairwise ratios imply p ≈ 0.28, 0.40, 0.47 for rust). Under the paper's weights, lesion pixels carry 1.9% (rust) and 4.2% (leaf spot) of the total cross-entropy weight, versus 0.17% and 0.39% unweighted. In the training kernels the Focal Tversky term (weight 0.7) is computed on the lesion class alone, so lesion rarity is additionally addressed by that term; how much of the correction each term contributes was not measured here.

## Suggested manuscript sentences

1. (Results or Supplement, validation.) "On the validation split, whose rust masks were produced with the same semi-automatic protocol as the test masks, leaf-level rust IoU was 0.776 (SegFormer-B5), 0.671 (UNet++) and 0.653 (DeepLabV3+), within 0.03 of the test values, and leaf-spot IoU was 0.859, 0.833 and 0.790 (Table Sx). Rust validation MAE was lower than on test (0.58–1.08 vs 0.99–1.77 percentage points), although the differences were within bootstrap uncertainty (e.g. SegFormer-B5, test minus validation +0.41 percentage points, 95% CI −0.23 to +1.11); the validation leaves were also less severe (maximum reference severity 6.3% vs 19.3%). Because the validation split was used for model selection, these values document consistency rather than an independent replication."
2. (Methods, data.) "All rust leaves were collected on a single day, so between-date variation could not be estimated for rust. American leaf spot leaves came from two dates that also differed in grade mix; resampling dates and then leaves widened the 95% CI of SegFormer-B5 test IoU from 0.868–0.900 to 0.862–0.917 and of MAE from 0.15–0.30 to 0.11–0.43 percentage points; with only two dates this indicates, but cannot reliably estimate, date-level uncertainty."
3. (Supplement, rust faces.) "On the adaxial face, where rust shows chlorosis rather than sporulation, test IoU was lower (0.613, 0.441 and 0.499 for SegFormer-B5, UNet++ and DeepLabV3+) than on the abaxial face (0.767, 0.647 and 0.632), and averaging the two faces gave intermediate values (0.690, 0.544 and 0.565; Table Sx)."
4. (Methods, loss; replaces the inverse-frequency claim.) "In the letterboxed training images, background, healthy leaf tissue and lesion made up 92.3%, 7.5% and 0.17% of rust pixels and 92.6%, 7.0% and 0.39% of leaf-spot pixels. The cross-entropy weights of 0.25, 0.5 and 3.0 were set heuristically to increase with class rarity and are much flatter than inverse-frequency weights, which would be about 0.25, 3.1 and 133 for rust. Lesion rarity is additionally addressed by the Focal Tversky term (weight 0.7), which is computed on the lesion class alone."
5. (Methods, data.) "The reference grade mix did not differ detectably between the training, validation and test splits (permutation tests: rust, p = 0.22; American leaf spot, p = 1.00 for visual and 0.93 for mask-derived grades; Figure Sx), although the rust validation leaves were milder than the test leaves (5 vs 9 of 21 leaves in G2; maximum reference severity 6.3% vs 19.3%). The mask-derived leaf-spot grade never exceeded the annotator's visual grade and was one grade lower for 38 of 369 leaves."

## Figures

- `figure_val_test.pdf` / `.png`: leaf lesion IoU (top) and severity MAE (bottom) on validation (open markers) and test (filled markers) for the six models, 95% leaf-bootstrap CIs. Suggested caption: "Validation and test results under the paper's leaf-level rules (rust: abaxial face; American leaf spot: mean of the two faces). Open markers, validation (21 rust and 55 leaf-spot leaves); filled markers, test (21 and 55 leaves). Bars are 95% bootstrap confidence intervals (10,000 leaf resamples). The validation split was used for model selection."
- `figure_grade_distribution.pdf` / `.png`: reference grade mix per split. Suggested caption: "Reference grade distribution of the diseased leaves in the training, validation and test splits. (a) Coffee leaf rust, grade derived from the abaxial reference mask (cut-points 0.5, 5 and 20%); the healthy control leaves of the rust pack (99, 21 and 21 locally; all G0) are not shown. (b) American leaf spot, the annotator's visual grade and the grade derived from the mean reference severity of the two faces. Numbers in the bars are leaf counts."

## Notes and caveats

- Only 21 of the 54 healthy rust-pack test leaves reported in the paper are present locally; no specificity number was recomputed here.
- The validation split informed checkpoint selection (all models) and the rust CNN false-positive penalty, so validation metrics may be slightly optimistic and are not an independent test.
- Rust IoU/MAE measure agreement with the semi-automatic (detector + SAM 2, spot-checked, unedited) reference, in validation as in test.
- Leaf spot has two collection dates and rust one; a date-clustered bootstrap with two clusters is illustrative only, and the two leaf-spot dates differ in grade mix.
- QWK bootstrap resamples in which kappa is undefined (a single grade in both ratings) were dropped; counts are in results.json (`qwk_n_valid_resamples`; at most 39 of 10,000 dropped).
