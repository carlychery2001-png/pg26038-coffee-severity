# Detection, specificity and cross-disease behavior (major revision, PG-26038)

Scripts: `verify_per_image.py` (pixel-count check), `detection_analysis.py` (all numbers -> `results.json`, `leaf_level.csv`), `make_figure_detection.py` (figure), `make_md.py` (this file). No model inference was re-run.

Leaf-level rules as in the paper: rust = abaxial face only; American leaf spot = mean of the two faces. A model's leaf-level output uses the rule of the disease it was trained for (rust models: abaxial face; leaf-spot models: mean of both faces); other rules are sensitivity analyses. Grades: G0 < 0.5% ≤ G1 < 5% ≤ G2 < 20% ≤ G3, so "≥0.5%" means "predicted grade G1 or higher". Proportions: 95% Wilson intervals. Bootstrap: 10,000 leaf resamples, percentile 95% CI, seed 20260927 (paired where two models are compared on the same leaves). Lesion-location measures in Section 3 (share of predicted lesion pixels inside the reference lesion mask, lesion IoU, off-lesion severity) are computed on the faces the model reads, i.e. the same faces behind its severity, grade and ≥0.5% values (rust models: abaxial; leaf-spot models: both faces); for cross-disease runs they are also given on the faces of the leaf's own-disease rule (rust leaves: abaxial; leaf-spot leaves: both faces). The two face sets are identical for correct-disease runs. All numbers are rounded once, from full-precision values in results.json.

## 0. Reproduction and input check

- **Published values reproduced: YES.** All six models match the published leaf IoU, MAE, bias, grade QWK and Bland–Altman limits at the published precision:

| Model | n | IoU | MAE | Bias | QWK | LoA |
|---|---|---|---|---|---|---|
| rust/SegFormer-B5 | 21 | 0.767 | 0.99 | +0.99 | 0.88 | −1.68 to +3.66 |
| rust/UNet++ | 21 | 0.647 | 1.77 | +1.75 | 0.79 | −2.68 to +6.17 |
| rust/DeepLabV3+ | 21 | 0.632 | 1.47 | +1.41 | 0.88 | −3.07 to +5.89 |
| leaf spot/SegFormer-B5 | 55 | 0.884 | 0.22 | +0.16 | 0.82 | −0.47 to +0.79 |
| leaf spot/UNet++ | 55 | 0.853 | 0.15 | −0.03 | 0.85 | −0.50 to +0.44 |
| leaf spot/DeepLabV3+ | 55 | 0.815 | 0.25 | +0.08 | 0.84 | −0.66 to +0.82 |

- `per_image.csv` was re-derived from the stored class maps and rebuilt reference labels (the `load()` of `run_inference.py`) for every test row used here: 1164/1164 rows match on all six pixel counts (0 mismatches).

## 1. Sensitivity on the diseased test leaves

| Model | n | >0% | ≥0.5% | ≥0.5%, reference mask ≥0.5% only | AUC vs 21 healthy |
|---|---|---|---|---|---|
| Rust SegFormer-B5 | 21 | 21/21 (100.0%; 84.5–100.0) | 20/21 (95.2%; 77.3–99.2) | 20/20 (100.0%; 83.9–100.0) | 1.000 (1.000–1.000) |
| Rust UNet++ | 21 | 21/21 (100.0%; 84.5–100.0) | 20/21 (95.2%; 77.3–99.2) | 20/20 (100.0%; 83.9–100.0) | 1.000 (1.000–1.000) |
| Rust DeepLabV3+ | 21 | 21/21 (100.0%; 84.5–100.0) | 20/21 (95.2%; 77.3–99.2) | 20/20 (100.0%; 83.9–100.0) | 1.000 (1.000–1.000) |
| Leaf spot SegFormer-B5 | 55 | 55/55 (100.0%; 93.5–100.0) | 47/55 (85.5%; 73.8–92.4) | 47/49 (95.9%; 86.3–98.9) | 0.998 (0.992–1.000) |
| Leaf spot UNet++ | 55 | 55/55 (100.0%; 93.5–100.0) | 49/55 (89.1%; 78.2–94.9) | 49/49 (100.0%; 92.7–100.0) | 0.990 (0.971–1.000) |
| Leaf spot DeepLabV3+ | 55 | 55/55 (100.0%; 93.5–100.0) | 48/55 (87.3%; 76.0–93.7) | 48/49 (98.0%; 89.3–99.6) | 0.991 (0.973–1.000) |

- Rust: the one diseased leaf below 0.5% for all three models is 20260819_Hvastatrix_GX_123, whose reference severity is itself 0.41% (reference grade G0). Every rust leaf with reference severity ≥0.5% was called ≥0.5% by every model.
- Leaf spot: 6 of the 55 test leaves carry the visual grade G1 but a reference mask covering <0.5% of the leaf. Every leaf missed at ≥0.5% by any model is a very mild leaf (reference mask severity 0.13–0.60%), and the predicted severity on these leaves is close to the reference (see `leaves_below_0p5` in results.json).
- ">0%" (any predicted lesion pixel) gives 100% sensitivity for all six models, but it is not a usable detection rule because it also flags most healthy leaves (Section 2). AUC caveat: for rust the diseased leaves are from Sarchí and the healthy leaves from EARTH, so rust separation is confounded with site.

## 2. Specificity on the 21 locally available healthy test leaves (EARTH controls)

Only 21 of the 54 healthy rust-pack test leaves reported in the paper are present locally (collected at EARTH on 12, 15 and 22 June 2026); the other 33 are not in the local pack, so **the published 54-leaf results cannot be reproduced locally beyond these 21 leaves.**

| Model (leaf rule) | FP >0% | Specificity >0% | FP ≥0.5% | Specificity ≥0.5% | Median / max predicted severity | Published (54, any pixel) |
|---|---|---|---|---|---|---|
| Rust SegFormer-B5 (abaxial) | 17/21 | 4/21 (19.0%; 7.7–40.0) | 0/21 | 21/21 (100.0%; 84.5–100.0) | 0.085% / 0.36% | 0/54 |
| Rust UNet++ (abaxial) | 15/21 | 6/21 (28.6%; 13.8–50.0) | 0/21 | 21/21 (100.0%; 84.5–100.0) | 0.007% / 0.16% | 1/54 |
| Rust DeepLabV3+ (abaxial) | 7/21 | 14/21 (66.7%; 45.4–82.8) | 0/21 | 21/21 (100.0%; 84.5–100.0) | 0.000% / 0.09% | 0/54 |
| Leaf spot SegFormer-B5 (mean of faces) | 19/21 | 2/21 (9.5%; 2.7–28.9) | 0/21 | 21/21 (100.0%; 84.5–100.0) | 0.031% / 0.22% | 0/54 |
| Leaf spot UNet++ (mean of faces) | 21/21 | 0/21 (0.0%; 0.0–15.5) | 0/21 | 21/21 (100.0%; 84.5–100.0) | 0.071% / 0.41% | 0/54 |
| Leaf spot DeepLabV3+ (mean of faces) | 21/21 | 0/21 (0.0%; 0.0–15.5) | 0/21 | 21/21 (100.0%; 84.5–100.0) | 0.070% / 0.36% | 1/54 |

Sensitivity analyses (flagged leaves out of 21):

| Model | abaxial >0 / ≥0.5 | adaxial >0 / ≥0.5 | either face >0 / ≥0.5 |
|---|---|---|---|
| Rust SegFormer-B5 | 17 / 0 | 14 / 0 | 19 / 0 |
| Rust UNet++ | 15 / 0 | 14 / 0 | 19 / 0 |
| Rust DeepLabV3+ | 7 / 0 | 8 / 0 | 13 / 0 |
| Leaf spot SegFormer-B5 | 16 / 0 | 9 / 0 | 19 / 0 |
| Leaf spot UNet++ | 21 / 0 | 20 / 1 | 21 / 1 |
| Leaf spot DeepLabV3+ | 21 / 0 | 20 / 0 | 21 / 0 |

- **Definition problem (must be fixed in the manuscript).** The manuscript defines a healthy-leaf false positive as ANY predicted lesion pixel and reports 0/54 for four models. Local re-inference with the exact published protocol (raw argmax in letterbox space; it reproduces Tables 2/4 exactly) gives at least one predicted lesion pixel on 7–21 of the 21 local healthy test leaves, depending on the model. These 21 leaves are part of the published 54, so the published counts cannot have come from the any-pixel rule under this protocol. They are compatible with a severity threshold: the highest leaf-level predicted severity on a local healthy leaf is 0.41% under the paper's leaf-level rules, and the highest single-face value is 0.54%. The training kernels define a healthy-leaf false positive as predicted severity ≥5% per image. The threshold behind the published 54-leaf counts could not be recovered.
- The predicted "lesion" on healthy leaves is a speck: median over leaves of the predicted lesion pixels on the rule's face(s) = rust/SegFormer-B5 94 px (max 229), rust/UNet++ 6 px (max 160), rust/DeepLabV3+ 0 px (max 88), leaf spot/SegFormer-B5 47 px (max 270), leaf spot/UNet++ 120 px (max 522), leaf spot/DeepLabV3+ 109 px (max 640) in 1024 × 1024 letterbox space, where a healthy abaxial face covers about 55,000–120,000 leaf pixels.
- At the ≥0.5% threshold, 0/21 false positives has a 95% Wilson upper bound of 15.5% for the false-positive rate (specificity ≥84.5%), much wider than the 6.6% bound quoted for 0/54. At >0%, specificity on the 21 leaves is 0–67% depending on the model.
- Supplementary, not independent (rust CNNs were selected with a healthy-validation penalty): rust models on the 21 healthy validation leaves flag SegFormer-B5 19/21 at >0% and 0/21 at ≥0.5%, UNet++ 20/21 at >0% and 0/21 at ≥0.5%, DeepLabV3+ 13/21 at >0% and 0/21 at ≥0.5%.

## 3. Cross-disease behavior

A cross-disease reading on a diseased leaf is **not** a false positive in the healthy-leaf sense: the leaf carries real lesions of the other disease. It tests whether a disease-specific model is disease-specific, i.e. whether it stays silent on the other disease's lesions. A reading can combine pixels on the other disease's lesions with pixels on tissue outside them, so where the predicted pixels fall is reported separately (second table in each block).

### Rust models on the 55 American leaf spot test leaves (vs the leaf-spot models)

Reference severity on these leaves: median 4.15% (IQR 0.87–7.45); reference grades G0/G1/G2/G3 = 0/30/23/2.

| Model | Role | Median severity (95% CI) | IQR | ≥0.5% | Grades G0/G1/G2/G3 | Same model on 21 healthy: ≥0.5% (max) |
|---|---|---|---|---|---|---|
| SegFormer-B5 | correct disease | 4.13% (2.44–5.90) | 0.83–7.90 | 47/55 (85.5%; 73.8–92.4) | 8/23/22/2 | 0/21 (0.22%) |
| UNet++ | correct disease | 4.06% (2.47–5.75) | 0.92–7.42 | 49/55 (89.1%; 78.2–94.9) | 6/25/22/2 | 0/21 (0.41%) |
| DeepLabV3+ | correct disease | 4.05% (2.24–5.78) | 0.86–7.67 | 48/55 (87.3%; 76.0–93.7) | 7/24/22/2 | 0/21 (0.36%) |
| SegFormer-B5 | cross-disease | 6.29% (4.67–7.56) | 2.24–11.19 | 54/55 (98.2%; 90.4–99.7) | 1/21/29/4 | 0/21 (0.36%) |
| UNet++ | cross-disease | 4.57% (3.33–6.86) | 1.26–10.95 | 51/55 (92.7%; 82.7–97.1) | 4/27/20/4 | 0/21 (0.16%) |
| DeepLabV3+ | cross-disease | 5.26% (2.98–7.36) | 0.95–11.83 | 48/55 (87.3%; 76.0–93.7) | 7/20/19/9 | 0/21 (0.09%) |

Where the predicted lesion pixels fall, relative to the leaf-spot reference lesion mask of these leaves. "Model's faces" are the faces behind the severity, grade and ≥0.5% values above; "leaf's faces" are the faces of the leaf's own-disease rule (identical for correct-disease models).

| Model | Role | Faces | Pred. lesion px inside reference lesion, pooled (95% CI) | Lesion IoU | Off-lesion severity, median (IQR) | Leaves with off-lesion severity ≥0.5% | ≥0.5% calls that need off-lesion px | Mean severity / mean reference: all px, inside-lesion px only |
|---|---|---|---|---|---|---|---|---|
| SegFormer-B5 | correct disease | both (model's = leaf's) | 93.9% (92.9–94.7) | 0.88 | 0.21% (0.06–0.49) | 13/55 | 0/47 | 1.03, 0.97 |
| UNet++ | correct disease | both (model's = leaf's) | 94.2% (93.5–94.7) | 0.85 | 0.26% (0.12–0.45) | 12/55 | 2/49 | 1.00, 0.94 |
| DeepLabV3+ | correct disease | both (model's = leaf's) | 91.1% (90.3–91.9) | 0.81 | 0.35% (0.13–0.69) | 21/55 | 0/48 | 1.02, 0.93 |
| SegFormer-B5 | cross-disease | abaxial (model's) | 67.5% (63.6–71.2) | 0.60 | 1.69% (0.97–3.40) | 47/55 | 5/54 | 1.41, 0.95 |
| SegFormer-B5 | cross-disease | both (leaf's) | 71.3% (67.4–74.9) | 0.64 | 1.34% (0.80–2.59) | 45/55 | 6/54 | 1.28, 0.92 |
| UNet++ | cross-disease | abaxial (model's) | 66.1% (61.2–71.4) | 0.60 | 1.02% (0.26–3.81) | 38/55 | 5/51 | 1.34, 0.89 |
| UNet++ | cross-disease | both (leaf's) | 62.3% (57.4–67.3) | 0.60 | 1.30% (0.41–4.23) | 38/55 | 4/50 | 1.41, 0.88 |
| DeepLabV3+ | cross-disease | abaxial (model's) | 56.7% (49.4–64.8) | 0.53 | 1.15% (0.30–4.07) | 39/55 | 2/48 | 1.63, 0.90 |
| DeepLabV3+ | cross-disease | both (leaf's) | 61.9% (55.6–68.5) | 0.58 | 1.23% (0.28–3.67) | 39/55 | 4/49 | 1.44, 0.88 |

Off-lesion severity = predicted lesion pixels outside the reference lesion mask, in % of the predicted leaf area on the same faces (combined over faces like the severity); it includes lesion margins as well as distant tissue, which were not separated. "≥0.5% calls that need off-lesion px" = leaves at ≥0.5% whose inside-lesion severity alone is <0.5% (out of the leaves at ≥0.5% on those faces). Ratios are to the mean leaf-level reference severity.

Paired, same architecture, same leaves (cross-disease minus correct-disease model):

| Architecture | Δ fraction ≥0.5% (95% CI) | Mean severity ratio cross/correct (95% CI) | Mean severity ratio cross/reference (95% CI) | Same predicted grade |
|---|---|---|---|---|
| SegFormer-B5 | +0.127 (0.036 to 0.218) | 1.37 (1.25–1.51) | 1.41 (1.29–1.55) | 35/55 |
| UNet++ | +0.036 (−0.055 to 0.127) | 1.35 (1.21–1.51) | 1.34 (1.20–1.50) | 43/55 |
| DeepLabV3+ | +0.000 (−0.055 to 0.055) | 1.61 (1.35–1.92) | 1.63 (1.37–1.95) | 40/55 |

Cross-disease models under the leaf's own-disease rule (mean): ≥0.5% in SegFormer-B5 54/55, UNet++ 50/55, DeepLabV3+ 49/55.

### Leaf-spot models on the 21 rust test leaves (vs the rust models)

Reference severity on these leaves: median 3.83% (IQR 1.89–7.46); reference grades G0/G1/G2/G3 = 1/11/9/0.

| Model | Role | Median severity (95% CI) | IQR | ≥0.5% | Grades G0/G1/G2/G3 | Same model on 21 healthy: ≥0.5% (max) |
|---|---|---|---|---|---|---|
| SegFormer-B5 | correct disease | 4.08% (2.20–8.23) | 2.12–9.10 | 20/21 (95.2%; 77.3–99.2) | 1/10/9/1 | 0/21 (0.36%) |
| UNet++ | correct disease | 5.01% (2.11–9.65) | 2.01–10.35 | 20/21 (95.2%; 77.3–99.2) | 1/9/9/2 | 0/21 (0.16%) |
| DeepLabV3+ | correct disease | 4.92% (2.17–7.61) | 1.78–12.01 | 20/21 (95.2%; 77.3–99.2) | 1/10/9/1 | 0/21 (0.09%) |
| SegFormer-B5 | cross-disease | 1.08% (0.41–2.07) | 0.41–2.35 | 14/21 (66.7%; 45.4–82.8) | 7/13/1/0 | 0/21 (0.22%) |
| UNet++ | cross-disease | 1.50% (0.98–2.39) | 0.98–3.20 | 19/21 (90.5%; 71.1–97.3) | 2/17/2/0 | 0/21 (0.41%) |
| DeepLabV3+ | cross-disease | 1.50% (1.22–2.68) | 1.16–3.14 | 19/21 (90.5%; 71.1–97.3) | 2/18/1/0 | 0/21 (0.36%) |

Where the predicted lesion pixels fall, relative to the rust reference lesion mask of these leaves. "Model's faces" are the faces behind the severity, grade and ≥0.5% values above; "leaf's faces" are the faces of the leaf's own-disease rule (identical for correct-disease models).

| Model | Role | Faces | Pred. lesion px inside reference lesion, pooled (95% CI) | Lesion IoU | Off-lesion severity, median (IQR) | Leaves with off-lesion severity ≥0.5% | ≥0.5% calls that need off-lesion px | Mean severity / mean reference: all px, inside-lesion px only |
|---|---|---|---|---|---|---|---|---|
| SegFormer-B5 | correct disease | abaxial (model's = leaf's) | 80.4% (76.5–83.5) | 0.77 | 0.60% (0.29–2.16) | 11/21 | 0/20 | 1.18, 0.95 |
| UNet++ | correct disease | abaxial (model's = leaf's) | 70.3% (64.9–74.8) | 0.65 | 1.20% (0.51–3.82) | 16/21 | 0/20 | 1.32, 0.91 |
| DeepLabV3+ | correct disease | abaxial (model's = leaf's) | 71.7% (64.5–77.4) | 0.63 | 1.12% (0.44–3.54) | 15/21 | 0/20 | 1.26, 0.88 |
| SegFormer-B5 | cross-disease | both (model's) | 98.5% (97.6–99.5) | 0.26 | 0.01% (0.00–0.02) | 0/21 | 1/14 | 0.30, 0.30 |
| SegFormer-B5 | cross-disease | abaxial (leaf's) | 99.4% (99.1–99.7) | 0.38 | 0.00% (0.00–0.02) | 0/21 | 0/17 | 0.45, 0.45 |
| UNet++ | cross-disease | both (model's) | 71.5% (62.0–78.3) | 0.24 | 0.51% (0.38–0.71) | 11/21 | 3/19 | 0.39, 0.28 |
| UNet++ | cross-disease | abaxial (leaf's) | 81.7% (72.9–87.4) | 0.34 | 0.39% (0.27–0.69) | 8/21 | 1/19 | 0.50, 0.41 |
| DeepLabV3+ | cross-disease | both (model's) | 62.0% (50.7–70.6) | 0.19 | 0.68% (0.54–1.09) | 17/21 | 5/19 | 0.39, 0.25 |
| DeepLabV3+ | cross-disease | abaxial (leaf's) | 71.5% (59.7–79.8) | 0.28 | 0.60% (0.45–1.10) | 14/21 | 5/21 | 0.52, 0.37 |

Off-lesion severity = predicted lesion pixels outside the reference lesion mask, in % of the predicted leaf area on the same faces (combined over faces like the severity); it includes lesion margins as well as distant tissue, which were not separated. "≥0.5% calls that need off-lesion px" = leaves at ≥0.5% whose inside-lesion severity alone is <0.5% (out of the leaves at ≥0.5% on those faces). Ratios are to the mean leaf-level reference severity.

Paired, same architecture, same leaves (cross-disease minus correct-disease model):

| Architecture | Δ fraction ≥0.5% (95% CI) | Mean severity ratio cross/correct (95% CI) | Mean severity ratio cross/reference (95% CI) | Same predicted grade |
|---|---|---|---|---|
| SegFormer-B5 | −0.286 (−0.476 to −0.095) | 0.26 (0.23–0.29) | 0.30 (0.26–0.34) | 5/21 |
| UNet++ | −0.048 (−0.143 to 0.000) | 0.30 (0.27–0.35) | 0.39 (0.34–0.47) | 10/21 |
| DeepLabV3+ | −0.048 (−0.143 to 0.000) | 0.31 (0.27–0.38) | 0.39 (0.33–0.49) | 11/21 |

Cross-disease models under the leaf's own-disease rule (abaxial): ≥0.5% in SegFormer-B5 17/21, UNet++ 19/21, DeepLabV3+ 21/21.

**Interpretation.**

- The models do not discriminate between the two diseases. The rust models called ≥0.5% on 54, 51, 48 of 55 leaf-spot leaves (SegFormer-B5, UNet++, DeepLabV3+), as many as or more than the leaf-spot models (47, 49, 48), and over-read their severity (mean ratio to the leaf-spot reference 1.41, 1.34, 1.63). The leaf-spot models called ≥0.5% on 14, 19, 19 of 21 rust leaves, but read only 26%, 30%, 31% of the rust models' mean severity.
- Rust models on leaf-spot leaves: most of their predicted lesion pixels lie on the leaf-spot lesions, but they also mark tissue outside the leaf-spot reference lesions. On the face the rust models read (abaxial; the face behind their severity and ≥0.5% values), 67.5%, 66.1%, 56.7% of their predicted lesion pixels fall inside the leaf-spot reference lesions, against 93.9%, 94.2%, 91.1% for the leaf-spot models on the same leaves (both faces), so 33–43% versus 6–9% fall outside. The rust models' median off-lesion severity is 1.69%, 1.02%, 1.15% of the leaf (leaf-spot models 0.21–0.35%). Off-lesion pixels alone reach ≥0.5% of the leaf on 47, 38, 39 of 55 leaves, and on 5, 5, 2 leaves the ≥0.5% call would not be made without them. They account for the rust models' excess severity: the mean predicted severity is 1.41, 1.34, 1.63 times the leaf-spot reference with all predicted pixels, and 0.95, 0.89, 0.90 times when only pixels inside the reference lesions are counted. The extreme leaf is 20260628_Mcitricolor_GX_078 (reference 11.1%): the rust models' off-lesion area on its abaxial face is 10.1%, 17.2%, 39.6% of the leaf, and rust DeepLabV3+ marks a large area of apparently lesion-free tissue (23,278 of 30,256 predicted lesion pixels outside the reference). On the leaf's own-disease faces (both faces) the rust models' inside shares are 71.3%, 62.3%, 61.9%, with the same conclusion.
- Leaf-spot models on rust leaves: on the faces they read (both faces), 98.5%, 71.5%, 62.0% of their predicted lesion pixels fall inside the rust reference lesions (99.4%, 81.7%, 71.5% on the abaxial face, the rust reference face); their median off-lesion severity is 0.01%, 0.51%, 0.68% of the leaf. The rust models themselves place 80.4%, 70.3%, 71.7% of their abaxial pixels inside the rust reference (median off-lesion severity 0.60%, 1.20%, 1.12%), in line with their positive bias. The leaf-spot models' mean severity on these leaves is 0.30, 0.39, 0.39 times the rust reference, so their main error on rust leaves is under-reading; still, off-lesion pixels are needed for 1/14, 3/19, 5/19 of their ≥0.5% calls. The rust reference is semi-automatic (detector + SAM 2, spot-checked), so "outside the reference" is less certain on rust leaves.
- On the 21 healthy EARTH leaves the same models stay below 0.5% leaf-level severity. The rust models' off-lesion marking on leaf-spot leaves (median 1.02–1.69% of the leaf) is much larger than their specks on healthy leaves (maximum 0.36%). What it covers on lesioned leaves (lesion margins, chlorotic or otherwise damaged tissue, or lesions missing from the reference) was not examined.
- Consequence for use: disease identity has to come from the user or from a separate classification step; the severity models cannot tell rust from American leaf spot, and a positive reading from the wrong model must not be read as evidence of that disease.
- Site confound: the rust models were trained on Sarchí rust leaves and EARTH healthy leaves, yet they respond strongly to lesioned EARTH leaves (the leaf-spot leaves), including tissue outside the lesions. So their low output on the healthy EARTH leaves is not simply a learned "EARTH means healthy" rule. This does not remove the site confound; rust-infected leaves from EARTH, or healthy leaves from Sarchí, would still be needed.

## 4. Suggested manuscript sentences

1. (Methods) "Leaf-level detection was scored at two predicted-severity thresholds, >0% (any predicted lesion pixel) and ≥0.5% (predicted grade G1 or higher), using the leaf-level rules of Section~\ref{sec:evaluation}; sensitivity was computed on the diseased test leaves and specificity on the healthy test leaves available for reanalysis (21 of the 54; the remaining 33 were not available), with 95% Wilson intervals."
2. (Results) "At ≥0.5%, all six models flagged none of the 21 reanalyzed healthy leaves (specificity 100%, 95% CI 84.5–100%). Sensitivity was 20/21 rust leaves for all three rust models; the leaf that was missed had a reference severity of 0.41% (reference grade G0). For leaf spot, sensitivity was 47/55, 49/55 and 48/55 (SegFormer-B5, UNet++, DeepLabV3+), with every missed leaf having a reference mask severity of 0.60% or less."
3. (Results) "Counting any predicted lesion pixel as a false positive is not informative under raw argmax: 17, 15 and 7 of the 21 healthy leaves carried at least one predicted lesion pixel with the rust models, and 19, 21 and 21 with the leaf-spot models, although no healthy leaf exceeded 0.41% leaf-level predicted severity (the highest single-face value was 0.54%)." [Replaces the any-pixel definition and the claim that no healthy leaf received any predicted lesion. % TODO(author): state the threshold that produced the published 0/54 and 1/54 counts, or report the 21-leaf ≥0.5% result instead.]
4. (Results) "The models were not disease-specific. Applied to the American leaf spot test leaves, the rust models reported ≥0.5% severity on 48–54 of 55 leaves, and the leaf-spot models reported ≥0.5% on 14–19 of the 21 rust leaves. Most of the predicted lesion pixels behind these readings (57–99%, counted on the leaf faces each model reads) lay within the other disease's reference lesions. The rust models also marked tissue outside the leaf-spot reference lesions (33–43% of their predicted lesion pixels, against 6–9% for the leaf-spot models on the same leaves), which raised their mean severity on these leaves to 1.34–1.63 times the reference (0.89–0.95 times when only pixels inside the reference lesions were counted). The disease must therefore be identified before a model is applied."
5. (Limitations) "Specificity was reanalyzed on only 21 healthy leaves from a single site; at the ≥0.5% threshold, the 95% upper bound of the false-positive rate is 15.5% (at >0%, specificity on these leaves ranged from 0% to 67%, depending on the model). It does not cover abiotic or other biotic disorders, and because the models respond to lesions of the other disease, they do not identify which disease is present."

## 5. Figure

`figure_detection.pdf` (vector, TrueType Arial) and `figure_detection.png` (600 dpi), 16 cm wide.

Suggested caption: "Leaf-level predicted severity of the three models trained for (a) coffee leaf rust and (b) American leaf spot. Each panel shows the target disease's test leaves, the other disease's test leaves (cross-disease) and the 21 healthy test leaves available for reanalysis (EARTH controls; the paper's full set had 54). One point per leaf. The rust models read the abaxial face and the leaf-spot models the mean of both faces. Black bars are medians. The dashed line is the 0.5% threshold (G0/G1 cut-point) and the dotted lines are the 5% and 20% cut-points. Numbers above the points give the leaves at ≥0.5%. The y-axis is linear from 0 to 0.1% and logarithmic above."
