# Regenerated Figures 4, 12, 13 and 14 (PG-26038 major revision)

Folder: `<ROOT>/reanalysis/results/figures/`. No model inference was re-run. Run order: `make_quant_figures.py`, `make_qualitative_figures.py`, `error_location_summary.py`, `write_results.py` (shared code in `common.py`). All numbers are in `results.json`.

| Paper figure | Files | Replaces manuscript file |
|---|---|---|
| Figure 4 | `figure4.pdf`, `figure4.png` (600 dpi) | `figures/figure4.png` |
| Figure 12 | `figure12.pdf`, `figure12.png` (600 dpi) | `figures/figure8.png` |
| Figure 13 | `figure13.pdf`, `figure13.png` (600 dpi) | `figures/figure9.png` |
| Figure 14 | `figure14.pdf`, `figure14.png` (600 dpi) | `figures/figure10.png` |

## 1. Reproduction of the published values (done first)

All reproduced: **True** (leaf IoU, MAE, bias, QWK and Bland–Altman LoA, at the printed precision, all six models). Rust = abaxial face, 21 test leaves with disease; American leaf spot = mean of both faces, 55 test leaves.

| Disease | Model | Leaf IoU | MAE (pp) | Bias (pp) | QWK | LoA (pp) |
|---|---|---|---|---|---|---|
| coffee leaf rust | SegFormer-B5 | 0.7673 | 0.990 | +0.990 | 0.878 | −1.68 to +3.66 |
| coffee leaf rust | UNet++ | 0.6465 | 1.769 | +1.749 | 0.788 | −2.68 to +6.17 |
| coffee leaf rust | DeepLabV3+ | 0.6316 | 1.474 | +1.411 | 0.878 | −3.07 to +5.89 |
| American leaf spot | SegFormer-B5 | 0.8845 | 0.220 | +0.164 | 0.825 | −0.47 to +0.79 |
| American leaf spot | UNet++ | 0.8530 | 0.145 | −0.027 | 0.852 | −0.50 to +0.44 |
| American leaf spot | DeepLabV3+ | 0.8147 | 0.247 | +0.080 | 0.838 | −0.66 to +0.82 |

## 2. Figure 4: leaf-level lesion IoU, 95% bootstrap CI

10,000 leaf resamples, percentile CI, seed 20260927 (fresh generator per disease; same indices for the three models). Printed labels equal the leaf-level means (asserted).

| Disease | Model | n | Leaf IoU | 95% CI |
|---|---|---|---|---|
| coffee leaf rust | SegFormer-B5 | 21 | **0.767** | 0.743–0.790 |
| coffee leaf rust | UNet++ | 21 | **0.647** | 0.607–0.684 |
| coffee leaf rust | DeepLabV3+ | 21 | **0.632** | 0.590–0.669 |
| American leaf spot | SegFormer-B5 | 55 | **0.884** | 0.868–0.900 |
| American leaf spot | UNet++ | 55 | **0.853** | 0.830–0.874 |
| American leaf spot | DeepLabV3+ | 55 | **0.815** | 0.790–0.837 |

Paired differences (same resamples):

| Disease | Contrast | Mean difference | 95% CI | Leaves where first is higher |
|---|---|---|---|---|
| coffee leaf rust | SegFormer-B5 - UNet++ | +0.121 | +0.093 to +0.151 | 21/21 |
| coffee leaf rust | SegFormer-B5 - DeepLabV3+ | +0.136 | +0.106 to +0.169 | 21/21 |
| coffee leaf rust | UNet++ - DeepLabV3+ | +0.015 | −0.003 to +0.034 | 13/21 |
| American leaf spot | SegFormer-B5 - UNet++ | +0.031 | +0.023 to +0.041 | 50/55 |
| American leaf spot | SegFormer-B5 - DeepLabV3+ | +0.070 | +0.061 to +0.080 | 55/55 |
| American leaf spot | UNet++ - DeepLabV3+ | +0.038 | +0.033 to +0.043 | 54/55 |

## 3. Figure 12: leaf-level IoU by reference severity band

Bands aligned with the grade cut-points 0.5 / 5 / 20% (G1 split at 2%), plus a `<0.5` band so that every test leaf is counted. Reference severity: rust abaxial face; leaf spot mean of both faces. CI omitted when n < 3 (open symbols in the figure).

| Disease | Band (%) | n | SegFormer-B5 | UNet++ | DeepLabV3+ |
|---|---|---|---|---|---|
| coffee leaf rust | <0.5 | 1 | 0.745 (no CI, n < 3) | 0.714 (no CI, n < 3) | 0.589 (no CI, n < 3) |
| coffee leaf rust | 0.5–<2 | 5 | 0.755 (0.696–0.814) | 0.584 (0.512–0.664) | 0.591 (0.512–0.664) |
| coffee leaf rust | 2–<5 | 6 | 0.794 (0.757–0.827) | 0.679 (0.640–0.722) | 0.653 (0.614–0.692) |
| coffee leaf rust | 5–<20 | 9 | 0.759 (0.719–0.794) | 0.652 (0.589–0.713) | 0.645 (0.566–0.708) |
| coffee leaf rust | ≥20 | 0 | – | – | – |
| American leaf spot | <0.5 | 6 | 0.801 (0.757–0.847) | 0.750 (0.685–0.820) | 0.724 (0.662–0.793) |
| American leaf spot | 0.5–<2 | 12 | 0.860 (0.823–0.892) | 0.798 (0.737–0.847) | 0.756 (0.688–0.812) |
| American leaf spot | 2–<5 | 14 | 0.893 (0.867–0.916) | 0.870 (0.842–0.894) | 0.831 (0.799–0.859) |
| American leaf spot | 5–<20 | 21 | 0.910 (0.892–0.925) | 0.894 (0.875–0.911) | 0.854 (0.830–0.875) |
| American leaf spot | ≥20 | 2 | 0.953 (no CI, n < 3) | 0.948 (no CI, n < 3) | 0.909 (no CI, n < 3) |

Supplementary trend statistic (not plotted): Spearman ρ between leaf reference severity and leaf IoU, paired bootstrap 95% CI.

| Disease | SegFormer-B5 | UNet++ | DeepLabV3+ |
|---|---|---|---|
| coffee leaf rust | +0.12 (−0.34 to +0.56) | +0.16 (−0.28 to +0.58) | +0.37 (−0.10 to +0.72) |
| American leaf spot | +0.56 (+0.33 to +0.74) | +0.67 (+0.47 to +0.81) | +0.57 (+0.33 to +0.77) |

## 4. Figures 13 and 14: SegFormer-B5 worst / median / best test leaves

Rendered in 1024 × 1024 letterbox space (image, reference and prediction share one pixel grid; no inverse transform), cropped to the leaf and all lesion pixels (+10 px), common pixel scale per figure. Columns: input | reference outline (green) | prediction outline (magenta) | error map (TP white, FP magenta, FN green, other leaf grey).

| Figure | Row | Rank | Leaf | Face shown | Leaf IoU | Abaxial IoU | Adaxial IoU | TP / FP / FN px (shown face) | Shown-face IoU recomputed from plotted masks | Leaf severity ref. / pred. (%) | Shown-face severity ref. / pred. (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 13 | Worst | 1/21 | 20260819_Hvastatrix_GX_074 | abaxial | **0.638** | 0.638 | 0.396 | 6,651 / 3,616 / 152 | 0.638353 (= 2217/3473) | **8.84 / 13.36** | = leaf (rust leaf = abaxial face) |
| 13 | Median | 11/21 | 20260819_Hvastatrix_GX_020 | abaxial | **0.767** | 0.767 | 0.662 | 8,737 / 2,355 / 306 | 0.766538 (= 8737/11398) | **19.34 / 23.73** | = leaf (rust leaf = abaxial face) |
| 13 | Best | 21/21 | 20260819_Hvastatrix_GX_013 | abaxial | **0.857** | 0.857 | 0.470 | 2,314 / 313 / 73 | 0.857037 (= 1157/1350) | **4.71 / 5.17** | = leaf (rust leaf = abaxial face) |
| 14 | Worst | 1/55 | 20260621_Mcitricolor_GX_246 | abaxial | **0.732** | 0.775 | 0.689 | 214 / 14 / 48 | 0.775362 (= 107/138) | **0.28 / 0.30** | 0.35 / 0.30 |
| 14 | Median | 28/55 | 20260621_Mcitricolor_GX_230 | abaxial | **0.904** | 0.901 | 0.907 | 3,890 / 299 / 129 | 0.900880 (= 1945/2159) | **3.92 / 4.09** | 4.33 / 4.50 |
| 14 | Best | 55/55 | 20260628_Mcitricolor_GX_007 | abaxial | **0.961** | 0.966 | 0.956 | 14,491 / 364 / 148 | 0.965873 (= 14491/15003) | **11.73 / 11.97** | 11.88 / 12.03 |

Verification (all asserted, all passed): (1) rebuilt reference and saved prediction reproduce the per_image.csv pixel counts; (2) IoU from the plotted crop arrays = IoU from the error-map raster = CSV value, as exact integer ratios, and no lesion pixel lies outside the crop; (3) pixel centres inside the drawn outline paths reproduce each mask exactly (0 mismatched pixels); (4) the error map and input crop embedded in each PDF are pixel-identical to the computed arrays, and a MuPDF render of the PDF (anti-aliasing off) sampled at every image-pixel centre reproduces them exactly, from which TP/FP/FN and the IoU were recounted; (5) leaf spot: leaf IoU = mean of the shown-face IoU (from plotted masks) and the other face's IoU (recomputed from its masks); (6) the printed leaf severity equals the leaf-table value that sets the Figure 12 band (rust: abaxial face; leaf spot: mean of both faces), and the leaf-spot shown-face severity is printed under its own 'abaxial-face severity' label. Only the shown-face IoU and TP/FP/FN counts are functions of the plotted masks alone; severities also use the predicted leaf mask (not plotted), and the leaf-spot leaf IoU and leaf severity also use the adaxial face.

Leaf-spot face: the requested rule (face IoU closest to the leaf mean) ties by construction, because the leaf IoU is the mean of exactly two face IoUs. The tie was broken by a pre-specified, IoU-independent rule: the **abaxial face** for all three leaves (stated in the figure, which also prints both face IoUs).

### Error location over all SegFormer-B5 test faces (supports the qualitative description)

| Disease | Faces | TP px | FP px | FN px | FP/FN | Faces with FP > FN | FP within 2 px (5 px) of a reference lesion | FP more than 2 px (5 px) from any reference lesion | FN within 2 px (5 px) of a predicted lesion |
|---|---|---|---|---|---|---|---|---|---|
| coffee leaf rust | 21 | 81,532 | 19,813 | 4,715 | 4.20 | 20/21 | 67% (83%) | 33% (17%) | 72% (93%) |
| American leaf spot | 110 | 442,353 | 28,954 | 15,754 | 1.84 | 69/110 | 85% (91%) | 15% (9%) | 61% (74%) |

Worst rust leaf (20260819_Hvastatrix_GX_074, Figure 13 top row): 3,616 FP px, of which 58% lie more than 2 px and 33% more than 5 px from any reference lesion (the FP clusters away from reference lesions visible in its error map).

## 5. Notes and caveats

- **reproduction.** All published leaf-level values (IoU, MAE, bias, QWK, Bland–Altman LoA) reproduced exactly at the printed precision for all six models before any figure was drawn (the script stops otherwise).
- **figure4 label fix.** Rust UNet++ leaf IoU is 0.646541 (exact mean of 21 abaxial-face IoUs), which rounds to 0.647; the current figure4.png label 0.646 does not match the leaf-level mean (cause not traced).
- **figure4 annotation.** The 'n = ... test leaves' note of each Figure 4 panel is right-aligned to the right end of the drawn x-axis (checked in rendered coordinates), so it no longer extends past the axis.
- **severity labels.** Figure 14 prints two severities per leaf, each labelled: the leaf severity (mean of the two faces; the value that sets the leaf's Figure 12 band and the value quoted in sentence 5) and the abaxial-face severity of the face shown. They differ, e.g. GX_246 reference 0.28% (leaf) vs 0.35% (abaxial face); GX_230 reference 3.92% (leaf) vs 4.33% (abaxial face); GX_007 reference 11.73% (leaf) vs 11.88% (abaxial face). Figure 13 prints the leaf severity, which for rust is by definition the abaxial-face severity (asserted equal). The printed leaf severities are asserted equal to the Figure 12 leaf table.
- **healthy leaves.** Only 21 of the 54 healthy rust-pack test leaves (EARTH site) are available locally. None of Figures 4, 12, 13 or 14 uses healthy leaves: leaf IoU, as in the paper, is computed on the test leaves with disease (rust 21, leaf spot 55), so this local gap does not affect these figures.
- **leaf spot face rule.** The requested rule 'face whose IoU is closest to the leaf mean' is degenerate: the leaf IoU is the mean of the two face IoUs, so both faces are exactly equidistant from it (asserted with exact fractions for the three selected leaves). A pre-specified, IoU-independent tie-break was used: the abaxial face (as for rust and for paper Figure 2). Both face IoUs are printed in Figure 14. Under an alternative reading (face closest to the disease-level mean 0.884) the faces would be abaxial, abaxial, adaxial (worst, median, best).
- **extra band.** Bands 0.5–<2, 2–<5, 5–<20 and ≥20% were used as requested. One rust and six leaf-spot test leaves have reference severity <0.5%; they are shown as an additional '<0.5' band (the G0 range) so that every test leaf is counted (n sums to 21 and 55). Dropping that band leaves the other four bands unchanged. The six leaf-spot leaves below 0.5% all carry annotator grade G1 (the leaf-spot grade is visual).
- **small n.** CIs are omitted for n < 3 (rust <0.5%: n = 1; leaf spot ≥20%: n = 2); no rust test leaf reached 20%. Percentile bootstrap CIs for bands with n = 5–6 rest on few distinct resamples (126 multisets for n = 5, 462 for n = 6) and are likely too narrow; treat per-band trends as descriptive.
- **rust reference.** The rust reference masks are semi-automatic (colour detector + SAM 2, spot-checked, not edited), so rust FP/FN in Figure 13 and in the error-location summary are disagreements with that reference.
- **cnn protocol.** Model contrasts compare the published single-run models, whose training protocols differ (CNNs: photometric jitter p = 0.8, 30 epochs, FP-penalised selection for rust). Matched retraining is in progress; Figures 4 and 12 will need regenerating with those models (% TODO(retraining)).
- **test leaves shown.** Figures 13 and 14 show three test leaves per disease, as the current Figures 9 and 10 already do. The data-availability statement says the held-out test set is withheld pending peer review; the author may wish to confirm that showing these six test scans is compatible with that.
- **selection unchanged.** The selected leaves have leaf IoUs 0.638/0.767/0.857 (rust) and 0.732/0.904/0.961 (leaf spot), matching the values printed in the current figure9/figure10 to two decimals, so the leaf selection appears unchanged; the overlays are what was wrong (they are now registered).
- **rendering.** Qualitative panels are drawn in the 1024 x 1024 letterbox space; the PDF embeds the native pixels (interpolation 'none'), the 600 dpi PNG upsamples them with nearest neighbour. Outlines are vector 0.5-level contours of the binary masks with a thin dark halo.
- **colours.** Model colours/markers as specified. The disease colours of figure5 (rendered ≈ #CC7951 rust, ≈ #448BB0 leaf spot) are not used, because model is the encoded variable in these figures.
- **file mapping.** figure4: replaces manuscript figures/figure4.png (label fig4); figure12: replaces manuscript figures/figure8.png (label fig8); figure13: replaces manuscript figures/figure9.png (label fig9); figure14: replaces manuscript figures/figure10.png (label fig10)

## 6. Suggested manuscript sentences (LaTeX-ready)

1. Leaf-level lesion IoU (Figure~\ref{fig4}) was 0.767 (95\% CI 0.743--0.790) for SegFormer-B5, 0.647 (0.607--0.684) for UNet++ and 0.632 (0.590--0.669) for DeepLabV3+ on the 21 rust test leaves, and 0.884 (0.868--0.900), 0.853 (0.830--0.874) and 0.815 (0.790--0.837) on the 55 American leaf spot test leaves.

2. In the paired leaf bootstrap, SegFormer-B5 exceeded UNet++ by 0.121 (0.093--0.151) on rust and by 0.031 (0.023--0.041) on American leaf spot; these are contrasts between the published single-run models, which were trained under different protocols, not between architectures as such.

3. With severity bands aligned to the grade cut-points (Figure~\ref{fig8}), American leaf spot IoU rose with reference severity for all three architectures, from 0.80, 0.75 and 0.72 (SegFormer-B5, UNet++, DeepLabV3+) below 0.5\% ($n = 6$) to 0.91, 0.89 and 0.85 in the 5 to $<$20\% band ($n = 21$; Spearman $\rho$ between leaf severity and IoU 0.56--0.67), whereas for rust no severity trend was evident (SegFormer-B5 0.76--0.79 across the three bands with five or more leaves; bootstrap intervals of Spearman $\rho$ included zero for all three models).

4. Bands with fewer than three leaves (rust $<$0.5\%, $n = 1$; American leaf spot $\geq$20\%, $n = 2$) are shown without intervals, no rust test leaf reached 20\%, and the intervals of the other bands rest on 5 to 21 leaves each, so the per-band pattern is descriptive.

5. In the qualitative panels (Figures~\ref{fig9} and~\ref{fig10}), the worst American leaf spot leaf was also one of the mildest (leaf reference severity 0.28\%, the second lowest of the 55 test leaves), so that its IoU rests on a few hundred lesion pixels per face (276 and 235 pixels in the lesion union of the abaxial and adaxial faces). For rust, SegFormer-B5 predicted more lesion area than the semi-automatic reference in 20 of the 21 test leaves; the false-positive pixels lay mostly at lesion margins (67\% within 2 pixels of a reference lesion), with the remainder forming additional detections away from reference lesions (17\% more than 5 pixels away overall, and 33\% on the worst rust leaf in Figure~\ref{fig9}), in line with the positive rust severity bias (+0.99 percentage points).

## 7. Suggested captions

- **Figure 4 (fig4).** Leaf-level lesion IoU by architecture and disease. Points are means over the test leaves with disease (coffee leaf rust, $n = 21$, abaxial face; American leaf spot, $n = 55$, mean of both faces); bars are 95\% percentile bootstrap confidence intervals (10,000 resamples of leaves).
- **Figure 12 (fig8).** Leaf-level lesion IoU by reference severity band. Bands follow the grade cut-points (0.5\%, 5\% and 20\%; the G1 range is split at 2\%); $n$ is the number of test leaves per band. Reference severity is that of the abaxial face for rust and the mean of both faces for American leaf spot, whose reference grade is the annotator's visual grade, so a leaf's band need not match its grade. Bars are 95\% percentile bootstrap confidence intervals (10,000 resamples of leaves within the band); open symbols mark bands with fewer than three leaves, shown without an interval. No rust test leaf had a reference severity of 20\% or more. Symbols are offset horizontally within each band for legibility.
- **Figure 13 (fig9).** Qualitative segmentation of coffee leaf rust by SegFormer-B5 for the worst, median and best of the 21 rust test leaves ranked by leaf-level lesion IoU (abaxial face). Columns: input scan; semi-automatic reference lesions (green outline); SegFormer-B5 predicted lesions (magenta outline); error map (white, lesion in both; magenta, predicted only; green, reference only; grey, leaf without lesion in either). Panels are shown in the $1024 \times 1024$ letterbox space used for evaluation, cropped to the leaf, at a common pixel scale, so image, reference and prediction share one pixel grid; the printed IoU and pixel counts were recomputed from the plotted masks and match the tabulated values. The printed reference and predicted severities are those of the abaxial face, which defines the rust leaf-level values.
- **Figure 14 (fig10).** Qualitative segmentation of American leaf spot by SegFormer-B5 for the worst, median and best of the 55 American leaf spot test leaves ranked by leaf-level lesion IoU (mean of the abaxial- and adaxial-face IoUs). Columns, colours, letterbox space, cropping and common pixel scale as in Figure~\ref{fig9}. The abaxial face is shown for every leaf: the two face IoUs are, by construction, equally distant from their mean, so the face closest to the leaf mean is not defined, and the abaxial face was used as a pre-specified tie-break. For the shown face, the IoU and the TP/FP/FN pixel counts were recomputed from the plotted masks; the printed leaf IoU is the mean of that value and the adaxial-face IoU, recomputed from the adaxial masks, which are not plotted. The printed leaf severity is the mean of the two face severities (the value that places the leaf in its Figure~\ref{fig8} band); the abaxial-face severity of the shown face is printed separately.
