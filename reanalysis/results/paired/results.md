# Paired comparisons between the published models within disease (PG-26038 revision)

Reviewer issue (R3, R2 partial): severity and grade differences between architectures were never tested, and the Conclusions implied non-inferiority in grading without a margin or a test. This analysis adds paired tests on the same test leaves. It uses the reproduced per-image outputs (`reanalysis/out/per_image.csv`). No model inference was re-run.

**What is compared.** The three **published trained models** per disease, not architectures under a controlled protocol. Training differed (revision brief facts 3–4). The published CNNs (UNet++, DeepLabV3+) used photometric jitter (p = 0.8), and the rust CNNs used at most 30 epochs, patience 12 and model selection penalised for false positives. SegFormer-B5 used geometric augmentation only, up to 60 epochs and patience 15. Any difference below may reflect training budget and augmentation as well as architecture. For rust, the matched-protocol retraining (brief item 11) indicates that training accounts for much of the published ranking (see §6).

**Design.** Leaf-level, test partition. Rust: abaxial face only, 21 diseased leaves. Reference grade = cut-points 0.5 / 5 / 20% on the semi-automatic reference-mask severity. American leaf spot: mean of the two faces, 55 leaves. Reference grade = the annotator's visual grade; the mask-derived grade (same cut-points on reference severity) is a secondary reference. Paired bootstrap: 10,000 resamples of leaves, seed 20260927, percentile intervals. One index matrix per disease is shared by all models, pairs and metrics. Differences are **first model minus second**: ΔMAE < 0 and Δ|bias| < 0 favour the first model; ΔQWK > 0 and Δaccuracy > 0 favour the first model. Grade correctness: exact McNemar test (b = leaves only the first model grades correctly, c = leaves only the second does). Holm correction across the three pairs within each disease. Accuracy and per-grade recall: Wilson 95% intervals. Paired Δaccuracy: Tango score 95% interval, because the percentile bootstrap collapses when discordant leaves are absent or all on one side. Descriptive TOST margins for Δaccuracy use the Tango 90% interval for the same reason. Scripts: `paired_analysis.py` (all numbers → `results.json`), `make_figures.py`, `make_md.py`, `verify_csv_against_preds.py`.

## 0. Reproduction of the published values (done first)

| Disease | Model | Leaf IoU | MAE (pp) | Bias (pp) | LoA (pp) | QWK | Grade accuracy | Match |
|---|---|---|---|---|---|---|---|---|
| Coffee leaf rust | SegFormer-B5 | 0.767 | 0.99 | +0.99 | −1.68 to +3.66 | 0.88 | 0.90 (19/21) | yes |
| Coffee leaf rust | UNet++ | 0.647 | 1.77 | +1.75 | −2.68 to +6.17 | 0.79 | 0.81 (17/21) | yes |
| Coffee leaf rust | DeepLabV3+ | 0.632 | 1.47 | +1.41 | −3.07 to +5.89 | 0.88 | 0.90 (19/21) | yes |
| American leaf spot | SegFormer-B5 | 0.884 | 0.22 | +0.16 | −0.47 to +0.79 | 0.82 | 0.84 (46/55) | yes |
| American leaf spot | UNet++ | 0.853 | 0.15 | −0.03 | −0.50 to +0.44 | 0.85 | 0.87 (48/55) | yes |
| American leaf spot | DeepLabV3+ | 0.815 | 0.25 | +0.08 | −0.66 to +0.82 | 0.84 | 0.85 (47/55) | yes |

All 42 published values (leaf IoU, MAE, bias, both LoA limits, QWK and accuracy × 6 models) are reproduced at the published rounding: **yes**. LoA = bias ± 1.96 SD (sample SD). The paired IoU differences reproduce Table 3: all six point estimates match at 3 dp. Two bootstrap CI bounds differ by 0.001 at 3 dp: the upper 95% CI bound for rust SegFormer-B5 vs UNet++ (0.15055, which rounds to 0.151, against a published 0.150); the lower 95% CI bound for leaf spot UNet++ vs DeepLabV3+ (0.03337, which rounds to 0.033, against a published 0.034). This is Monte Carlo variation from the new seed. Rust QWK CIs are [0.68, 1.00] with this seed (published [0.69, 1.00]). Integrity check: the pixel counts of all 708 per-image rows used here were recomputed from the stored class maps and the rebuilt letterbox reference labels, with 0 mismatches.

## 1. Paired differences (first minus second model)

### Coffee leaf rust (n = 21 leaves)

| Comparison | ΔMAE, pp [95% CI] | Δ\|bias\|, pp [95% CI] | ΔQWK [95% CI] | ΔAccuracy [Tango 95% CI] (bootstrap CI) | McNemar b / c; exact p (Holm) | Same grade |
|---|---|---|---|---|---|---|
| SegFormer-B5 vs DeepLabV3+ | −0.48 [−1.40, +0.25] | −0.42 [−1.34, +0.31] | 0.000 [0.000, 0.000] | 0.000 [−0.155, +0.155] (0.000, 0.000) | 0 / 0; 1.00 (1.00) | 21/21 |
| SegFormer-B5 vs UNet++ | −0.78 [−1.46, −0.26] | −0.76 [−1.45, −0.24] | +0.090 [0.000, +0.255] | +0.095 [−0.074, +0.289] (0.000, +0.238) | 2 / 0; 0.50 (1.00) | 19/21 |
| UNet++ vs DeepLabV3+ | +0.30 [−0.20, +0.76] | +0.34 [−0.16, +0.80] | −0.090 [−0.255, 0.000] | −0.095 [−0.289, +0.074] (−0.238, 0.000) | 0 / 2; 0.50 (1.00) | 19/21 |

Bootstrap two-sided p (inverts the percentile CI; Holm across the three pairs within disease), for ΔMAE / Δ|bias|: SegFormer-B5 vs DeepLabV3+ 0.25 (Holm 0.46) / 0.34 (Holm 0.34); SegFormer-B5 vs UNet++ <0.001 (Holm <0.001) / <0.001 (Holm <0.001); UNet++ vs DeepLabV3+ 0.23 (Holm 0.46) / 0.17 (Holm 0.34).

### American leaf spot (n = 55 leaves)

| Comparison | ΔMAE, pp [95% CI] | Δ\|bias\|, pp [95% CI] | ΔQWK [95% CI] | ΔAccuracy [Tango 95% CI] (bootstrap CI) | McNemar b / c; exact p (Holm) | Same grade |
|---|---|---|---|---|---|---|
| SegFormer-B5 vs DeepLabV3+ | −0.027 [−0.083, +0.028] | +0.084 [−0.003, +0.183] | −0.013 [−0.046, 0.000] | −0.018 [−0.096, +0.048] (−0.055, 0.000) | 0 / 1; 1.00 (1.00) | 54/55 |
| SegFormer-B5 vs UNet++ | +0.075 [+0.013, +0.141] | +0.138 [+0.018, +0.235] | −0.028 [−0.074, 0.000] | −0.036 [−0.123, +0.031] (−0.091, 0.000) | 0 / 2; 0.50 (1.00) | 53/55 |
| UNet++ vs DeepLabV3+ | −0.102 [−0.152, −0.057] | −0.053 [−0.157, +0.074] | +0.014 [0.000, +0.051] | +0.018 [−0.048, +0.096] (0.000, +0.055) | 1 / 0; 1.00 (1.00) | 54/55 |

Bootstrap two-sided p (inverts the percentile CI; Holm across the three pairs within disease), for ΔMAE / Δ|bias|: SegFormer-B5 vs DeepLabV3+ 0.35 (Holm 0.35) / 0.060 (Holm 0.12); SegFormer-B5 vs UNet++ 0.020 (Holm 0.040) / 0.026 (Holm 0.079); UNet++ vs DeepLabV3+ <0.001 (Holm <0.001) / 0.48 (Holm 0.48).

Reading the table:
- **Rust severity.** The published SegFormer-B5 model had a lower MAE and a smaller |bias| than the published UNet++ model. Both intervals exclude 0 and survive Holm. The training protocols differed, so this cannot be attributed to architecture. The difference between the published SegFormer-B5 and DeepLabV3+ models was not resolved (ΔMAE −0.48 pp, [−1.40, +0.25]). The two CNNs were not shown to differ. All three rust mean biases are positive (SegFormer-B5 +0.99 pp, UNet++ +1.75 pp, DeepLabV3+ +1.41 pp), so Δ|bias| equals Δbias for rust. This does not hold leaf by leaf. Signed severity errors are negative on 1/21 (SegFormer-B5, min −0.003 pp), 3/21 (UNet++, min −0.138 pp), 5/21 (DeepLabV3+, min −0.467 pp) rust leaves. All rust *grade* errors, however, are over-grades.
- **Leaf-spot severity.** The published UNet++ model had the lowest MAE: 0.075 pp below SegFormer-B5 (Holm p = 0.040) and 0.102 pp below DeepLabV3+ (Holm p <0.001). These differences are a few hundredths to a tenth of a percentage point. UNet++ also had a smaller |bias| than SegFormer-B5 (+0.138 pp, [+0.018, +0.235]), but this does not survive Holm (p = 0.079).
- **Grades.** No pair differed detectably in grade correctness. There were 0–2 discordant leaves per pair, all exact McNemar p ≥ 0.50, and all Holm-adjusted p = 1. Rust SegFormer-B5 and DeepLabV3+ gave the **same grade to all 21 leaves**, so the bootstrap intervals for ΔQWK and Δaccuracy are [0, 0]. These are degenerate intervals, not evidence of equivalence. The Tango interval for that pair is [−0.155, +0.155]. When all discordant leaves favour one model, the percentile bootstrap bound sits exactly at 0 (e.g., rust SegFormer-B5 vs UNet++: 2 / 0 discordant). This means 0 is not excluded.
- **No equivalence.** No margin was pre-specified. For Δaccuracy the smallest symmetric margin a TOST at α = 0.05 would accept, taken from the **Tango 90% score interval**, is SegFormer-B5 vs DeepLabV3+ 0.11 ([−0.114, +0.114]); SegFormer-B5 vs UNet++ 0.25 ([−0.030, +0.251]); UNet++ vs DeepLabV3+ 0.25 ([−0.251, +0.030]) for rust, and SegFormer-B5 vs DeepLabV3+ 0.077 ([−0.077, +0.030]); SegFormer-B5 vs UNet++ 0.104 ([−0.104, +0.012]); UNet++ vs DeepLabV3+ 0.077 ([−0.030, +0.077]) for leaf spot. The 90% percentile bootstrap is not used here: it is [0, 0] for rust SegFormer-B5 vs DeepLabV3+ and has a bound pinned at 0 for the other five pairs, so it would understate the margin. For ΔQWK the 90% bootstrap interval is pinned at 0 for every pair, so no margin is given (`tost_margin_needed` = null). Even for leaf spot, equivalence would therefore need a margin of at least about 8–10 accuracy points, and for rust about 11–25 points. Such a margin would also have to be pre-specified. Equivalence in grading is not claimed for either disease.

Unadjusted 95% intervals are reported. Bonferroni 98.33% intervals are in `results.json` (`ci98_33_bonferroni3`). Under those intervals the rust SegFormer-B5 vs UNet++ and leaf-spot UNet++ vs DeepLabV3+ MAE differences still exclude 0, and the leaf-spot SegFormer-B5 vs UNet++ MAE difference no longer does ([−0.003, +0.156]).

## 2. Grade agreement per model (Table 5 update), with the leaf-spot mask-derived reference

| Disease | Model | Reference | Accuracy (k/n) [Wilson 95% CI] | QWK [bootstrap 95% CI] | Errors (over / under; all adjacent) |
|---|---|---|---|---|---|
| Coffee leaf rust | SegFormer-B5 | mask-derived (primary) | 0.90 (19/21) [0.71, 0.97] | 0.88 [0.68, 1.00] | 2 / 0; yes |
| Coffee leaf rust | UNet++ | mask-derived (primary) | 0.81 (17/21) [0.60, 0.92] | 0.79 [0.57, 0.94] | 4 / 0; yes |
| Coffee leaf rust | DeepLabV3+ | mask-derived (primary) | 0.90 (19/21) [0.71, 0.97] | 0.88 [0.68, 1.00] | 2 / 0; yes |
| American leaf spot | SegFormer-B5 | visual (primary) | 0.84 (46/55) [0.72, 0.91] | 0.82 [0.73, 0.91] | 0 / 9; yes |
| American leaf spot | SegFormer-B5 | mask-derived (secondary) | 0.95 (52/55) [0.85, 0.98] | 0.95 [0.88, 1.00] | 1 / 2; yes |
| American leaf spot | UNet++ | visual (primary) | 0.87 (48/55) [0.76, 0.94] | 0.85 [0.75, 0.94] | 0 / 7; yes |
| American leaf spot | UNet++ | mask-derived (secondary) | 0.98 (54/55) [0.90, 1.00] | 0.98 [0.94, 1.00] | 1 / 0; yes |
| American leaf spot | DeepLabV3+ | visual (primary) | 0.85 (47/55) [0.74, 0.92] | 0.84 [0.74, 0.92] | 0 / 8; yes |
| American leaf spot | DeepLabV3+ | mask-derived (secondary) | 0.96 (53/55) [0.88, 0.99] | 0.97 [0.91, 1.00] | 1 / 1; yes |

Leaf-spot reference standards against each other: the visual grade equals the mask-derived grade for 47/55 leaves (QWK 0.83). All disagreements are single-grade (yes): 6 leaves are visual G1 but mask G0 (reference severity 0.13–0.40%), and 2 are visual G2 but mask G1 (4.80% and 4.92%). This reproduces the manuscript's 47/55 statement. Against the mask-derived grade, the models reach 0.95–0.98 accuracy and QWK 0.95–0.98, compared with 0.84–0.87 and 0.82–0.85 against the visual grade. This is expected, because the models were trained on these masks. It is not an independent check. It does show that most leaf-spot grade errors against the visual reference are leaves where the two references disagree at the G0–G1 boundary.

## 3. Per-grade recall with Wilson 95% intervals

Recall = share of leaves with reference grade g that the model assigned to g. Healthy† = rust-pack healthy TEST leaves from the EARTH site: **21 available locally; the paper used 54**. They are scored by the rust models (abaxial face) and by the leaf-spot models (cross-disease run, mean of both faces). Healthy leaves are never mixed into accuracy or QWK.

| Disease / reference | Grade (n) | SegFormer-B5 | UNet++ | DeepLabV3+ |
|---|---|---|---|---|
| Rust / mask-derived | G0 (1) | 1/1 = 1.00 [0.21, 1.00] | 1/1 = 1.00 [0.21, 1.00] | 1/1 = 1.00 [0.21, 1.00] |
| Rust / mask-derived | G1 (11) | 10/11 = 0.91 [0.62, 0.98] | 9/11 = 0.82 [0.52, 0.95] | 10/11 = 0.91 [0.62, 0.98] |
| Rust / mask-derived | G2 (9) | 8/9 = 0.89 [0.57, 0.98] | 7/9 = 0.78 [0.45, 0.94] | 8/9 = 0.89 [0.57, 0.98] |
| Rust / mask-derived | G3 (0) | no leaves | no leaves | no leaves |
| Rust / mask-derived | Healthy† G0 (21) | 21/21 = 1.00 [0.85, 1.00] | 21/21 = 1.00 [0.85, 1.00] | 21/21 = 1.00 [0.85, 1.00] |
| Leaf spot / visual | G0 (0) | no leaves | no leaves | no leaves |
| Leaf spot / visual | G1 (30) | 22/30 = 0.73 [0.56, 0.86] | 24/30 = 0.80 [0.63, 0.90] | 23/30 = 0.77 [0.59, 0.88] |
| Leaf spot / visual | G2 (23) | 22/23 = 0.96 [0.79, 0.99] | 22/23 = 0.96 [0.79, 0.99] | 22/23 = 0.96 [0.79, 0.99] |
| Leaf spot / visual | G3 (2) | 2/2 = 1.00 [0.34, 1.00] | 2/2 = 1.00 [0.34, 1.00] | 2/2 = 1.00 [0.34, 1.00] |
| Leaf spot / visual | Healthy† G0 (21) | 21/21 = 1.00 [0.85, 1.00] | 21/21 = 1.00 [0.85, 1.00] | 21/21 = 1.00 [0.85, 1.00] |
| Leaf spot / mask-derived | G0 (6) | 6/6 = 1.00 [0.61, 1.00] | 6/6 = 1.00 [0.61, 1.00] | 6/6 = 1.00 [0.61, 1.00] |
| Leaf spot / mask-derived | G1 (26) | 23/26 = 0.88 [0.71, 0.96] | 25/26 = 0.96 [0.81, 0.99] | 24/26 = 0.92 [0.76, 0.98] |
| Leaf spot / mask-derived | G2 (21) | 21/21 = 1.00 [0.85, 1.00] | 21/21 = 1.00 [0.85, 1.00] | 21/21 = 1.00 [0.85, 1.00] |
| Leaf spot / mask-derived | G3 (2) | 2/2 = 1.00 [0.34, 1.00] | 2/2 = 1.00 [0.34, 1.00] | 2/2 = 1.00 [0.34, 1.00] |

Leaf-spot G1 has the lowest recall (0.73–0.80), and every G1 error is an under-grade to G0. Rust G1 and G2 errors are all over-grades. G3 is untested for rust (no leaves), rests on 2 leaf-spot leaves (Wilson lower bound 0.34), and rust G0 on a single diseased leaf.

**Caution for the specificity text (outside this task, found in passing).** On the 21 local healthy leaves, every model's leaf-level predicted severity is < 0.5% (all G0). However, under the reproduced raw-argmax protocol, many of these leaves carry at least one predicted lesion pixel. The count depends on which faces are examined. **Rust models, abaxial face only** (the rust leaf-level rule used throughout this analysis): SegFormer-B5 17/21, UNet++ 15/21, DeepLabV3+ 7/21 (max leaf severity 0.36%, 0.16%, 0.09%). **Rust models, a pixel on either face** (maximum over both faces, the rule in revision brief item 10): SegFormer-B5 19/21, UNet++ 19/21, DeepLabV3+ 13/21 (max single-face severity 0.38%, 0.27%, 0.09%). **Leaf-spot models** (cross-disease run, leaf = mean of both faces, so a pixel on either face counts): SegFormer-B5 19/21, UNet++ 21/21, DeepLabV3+ 21/21 (max leaf severity 0.22%, 0.41%, 0.36%; max single-face severity 0.23%, 0.54%, 0.41%). A single face can exceed 0.5% while the leaf mean stays in G0 (leaf-spot UNet++), so G0 counts also depend on the face rule. The manuscript should quote one rule and state it. The either-face rule matches brief item 10 and is the stricter test of 'no predicted lesion'. The manuscript defines a false positive as 'any predicted lesion pixel' and reports 0/54 for four models. That definition cannot be reconciled with these local outputs. The published counts behave like a severity or grade threshold, not a > 0 pixel rule. The FP definition should be checked against the original evaluation code before the specificity paragraph is finalised.

## 4. Figures

- `figure_paired_differences.pdf/.png`: paired differences (first minus second model) in MAE, |bias|, QWK (bootstrap 95% CI) and grade accuracy (Tango score 95% CI). Right column: McNemar discordant counts b / c and exact p. (a) rust, (b) American leaf spot.
- `figure_grade_recall.pdf/.png`: per-grade recall with Wilson 95% CIs for every model. (a) rust, mask-derived reference; (b) leaf spot, visual reference; (c) leaf spot, mask-derived reference; (d) leaf-spot accuracy (Wilson CI) and QWK (bootstrap CI) against the visual (filled) and mask-derived (open) grades. The dotted line in (d) is the agreement between the two reference standards.

Suggested caption (paired differences): *Paired differences between the published models on the same test leaves (first-named model minus second; 21 rust and 55 American leaf spot leaves; training protocols differed between models, see Section X). Points are observed differences. Bars are 95% percentile intervals from 10,000 paired bootstrap resamples of leaves; for grade accuracy they are Tango score intervals. For MAE and |bias| negative values favour the first model, and for QWK and accuracy positive values do. b / c are leaves graded correctly only by the first / only by the second model, and p is from the exact McNemar test. Rust SegFormer-B5 and DeepLabV3+ assigned identical grades to all 21 leaves. pp = percentage points; QWK = quadratic-weighted kappa.*

Suggested caption (recall): *Per-grade recall of each model with 95% Wilson intervals. (a) Coffee leaf rust, reference grade derived from the semi-automatic reference masks. (b) American leaf spot, annotator's visual grade. (c) American leaf spot, grade derived from the reference masks through the same cut-points. (d) American leaf spot grade accuracy and quadratic-weighted kappa against the visual (filled) and mask-derived (open) grades. The dotted line marks the agreement between the two reference standards. †Healthy test leaves from the EARTH site available locally (21 of the 54 used for specificity), scored by the rust models on the abaxial face and by the leaf-spot models as the mean of both faces.*

## 5. Suggested manuscript sentences (precise, no equivalence claim)

Plain text (LaTeX versions below):

1. For rust, the published SegFormer-B5 model had a lower MAE than the published UNet++ model (training protocols differed; paired difference −0.78 pp, 95% CI −1.46 to −0.26; Holm-adjusted bootstrap p <0.001), but its difference from DeepLabV3+ was not resolved (−0.48 pp, −1.40 to +0.25). For American leaf spot, the published UNet++ model had the lowest MAE, 0.075 pp below SegFormer-B5 (95% CI 0.013 to 0.141; Holm-adjusted bootstrap p = 0.040) and 0.102 pp below DeepLabV3+ (0.057 to 0.152; p <0.001), differences of about a tenth of a percentage point or less.
2. No pair of published models differed detectably in per-leaf grade correctness: within each disease, two models disagreed on whether a leaf was graded correctly for at most two leaves (exact McNemar p ≥ 0.50), and for rust, SegFormer-B5 and DeepLabV3+ assigned the same grade to all 21 leaves.
3. These grade comparisons have little power and do not show equivalence. The 95% interval for the difference in rust grade accuracy was −0.07 to +0.29 between SegFormer-B5 and UNet++, and −0.15 to +0.15 even between SegFormer-B5 and DeepLabV3+, which graded all 21 leaves identically. For leaf spot it was −0.12 to +0.03 between SegFormer-B5 and UNet++.
4. Against grades derived from the reference masks, leaf-spot grade accuracy was 0.95, 0.98 and 0.96 and quadratic-weighted kappa 0.95, 0.98 and 0.97 for SegFormer-B5, UNet++ and DeepLabV3+. Against the visual grades, accuracy was 0.84–0.87 and kappa 0.82–0.85. Because the models were trained on these masks, this is not an independent check, but it locates most leaf-spot grade errors at the G0–G1 boundary, where the visual and mask-derived references themselves disagree.
5. Per-grade recall was lowest for leaf-spot G1 (0.73–0.80; Wilson 95% intervals 0.56–0.90), where every error was an under-grade to G0. G3 recall rests on two leaf-spot leaves and could not be assessed for rust. Because these tests compare the published models, whose training protocols differed, they do not isolate an effect of architecture; differences in the ordinal grade were not detected and cannot be excluded at this sample size.

```latex
For rust, the published SegFormer-B5 model had a lower MAE than the published UNet++ model (training protocols differed; paired difference $-$0.78 pp, 95\% CI $-$1.46 to $-$0.26; Holm-adjusted bootstrap p $<$ 0.001), but its difference from DeepLabV3+ was not resolved ($-$0.48 pp, $-$1.40 to +0.25). For American leaf spot, the published UNet++ model had the lowest MAE, 0.075 pp below SegFormer-B5 (95\% CI 0.013 to 0.141; Holm-adjusted bootstrap p = 0.040) and 0.102 pp below DeepLabV3+ (0.057 to 0.152; p $<$ 0.001), differences of about a tenth of a percentage point or less.

No pair of published models differed detectably in per-leaf grade correctness: within each disease, two models disagreed on whether a leaf was graded correctly for at most two leaves (exact McNemar p $\geq$ 0.50), and for rust, SegFormer-B5 and DeepLabV3+ assigned the same grade to all 21 leaves.

These grade comparisons have little power and do not show equivalence. The 95\% interval for the difference in rust grade accuracy was $-$0.07 to +0.29 between SegFormer-B5 and UNet++, and $-$0.15 to +0.15 even between SegFormer-B5 and DeepLabV3+, which graded all 21 leaves identically. For leaf spot it was $-$0.12 to +0.03 between SegFormer-B5 and UNet++.

Against grades derived from the reference masks, leaf-spot grade accuracy was 0.95, 0.98 and 0.96 and quadratic-weighted kappa 0.95, 0.98 and 0.97 for SegFormer-B5, UNet++ and DeepLabV3+. Against the visual grades, accuracy was 0.84--0.87 and kappa 0.82--0.85. Because the models were trained on these masks, this is not an independent check, but it locates most leaf-spot grade errors at the G0--G1 boundary, where the visual and mask-derived references themselves disagree.

Per-grade recall was lowest for leaf-spot G1 (0.73--0.80; Wilson 95\% intervals 0.56--0.90), where every error was an under-grade to G0. G3 recall rests on two leaf-spot leaves and could not be assessed for rust. Because these tests compare the published models, whose training protocols differed, they do not isolate an effect of architecture; differences in the ordinal grade were not detected and cannot be excluded at this sample size.

```

Methods addition: *Differences between the published models in MAE, absolute bias, quadratic-weighted kappa and grade accuracy were estimated on the same leaves with paired percentile bootstrap intervals (10,000 resamples of leaves, seed 20260927). For grade accuracy we also report Tango's score interval. Per-leaf grade correctness was compared with the exact McNemar test with Holm correction within disease. Accuracy and per-grade recall carry Wilson score intervals. No equivalence margin was pre-specified.*

Conclusions replacement for the sentence 'grade and severity differences between architectures were not formally tested…': *On rust, the published DeepLabV3+ model assigned the same grade as the published SegFormer-B5 model to all 21 test leaves, and the published UNet++ model differed on two. No grade difference was detected, but with 21 leaves, differences in grade accuracy of up to about 15 points (SegFormer-B5 vs DeepLabV3+) and 29 points (SegFormer-B5 vs UNet++) cannot be excluded. The published SegFormer-B5 model had a lower rust severity MAE than the published UNet++ model, but the training protocols differed, so these comparisons do not establish an advantage of one architecture.* (If the matched-protocol retraining is reported in the manuscript, cite it here, because for rust the SegFormer-B5 advantage largely disappears under matched training; see §6. `% TODO(retraining)` for leaf spot.)

## 6. Caveats

- Each published model was trained once, and the leaf-spot models were never re-seeded. All intervals reflect test-leaf sampling only, not training variability.
- **Published models, not architectures.** Training protocols differed (brief facts 3–4), so no difference here is a controlled architecture effect. The matched-protocol **rust** retraining is available (brief item 11: 60 epochs, patience 15, geometric augmentation only, LR 6e-5/6e-4 chosen on validation; evaluated with the paper protocol on the same 21 test leaves; numbers quoted from the brief, not recomputed here). Retrained UNet++ reached a leaf IoU of 0.776 and an MAE of 0.50 pp (bias +0.47 pp, grade accuracy 0.95, QWK 0.94), and retrained DeepLabV3+ reached 0.721 and 0.87 pp, against 0.767 and 0.99 pp for the published SegFormer-B5. The brief concludes that the published rust ranking is a training-budget effect. The rust SegFormer-B5 vs UNet++ severity difference reported here therefore describes the published models only. The paired tests were not re-run on the retrained models, because their per-leaf outputs are not in this folder's input. **Leaf-spot** retraining is still running (`% TODO(retraining)`).
- Rust reference severity and grades come from the semi-automatic (detector + SAM 2, human-spot-checked) masks. The rust grade comparisons measure agreement with that reference.
- The percentile bootstrap is degenerate when two models grade every leaf identically (rust SegFormer-B5 vs DeepLabV3+) and pins a bound at 0 when discordance is one-sided. The McNemar counts and Tango intervals are the informative summaries for grades, and TOST margins for Δaccuracy come from the Tango 90% interval. No TOST margin is given for ΔQWK. No QWK resample was undefined (0/0) in any analysis.
- Leaf-level bootstrap only. A session-clustered bootstrap is a separate TODO.
- Healthy leaves: 21 of the paper's 54 are available locally, all from EARTH, so they are confounded with site for rust.
