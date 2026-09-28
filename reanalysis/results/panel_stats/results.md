# Expert-panel secondary statistics (PG-26038 revision)

> **Public-release copy.** Per-evaluator values were removed and per-leaf panel grades were removed (the sorted four-evaluator grades, set against the paper's five-evaluator per-leaf figure, would reveal the added evaluator's grades). Every number shown is unchanged; local paths are written as `<ROOT>` (the working tree) and `<USER_HOME>`.

Primary set: 53 leaves (30 American leaf spot, 23 rust); the 7 rust-training healthy leaves are excluded. Rust grade from the abaxial face, leaf spot from the mean of both faces. Statistic: mean leave-one-evaluator-out (LOO) QWK of the model minus that of the evaluators (as in `tiefree.py`). Bootstrap: 10,000 leaf resamples, seed 7 (the tiefree seed). Evaluators are held fixed, so inference is conditional on these four readers. Two-sided p = min(1, 2·min(P*(d ≤ 0), P*(d ≥ 0))), which inverts the percentile interval. Script: `panel_stats.py`; all numbers: `results.json`.

## 1. Model-minus-evaluator contrasts

| Disease | Model | Role | Model LOO QWK | Evaluators LOO QWK | Difference [95% CI] | p (unadjusted) | p Holm, 4 secondary | p Holm, all 6 |
|---|---|---|---|---|---|---|---|---|
| American leaf spot | SegFormer-B5 | primary | 0.884 | 0.765 | +0.119 [0.061, 0.188] | <0.001 | — (unadjusted) | 0.002 |
| American leaf spot | DeepLabV3+ | secondary | 0.860 | 0.765 | +0.095 [0.020, 0.175] | 0.016 | 0.066 | 0.082 |
| American leaf spot | UNet++ | secondary | 0.747 | 0.765 | −0.018 [−0.131, 0.093] | 0.76 | 0.93 | 0.93 |
| Coffee leaf rust | SegFormer-B5 | primary | 0.815 | 0.728 | +0.087 [−0.015, 0.218] | 0.084 | — (unadjusted) | 0.34 |
| Coffee leaf rust | DeepLabV3+ | secondary | 0.777 | 0.728 | +0.049 [−0.077, 0.263] | 0.46 | 0.93 | 0.93 |
| Coffee leaf rust | UNet++ | secondary | 0.794 | 0.728 | +0.066 [−0.035, 0.204] | 0.18 | 0.54 | 0.54 |

Pooled (descriptive only: it mixes two disease-specific models and belongs to no test family): SegFormer-B5 +0.107 [0.052, 0.169]; DeepLabV3+ +0.079 [−0.0004, 0.166]; UNet++ +0.022 [−0.052, 0.099].

**Reproduction check:** all 9 CIs (6 per-disease and 3 pooled) equal `validation/tiefree_results.json` `diff_ci` to 3 dp (max |difference| = 0.0), and the point estimates match `final_results.json`. **Monte Carlo check** (independent seed 20260927): CI bounds moved by at most 0.010; p-values: ojo_segformer <0.001, ojo_deeplab 0.016, ojo_unetpp 0.74, roya_segformer 0.093, roya_deeplab 0.47, roya_unetpp 0.18. No conclusion changes.

## 2. Direction of error against the four-evaluator median (both tie rules)

Two-against-two splits among the primary leaves: 7 of 30 leaf-spot leaves and 9 of 23 rust leaves. Counts are model grade lower / equal / higher than the median. Rounding ties up gives the higher reference, and rounding down the lower one. The last column is tie-free: it compares the model grade with the lowest and highest of the four panel grades.

| Disease | Model | Ties up: lower / equal / higher (bias) | Ties down: lower / equal / higher (bias) | Panel-G0 leaves graded ≥ G1 (up; down) | Tie-free: below min / inside / above max |
|---|---|---|---|---|---|
| American leaf spot | SegFormer-B5 | 1 / 27 / 2 (+0.03) | 0 / 22 / 8 (+0.27) | 0 of 7; 3 of 10 | 0 / 30 / 0 |
| American leaf spot | DeepLabV3+ | 1 / 26 / 3 (+0.07) | 0 / 21 / 9 (+0.30) | 1 of 7; 4 of 10 | 0 / 30 / 0 |
| American leaf spot | UNet++ | 1 / 22 / 7 (+0.20) | 0 / 17 / 13 (+0.43) | 5 of 7; 8 of 10 | 0 / 30 / 0 |
| Coffee leaf rust | SegFormer-B5 | 2 / 19 / 2 (+0.00) | 0 / 14 / 9 (+0.39) | 0 of 3; 4 of 7 | 0 / 21 / 2 |
| Coffee leaf rust | DeepLabV3+ | 6 / 17 / 0 (−0.26) | 2 / 16 / 5 (+0.13) | 0 of 3; 3 of 7 | 2 / 21 / 0 |
| Coffee leaf rust | UNet++ | 2 / 18 / 3 (+0.04) | 0 / 14 / 9 (+0.43) | 0 of 3; 3 of 7 | 0 / 20 / 3 |

Grades outside the panel range (all rust; none for leaf spot): SegFormer-B5 roya13 graded G2 (abaxial 5.4%) (panel grades omitted); SegFormer-B5 roya21 graded G2 (abaxial 8.81%) (panel grades omitted); DeepLabV3+ roya07 graded G2 (abaxial 15.2%) (panel grades omitted); DeepLabV3+ roya15 graded G2 (abaxial 17.14%) (panel grades omitted); UNet++ roya17 graded G3 (abaxial 24.3%) (panel grades omitted); UNet++ roya21 graded G2 (abaxial 11.26%) (panel grades omitted); UNet++ roya23 graded G2 (abaxial 5.51%) (panel grades omitted).

## 3. Per-disease primary values

| Disease (n) | Rater | LOO QWK | Difference [95% CI] | Mean QWK vs each evaluator | QWK vs 4-evaluator median (down–up) | Exact vs median (down–up) | Exact vs LOO consensus | Inside panel range | Bias (down; up) |
|---|---|---|---|---|---|---|---|---|---|
| American leaf spot (30) | Evaluators (mean; range) | 0.77 (0.67–0.82) | — | 0.72 (pairwise) | — | — | 61% | — | — |
| | SegFormer-B5 | 0.88 | +0.12 [0.06, 0.19] | 0.81 | 0.83–0.94 | 73–90% | 82% | 30/30 | +0.27; +0.03 |
| | DeepLabV3+ | 0.86 | +0.09 [0.02, 0.17] | 0.79 | 0.81–0.91 | 70–87% | 78% | 30/30 | +0.30; +0.07 |
| | UNet++ | 0.75 | −0.02 [−0.13, 0.09] | 0.72 | 0.70–0.80 | 57–73% | 65% | 30/30 | +0.43; +0.20 |
| Coffee leaf rust (23) | Evaluators (mean; range) | 0.73 (0.69–0.75) | — | 0.77 (pairwise) | — | — | 55% | — | — |
| | SegFormer-B5 | 0.82 | +0.09 [−0.01, 0.22] | 0.78 | 0.76–0.88 | 61–83% | 72% | 21/23 | +0.39; +0.00 |
| | DeepLabV3+ | 0.78 | +0.05 [−0.08, 0.26] | 0.73 | 0.76–0.80 | 70–74% | 72% | 21/23 | +0.13; −0.26 |
| | UNet++ | 0.79 | +0.07 [−0.03, 0.20] | 0.75 | 0.73–0.87 | 61–78% | 70% | 20/23 | +0.43; +0.04 |
| Pooled (53) [descriptive] | Evaluators (mean; range) | 0.75 (0.69–0.79) | — | 0.74 (pairwise) | — | — | 58% | — | — |
| | SegFormer-B5 | 0.86 | +0.11 [0.05, 0.17] | 0.80 | 0.80–0.91 | 68–87% | 77% | 51/53 | +0.32; +0.02 |
| | DeepLabV3+ | 0.83 | +0.08 [−0.0004, 0.17] | 0.77 | 0.79–0.87 | 70–81% | 75% | 51/53 | +0.23; −0.08 |
| | UNet++ | 0.77 | +0.02 [−0.05, 0.10] | 0.74 | 0.71–0.84 | 58–75% | 67% | 50/53 | +0.43; +0.13 |

## Suggested sentences for the paper

Three sentences, LaTeX-ready. The fourth is optional.

```latex
% Methods (Section~\ref{sec:expert_methods})
The two SegFormer-B5 contrasts, one per disease, were designated primary and tested without adjustment, the four DeepLabV3+ and UNet++ contrasts were Holm-corrected~\cite{ref47}, and two-sided $p$-values were obtained by inverting the percentile bootstrap interval with the evaluators held fixed, so inference is conditional on this panel.

% Results (Section~\ref{sec:expert_results})
SegFormer-B5 exceeded the evaluators for American leaf spot (+0.12 [0.06, 0.19], $p < 0.001$) but not significantly for rust (+0.09 [$-$0.01, 0.22], $p = 0.084$), and no secondary contrast remained significant after Holm correction, including DeepLabV3+ on American leaf spot (+0.09 [0.02, 0.17]; unadjusted $p = 0.016$, Holm $p = 0.066$; other contrasts Holm $p \geq 0.54$).

The direction of the rust errors depended on how two-against-two panel splits were resolved: with ties rounded up, DeepLabV3+ graded 6 of 23 leaves below the four-evaluator median and 0 above it, but with ties rounded down 2 below and 5 above (SegFormer-B5: 2 below and 2 above, against 0 and 9), whereas UNet++ over-graded near-healthy leaf-spot leaves under either rule, grading 5 of the 7 (ties up) or 8 of the 10 (ties down) panel-G0 leaves as G1.

% Optional, tie-free
Against the full range of panel grades, which needs no tie rule, DeepLabV3+ fell below the lowest panel grade on 2 rust leaves, both rated G3 by all four evaluators, and SegFormer-B5 above the highest on 2, both rated G1 by all four.
```

## Notes

- **Text that changes:** Section 3.5 now says DeepLabV3+ "also exceeded the evaluators on leaf spot (+0.09 [0.02, 0.17])". The CI is correct, but the contrast is secondary and its Holm-adjusted p is 0.066 (0.082 with Holm over all six). Present it as not significant after correction.
- **Text that changes:** the claims that DeepLabV3+ "under-graded rust … never a higher one" and that SegFormer-B5 rust errors were "balanced (2 lower, 2 higher)" hold only when ties are rounded up. When ties are rounded down, every model grades higher on average, because that reference is lower. Across both rules, DeepLabV3+ has the lowest rust bias of the three models (−0.26 ties up, +0.13 ties down). The UNet++ leaf-spot over-grading holds under both rules.
- Only SegFormer-B5 on leaf spot stays significant under Holm over all six contrasts (p = 0.002).
- Do not quote tiefree's `p_le0` (for example, 0.042 for SegFormer-B5 on rust) as a p-value. It is one-sided; the two-sided value is 0.084.
- Per-evaluator values are omitted from this public copy (results.md and results.json).
- Not done: an MRMC (Obuchowski–Rockette) analysis or a reader-and-leaf bootstrap. Evaluators are fixed, so claims about "a typical expert" should be worded as conditional on this panel.
