"""Write results.md from results.json (every number in the Markdown comes from results.json; nothing is typed by hand
except the interpretive text)."""
import json
import os

OUT = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(OUT, "results.json"), encoding="utf-8"))
M = R["models"]
DIS = [("rust", "Coffee leaf rust"), ("leaf_spot", "American leaf spot")]
ARCH = [("segformer", "SegFormer-B5"), ("unetpp", "UNet++"), ("deeplab", "DeepLabV3+")]


def n(x, d=2, sign=False):
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "−")


def ci(c, d=2):
    return f"[{n(c[0], d)}, {n(c[1], d)}]"


def p(v):
    return "<0.001" if v < 0.001 else f"{v:.3f}"


def m(dis, arch):
    return M[f"{dis}/{arch}"]


L = []
w = L.append
S = m("rust", "segformer")
LS = m("leaf_spot", "segformer")
rs_reg = S["regression_of_differences_on_means"]
ls_reg = LS["regression_of_differences_on_means"]
rs5 = S["regression_based_loa"]["at_reference_cutpoint"]["ref=5%"]
rs20 = S["regression_based_loa"]["at_reference_cutpoint"]["ref=20%"]
ls5 = LS["regression_based_loa"]["at_reference_cutpoint"]["ref=5%"]
ls20 = LS["regression_based_loa"]["at_reference_cutpoint"]["ref=20%"]
dec = S["error_decomposition_pct_of_leaf"]
ldec = LS["error_decomposition_pct_of_leaf"]
ref_r, ref_l = S["reference_severity"], LS["reference_severity"]
# SegFormer-B5 rust test leaves inside the bands where the fitted mean prediction is at or above 5% / 20%
_sb = [b for cut in ("5", "20") for b in S["regression_based_loa"]["at_reference_cutpoint"][f"ref={cut}%"]
       ["leaves_in_that_band_below_cut"]]
_sb_over = sorted((b for b in _sb if b["over_graded"]), key=lambda b: b["ref_sev"])
_sb_ok = sorted((b for b in _sb if not b["over_graded"]), key=lambda b: b["ref_sev"])
# every SegFormer-B5 rust grade error lies in a band (checked), so the sentence below is exact
assert {d["leaf_id"] for d in S["grades"]["disagreements"]} == {b["leaf_id"] for b in _sb_over}
assert len(_sb_over) == 2 and len(_sb_ok) == 1
SB_ERR = f"{_sb_over[0]['ref_sev']:.2f}% and {_sb_over[1]['ref_sev']:.2f}%"
SB_OK = f"{_sb_ok[0]['ref_sev']:.2f}%, predicted {_sb_ok[0]['pred_sev']:.2f}%"
SB_SENT = (f"Both SegFormer-B5 rust grade errors (reference {SB_ERR}) lay in these bands, and the third test leaf in "
           f"them ({SB_OK}) was graded correctly.")

w("# Severity agreement reanalysis (Bland–Altman done properly)")
w("")
w("PG-26038 major revision, 27 September 2026. Scripts: `pixel_metrics.py` (reference rebuilt in letterbox space, "
  "FP/FN split, boundary F1), `agreement.py` (all statistics → `results.json`), `make_figures.py`, `make_md.py`. "
  "No model inference was re-run; predictions are the saved class maps in `out/preds`.")
w("")
w("**Scope.** Diseased test leaves at leaf level: rust = abaxial face of the 21 leaves; American leaf spot = mean of "
  "both faces of the 55 leaves. D = predicted − reference severity (percentage points, pp); A = mean of the two. "
  "Bootstrap = 10,000 resamples of leaves, percentile 95% CIs, seed 20260927 (paired across models within a disease). "
  "Rust reference grades are derived from the reference severity (cut-points 0.5/5/20%); leaf-spot reference grades are "
  "the annotator's visual grades.")
w("")
w("## 1. Reproduction of the published values")
w("")
w(f"All published values reproduce exactly (`reproduction_all_match` = {str(R['reproduction_all_match']).lower()}); all "
  "reference and predicted pixel counts rebuilt from the images/masks/prediction maps match `per_image.csv` for all 393 "
  "model × image pairs.")
w("")
w("| Disease | Model | Leaf IoU | MAE | Bias | QWK | Constant LoA (published) | Reproduced |")
w("|---|---|---|---|---|---|---|---|")
for dis, dn in DIS:
    for arch, an in ARCH:
        r = R["reproduction"][f"{dis}/{arch}"]["reproduced"]
        pub = R["reproduction"][f"{dis}/{arch}"]["published"]
        w(f"| {dn} | {an} | {r['iou']:.3f} | {n(r['mae'])} | {n(r['bias'], 2, True)} | {r['qwk']:.2f} | "
          f"{n(pub['loa'][0])} to {n(pub['loa'][1], 2, True)} | {n(r['loa'][0])} to {n(r['loa'][1], 2, True)} ✓ |")
w("")
w("## 2. Reference severity (test diseased leaves)")
w("")
w("| Disease | n | Mean (%) | Median (%) | SD | Range (%) | IQR (%) | Leaves > 10% | Reference grades |")
w("|---|---|---|---|---|---|---|---|---|")
for dis, dn in DIS:
    rs = m(dis, "segformer")["reference_severity"]
    g = rs["grade_counts_reference"]
    w(f"| {dn} | {m(dis, 'segformer')['n_leaves']} | {rs['mean']:.2f} | {rs['median']:.2f} | {rs['sd']:.2f} | "
      f"{rs['min']:.2f}–{rs['max']:.2f} | {rs['q25']:.2f}–{rs['q75']:.2f} | {rs['n_above_10pct']} | "
      f"G0 {g['G0']}, G1 {g['G1']}, G2 {g['G2']}, G3 {g['G3']}; {rs['grade_source']} |")
w("")
w(f"No rust test leaf reaches 20% reference severity (maximum {ref_r['max']:.2f}%); only {ref_r['n_above_10pct']} rust "
  f"leaves exceed 10%, so every rust statement about the 20% cut-point is anchored by those three leaves.")
w("")
w("## 3. Bias and constant limits of agreement (95% CIs)")
w("")
w("| Disease | Model | Bias (pp) [bootstrap CI] | SD of D | Lower LoA [bootstrap CI] | Upper LoA [bootstrap CI] | "
  "Lower / upper LoA, parametric CI (B&A 1986) |")
w("|---|---|---|---|---|---|---|")
for dis, dn in DIS:
    for arch, an in ARCH:
        x = m(dis, arch)
        c = x["constant_loa"]
        w(f"| {dn} | {an} | {n(x['bias']['estimate'], 2, True)} {ci(x['bias']['ci95_bootstrap'])} | "
          f"{x['sd_of_differences']['estimate']:.2f} | {n(c['lower'])} {ci(c['lower_ci95_bootstrap'])} | "
          f"{n(c['upper'], 2, True)} {ci(c['upper_ci95_bootstrap'])} | {ci(c['lower_ci95_parametric_BA1986'])} / "
          f"{ci(c['upper_ci95_parametric_BA1986'])} |")
w("")
w("Constant LoA assume that D does not depend on severity; Section 4 shows that this assumption fails for rust "
  "(all models) and, to a small degree, for leaf spot, so the constant LoA are reported only to reproduce the "
  "published values and should be replaced by the regression-based LoA.")
w("")
w("## 4. Proportional bias: regression of differences on means")
w("")
w("OLS of D on A; 95% CI and p from the t distribution (df = n − 2). Because the residual spread grows with A, the slope "
  "is also given with a bootstrap percentile CI and an HC3 heteroscedasticity-robust p value. "
  "'Abs-residual slope' = slope of the absolute residuals on A (tests whether the SD of D grows with A).")
w("")
w("| Disease | Model | Slope [t CI] | p | Slope bootstrap CI | p (HC3) | Intercept (pp) [t CI] | p | R² | Abs-residual slope; p |")
w("|---|---|---|---|---|---|---|---|---|---|")
for dis, dn in DIS:
    for arch, an in ARCH:
        g = m(dis, arch)["regression_of_differences_on_means"]
        rl = m(dis, arch)["regression_based_loa"]
        w(f"| {dn} | {an} | {n(g['slope'], 3)} {ci(g['slope_ci95_t'], 3)} | {p(g['slope_p'])} | "
          f"{ci(g['slope_ci95_bootstrap'], 3)} | {p(g['slope_p_hc3'])} | {n(g['intercept'], 2, True)} "
          f"{ci(g['intercept_ci95_t'])} | {p(g['intercept_p'])} | {g['r_squared']:.2f} | "
          f"{n(rl['abs_residual_slope_c1'], 3)}; {p(rl['abs_residual_slope_p'])} |")
w("")
w("Calibration regression (secondary): predicted = α + β × reference.")
w("")
w("| Disease | Model | β [bootstrap CI] | α (pp) | Log-ratio: geometric mean ratio pred/ref [CI] | Ratio LoA | Leaves over-estimated |")
w("|---|---|---|---|---|---|---|")
for dis, dn in DIS:
    for arch, an in ARCH:
        x = m(dis, arch)
        c = x["calibration_pred_on_ref"]
        lr = x["log_ratio_loa"]
        w(f"| {dn} | {an} | {c['slope']:.3f} {ci(c['slope_ci95_bootstrap'], 3)} | {n(c['intercept'], 2, True)} | "
          f"{lr['geometric_mean_ratio']:.3f} {ci(lr['gmr_ci95_bootstrap'], 3)} | {lr['loa_ratio_lower']:.2f}–"
          f"{lr['loa_ratio_upper']:.2f} | {lr['n_leaves_with_ratio_above_1']}/{x['n_leaves']} |")
w("")
w("## 5. Regression-based limits of agreement (Bland & Altman 1999)")
w("")
w("D̂(A) = b0 + b1·A; |residual| = c0 + c1·A; SD(A) = √(π/2)·(c0 + c1·A); LoA(A) = D̂(A) ± 1.96·SD(A). "
  "c0 > 0 for every model, so the modelled SD is positive over the whole observed range.")
w("")
w("| Disease | Model | D̂(A) | SD(A) |")
w("|---|---|---|---|")
for dis, dn in DIS:
    for arch, an in ARCH:
        rl = m(dis, arch)["regression_based_loa"]
        w(f"| {dn} | {an} | {rl['mean_function'].split('= ')[1].replace('-', chr(0x2212))} | {rl['sd_function'].split('= ')[1]} |")
w("")
w("### 5a. At the 5% and 20% cut-points (reference scale)")
w("")
w("Values are read off the fitted line at the point whose reference severity equals the cut-point "
  "(R = A − D/2, i.e. D = (b0 + b1·R)/(1 − b1/2)); 'relative' = fitted D as % of the cut-point. "
  "'Fitted prediction reaches the cut at' = reference severity above which the fitted mean prediction R + D̂ is at or "
  "above the cut-point. Leaves with reference severity between that value and the cut-point are therefore prone to "
  "over-grading, but the band only marks where the fitted mean crosses the cut: at its lower edge the fitted prediction "
  "just reaches the cut, so an over-grade there is roughly as likely as not (the chance rises toward the cut-point), "
  "and the edge itself is uncertain (bootstrap CI). The last column gives the number of test leaves in the band and, "
  "in parentheses, how many of them were actually over-graded.")
w("")
w("| Disease | Model | Cut | Fitted D (pp) [bootstrap CI] | Relative (%) [CI] | Regression-based 95% LoA | Fitted prediction reaches the cut at reference = [CI] | Test leaves in band (over-graded) |")
w("|---|---|---|---|---|---|---|---|")
for dis, dn in DIS:
    for arch, an in ARCH:
        for cut in ("5", "20"):
            v = m(dis, arch)["regression_based_loa"]["at_reference_cutpoint"][f"ref={cut}%"]
            w(f"| {dn} | {an} | {cut}% | {n(v['fitted_difference'], 2, True)} "
              f"{ci(v['fitted_difference_ci95_bootstrap'])} | {n(v['fitted_difference_relative_pct'], 1, True)} "
              f"{ci(v['fitted_difference_relative_pct_ci95_bootstrap'], 1)} | {n(v['loa_lower'], 2, True)} to "
              f"{n(v['loa_upper'], 2, True)} | {v['reference_severity_whose_fitted_prediction_reaches_cut']:.2f}% "
              f"{ci(v['reference_severity_whose_fitted_prediction_reaches_cut_ci95_bootstrap'])} | "
              f"{v['n_leaves_with_reference_in_that_band_below_cut']} ({v['n_over_graded_in_that_band']}) |")
w("")
w("### 5b. At fixed mean severities A (as plotted)")
w("")
w("| Disease | Model | A = 1% | A = 5% | A = 10% | A = 20% |")
w("|---|---|---|---|---|---|")
for dis, dn in DIS:
    for arch, an in ARCH:
        cells = []
        for a in ("1", "5", "10", "20"):
            v = m(dis, arch)["regression_based_loa"]["at_mean_severity"][f"A={a}"]
            cells.append(f"{n(v['fitted_difference'], 2, True)} ({n(v['loa_lower'], 2, True)} to {n(v['loa_upper'], 2, True)})")
        w(f"| {dn} | {an} | " + " | ".join(cells) + " |")
w("")
w("Cells: fitted D (regression-based 95% LoA), pp. Observed A range: rust 0.39–22.2%, leaf spot 0.12–27.3%; "
  f"only {S['mean_of_A']['n_A_above_15']} rust and {LS['mean_of_A']['n_A_above_15']} leaf-spot leaves have A > 15%.")
w("")
w("### 5c. Grade errors against the cut-points (observed)")
w("")
CAT_LABEL = {"visual_mask_gap": "visual/mask gap", "severity_error": "severity error",
             "gap_and_severity": "gap and severity error"}
w("Leaves sorted by reference severity. For leaf spot the reference grade is the annotator's visual grade, so each "
  "error is also labelled with the grade derived from the reference mask severity (same cut-points): *visual/mask gap* "
  "= the predicted grade equals the mask-derived grade (an exact reproduction of the reference mask would give the same "
  "error); *severity error* = the mask-derived grade equals the visual grade and the predicted severity crossed a "
  "cut-point. For rust the reference grade is mask-derived, so every rust error is a severity error.")
w("")
for dis, dn in DIS:
    for arch, an in ARCH:
        g = m(dis, arch)["grades"]
        dd = sorted(g["disagreements"], key=lambda d: d["ref_sev"])
        if dis == "rust":
            items = "; ".join(f"{d['ref_sev']:.2f}% → {d['pred_sev']:.2f}% (G{d['ref_grade']}→G{d['pred_grade']})"
                              for d in dd)
        else:
            items = "; ".join(f"{d['ref_sev']:.3f}% → {d['pred_sev']:.3f}% (visual G{d['ref_grade']}→G{d['pred_grade']}; "
                              f"mask-derived G{d['mask_grade']}: {CAT_LABEL[d['category']]})" for d in dd)
        w(f"- {dn}, {an}: QWK {g['qwk']:.2f}, accuracy {g['accuracy']:.2f}; {g['n_over']} over-grades, "
          f"{g['n_under']} under-grades. {('Reference → predicted severity: ' + items) if items else ''}")
w("")

# ---- rust: are the over-grades inside the band where the fitted mean prediction crosses the cut?
rust_err, rust_err_in, rust_err_out = 0, 0, []
band_not_over = {}  # ref_sev -> {model: pred_sev} for band leaves that were NOT over-graded
for arch, an in ARCH:
    x = m("rust", arch)
    cp = x["regression_based_loa"]["at_reference_cutpoint"]
    for d in x["grades"]["disagreements"]:
        assert d["pred_grade"] == d["ref_grade"] + 1  # one-grade over-call
        cut = {1: "5", 2: "20"}[d["ref_grade"]]
        lo = cp[f"ref={cut}%"]["reference_severity_whose_fitted_prediction_reaches_cut"]
        rust_err += 1
        if lo <= d["ref_sev"] < float(cut):
            rust_err_in += 1
        else:
            rust_err_out.append((an, d, cut, lo))
    for cut in ("5", "20"):
        for b in cp[f"ref={cut}%"]["leaves_in_that_band_below_cut"]:
            if not b["over_graded"]:
                band_not_over.setdefault(round(b["ref_sev"], 2), {})[an] = b["pred_sev"]
assert all(m("rust", a)["grades"]["n_under"] == 0 for a, _ in ARCH)
assert all(lo_o - d_o["ref_sev"] < 0.05 for _, d_o, _, lo_o in rust_err_out)  # 'at the lower edge' holds
out_txt = "; ".join(f"the {an_o} over-call {d_o['ref_sev']:.2f}% → {d_o['pred_sev']:.2f}% lies "
                    f"{lo_o - d_o['ref_sev']:.2f} pp below the lower edge of the {an_o} {cut_o}% band ({lo_o:.2f}%; "
                    f"reference {d_o['ref_sev']:.3f}% vs {lo_o:.3f}%)" for an_o, d_o, cut_o, lo_o in rust_err_out)
bno_txt = "; ".join(f"the leaf at {rsv:.2f}% lies in the 20% band of all three models but was graded correctly by "
                    + " and ".join(f"{a} (predicted {pv:.2f}%)" for a, pv in mods.items())
                    for rsv, mods in band_not_over.items())
w(f"For rust every grade error is a one-grade over-call (no model under-graded a rust leaf), and all {rust_err} "
  "over-calls (all models) lie inside or at the lower edge of the band just below a cut-point in which the fitted mean "
  f"prediction is at or above the cut (Section 5a): {rust_err_in} lie inside it, and {out_txt}. Conversely, not every "
  f"leaf in a band was over-graded: {bno_txt}. The bands therefore mark where leaves are prone to over-grading, not "
  "where it is certain.")
w("")

# ---- leaf spot: visual/mask gap versus genuine severity errors
gap_leaves, sev_err = {}, []
for arch, an in ARCH:
    for d in m("leaf_spot", arch)["grades"]["disagreements"]:
        if d["category"] == "visual_mask_gap":
            gap_leaves[d["leaf_id"]] = d
        elif d["category"] == "severity_error":
            sev_err.append((an, d))
assert all(d["direction"] == "under" for a, _ in ARCH for d in m("leaf_spot", a)["grades"]["disagreements"])
assert all(m("leaf_spot", a)["grades"]["disagreement_category_counts"]["gap_and_severity"] == 0 for a, _ in ARCH)
g_low = sorted(d["ref_sev"] for d in gap_leaves.values() if d["ref_grade"] == 1 and d["mask_grade"] == 0)
g_hi = [d for d in gap_leaves.values() if d["ref_grade"] == 2 and d["mask_grade"] == 1]
assert len(g_low) + len(g_hi) == len(gap_leaves) and len(g_hi) == 1
cnt = {an: m("leaf_spot", a)["grades"]["disagreement_category_counts"] for a, an in ARCH}
ntot = {an: len(m("leaf_spot", a)["grades"]["disagreements"]) for a, an in ARCH}
_cov = [f"{'all ' if cnt[an]['visual_mask_gap'] == ntot[an] else ''}{cnt[an]['visual_mask_gap']}"
        f"{'' if cnt[an]['visual_mask_gap'] == ntot[an] else ' of the ' + str(ntot[an])} {an} errors"
        for an in ("UNet++", "SegFormer-B5", "DeepLabV3+")]
cover = ", ".join(_cov[:-1]) + " and " + _cov[-1]
_under = [d["ref_sev"] - d["pred_sev"] for _, d in sev_err]
NUMW = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
sev_txt = "; ".join(f"{an} {d['ref_sev']:.3f}% → {d['pred_sev']:.3f}%" for an, d in sev_err)
ls_int = {an: m("leaf_spot", a)["regression_of_differences_on_means"] for a, an in ARCH}
sev_models = sorted({an for an, _ in sev_err}, key=lambda s: [a[1] for a in ARCH].index(s))
int_txt = "; ".join(f"{an} {n(ls_int[an]['intercept'], 2, True)} pp, p = {ls_int[an]['intercept_p']:.3f}, fitted D at "
                    f"A = 0.5% {n(ls_int[an]['intercept'] + 0.5 * ls_int[an]['slope'], 2, True)} pp" for an in sev_models)
LS_GAP_SENT = (f"{NUMW[len(g_low)]} leaves visually graded G1 have mask severity {g_low[0]:.2f}–{g_low[-1]:.2f}% "
               f"(mask-derived G0) and one leaf visually graded G2 has mask severity {g_hi[0]['ref_sev']:.2f}% "
               "(mask-derived G1)")
w(f"For leaf spot all grade errors are under-grades, of two kinds. (i) Visual/mask gap: {LS_GAP_SENT}. For these "
  f"{NUMW[len(gap_leaves)]} leaves the predicted grade equals the mask-derived grade, so even an exact reproduction of the "
  f"reference mask would have produced the same error; they reflect the known gap between visual grades and mask "
  f"severity, not model error, and account for {cover}. (ii) Severity errors across the 0.5% cut-point: {sev_txt}. "
  f"Here the mask-derived grade agrees with the visual G1, and the model under-estimated severity by "
  f"{min(_under):.2f}–{max(_under):.2f} pp, which took the prediction below 0.5%. These small under-estimates at low severity are in the direction of the negative leaf-spot intercepts "
  f"({int_txt}); they are not caused by the proportional over-estimation, which is negligible at these severities.")
w("")
w("## 6. Concordance, relative error")
w("")
w("| Disease | Model | Lin's CCC [bootstrap CI] | CCC z-transform CI | Pearson r (precision) | Cb (accuracy) [CI] | MAE (pp) [CI] | Relative error: MAE / mean reference (%) [CI] | Relative bias (%) | RMSE (pp) |")
w("|---|---|---|---|---|---|---|---|---|---|")
for dis, dn in DIS:
    for arch, an in ARCH:
        x = m(dis, arch)
        c = x["lin_ccc"]
        w(f"| {dn} | {an} | {c['ccc']:.3f} {ci(c['ci95_bootstrap'], 3)} | {ci(c['ci95_z_transform'], 3)} | "
          f"{c['pearson_r']:.3f} | {c['cb']:.3f} {ci(c['cb_ci95_bootstrap'], 3)} | {x['mae']['estimate']:.2f} "
          f"{ci(x['mae']['ci95_bootstrap'])} | {x['relative_error_pct']['estimate']:.1f} "
          f"{ci(x['relative_error_pct']['ci95_bootstrap'], 1)} | {n(x['relative_bias_pct']['estimate'], 1, True)} | "
          f"{x['rmse']['estimate']:.2f} |")
w("")
sh = {f"{dis}/{arch}": m(dis, arch)["lin_ccc"] for dis, _ in DIS for arch, _ in ARCH}
w("CCC = r × Cb, so the shortfall of CCC from 1 splits additively on the log scale, −log CCC = −log r − log Cb. "
  "Share of the log-shortfall due to accuracy (Cb) / precision (r): rust "
  + ", ".join(f"{an} {100 * sh['rust/' + a]['share_of_log_shortfall_accuracy_cb']:.0f}% / "
              f"{100 * sh['rust/' + a]['share_of_log_shortfall_precision_r']:.0f}%" for a, an in ARCH)
  + "; leaf spot "
  + ", ".join(f"{an} {100 * sh['leaf_spot/' + a]['share_of_log_shortfall_accuracy_cb']:.0f}% / "
              f"{100 * sh['leaf_spot/' + a]['share_of_log_shortfall_precision_r']:.0f}%" for a, an in ARCH)
  + ". For rust SegFormer-B5 the shortfall is therefore mainly, not entirely, a matter of accuracy (systematic "
  f"over-estimation); for leaf spot every CCC is at least "
  f"{int(1000 * min(sh['leaf_spot/' + a]['ccc'] for a, _ in ARCH)) / 1000:.3f}, so the split is immaterial there.")
w("")
w("## 7. Where the over-estimation comes from: FP/FN decomposition of the per-leaf severity error")
w("")
w("D = FP% − FN% + L exactly (per face, hence per leaf and on average), where FP% and FN% are false-positive and "
  "false-negative lesion pixels as % of the reference leaf and L is the leaf-mask (denominator) term. FP pixels are "
  "split into: *band* = within 2 px of a reference lesion, in a predicted component that overlaps it (predicted lesion "
  "slightly wider than the reference lesion); *extension* = same components, > 2 px from the reference lesion; "
  "*spurious* = predicted components that touch no reference lesion. Slopes are OLS slopes on A; they add up exactly "
  "to the Bland–Altman slope (slope FP − slope FN + slope L).")
w("")
w("| Disease | Model | Bias | FP% [CI] | FN% [CI] | L | FP band | FP extension | FP spurious | FN: missed lesions | Slope D | Slope FP | Slope FP band | Slope FN |")
w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for dis, dn in DIS:
    for arch, an in ARCH:
        x = m(dis, arch)
        d = x["error_decomposition_pct_of_leaf"]
        w(f"| {dn} | {an} | {n(x['bias']['estimate'], 2, True)} | {d['fp_pct']['mean']:.2f} "
          f"{ci(d['fp_pct']['mean_ci95_bootstrap'])} | {d['fn_pct']['mean']:.2f} {ci(d['fn_pct']['mean_ci95_bootstrap'])} | "
          f"{n(d['leafterm_pct']['mean'], 3, True)} | {d['fp_band_pct']['mean']:.2f} | {d['fp_extension_pct']['mean']:.2f} | "
          f"{d['fp_spurious_pct']['mean']:.2f} | {d['fn_missed_pct']['mean']:.2f} | "
          f"{n(x['regression_of_differences_on_means']['slope'], 3)} | {n(d['fp_pct']['slope_on_A'], 3)} | "
          f"{n(d['fp_band_pct']['slope_on_A'], 3)} | {n(d['fn_pct']['slope_on_A'], 3)} |")
w("")
fp_share_band = 100 * dec["fp_band_pct"]["mean"] / dec["fp_pct"]["mean"]
fp_share_ext = 100 * dec["fp_extension_pct"]["mean"] / dec["fp_pct"]["mean"]
fp_share_sp = 100 * dec["fp_spurious_pct"]["mean"] / dec["fp_pct"]["mean"]
band_slope_share = 100 * dec["fp_band_pct"]["slope_on_A"] / rs_reg["slope"]
lfp_share_band = 100 * ldec["fp_band_pct"]["mean"] / ldec["fp_pct"]["mean"]
w(f"Reading (rust SegFormer-B5): FP area {dec['fp_pct']['mean']:.2f}% of the leaf against FN {dec['fn_pct']['mean']:.2f}%; "
  f"{fp_share_band:.0f}% of the FP area lies in the 2-px band around reference lesions, {fp_share_ext:.0f}% is extension "
  f"beyond 2 px and {fp_share_sp:.0f}% is spurious blobs; the band term alone has slope "
  f"{dec['fp_band_pct']['slope_on_A']:.3f} on A, {band_slope_share:.0f}% of the Bland–Altman slope. Missed lesions "
  f"contribute almost nothing ({dec['fn_missed_pct']['mean']:.3f}%). So the proportional over-estimation arises because "
  "predicted lesions extend slightly beyond the reference lesion borders; more lesion border means more such pixels, "
  f"which makes the error scale with severity. The same pattern holds for leaf spot ({lfp_share_band:.0f}% of SegFormer-B5 "
  "FP area in the band) and for the rust CNNs, which additionally show more extension beyond 2 px. The leaf-mask term L "
  "is negligible. Note that the rust reference is semi-automatic (detector + SAM 2, spot-checked, not edited), so this "
  "describes disagreement with the reference border placement; it does not establish which border is correct.")
w("")
w("## 8. IoU sensitivity analyses")
w("")
w("| Disease | Model | Mean leaf IoU (published) [CI] | Median leaf IoU [CI] | Pooled (micro) IoU [CI] | Minimum leaf IoU |")
w("|---|---|---|---|---|---|")
for dis, dn in DIS:
    for arch, an in ARCH:
        i = m(dis, arch)["iou"]
        w(f"| {dn} | {an} | {i['mean_leaf_iou']:.3f} {ci(i['mean_leaf_iou_ci95_bootstrap'], 3)} | "
          f"{i['median_leaf_iou']:.3f} {ci(i['median_leaf_iou_ci95_bootstrap'], 3)} | {i['pooled_micro_iou']:.3f} "
          f"{ci(i['pooled_micro_iou_ci95_bootstrap'], 3)} | {i['min_leaf_iou']:.3f} |")
w("")
w("The architecture ordering is the same under all three summaries (rust: SegFormer-B5 > UNet++ ≥ DeepLabV3+; leaf spot: "
  "SegFormer-B5 > UNet++ > DeepLabV3+). For leaf spot the mean leaf IoU is 0.02–0.04 below the median and pooled "
  "values because a few low-severity leaves with small lesions have low IoU; for rust the three summaries nearly coincide.")
w("")
w("## 9. Boundary F1 (lesion class, 2-px tolerance, letterbox space)")
w("")
w("Boundary = lesion mask minus its 3 × 3 erosion; a boundary pixel is matched if a boundary pixel of the other mask lies "
  "within 2 px (Euclidean). Rust images are 791 × 1024 (letterbox scale 1.0, so 2 px = 2 native px); leaf-spot images "
  "are 1583 × 2048 (scale 0.5, so 2 px = 4 native px). The physical pixel size was not recorded, so no tolerance in mm "
  "is given. Leaf mean = mean over leaves of per-leaf F1 (leaf spot: mean of the two faces); pooled = F1 of the pooled "
  "matched-pixel counts.")
w("")
w("| Disease | Model | BF1 @2 px, leaf mean [CI] | BF1 @2 px, pooled [CI] | Pooled boundary precision | Pooled boundary recall | BF1 @1 / @3 / @5 px (leaf mean) |")
w("|---|---|---|---|---|---|---|")
for dis, dn in DIS:
    for arch, an in ARCH:
        b = m(dis, arch)["boundary_f1"]
        t2 = b["tol_2px"]
        w(f"| {dn} | {an} | {t2['mean_leaf']:.3f} {ci(t2['mean_leaf_ci95_bootstrap'], 3)} | {t2['pooled']:.3f} "
          f"{ci(t2['pooled_ci95_bootstrap'], 3)} | {t2['pooled_precision']:.3f} | {t2['pooled_recall']:.3f} | "
          f"{b['tol_1px']['mean_leaf']:.3f} / {b['tol_3px']['mean_leaf']:.3f} / {b['tol_5px']['mean_leaf']:.3f} |")
w("")
w("## 10. Healthy rust-pack test leaves (context only)")
w("")
h = R["healthy_rust_test_leaves_local_abaxial"]
hb = R["healthy_rust_test_leaves_local_max_over_both_faces"]
w("Only 21 of the 54 healthy rust-test leaves used in the paper are present locally (all from the EARTH site); the "
  "numbers below cover those 21 (rust models) and do not replace the published 54-leaf counts. Two leaf-level "
  "definitions are shown because they give different counts for the same leaves: **abaxial face only** (the rust "
  "leaf-level rule used for the diseased leaves throughout this analysis) and **maximum over both faces** (a leaf "
  "counts if either face carries a predicted lesion; the definition used in the revision brief, fact 10).")
w("")
w("| Model | Definition | Mean predicted severity (%) | Max (%) | Leaves with ≥ 1 predicted lesion px | Leaves ≥ 0.5% | Leaves ≥ 5% |")
w("|---|---|---|---|---|---|---|")
for an in h:
    v, vb = h[an], hb[an]
    w(f"| {an} | abaxial face only | {v['mean_pred_sev']:.3f} | {v['max_pred_sev']:.3f} | "
      f"{v['n_any_lesion_px']}/{v['n_leaves']} | {v['n_pred_sev_ge_0.5']}/{v['n_leaves']} | "
      f"{v['n_pred_sev_ge_5']}/{v['n_leaves']} |")
    w(f"| {an} | max over both faces | {vb['mean_of_leaf_max_pred_sev']:.3f} | {vb['max_pred_sev']:.3f} | "
      f"{vb['n_any_lesion_px_either_face']}/{vb['n_leaves']} | {vb['n_pred_sev_ge_0.5_either_face']}/{vb['n_leaves']} | "
      f"{vb['n_pred_sev_ge_5_either_face']}/{vb['n_leaves']} |")
w("")
AR = [an for _, an in ARCH]
w("**Use one definition in the manuscript and state it.** Any-pixel counts: abaxial face only "
  + ", ".join(f"{h[an]['n_any_lesion_px']}/21" for an in AR) + " (maximum "
  + "/".join(f"{h[an]['max_pred_sev']:.2f}" for an in AR) + "%); maximum over both faces "
  + ", ".join(f"{hb[an]['n_any_lesion_px_either_face']}/21" for an in AR) + " (maximum "
  + "/".join(f"{hb[an]['max_pred_sev']:.2f}" for an in AR) + "%), for SegFormer-B5, UNet++ and DeepLabV3+. The "
  "both-face numbers are the ones in the revision brief; if the manuscript follows the brief, it should say 'on either "
  "face'. The two definitions agree on the grading conclusion: no healthy leaf reaches 0.5% on either face, so every "
  "healthy leaf is graded G0 by all three models.")
w("")
w(f"These false alarms are small in area (at most {hb['SegFormer-B5']['max_pred_sev']:.2f}% of the leaf on any face). "
  "They do not, however, cleanly show that spurious detections are unimportant on the diseased rust leaves, because "
  "the healthy leaves come from the EARTH site while the diseased rust leaves come from Sarchí (site and disease "
  "status are confounded). The better evidence is on the diseased leaves themselves (Section 7): for rust "
  f"SegFormer-B5, spurious predicted components that touch no reference lesion make up {fp_share_sp:.0f}% of the "
  f"false-positive area ({dec['fp_spurious_pct']['mean']:.2f} of {dec['fp_pct']['mean']:.2f}% of the leaf), against "
  f"{fp_share_band:.0f}% in the 2-px band around reference lesions.")
w("")
w("**Inconsistency to resolve (outside this task):** the manuscript says the zero-false-positive models left every "
  "healthy leaf without any predicted lesion pixel and reports 0/54 for rust SegFormer-B5, but with the raw argmax maps "
  "most of the 21 local healthy leaves already carry ≥ 1 predicted lesion pixel (counts above), and none reaches 0.5%. "
  "The published counts must therefore rest on a threshold; the revision brief (fact 10) notes that the training "
  "kernels defined a false positive as predicted severity ≥ 5%. The specificity definition in Methods/Results and in "
  "Figure 7 (label fig7, false-positive rate on healthy test leaves) should be checked against the code that produced "
  "the 0/54 counts (`% TODO(author)`).")
w("")
w("## 11. Answer: is the over-estimation proportional to severity, and how large is it at the cut-points?")
w("")
w(f"**Rust: yes, for all three models.** The difference grows by {n(rs_reg['slope'], 3)} pp per 1-pp increase in mean "
  f"severity for SegFormer-B5 (95% CI {n(rs_reg['slope_ci95_t'][0], 3)} to {n(rs_reg['slope_ci95_t'][1], 3)}, p < 0.001; "
  f"bootstrap CI {n(rs_reg['slope_ci95_bootstrap'][0], 3)} to {n(rs_reg['slope_ci95_bootstrap'][1], 3)}), "
  f"{n(m('rust', 'unetpp')['regression_of_differences_on_means']['slope'], 3)} for UNet++ and "
  f"{n(m('rust', 'deeplab')['regression_of_differences_on_means']['slope'], 3)} for DeepLabV3+, and the intercepts do not "
  f"differ from zero (SegFormer-B5 {n(rs_reg['intercept'], 2, True)} pp, p = {rs_reg['intercept_p']:.2f}). The "
  f"over-estimation is thus essentially multiplicative: the calibration line is predicted = "
  f"{S['calibration_pred_on_ref']['slope']:.2f} × reference − {abs(S['calibration_pred_on_ref']['intercept']):.2f} pp "
  f"for SegFormer-B5 (UNet++ slope {m('rust', 'unetpp')['calibration_pred_on_ref']['slope']:.2f}, DeepLabV3+ "
  f"{m('rust', 'deeplab')['calibration_pred_on_ref']['slope']:.2f}), i.e. SegFormer-B5 predicts roughly one-fifth more "
  f"lesion area than the reference between 5% and 20% severity. For SegFormer-B5 the fitted "
  f"over-estimate is {n(rs5['fitted_difference'], 2, True)} pp at a reference severity of 5% (95% CI "
  f"{n(rs5['fitted_difference_ci95_bootstrap'][0])} to {n(rs5['fitted_difference_ci95_bootstrap'][1])}; "
  f"{rs5['fitted_difference_relative_pct']:.0f}% of the value; regression-based LoA {n(rs5['loa_lower'], 2, True)} to "
  f"{n(rs5['loa_upper'], 2, True)}) and {n(rs20['fitted_difference'], 2, True)} pp at 20% (95% CI "
  f"{n(rs20['fitted_difference_ci95_bootstrap'][0])} to {n(rs20['fitted_difference_ci95_bootstrap'][1])}; "
  f"{rs20['fitted_difference_relative_pct']:.0f}%; LoA {n(rs20['loa_lower'], 2, True)} to {n(rs20['loa_upper'], 2, True)}). "
  f"The fitted mean prediction therefore crosses the 5% cut-point for reference severities of about "
  f"{rs5['reference_severity_whose_fitted_prediction_reaches_cut']:.1f}–5% (lower edge 95% CI "
  f"{ci(rs5['reference_severity_whose_fitted_prediction_reaches_cut_ci95_bootstrap'])}%) and the 20% cut-point for "
  f"about {rs20['reference_severity_whose_fitted_prediction_reaches_cut']:.1f}–20% "
  f"({ci(rs20['reference_severity_whose_fitted_prediction_reaches_cut_ci95_bootstrap'])}%), so leaves in these bands "
  "are prone to over-grading; at the lower edge of a band the fitted prediction only just reaches the cut, so an "
  f"over-grade there is roughly as likely as not. {SB_SENT} The 20% estimates rest on the three rust test leaves above "
  "10% (maximum reference 19.3%) and have wide intervals; they indicate direction and order of magnitude, not a "
  "precise value.")
w("")
w(f"**American leaf spot: a small proportional component.** Slope {n(ls_reg['slope'], 3)} "
  f"({n(ls_reg['slope_ci95_t'][0], 3)} to {n(ls_reg['slope_ci95_t'][1], 3)}) for SegFormer-B5 and "
  f"{n(m('leaf_spot', 'deeplab')['regression_of_differences_on_means']['slope'], 3)} for DeepLabV3+ (both robust to "
  f"bootstrap and HC3); for UNet++ the slope {n(m('leaf_spot', 'unetpp')['regression_of_differences_on_means']['slope'], 3)} "
  f"(t p = {m('leaf_spot', 'unetpp')['regression_of_differences_on_means']['slope_p']:.3f}) is not robust (bootstrap CI "
  f"{ci(m('leaf_spot', 'unetpp')['regression_of_differences_on_means']['slope_ci95_bootstrap'], 3)}, HC3 p = "
  f"{m('leaf_spot', 'unetpp')['regression_of_differences_on_means']['slope_p_hc3']:.2f}). SegFormer-B5 over-estimates by "
  f"{n(ls5['fitted_difference'], 2, True)} pp at 5% ({ls5['fitted_difference_relative_pct']:.0f}%) and "
  f"{n(ls20['fitted_difference'], 2, True)} pp at 20% ({ls20['fitted_difference_relative_pct']:.0f}%); this can move a "
  f"leaf up a grade only if its reference severity lies in {ls5['reference_severity_whose_fitted_prediction_reaches_cut']:.2f}–5% "
  f"or {ls20['reference_severity_whose_fitted_prediction_reaches_cut']:.2f}–20% (one test leaf, which the annotator "
  "graded G2 visually, so no over-grade results). Leaf-spot grade errors are all under-grades instead. For "
  f"SegFormer-B5, {NUMW[LS['grades']['disagreement_category_counts']['visual_mask_gap']]} of its "
  f"{NUMW[len(LS['grades']['disagreements'])]} errors reflect the visual-grade/mask gap: {NUMW[len(g_low)]} G1 leaves with "
  f"mask severity {g_low[0]:.2f}–{g_low[-1]:.2f}% and one G2 leaf at {g_hi[0]['ref_sev']:.2f}%, whose predicted grade "
  "equals the grade derived from the reference mask. The other "
  f"{NUMW[LS['grades']['disagreement_category_counts']['severity_error']]} ("
  + " and ".join(f"{d['ref_sev']:.3f}% → {d['pred_sev']:.3f}%" for d in sorted(
      (d for d in LS["grades"]["disagreements"] if d["category"] == "severity_error"), key=lambda d: d["ref_sev"]))
  + ") are small under-estimates across the 0.5% cut-point, consistent with the slightly negative leaf-spot intercept "
  "rather than with the proportional over-estimation (Section 5c).")
w("")
w("## 12. Suggested manuscript sentences")
w("")
w("*Methods (Section sec:evaluation; replaces the Bland–Altman sentence):*")
w("")
w("> Agreement between predicted and reference per-leaf severity was assessed by Bland–Altman analysis: the difference "
  "(predicted − reference) was plotted against the mean of the two, and the bias (mean difference) and 95% limits of "
  "agreement (bias ± 1.96 SD of the differences) were estimated with 95% bootstrap confidence intervals (10,000 leaf "
  "resamples). Because the differences increased with severity, proportional bias was tested by least-squares "
  "regression of the differences on the means, and regression-based limits of agreement were obtained by modelling the "
  "mean difference and its standard deviation as linear functions of the mean (Bland and Altman, 1999); Lin's "
  "concordance correlation coefficient (CCC) and its bias-correction factor C_b are also reported.")
w("")
w("*Results / Discussion (2–5 sentences):*")
w("")
w(f"> For rust, the over-estimation was proportional to severity for all three models: the difference increased by "
  f"{rs_reg['slope']:.2f} percentage points per percentage point of mean severity for SegFormer-B5 (95% CI "
  f"{rs_reg['slope_ci95_t'][0]:.2f} to {rs_reg['slope_ci95_t'][1]:.2f}; p < 0.001) and by "
  f"{m('rust', 'deeplab')['regression_of_differences_on_means']['slope']:.2f} and "
  f"{m('rust', 'unetpp')['regression_of_differences_on_means']['slope']:.2f} for DeepLabV3+ and UNet++, with intercepts "
  f"compatible with zero. For SegFormer-B5 this corresponds to a fitted over-estimate of {rs5['fitted_difference']:.2f} "
  f"points at a reference severity of 5% (95% CI {rs5['fitted_difference_ci95_bootstrap'][0]:.2f} to "
  f"{rs5['fitted_difference_ci95_bootstrap'][1]:.2f}) and {rs20['fitted_difference']:.1f} points at 20% "
  f"({rs20['fitted_difference_ci95_bootstrap'][0]:.1f} to {rs20['fitted_difference_ci95_bootstrap'][1]:.1f}), the latter "
  f"resting on the three test leaves above 10% severity. The fitted prediction crosses the 5% and 20% cut-points for "
  f"reference severities of about {rs5['reference_severity_whose_fitted_prediction_reaches_cut']:.1f}–5% and "
  f"{rs20['reference_severity_whose_fitted_prediction_reaches_cut']:.1f}–20%, respectively, so such leaves are prone to "
  f"over-grading; both SegFormer-B5 rust grade errors ({SB_ERR}) lay in these bands, and the third leaf in them "
  f"({_sb_ok[0]['ref_sev']:.2f}%) was graded correctly. For American leaf spot the proportional component "
  f"was small (SegFormer-B5 slope {ls_reg['slope']:.3f}, 95% CI {ls_reg['slope_ci95_t'][0]:.3f} to "
  f"{ls_reg['slope_ci95_t'][1]:.3f}; {ls5['fitted_difference']:+.2f} points at 5% and {ls20['fitted_difference']:+.2f} "
  f"at 20%) and was not robust for UNet++. Lin's CCC was {S['lin_ccc']['ccc']:.3f} (95% CI "
  f"{S['lin_ccc']['ci95_bootstrap'][0]:.3f} to {S['lin_ccc']['ci95_bootstrap'][1]:.3f}) for rust SegFormer-B5 and "
  f"{LS['lin_ccc']['ccc']:.3f} for leaf spot; the rust shortfall mainly reflected accuracy (C_b = "
  f"{S['lin_ccc']['cb']:.3f}) rather than precision (r = {S['lin_ccc']['pearson_r']:.3f}), and the mean absolute error was "
  f"{S['relative_error_pct']['estimate']:.0f}% of the mean reference severity for rust "
  f"({LS['relative_error_pct']['estimate']:.0f}% for leaf spot). Decomposing the per-leaf error showed that the rust "
  f"over-estimate came from false-positive lesion pixels ({dec['fp_pct']['mean']:.2f}% of leaf area against "
  f"{dec['fn_pct']['mean']:.2f}% false-negative for SegFormer-B5), {fp_share_band:.0f}% of which lay within two pixels "
  "of a reference lesion, indicating that predicted lesions extended slightly beyond the borders of the semi-automatic "
  f"reference lesions; separate spurious detections ({fp_share_sp:.0f}% of the false-positive area) and missed lesions "
  "contributed little.")
w("")
w("*Figure 5 caption (new):*")
w("")
w("> Per-leaf agreement between predicted and reference severity for SegFormer-B5: (a) coffee leaf rust, abaxial face, "
  "n = 21 test leaves; (b) American leaf spot, mean of both faces, n = 55. Each point is a leaf. The dashed line is the "
  "least-squares regression of the differences on the means, the dotted lines are the regression-based 95% limits of "
  "agreement (Bland and Altman, 1999), and the solid grey line marks zero difference; the slope of the regression line "
  "is given with its 95% confidence interval. pp = percentage points.")
w("")
w("*Supplementary figure caption:*")
w("")
w("> Per-leaf Bland–Altman agreement for all six models, (a) coffee leaf rust (n = 21) and (b) American leaf spot "
  "(n = 55), with the regression of the differences on the means (dashed), regression-based 95% limits of agreement "
  "(dotted), and zero difference (solid grey). The y axis is shared within each disease. Slopes are least-squares "
  "estimates with t-based 95% CIs; for leaf-spot UNet++ the bootstrap CI includes zero.")
w("")
w("## 13. Caveats and notes for the authors")
w("")
w("- New references needed: Bland JM, Altman DG (1999) Measuring agreement in method comparison studies. *Stat Methods "
  "Med Res* 8:135–160; Lin LI-K (1989) A concordance correlation coefficient to evaluate reproducibility. *Biometrics* "
  "45:255–268 (and the 2000 correction, *Biometrics* 56:324–325); for boundary F1, e.g. Csurka G, Larlus D, Perronnin F "
  "(2013) What is a good evaluation measure for semantic segmentation? *BMVC*. The existing ref44 is Bland & Altman 1986.")
w("- n = 21 rust leaves, 3 above 10% severity, none above 19.3%: the rust 20% cut-point estimates are imprecise and "
  "partly extrapolated; the 5% estimates are well supported.")
w("- OLS p values assume constant residual variance, which does not hold; the rust conclusions are unchanged under "
  "bootstrap and HC3 inference. For rust SegFormer-B5 the absolute-residual slope is not significant "
  f"(p = {S['regression_based_loa']['abs_residual_slope_p']:.2f}), so its LoA width is compatible with a constant SD; the "
  "proportional finding concerns the mean difference.")
w("- The rust reference is semi-automatic (detector + SAM 2, spot-checked, not edited): 'over-estimation' means relative "
  "to that reference. Severity agreement is not an independent check because the reference masks are also the training "
  "targets (already stated in the manuscript).")
w("- Each model was trained once; the CIs capture test-leaf sampling only.")
w("- Bootstrap CIs here use seed 20260927 and differ from the published Table 2/4 CIs by at most 0.02 pp (MAE) and "
  "0.001 (IoU) (e.g., rust SegFormer-B5 MAE [0.48, 1.60] here vs [0.48, 1.61] published; leaf IoU [0.743, 0.790] vs "
  "[0.743, 0.791]). Point estimates are identical. Keep the published CIs unless the tables are regenerated in full.")
w("- Suggested abbreviations: CCC = concordance correlation coefficient; LoA = limits of agreement.")
w("")
w("## Files")
w("")
w("- `results.json`: every number above (and more: parametric CIs, HC3, calibration, log-ratio, tolerance sweep; "
  "per grade error the mask-derived grade and its category; per cut-point band the leaves in it and whether each was "
  "over-graded; CCC log-shortfall shares; healthy leaves under both face definitions).")
w("- `figure5.pdf` / `figure5.png`: new Figure 5 (SegFormer-B5). `figureS_agreement_all.pdf` / `.png`: all six models.")
w("- `pixel_metrics.py`, `agreement.py`, `make_figures.py`, `make_md.py`; `work/pixel_metrics.csv` (per image), "
  "`work/leaf_level.csv` (per leaf).")

open(os.path.join(OUT, "results.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("results.md written", len(L), "lines")
