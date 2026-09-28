"""Write results.md from results.json and verify_per_image.json (so every number in the text comes from the JSON)."""
import json
import os

OUT = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(f"{OUT}/results.json"))
V = json.load(open(f"{OUT}/verify_per_image.json"))
M = ["SegFormer-B5", "UNet++", "DeepLabV3+"]
DN = {"rust": "Rust", "leaf_spot": "Leaf spot"}
FACES = {"abaxial": "abaxial", "abaxial+adaxial": "both"}


def pc(p):   # "20/21 (95.2%; 77.3–99.2)"
    lo, hi = p["wilson95_pct_1dp"]
    return f"{p['k']}/{p['n']} ({p['pct_1dp']:.1f}%; {lo:.1f}–{hi:.1f})"


def f(x, d=2):
    return f"{x:.{d}f}"


def ci(c, d=2):
    return f"{c[0]:.{d}f}–{c[1]:.{d}f}"


def sg(x, d=2, plus=False):   # signed number with a true minus sign
    t = f"{abs(x):.{d}f}"
    return ("−" + t) if x < 0 and float(t) != 0 else (("+" if plus else "") + t)


def sci(c, d=3):              # signed interval "a to b"
    return f"{sg(c[0], d)} to {sg(c[1], d)}"


L = []
w = L.append
w("# Detection, specificity and cross-disease behavior (major revision, PG-26038)")
w("")
w("Scripts: `verify_per_image.py` (pixel-count check), `detection_analysis.py` (all numbers -> `results.json`, "
  "`leaf_level.csv`), `make_figure_detection.py` (figure), `make_md.py` (this file). No model inference was re-run.")
w("")
w("Leaf-level rules as in the paper: rust = abaxial face only; American leaf spot = mean of the two faces. A model's "
  "leaf-level output uses the rule of the disease it was trained for (rust models: abaxial face; leaf-spot models: mean "
  "of both faces); other rules are sensitivity analyses. Grades: G0 < 0.5% ≤ G1 < 5% ≤ G2 < 20% ≤ G3, so "
  "\"≥0.5%\" means \"predicted grade G1 or higher\". Proportions: 95% Wilson intervals. Bootstrap: 10,000 leaf "
  "resamples, percentile 95% CI, seed 20260927 (paired where two models are compared on the same leaves). "
  "Lesion-location measures in Section 3 (share of predicted lesion pixels inside the reference lesion mask, lesion "
  "IoU, off-lesion severity) are computed on the faces the model reads, i.e. the same faces behind its severity, grade "
  "and ≥0.5% values (rust models: abaxial; leaf-spot models: both faces); for cross-disease runs they are also given "
  "on the faces of the leaf's own-disease rule (rust leaves: abaxial; leaf-spot leaves: both faces). The two face "
  "sets are identical for correct-disease runs. All numbers are rounded once, from full-precision values in "
  "results.json.")
w("")
w("## 0. Reproduction and input check")
w("")
w(f"- **Published values reproduced: {'YES' if R['reproduction']['all_match'] else 'NO'}.** All six models match "
  "the published leaf IoU, MAE, bias, grade QWK and Bland–Altman limits at the published precision:")
w("")
w("| Model | n | IoU | MAE | Bias | QWK | LoA |")
w("|---|---|---|---|---|---|---|")
for k, v in R["reproduction"]["models"].items():
    c = v["computed"]
    w(f"| {k.replace('leaf_spot', 'leaf spot')} | {v['n_leaves']} | {c['iou']:.3f} | {c['mae']:.2f} | {sg(c['bias'], plus=True)} | "
      f"{c['qwk']:.2f} | {sg(c['loa'][0])} to {sg(c['loa'][1], plus=True)} |")
w("")
w(f"- `per_image.csv` was re-derived from the stored class maps and rebuilt reference labels (the `load()` of "
  f"`run_inference.py`) for every test row used here: {V['rows_matching_all_pixel_counts']}/{V['rows_checked']} rows "
  f"match on all six pixel counts ({V['n_mismatches']} mismatches).")
w("")

w("## 1. Sensitivity on the diseased test leaves")
w("")
w("| Model | n | >0% | ≥0.5% | ≥0.5%, reference mask ≥0.5% only | AUC vs 21 healthy |")
w("|---|---|---|---|---|---|")
for k, v in R["sensitivity_test_diseased"].items():
    d, m = k.split("/")
    rr = v["ref_mask_sev_ge05_only"]
    w(f"| {DN[d]} {m} | {v['n_diseased_leaves']} | {pc(v['gt0'])} | {pc(v['ge05'])} | {pc(rr['ge05'])} | "
      f"{v['auc_vs_21_healthy']:.3f} ({ci(v['auc_vs_21_healthy_ci'], 3)}) |")
S = R["sensitivity_test_diseased"]
miss_ls = [x["ref_sev"] for k, v in S.items() if k.startswith("leaf_spot") for x in v["leaves_below_0p5"]]
miss_ru = [x["ref_sev"] for k, v in S.items() if k.startswith("rust") for x in v["leaves_below_0p5"]]
w("")
w(f"- Rust: the one diseased leaf below 0.5% for all three models is {S['rust/SegFormer-B5']['leaves_below_0p5'][0]['leaf_id']}, "
  f"whose reference severity is itself {S['rust/SegFormer-B5']['leaves_below_0p5'][0]['ref_sev']:.2f}% (reference grade G0). "
  "Every rust leaf with reference severity ≥0.5% was called ≥0.5% by every model.")
w(f"- Leaf spot: {S['leaf_spot/SegFormer-B5']['n_ref_mask_sev_lt05']} of the 55 test leaves carry the visual grade G1 "
  "but a reference mask covering <0.5% of the leaf. Every leaf missed at ≥0.5% by any model is a very mild leaf "
  f"(reference mask severity {min(miss_ls):.2f}–{max(miss_ls):.2f}%), and the predicted severity on these leaves is close "
  "to the reference (see `leaves_below_0p5` in results.json).")
w("- \">0%\" (any predicted lesion pixel) gives 100% sensitivity for all six models, but it is not a usable detection "
  "rule because it also flags most healthy leaves (Section 2). AUC caveat: for rust the diseased leaves are from Sarchí "
  "and the healthy leaves from EARTH, so rust separation is confounded with site.")
w("")

w("## 2. Specificity on the 21 locally available healthy test leaves (EARTH controls)")
w("")
w("Only 21 of the 54 healthy rust-pack test leaves reported in the paper are present locally (collected at EARTH on "
  "12, 15 and 22 June 2026); the other 33 are not in the local pack, so **the published 54-leaf results cannot be "
  "reproduced locally beyond these 21 leaves.**")
w("")
w("| Model (leaf rule) | FP >0% | Specificity >0% | FP ≥0.5% | Specificity ≥0.5% | Median / max predicted severity | Published (54, any pixel) |")
w("|---|---|---|---|---|---|---|")
P = R["specificity_published_54"]
for k, v in R["specificity_local_healthy_test"].items():
    d, m = k.split("/")
    e = v["primary_model_rule"]
    w(f"| {DN[d]} {m} ({'abaxial' if v['rule'] == 'abaxial' else 'mean of faces'}) | {e['fp_gt0']}/21 | "
      f"{pc(e['specificity_gt0'])} | {e['fp_ge05']}/21 | {pc(e['specificity_ge05'])} | "
      f"{v['pred_sev_median_primary']:.3f}% / {e['max_pred_sev']:.2f}% | {P[d][m]} |")
w("")
w("Sensitivity analyses (flagged leaves out of 21):")
w("")
w("| Model | abaxial >0 / ≥0.5 | adaxial >0 / ≥0.5 | either face >0 / ≥0.5 |")
w("|---|---|---|---|")
for k, v in R["specificity_local_healthy_test"].items():
    d, m = k.split("/")
    w(f"| {DN[d]} {m} | {v['abaxial']['fp_gt0']} / {v['abaxial']['fp_ge05']} | {v['adaxial']['fp_gt0']} / "
      f"{v['adaxial']['fp_ge05']} | {v['either_face_max']['fp_gt0']} / {v['either_face_max']['fp_ge05']} |")
w("")
w("- **Definition problem (must be fixed in the manuscript).** " + R["definition_check"])
w("- The predicted \"lesion\" on healthy leaves is a speck: median over leaves of the predicted lesion pixels on the "
  "rule's face(s) = " + ", ".join(f"{k.replace('leaf_spot', 'leaf spot')} {v['pred_lesion_px_on_rule_faces']['median']:.0f} px "
                                  f"(max {v['pred_lesion_px_on_rule_faces']['max']})"
                                  for k, v in R["specificity_local_healthy_test"].items())
  + " in 1024 × 1024 letterbox space, where a healthy abaxial face covers about 55,000–120,000 leaf pixels.")
_sp05 = R["specificity_local_healthy_test"]["rust/SegFormer-B5"]["primary_model_rule"]["specificity_ge05"]
FP_UB = 100 * (1 - _sp05["wilson95"][0])                     # 0/21 false positives -> Wilson upper bound of the FP rate
_sp0 = [v["primary_model_rule"]["specificity_gt0"]["p"] for v in R["specificity_local_healthy_test"].values()]
SP0_RANGE = (100 * min(_sp0), 100 * max(_sp0))
w(f"- At the ≥0.5% threshold, 0/21 false positives has a 95% Wilson upper bound of {FP_UB:.1f}% for the false-positive "
  f"rate (specificity ≥{100 - FP_UB:.1f}%), much wider than the 6.6% bound quoted for 0/54. At >0%, specificity on the "
  f"21 leaves is {SP0_RANGE[0]:.0f}–{SP0_RANGE[1]:.0f}% depending on the model.")
sup = R["supplementary_rust_models_on_healthy_validation_leaves_NOT_independent"]
w("- Supplementary, not independent (rust CNNs were selected with a healthy-validation penalty): rust models on the 21 "
  "healthy validation leaves flag " + ", ".join(f"{k.split('/')[1]} {v['fp_gt0']}/21 at >0% and {v['fp_ge05']}/21 at ≥0.5%"
                                               for k, v in sup.items()) + ".")
w("")

w("## 3. Cross-disease behavior")
w("")
w("A cross-disease reading on a diseased leaf is **not** a false positive in the healthy-leaf sense: the leaf carries "
  "real lesions of the other disease. It tests whether a disease-specific model is disease-specific, i.e. whether it "
  "stays silent on the other disease's lesions. A reading can combine pixels on the other disease's lesions with pixels "
  "on tissue outside them, so where the predicted pixels fall is reported separately (second table in each block).")
w("")
for blk, title in (("on_leaf_spot_leaves", "Rust models on the 55 American leaf spot test leaves (vs the leaf-spot models)"),
                   ("on_rust_leaves", "Leaf-spot models on the 21 rust test leaves (vs the rust models)")):
    b = R["cross_disease"][blk]
    w(f"### {title}")
    w("")
    w(f"Reference severity on these leaves: median {b['reference']['median_ref_sev']:.2f}% (IQR "
      f"{b['reference']['iqr'][0]:.2f}–{b['reference']['iqr'][1]:.2f}); reference grades G0/G1/G2/G3 = "
      f"{'/'.join(map(str, b['reference']['grade_counts_G0_G3']))}.")
    w("")
    w("| Model | Role | Median severity (95% CI) | IQR | ≥0.5% | Grades G0/G1/G2/G3 | Same model on 21 healthy: ≥0.5% (max) |")
    w("|---|---|---|---|---|---|---|")
    for role, lab in (("correct_models", "correct disease"), ("cross_models", "cross-disease")):
        for m in M:
            e = b[role][m]
            h = e["same_model_on_21_healthy_EARTH_leaves"]
            w(f"| {m} | {lab} | {e['pred_sev_median']:.2f}% ({ci(e['pred_sev_median_ci'])}) | "
              f"{e['pred_sev_iqr'][0]:.2f}–{e['pred_sev_iqr'][1]:.2f} | {pc(e['ge05'])} | "
              f"{'/'.join(map(str, e['pred_grade_counts_G0_G3']))} | {h['ge05']} ({h['max']:.2f}%) |")
    w("")
    w(f"Where the predicted lesion pixels fall, relative to the {'rust' if blk == 'on_rust_leaves' else 'leaf-spot'} "
      "reference lesion mask of these leaves. \"Model's faces\" are the faces behind the severity, grade and ≥0.5% "
      "values above; \"leaf's faces\" are the faces of the leaf's own-disease rule (identical for correct-disease models).")
    w("")
    w("| Model | Role | Faces | Pred. lesion px inside reference lesion, pooled (95% CI) | Lesion IoU | Off-lesion severity, median (IQR) | Leaves with off-lesion severity ≥0.5% | ≥0.5% calls that need off-lesion px | Mean severity / mean reference: all px, inside-lesion px only |")
    w("|---|---|---|---|---|---|---|---|---|")
    for role, lab in (("correct_models", "correct disease"), ("cross_models", "cross-disease")):
        for m in M:
            e = b[role][m]
            sets = [("model_rule_faces", "model's = leaf's")] if role == "correct_models" else \
                   [("model_rule_faces", "model's"), ("leaf_rule_faces", "leaf's")]
            for key, flab in sets:
                q = e["lesion_location"][key]
                w(f"| {m} | {lab} | {FACES[q['faces']]} ({flab}) | "
                  f"{100 * q['share_pred_lesion_px_inside_reference_pooled']:.1f}% ({ci([100 * x for x in q['share_inside_ci']], 1)}) | "
                  f"{q['lesion_iou_mean']:.2f} | {q['off_lesion_pred_sev_median']:.2f}% "
                  f"({q['off_lesion_pred_sev_iqr'][0]:.2f}–{q['off_lesion_pred_sev_iqr'][1]:.2f}) | "
                  f"{q['n_leaves_off_lesion_sev_ge05']}/{b['n']} | {q['n_leaves_ge05_only_with_off_lesion_px']}/{q['n_leaves_pred_sev_ge05']} | "
                  f"{q['ratio_mean_pred_sev_over_reference']:.2f}, {q['ratio_mean_in_lesion_sev_over_reference']:.2f} |")
    w("")
    w("Off-lesion severity = predicted lesion pixels outside the reference lesion mask, in % of the predicted leaf area "
      "on the same faces (combined over faces like the severity); it includes lesion margins as well as distant tissue, "
      "which were not separated. \"≥0.5% calls that need off-lesion px\" = leaves at ≥0.5% whose inside-lesion severity "
      "alone is <0.5% (out of the leaves at ≥0.5% on those faces). Ratios are to the mean leaf-level reference severity.")
    w("")
    w("Paired, same architecture, same leaves (cross-disease minus correct-disease model):")
    w("")
    w("| Architecture | Δ fraction ≥0.5% (95% CI) | Mean severity ratio cross/correct (95% CI) | Mean severity ratio cross/reference (95% CI) | Same predicted grade |")
    w("|---|---|---|---|---|")
    for m in M:
        p = b["paired_cross_vs_correct_same_arch"][m]
        w(f"| {m} | {sg(p['diff_frac_ge05_cross_minus_correct'], 3, plus=True)} ({sci(p['diff_frac_ge05_ci'])}) | "
          f"{p['ratio_mean_pred_sev_cross_over_correct']:.2f} ({ci(p['ratio_mean_ci'])}) | "
          f"{p['ratio_mean_pred_sev_cross_over_reference']:.2f} ({ci(p['ratio_mean_ref_ci'])}) | "
          f"{p['n_leaves_cross_grade_equal_correct_grade']}/{b['n']} |")
    w("")
    alt = ", ".join(f"{m} {b['cross_models'][m]['alt_rule_leaf_disease']['ge05']['k']}/{b['n']}" for m in M)
    w(f"Cross-disease models under the leaf's own-disease rule ({b['cross_models'][M[0]]['alt_rule_leaf_disease']['rule']}): "
      f"≥0.5% in {alt}.")
    w("")
X = R["cross_disease"]
w("**Interpretation.**")
w("")
def loc(blk, role, m, key="model_rule_faces"):
    return X[blk][role][m]["lesion_location"][key]


def lst(vals, fmt):
    return ", ".join(format(v, fmt) for v in vals)


def lstp(vals, fmt):
    return ", ".join(format(v, fmt) + "%" for v in vals)


def rng(vals, fmt):
    return f"{format(min(vals), fmt)}–{format(max(vals), fmt)}"


LSL, RUL = "on_leaf_spot_leaves", "on_rust_leaves"
in_x_ls = [100 * loc(LSL, "cross_models", m)["share_pred_lesion_px_inside_reference_pooled"] for m in M]
in_c_ls = [100 * loc(LSL, "correct_models", m)["share_pred_lesion_px_inside_reference_pooled"] for m in M]
in_x_ls_leaf = [100 * loc(LSL, "cross_models", m, "leaf_rule_faces")["share_pred_lesion_px_inside_reference_pooled"] for m in M]
in_x_ru = [100 * loc(RUL, "cross_models", m)["share_pred_lesion_px_inside_reference_pooled"] for m in M]
in_x_ru_leaf = [100 * loc(RUL, "cross_models", m, "leaf_rule_faces")["share_pred_lesion_px_inside_reference_pooled"] for m in M]
in_c_ru = [100 * loc(RUL, "correct_models", m)["share_pred_lesion_px_inside_reference_pooled"] for m in M]
off_x_ls = [loc(LSL, "cross_models", m)["off_lesion_pred_sev_median"] for m in M]
off_c_ls = [loc(LSL, "correct_models", m)["off_lesion_pred_sev_median"] for m in M]
off_x_ru = [loc(RUL, "cross_models", m)["off_lesion_pred_sev_median"] for m in M]
off_c_ru = [loc(RUL, "correct_models", m)["off_lesion_pred_sev_median"] for m in M]
n_off = [loc(LSL, "cross_models", m)["n_leaves_off_lesion_sev_ge05"] for m in M]
n_dep = [loc(LSL, "cross_models", m)["n_leaves_ge05_only_with_off_lesion_px"] for m in M]
rat_all = [loc(LSL, "cross_models", m)["ratio_mean_pred_sev_over_reference"] for m in M]
rat_in = [loc(LSL, "cross_models", m)["ratio_mean_in_lesion_sev_over_reference"] for m in M]
tops = {m: loc(LSL, "cross_models", m)["top3_off_lesion_leaves"] for m in M}
rat_ru = [loc(RUL, "cross_models", m)["ratio_mean_pred_sev_over_reference"] for m in M]
dep_ru = [(loc(RUL, "cross_models", m)["n_leaves_ge05_only_with_off_lesion_px"], loc(RUL, "cross_models", m)["n_leaves_pred_sev_ge05"]) for m in M]
ext = tops["DeepLabV3+"][0]["leaf_id"]
ext_rows = []
for m in M:
    hit = [t for t in tops[m] if t["leaf_id"] == ext]
    assert hit, f"{ext} not in the top-3 off-lesion leaves of rust/{m}"
    ext_rows.append(hit[0])
ext_ab = [t["per_face"]["abaxial"] for t in ext_rows]
hmax_rust = max(X[LSL]["cross_models"][m]["same_model_on_21_healthy_EARTH_leaves"]["max"] for m in M)

w("- The models do not discriminate between the two diseases. The rust models called ≥0.5% on "
  + ", ".join(f"{X[LSL]['cross_models'][m]['ge05']['k']}" for m in M)
  + " of 55 leaf-spot leaves (SegFormer-B5, UNet++, DeepLabV3+), as many as or more than the leaf-spot models ("
  + ", ".join(f"{X[LSL]['correct_models'][m]['ge05']['k']}" for m in M)
  + "), and over-read their severity (mean ratio to the leaf-spot reference "
  + lst(rat_all, ".2f")
  + "). The leaf-spot models called ≥0.5% on "
  + ", ".join(f"{X[RUL]['cross_models'][m]['ge05']['k']}" for m in M)
  + " of 21 rust leaves, but read only "
  + ", ".join(f"{100 * X[RUL]['paired_cross_vs_correct_same_arch'][m]['ratio_mean_pred_sev_cross_over_correct']:.0f}%" for m in M)
  + " of the rust models' mean severity.")
w("- Rust models on leaf-spot leaves: most of their predicted lesion pixels lie on the leaf-spot lesions, but they also "
  "mark tissue outside the leaf-spot reference lesions. On the face the rust models read (abaxial; the face behind "
  f"their severity and ≥0.5% values), {lstp(in_x_ls, '.1f')} of their predicted lesion pixels fall inside the leaf-spot "
  f"reference lesions, against {lstp(in_c_ls, '.1f')} for the leaf-spot models on the same leaves (both faces), so "
  f"{rng([100 - v for v in in_x_ls], '.0f')}% versus {rng([100 - v for v in in_c_ls], '.0f')}% fall outside. The rust models' "
  f"median off-lesion severity is {lstp(off_x_ls, '.2f')} of the leaf (leaf-spot models {rng(off_c_ls, '.2f')}%). "
  f"Off-lesion pixels alone reach ≥0.5% of the leaf on {lst(n_off, 'd')} of 55 leaves, and on {lst(n_dep, 'd')} leaves "
  "the ≥0.5% call would not be made without them. They account for the rust models' excess severity: the mean predicted "
  f"severity is {lst(rat_all, '.2f')} times the leaf-spot reference with all predicted pixels, and {lst(rat_in, '.2f')} "
  f"times when only pixels inside the reference lesions are counted. The extreme leaf is {ext} (reference "
  f"{ext_rows[0]['ref_sev_leafrule']:.1f}%): the rust models' off-lesion area on its abaxial face is "
  + ", ".join(f"{t['off_lesion_pct_of_leaf']:.1f}%" for t in ext_ab)
  + " of the leaf, and rust DeepLabV3+ marks a large area of apparently lesion-free tissue "
  f"({ext_ab[2]['pred_lesion_px_outside_reference']:,} of {ext_ab[2]['pred_lesion_px']:,} predicted lesion pixels "
  "outside the reference). On the leaf's own-disease faces (both faces) the rust models' inside shares are "
  f"{lstp(in_x_ls_leaf, '.1f')}, with the same conclusion.")
w("- Leaf-spot models on rust leaves: on the faces they read (both faces), "
  f"{lstp(in_x_ru, '.1f')} of their predicted lesion pixels fall inside the rust reference lesions "
  f"({lstp(in_x_ru_leaf, '.1f')} on the abaxial face, the rust reference face); their median off-lesion severity is "
  f"{lstp(off_x_ru, '.2f')} of the leaf. The rust models themselves place {lstp(in_c_ru, '.1f')} of their abaxial "
  f"pixels inside the rust reference (median off-lesion severity {lstp(off_c_ru, '.2f')}), in line with their positive "
  f"bias. The leaf-spot models' mean severity on these leaves is {lst(rat_ru, '.2f')} times the rust reference, so their "
  "main error on rust leaves is under-reading; still, off-lesion pixels are needed for "
  + ", ".join(f"{d}/{t}" for d, t in dep_ru)
  + " of their ≥0.5% calls. The rust reference is semi-automatic (detector + SAM 2, spot-checked), so \"outside the "
  "reference\" is less certain on rust leaves.")
w(f"- On the 21 healthy EARTH leaves the same models stay below 0.5% leaf-level severity. The rust models' off-lesion "
  f"marking on leaf-spot leaves (median {rng(off_x_ls, '.2f')}% of the leaf) is much larger than their specks on healthy "
  f"leaves (maximum {hmax_rust:.2f}%). What it covers on lesioned leaves (lesion margins, chlorotic or otherwise damaged "
  "tissue, or lesions missing from the reference) was not examined.")
w("- Consequence for use: disease identity has to come from the user or from a separate classification step; "
  "the severity models cannot tell rust from American leaf spot, and a positive reading from the wrong model "
  "must not be read as evidence of that disease.")
w("- Site confound: the rust models were trained on Sarchí rust leaves and EARTH healthy leaves, yet they respond "
  "strongly to lesioned EARTH leaves (the leaf-spot leaves), including tissue outside the lesions. So their low output on the healthy EARTH leaves is not "
  "simply a learned \"EARTH means healthy\" rule. This does not remove the site confound; rust-infected leaves from "
  "EARTH, or healthy leaves from Sarchí, would still be needed.")
w("")

w("## 4. Suggested manuscript sentences")
w("")
sp = R["specificity_local_healthy_test"]
w("1. (Methods) \"Leaf-level detection was scored at two predicted-severity thresholds, >0% (any predicted lesion "
  "pixel) and ≥0.5% (predicted grade G1 or higher), using the leaf-level rules of Section~\\ref{sec:evaluation}; "
  "sensitivity was computed on the diseased test leaves and specificity on the healthy test leaves available for "
  "reanalysis (21 of the 54; the remaining 33 were not available), with 95% Wilson intervals.\"")
w(f"2. (Results) \"At ≥0.5%, all six models flagged none of the 21 reanalyzed healthy leaves (specificity 100%, "
  f"95% CI {100 - FP_UB:.1f}–100%). Sensitivity was 20/21 rust leaves for all three rust models; the leaf that was missed had a reference "
  f"severity of {max(miss_ru):.2f}% (reference grade G0). For leaf spot, sensitivity was "
  f"{S['leaf_spot/SegFormer-B5']['ge05']['k']}/55, {S['leaf_spot/UNet++']['ge05']['k']}/55 and "
  f"{S['leaf_spot/DeepLabV3+']['ge05']['k']}/55 (SegFormer-B5, UNet++, DeepLabV3+), with every "
  f"missed leaf having a reference mask severity of {max(miss_ls):.2f}% or less.\"")
w(f"3. (Results) \"Counting any predicted lesion pixel as a false positive is not informative under raw argmax: "
  f"{sp['rust/SegFormer-B5']['primary_model_rule']['fp_gt0']}, {sp['rust/UNet++']['primary_model_rule']['fp_gt0']} and "
  f"{sp['rust/DeepLabV3+']['primary_model_rule']['fp_gt0']} of the 21 healthy leaves carried at least one predicted "
  f"lesion pixel with the rust models, and {sp['leaf_spot/SegFormer-B5']['primary_model_rule']['fp_gt0']}, "
  f"{sp['leaf_spot/UNet++']['primary_model_rule']['fp_gt0']} and {sp['leaf_spot/DeepLabV3+']['primary_model_rule']['fp_gt0']} "
  f"with the leaf-spot models, although no healthy leaf exceeded "
  f"{R['definition_check_numbers']['max_leaf_level_pred_sev_healthy']:.2f}% leaf-level predicted severity (the highest "
  f"single-face value was {R['definition_check_numbers']['max_single_face_pred_sev_healthy']:.2f}%).\" "
  "[Replaces the any-pixel definition and the claim that no healthy leaf received any predicted lesion. "
  "% TODO(author): state the threshold that produced the published 0/54 and 1/54 counts, or report the 21-leaf ≥0.5% result instead.]")
in_all = in_x_ls + in_x_ru
w("4. (Results) \"The models were not disease-specific. Applied to the American leaf spot test leaves, the rust "
  f"models reported ≥0.5% severity on {min(X[LSL]['cross_models'][m]['ge05']['k'] for m in M)}–"
  f"{max(X[LSL]['cross_models'][m]['ge05']['k'] for m in M)} of 55 leaves, and the leaf-spot models reported ≥0.5% on "
  f"{min(X[RUL]['cross_models'][m]['ge05']['k'] for m in M)}–{max(X[RUL]['cross_models'][m]['ge05']['k'] for m in M)} "
  f"of the 21 rust leaves. Most of the predicted lesion pixels behind these readings ({rng(in_all, '.0f')}%, counted on "
  "the leaf faces each model reads) lay within the other disease's reference lesions. The rust models also marked "
  f"tissue outside the leaf-spot reference lesions ({rng([100 - v for v in in_x_ls], '.0f')}% of their predicted "
  f"lesion pixels, against {rng([100 - v for v in in_c_ls], '.0f')}% for the leaf-spot models on the same leaves), "
  f"which raised their mean severity on these leaves to {rng(rat_all, '.2f')} times the reference "
  f"({rng(rat_in, '.2f')} times when only pixels inside the reference lesions were counted). The disease must "
  "therefore be identified before a model is applied.\"")
w(f"5. (Limitations) \"Specificity was reanalyzed on only 21 healthy leaves from a single site; at the ≥0.5% threshold, "
  f"the 95% upper bound of the false-positive rate is {FP_UB:.1f}% (at >0%, specificity on these leaves ranged from "
  f"{SP0_RANGE[0]:.0f}% to {SP0_RANGE[1]:.0f}%, depending on the model). It does not cover abiotic or other biotic "
  "disorders, and because the models respond to lesions of the other disease, they do not identify which disease is "
  "present.\"")
w("")
w("## 5. Figure")
w("")
w("`figure_detection.pdf` (vector, TrueType Arial) and `figure_detection.png` (600 dpi), 16 cm wide.")
w("")
w("Suggested caption: \"Leaf-level predicted severity of the three models trained for (a) coffee leaf rust and "
  "(b) American leaf spot. Each panel shows the target disease's test leaves, the other disease's test leaves "
  "(cross-disease) and the 21 healthy test leaves available for reanalysis (EARTH controls; the paper's full set "
  "had 54). One point per leaf. The rust models read the abaxial face and the leaf-spot models the mean of both faces. "
  "Black bars are medians. The dashed line is the 0.5% threshold (G0/G1 cut-point) and the dotted lines are the 5% and 20% "
  "cut-points. Numbers above the points give the leaves at ≥0.5%. The y-axis is linear from 0 to 0.1% and logarithmic above.\"")
open(f"{OUT}/results.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("results.md written")
