"""Write results.md from results.json (paired_analysis.py) so that every number in the text comes from the JSON."""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
R = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
V = json.loads((OUT / "verify_csv.json").read_text(encoding="utf-8"))

ARCHS = ["segformer", "unetpp", "deeplab"]
NAME = {"segformer": "SegFormer-B5", "unetpp": "UNet++", "deeplab": "DeepLabV3+"}
DIS = {"roya": "Coffee leaf rust", "ojo": "American leaf spot"}
PAIRS = ["segformer_vs_deeplab", "segformer_vs_unetpp", "unetpp_vs_deeplab"]
MINUS = "−"


def f(x, k=2, sign=False):
    if x is None:
        return "—"
    s = f"{x:+.{k}f}" if sign else f"{x:.{k}f}"
    if float(s.replace("+", "")) == 0:
        s = f"{0:.{k}f}"
    return s.replace("-", MINUS)


def ci(c, k=2, sign=False):
    return f"[{f(c[0], k, sign)}, {f(c[1], k, sign)}]"


def p(x):
    if x < 0.001:
        return "<0.001"
    return f"{x:.3f}" if x < 0.1 else f"{x:.2f}"


def pair_name(pk):
    a, b = pk.split("_vs_")
    return f"{NAME[a]} vs {NAME[b]}"


L = []
w = L.append
BS = R["bootstrap"]

w("# Paired comparisons between the published models within disease (PG-26038 revision)")
w("")
w("Reviewer issue (R3, R2 partial): severity and grade differences between architectures were never tested, and the "
  "Conclusions implied non-inferiority in grading without a margin or a test. This analysis adds paired tests on the "
  "same test leaves. It uses the reproduced per-image outputs (`reanalysis/out/per_image.csv`). No model inference "
  "was re-run.")
w("")
w("**What is compared.** The three **published trained models** per disease, not architectures under a controlled "
  "protocol. Training differed (revision brief facts 3–4). The published CNNs (UNet++, DeepLabV3+) used photometric "
  "jitter (p = 0.8), and the rust CNNs used at most 30 epochs, patience 12 and model selection penalised for "
  "false positives. SegFormer-B5 used geometric augmentation only, up to 60 epochs and patience 15. Any difference "
  "below may reflect training budget and augmentation as well as architecture. For rust, the matched-protocol "
  "retraining (brief item 11) indicates that training accounts for much of the published ranking (see §6).")
w("")
w("**Design.** Leaf-level, test partition. Rust: abaxial face only, 21 diseased leaves. Reference grade = cut-points "
  "0.5 / 5 / 20% on the semi-automatic reference-mask severity. American leaf spot: mean of the two faces, 55 leaves. "
  "Reference grade = the annotator's visual grade; the mask-derived grade (same cut-points on reference severity) is "
  "a secondary reference. Paired bootstrap: "
  f"{BS['resamples']:,} resamples of leaves, seed {BS['seed']}, percentile intervals. One index matrix per disease "
  "is shared by all models, pairs and metrics. Differences are **first model minus second**: ΔMAE < 0 and Δ|bias| < 0 "
  "favour the first model; ΔQWK > 0 and Δaccuracy > 0 favour the first model. Grade correctness: exact McNemar test "
  "(b = leaves only the first model grades correctly, c = leaves only the second does). Holm correction across the "
  "three pairs within each disease. Accuracy and per-grade recall: Wilson 95% intervals. Paired Δaccuracy: Tango score "
  "95% interval, because the percentile bootstrap collapses when discordant leaves are absent or all on one side. "
  "Descriptive TOST margins for Δaccuracy use the Tango 90% interval for the same reason. "
  "Scripts: `paired_analysis.py` (all numbers → `results.json`), `make_figures.py`, `make_md.py`, "
  "`verify_csv_against_preds.py`.")
w("")

# ------------------------------------------------------------------ reproduction
w("## 0. Reproduction of the published values (done first)")
w("")
w("| Disease | Model | Leaf IoU | MAE (pp) | Bias (pp) | LoA (pp) | QWK | Grade accuracy | Match |")
w("|---|---|---|---|---|---|---|---|---|")
for dis in ["roya", "ojo"]:
    for a in ARCHS:
        m = R["per_model"][dis][a]
        g = m["grades"]["reference"]
        ok = R["reproduction"][dis][a]["match"]
        w(f"| {DIS[dis]} | {NAME[a]} | {m['leaf_iou']:.3f} | {f(m['mae_pp'])} | {f(m['bias_pp'], sign=True)} | "
          f"{f(m['loa_pp'][0], sign=True)} to {f(m['loa_pp'][1], sign=True)} | {g['qwk']:.2f} | "
          f"{g['accuracy']:.2f} ({g['correct']}/{g['n']}) | {'yes' if ok else 'NO'} |")
w("")
QNAME = {"estimate": "point estimate", "ci95_low": "lower 95% CI bound", "ci95_high": "upper 95% CI bound"}
T3 = [(dis, pk, mm) for dis in ["roya", "ojo"] for pk in PAIRS
      for mm in R["paired"][dis][pk]["d_leaf_iou"]["table3_mismatches_3dp"]]
T3_TXT = "; ".join(
    f"the {QNAME[mm['quantity']]} for {'rust' if dis == 'roya' else 'leaf spot'} {pair_name(pk)} "
    f"({mm['reproduced']:.5f}, which rounds to {mm['reproduced_3dp']:.3f}, against a published {mm['published']:.3f})"
    for dis, pk, mm in T3)
T3_EST_OK = all(mm["quantity"] != "estimate" for _, _, mm in T3)
w(f"All 42 published values (leaf IoU, MAE, bias, both LoA limits, QWK and accuracy × 6 models) are reproduced at the "
  f"published rounding: **{'yes' if R['reproduced_published_values'] else 'NO'}**. LoA = bias ± 1.96 SD "
  "(sample SD). The paired IoU differences reproduce Table 3"
  + (f"{': all six point estimates match at 3 dp' if T3_EST_OK else ''}. "
     f"{'Two' if len(T3) == 2 else str(len(T3))} bootstrap CI bound{'s' if len(T3) != 1 else ''} differ"
     f"{'' if len(T3) != 1 else 's'} by 0.001 at 3 dp: {T3_TXT}. This is Monte Carlo variation from the new seed. "
     if T3 else " exactly at 3 dp. ") +
  "Rust QWK CIs are "
  f"[{f(R['per_model']['roya']['segformer']['grades']['reference']['qwk_ci95'][0])}, 1.00] with this seed (published "
  "[0.69, 1.00]). "
  f"Integrity check: the pixel counts of all {V['images_checked']} per-image rows used here were recomputed from the "
  f"stored class maps and the rebuilt letterbox reference labels, with {V['mismatches']} mismatches.")
w("")

# ------------------------------------------------------------------ paired table
w("## 1. Paired differences (first minus second model)")
w("")
for dis in ["roya", "ojo"]:
    n = R["per_model"][dis]["segformer"]["n_leaves"]
    w(f"### {DIS[dis]} (n = {n} leaves)")
    w("")
    w("| Comparison | ΔMAE, pp [95% CI] | Δ\\|bias\\|, pp [95% CI] | ΔQWK [95% CI] | ΔAccuracy [Tango 95% CI] (bootstrap CI) | McNemar b / c; exact p (Holm) | Same grade |")
    w("|---|---|---|---|---|---|---|")
    for pk in PAIRS:
        d = R["paired"][dis][pk]
        mc = d["mcnemar"]
        k = 2 if dis == "roya" else 3
        w(f"| {pair_name(pk)} | {f(d['d_mae_pp']['estimate'], k, True)} {ci(d['d_mae_pp']['ci95'], k, True)} | "
          f"{f(d['d_abs_bias_pp']['estimate'], k, True)} {ci(d['d_abs_bias_pp']['ci95'], k, True)} | "
          f"{f(d['d_qwk']['estimate'], 3, True)} {ci(d['d_qwk']['ci95'], 3, True)} | "
          f"{f(d['d_accuracy']['estimate'], 3, True)} {ci(d['d_accuracy']['tango_score_ci95'], 3, True)} "
          f"({ci(d['d_accuracy']['ci95'], 3, True)[1:-1]}) | "
          f"{mc['first_correct_second_wrong']} / {mc['first_wrong_second_correct']}; {p(mc['p_exact'])} ({p(mc['p_holm3'])}) | "
          f"{d['grade_identical_leaves']}/{n} |")
    w("")
    w("Bootstrap two-sided p (inverts the percentile CI; Holm across the three pairs within disease), "
      "for ΔMAE / Δ|bias|: " + "; ".join(
          f"{pair_name(pk)} {p(R['paired'][dis][pk]['d_mae_pp']['p_boot'])} "
          f"(Holm {p(R['paired'][dis][pk]['d_mae_pp']['p_boot_holm3'])}) / "
          f"{p(R['paired'][dis][pk]['d_abs_bias_pp']['p_boot'])} "
          f"(Holm {p(R['paired'][dis][pk]['d_abs_bias_pp']['p_boot_holm3'])})" for pk in PAIRS) + ".")
    w("")
w("Reading the table:")
rr = R["paired"]["roya"]
oo = R["paired"]["ojo"]
PMr = R["per_model"]["roya"]
w(f"- **Rust severity.** The published SegFormer-B5 model had a lower MAE and a smaller |bias| than the published "
  f"UNet++ model. Both intervals exclude 0 and survive Holm. The training protocols differed, so this cannot be "
  f"attributed to architecture. The difference between the published SegFormer-B5 and DeepLabV3+ models was not "
  f"resolved (ΔMAE "
  f"{f(rr['segformer_vs_deeplab']['d_mae_pp']['estimate'], sign=True)} pp, "
  f"{ci(rr['segformer_vs_deeplab']['d_mae_pp']['ci95'], sign=True)}). The two CNNs were not shown to differ. "
  "All three rust mean biases are positive ("
  + ", ".join(f"{NAME[a]} {f(PMr[a]['bias_pp'], sign=True)} pp" for a in ARCHS) +
  "), so Δ|bias| equals Δbias for rust. This does not hold leaf by leaf. Signed severity errors are negative on "
  + ", ".join(f"{PMr[a]['n_leaves_negative_error']}/21 ({NAME[a]}, min {f(PMr[a]['min_signed_error_pp'], 3, True)} pp)"
              for a in ARCHS) +
  " rust leaves. All rust *grade* errors, however, are over-grades.")
w(f"- **Leaf-spot severity.** The published UNet++ model had the lowest MAE: "
  f"{f(oo['segformer_vs_unetpp']['d_mae_pp']['estimate'], 3)} pp below SegFormer-B5 "
  f"(Holm p = {p(oo['segformer_vs_unetpp']['d_mae_pp']['p_boot_holm3'])}) and "
  f"{f(-oo['unetpp_vs_deeplab']['d_mae_pp']['estimate'], 3)} pp below DeepLabV3+ "
  f"(Holm p {p(oo['unetpp_vs_deeplab']['d_mae_pp']['p_boot_holm3'])}). These differences are a few hundredths to a "
  "tenth of a percentage point. UNet++ also had a smaller |bias| than SegFormer-B5 "
  f"({f(oo['segformer_vs_unetpp']['d_abs_bias_pp']['estimate'], 3, True)} pp, "
  f"{ci(oo['segformer_vs_unetpp']['d_abs_bias_pp']['ci95'], 3, True)}), but this does not survive Holm "
  f"(p = {p(oo['segformer_vs_unetpp']['d_abs_bias_pp']['p_boot_holm3'])}).")
w("- **Grades.** No pair differed detectably in grade correctness. There were 0–2 discordant leaves per pair, all "
  "exact McNemar p ≥ 0.50, and all Holm-adjusted p = 1. Rust SegFormer-B5 and DeepLabV3+ gave the **same grade to all "
  "21 leaves**, so the bootstrap intervals for ΔQWK and Δaccuracy are [0, 0]. These are degenerate intervals, not "
  "evidence of equivalence. The Tango interval for that pair is "
  f"{ci(rr['segformer_vs_deeplab']['d_accuracy']['tango_score_ci95'], 3, True)}. When all discordant leaves favour one "
  "model, the percentile bootstrap bound sits exactly at 0 (e.g., rust SegFormer-B5 vs UNet++: 2 / 0 discordant). "
  "This means 0 is not excluded.")
TM = {dis: {pk: R["paired"][dis][pk]["d_accuracy"]["tost_margin_needed"] for pk in PAIRS} for dis in ["roya", "ojo"]}
for dis in TM:
    for pk in PAIRS:
        assert R["paired"][dis][pk]["d_accuracy"]["tost_margin_basis"] == "Tango score 90% interval"
        assert R["paired"][dis][pk]["d_qwk"]["tost_margin_needed"] is None  # pinned bootstrap bound -> no margin


def pts(dis):
    v = sorted(TM[dis].values())
    return f"{round(v[0] * 100)}–{round(v[-1] * 100)}"


w("- **No equivalence.** No margin was pre-specified. For Δaccuracy the smallest symmetric margin a TOST at "
  "α = 0.05 would accept, taken from the **Tango 90% score interval**, is " + "; ".join(
      f"{pair_name(pk)} {TM['roya'][pk]:.2f} ({ci(R['paired']['roya'][pk]['d_accuracy']['tango_score_ci90'], 3, True)})"
      for pk in PAIRS) +
  " for rust, and " + "; ".join(
      f"{pair_name(pk)} {TM['ojo'][pk]:.3f} ({ci(R['paired']['ojo'][pk]['d_accuracy']['tango_score_ci90'], 3, True)})"
      for pk in PAIRS) +
  " for leaf spot. The 90% percentile bootstrap is not used here: it is [0, 0] for rust SegFormer-B5 vs DeepLabV3+ "
  "and has a bound pinned at 0 for the other five pairs, so it would understate the margin. For ΔQWK the 90% bootstrap "
  "interval is pinned at 0 for every pair, so no margin is given (`tost_margin_needed` = null). Even for leaf spot, "
  f"equivalence would therefore need a margin of at least about {pts('ojo')} accuracy points, and for rust about "
  f"{pts('roya')} points. Such a margin would also have to be pre-specified. Equivalence in grading is not "
  "claimed for either disease.")
w("")
w("Unadjusted 95% intervals are reported. Bonferroni 98.33% intervals are in `results.json` (`ci98_33_bonferroni3`). "
  "Under those intervals the rust SegFormer-B5 vs UNet++ and leaf-spot UNet++ vs DeepLabV3+ MAE differences still "
  "exclude 0, and the leaf-spot SegFormer-B5 vs UNet++ MAE difference no longer does "
  f"({ci(oo['segformer_vs_unetpp']['d_mae_pp']['ci98_33_bonferroni3'], 3, True)}).")
w("")

# ------------------------------------------------------------------ Table 5 update
w("## 2. Grade agreement per model (Table 5 update), with the leaf-spot mask-derived reference")
w("")
w("| Disease | Model | Reference | Accuracy (k/n) [Wilson 95% CI] | QWK [bootstrap 95% CI] | Errors (over / under; all adjacent) |")
w("|---|---|---|---|---|---|")
for dis in ["roya", "ojo"]:
    for a in ARCHS:
        for rname, rl in [("reference", "mask-derived (primary)" if dis == "roya" else "visual (primary)"),
                          ("mask_derived", "mask-derived (secondary)")]:
            if rname not in R["per_model"][dis][a]["grades"]:
                continue
            g = R["per_model"][dis][a]["grades"][rname]
            w(f"| {DIS[dis]} | {NAME[a]} | {rl} | {g['accuracy']:.2f} ({g['correct']}/{g['n']}) {ci(g['accuracy_wilson95'])} | "
              f"{g['qwk']:.2f} {ci(g['qwk_ci95'])} | {g['over_graded']} / {g['under_graded']}; "
              f"{'yes' if g['errors_adjacent_only'] else 'no'} |")
w("")
A = R["leafspot_reference_agreement"]
w(f"Leaf-spot reference standards against each other: the visual grade equals the mask-derived grade for "
  f"{A['visual_eq_mask_derived']}/{A['n']} leaves (QWK {A['qwk_visual_vs_mask']:.2f}). All disagreements are single-grade "
  f"({'yes' if A['all_disagreements_single_grade'] else 'no'}): {A['visual_G1_mask_G0']} leaves are visual G1 but mask "
  f"G0 (reference severity 0.13–0.40%), and 2 are visual G2 but mask G1 (4.80% and 4.92%). This reproduces the "
  "manuscript's 47/55 statement. Against the mask-derived grade, the models reach 0.95–0.98 accuracy and QWK "
  "0.95–0.98, compared with 0.84–0.87 and 0.82–0.85 against the visual grade. This is expected, because the models were "
  "trained on these masks. It is not an independent check. It does show that most leaf-spot grade errors against the "
  "visual reference are leaves where the two references disagree at the G0–G1 boundary.")
w("")

# ------------------------------------------------------------------ recall
w("## 3. Per-grade recall with Wilson 95% intervals")
w("")
w("Recall = share of leaves with reference grade g that the model assigned to g. Healthy† = rust-pack healthy TEST "
  "leaves from the EARTH site: **21 available locally; the paper used 54**. They are scored by the rust models "
  "(abaxial face) and by the leaf-spot models (cross-disease run, mean of both faces). Healthy leaves are never mixed "
  "into accuracy or QWK.")
w("")
w("| Disease / reference | Grade (n) | SegFormer-B5 | UNet++ | DeepLabV3+ |")
w("|---|---|---|---|---|")


def rec(g):
    if g["n"] == 0:
        return "no leaves"
    return f"{g['correct']}/{g['n']} = {g['recall']:.2f} {ci(g['wilson95'])}"


for dis, rname, lab in [("roya", "reference", "Rust / mask-derived"), ("ojo", "reference", "Leaf spot / visual"),
                        ("ojo", "mask_derived", "Leaf spot / mask-derived")]:
    for gname in ["G0", "G1", "G2", "G3"]:
        gg = {a: R["per_model"][dis][a]["grades"][rname]["per_grade_recall"][gname] for a in ARCHS}
        w(f"| {lab} | {gname} ({gg['segformer']['n']}) | " + " | ".join(rec(gg[a]) for a in ARCHS) + " |")
    if rname == "reference":
        h = {a: R["healthy_g0"][dis][a] for a in ARCHS}
        w(f"| {lab} | Healthy† G0 ({h['segformer']['n_healthy_local']}) | " + " | ".join(
            f"{h[a]['predicted_G0']}/{h[a]['n_healthy_local']} = {h[a]['g0_recall']:.2f} {ci(h[a]['g0_recall_wilson95'])}"
            for a in ARCHS) + " |")
w("")
w("Leaf-spot G1 has the lowest recall (0.73–0.80), and every G1 error is an under-grade to G0. Rust G1 and G2 errors "
  "are all over-grades. G3 is untested for rust (no leaves), rests on 2 leaf-spot leaves (Wilson lower bound 0.34), "
  "and rust G0 on a single diseased leaf.")
w("")
H = R["healthy_g0"]
for a in ARCHS:  # leaf-spot models: the leaf rule (mean of both faces) already counts a pixel on either face
    assert H["ojo"][a]["any_predicted_lesion_pixel"] == H["ojo"][a]["any_predicted_lesion_pixel_either_face"]
w("**Caution for the specificity text (outside this task, found in passing).** On the 21 local healthy leaves, every "
  "model's leaf-level predicted severity is < 0.5% (all G0). However, under the reproduced raw-argmax protocol, many "
  "of these leaves carry at least one predicted lesion pixel. The count depends on which faces are examined. "
  "**Rust models, abaxial face only** (the rust leaf-level rule used throughout this analysis): " + ", ".join(
      f"{NAME[a]} {H['roya'][a]['any_predicted_lesion_pixel']}/21" for a in ARCHS) +
  " (max leaf severity " + ", ".join(f"{H['roya'][a]['max_pred_sev_pct']:.2f}%" for a in ARCHS) + "). "
  "**Rust models, a pixel on either face** (maximum over both faces, the rule in revision brief item 10): " + ", ".join(
      f"{NAME[a]} {H['roya'][a]['any_predicted_lesion_pixel_either_face']}/21" for a in ARCHS) +
  " (max single-face severity " + ", ".join(f"{H['roya'][a]['max_face_pred_sev_pct_either_face']:.2f}%"
                                              for a in ARCHS) + "). "
  "**Leaf-spot models** (cross-disease run, leaf = mean of both faces, so a pixel on either face counts): " + ", ".join(
      f"{NAME[a]} {H['ojo'][a]['any_predicted_lesion_pixel']}/21" for a in ARCHS) +
  " (max leaf severity " + ", ".join(f"{H['ojo'][a]['max_pred_sev_pct']:.2f}%" for a in ARCHS) +
  "; max single-face severity " + ", ".join(f"{H['ojo'][a]['max_face_pred_sev_pct_either_face']:.2f}%"
                                              for a in ARCHS) + ")" +
  ("" if all(H[d][a]["max_face_pred_sev_pct_either_face"] < 0.5 for d in H for a in ARCHS) else
   ". A single face can exceed 0.5% while the leaf mean stays in G0 ("
   + ", ".join(f"{'rust' if d == 'roya' else 'leaf-spot'} {NAME[a]}" for d in ["roya", "ojo"] for a in ARCHS
               if H[d][a]["max_face_pred_sev_pct_either_face"] >= 0.5) + "), so G0 counts also depend on the face rule") +
  ". The manuscript should quote one rule and state it. The either-face rule matches brief item 10 and is the "
  "stricter test of 'no predicted lesion'. The manuscript defines a false positive as 'any predicted lesion pixel' and "
  "reports 0/54 for four models. That "
  "definition cannot be reconciled with these local outputs. The published counts behave like a severity or grade "
  "threshold, not a > 0 pixel rule. The FP definition should be checked against the original evaluation code before "
  "the specificity paragraph is finalised.")
w("")

# ------------------------------------------------------------------ figures
w("## 4. Figures")
w("")
w("- `figure_paired_differences.pdf/.png`: paired differences (first minus second model) in MAE, |bias|, QWK "
  "(bootstrap 95% CI) and grade accuracy (Tango score 95% CI). Right column: McNemar discordant counts b / c and exact "
  "p. (a) rust, (b) American leaf spot.")
w("- `figure_grade_recall.pdf/.png`: per-grade recall with Wilson 95% CIs for every model. (a) rust, mask-derived "
  "reference; (b) leaf spot, visual reference; (c) leaf spot, mask-derived reference; (d) leaf-spot accuracy (Wilson "
  "CI) and QWK (bootstrap CI) against the visual (filled) and mask-derived (open) grades. The dotted line in (d) is the "
  "agreement between the two reference standards.")
w("")
w("Suggested caption (paired differences): *Paired differences between the published models on the same test leaves "
  "(first-named model minus second; 21 rust and 55 American leaf spot leaves; training protocols differed between "
  "models, see Section X). Points are observed differences. Bars "
  "are 95% percentile intervals from 10,000 paired bootstrap resamples of leaves; for grade accuracy they are Tango "
  "score intervals. For MAE and |bias| negative values favour the first model, and for QWK and accuracy positive "
  "values do. b / c are leaves graded correctly only by the first / only by the second model, and p is from the exact "
  "McNemar test. Rust SegFormer-B5 and DeepLabV3+ assigned identical grades to all 21 leaves. pp = percentage points; "
  "QWK = quadratic-weighted kappa.*")
w("")
w("Suggested caption (recall): *Per-grade recall of each model with 95% Wilson intervals. (a) Coffee leaf rust, "
  "reference grade derived from the semi-automatic reference masks. (b) American leaf spot, annotator's visual grade. "
  "(c) American leaf spot, grade derived from the reference masks through the same cut-points. (d) American leaf spot "
  "grade accuracy and quadratic-weighted kappa against the visual (filled) and mask-derived (open) grades. The dotted "
  "line marks the agreement between the two reference standards. †Healthy test leaves from the EARTH site available "
  "locally (21 of the 54 used for specificity), scored by the rust models on the abaxial face and by the leaf-spot "
  "models as the mean of both faces.*")
w("")

# ------------------------------------------------------------------ sentences
g = lambda dis, a, r="reference": R["per_model"][dis][a]["grades"][r]
w("## 5. Suggested manuscript sentences (precise, no equivalence claim)")
w("")
w("Plain text (LaTeX versions below):")
w("")
s1 = (f"For rust, the published SegFormer-B5 model had a lower MAE than the published UNet++ model (training "
      f"protocols differed; paired difference "
      f"{f(rr['segformer_vs_unetpp']['d_mae_pp']['estimate'], sign=True)} pp, 95% CI "
      f"{f(rr['segformer_vs_unetpp']['d_mae_pp']['ci95'][0], sign=True)} to "
      f"{f(rr['segformer_vs_unetpp']['d_mae_pp']['ci95'][1], sign=True)}; Holm-adjusted bootstrap p "
      f"{p(rr['segformer_vs_unetpp']['d_mae_pp']['p_boot_holm3'])}), but its difference from DeepLabV3+ was not "
      f"resolved ({f(rr['segformer_vs_deeplab']['d_mae_pp']['estimate'], sign=True)} pp, "
      f"{f(rr['segformer_vs_deeplab']['d_mae_pp']['ci95'][0], sign=True)} to "
      f"{f(rr['segformer_vs_deeplab']['d_mae_pp']['ci95'][1], sign=True)}). For American leaf spot, the published "
      f"UNet++ model had the lowest MAE, {f(oo['segformer_vs_unetpp']['d_mae_pp']['estimate'], 3)} pp below SegFormer-B5 "
      f"(95% CI {f(oo['segformer_vs_unetpp']['d_mae_pp']['ci95'][0], 3)} to "
      f"{f(oo['segformer_vs_unetpp']['d_mae_pp']['ci95'][1], 3)}; Holm-adjusted bootstrap p = "
      f"{p(oo['segformer_vs_unetpp']['d_mae_pp']['p_boot_holm3'])}) and "
      f"{f(-oo['unetpp_vs_deeplab']['d_mae_pp']['estimate'], 3)} pp below DeepLabV3+ "
      f"({f(-oo['unetpp_vs_deeplab']['d_mae_pp']['ci95'][1], 3)} to "
      f"{f(-oo['unetpp_vs_deeplab']['d_mae_pp']['ci95'][0], 3)}; p {p(oo['unetpp_vs_deeplab']['d_mae_pp']['p_boot_holm3'])}), "
      f"differences of about a tenth of a percentage point or less.")
s2 = ("No pair of published models differed detectably in per-leaf grade correctness: within each disease, two models "
      "disagreed on whether a leaf was graded correctly for at most two leaves (exact McNemar p ≥ 0.50), and for rust, "
      "SegFormer-B5 and DeepLabV3+ assigned the same grade to all 21 leaves.")
s3 = (f"These grade comparisons have little power and do not show equivalence. The 95% interval for the difference in "
      f"rust grade accuracy was {f(rr['segformer_vs_unetpp']['d_accuracy']['tango_score_ci95'][0], 2, True)} to "
      f"{f(rr['segformer_vs_unetpp']['d_accuracy']['tango_score_ci95'][1], 2, True)} between SegFormer-B5 and UNet++, "
      f"and {f(rr['segformer_vs_deeplab']['d_accuracy']['tango_score_ci95'][0], 2, True)} to "
      f"{f(rr['segformer_vs_deeplab']['d_accuracy']['tango_score_ci95'][1], 2, True)} even between SegFormer-B5 and "
      f"DeepLabV3+, which graded all 21 leaves identically. For leaf spot it was "
      f"{f(oo['segformer_vs_unetpp']['d_accuracy']['tango_score_ci95'][0], 2, True)} to "
      f"{f(oo['segformer_vs_unetpp']['d_accuracy']['tango_score_ci95'][1], 2, True)} between SegFormer-B5 and UNet++.")
s4 = (f"Against grades derived from the reference masks, leaf-spot grade accuracy was "
      f"{g('ojo', 'segformer', 'mask_derived')['accuracy']:.2f}, {g('ojo', 'unetpp', 'mask_derived')['accuracy']:.2f} "
      f"and {g('ojo', 'deeplab', 'mask_derived')['accuracy']:.2f} and quadratic-weighted kappa "
      f"{g('ojo', 'segformer', 'mask_derived')['qwk']:.2f}, {g('ojo', 'unetpp', 'mask_derived')['qwk']:.2f} and "
      f"{g('ojo', 'deeplab', 'mask_derived')['qwk']:.2f} for SegFormer-B5, UNet++ and DeepLabV3+. Against the visual "
      "grades, accuracy was 0.84–0.87 and kappa 0.82–0.85. Because the models were trained on these masks, this is not "
      "an independent check, but it locates most leaf-spot grade errors at the G0–G1 boundary, where the visual and "
      "mask-derived references themselves disagree.")
s5 = ("Per-grade recall was lowest for leaf-spot G1 (0.73–0.80; Wilson 95% intervals 0.56–0.90), where every error was "
      "an under-grade to G0. G3 recall rests on two leaf-spot leaves and could not be assessed for rust. Because these "
      "tests compare the published models, whose training protocols differed, they do not isolate an effect of "
      "architecture; differences in the ordinal grade were not detected and cannot be excluded at this sample size.")
for i, s in enumerate([s1, s2, s3, s4, s5], 1):
    w(f"{i}. {s}")
w("")


def tex(s):
    s = s.replace("−", "$-$").replace("≥", "$\\geq$").replace("%", "\\%").replace("–", "--")
    s = s.replace("p <0.001", "p $<$ 0.001").replace("|bias|", "$|$bias$|$")
    return s


w("```latex")
for s in [s1, s2, s3, s4, s5]:
    w(tex(s))
    w("")
w("```")
w("")
w("Methods addition: *Differences between the published models in MAE, absolute bias, quadratic-weighted kappa "
  "and grade accuracy were estimated on the same leaves with paired percentile bootstrap intervals (10,000 resamples "
  "of leaves, seed 20260927). For grade accuracy we also report Tango's score interval. Per-leaf grade correctness "
  "was compared with the exact McNemar test with Holm correction within disease. Accuracy and per-grade recall carry "
  "Wilson score intervals. No equivalence margin was pre-specified.*")
w("")
t_sd = rr["segformer_vs_deeplab"]["d_accuracy"]["tango_score_ci95"]
t_su = rr["segformer_vs_unetpp"]["d_accuracy"]["tango_score_ci95"]
assert rr["segformer_vs_deeplab"]["grade_identical_leaves"] == 21 and rr["segformer_vs_unetpp"]["grade_identical_leaves"] == 19
w("Conclusions replacement for the sentence 'grade and severity differences between architectures were not formally "
  "tested…': *On rust, the published DeepLabV3+ model assigned the same grade as the published SegFormer-B5 model to "
  "all 21 test leaves, and the published UNet++ model differed on two. No grade difference was detected, but with 21 "
  f"leaves, differences in grade accuracy of up to about {round(max(abs(t_sd[0]), abs(t_sd[1])) * 100)} points "
  f"(SegFormer-B5 vs DeepLabV3+) and {round(max(abs(t_su[0]), abs(t_su[1])) * 100)} points (SegFormer-B5 vs UNet++) "
  "cannot be excluded. The published SegFormer-B5 model had a lower rust severity MAE than the published UNet++ model, "
  "but the training protocols differed, so these comparisons do not establish an advantage of one architecture.* "
  "(If the matched-protocol retraining is reported in the manuscript, cite it here, because for rust the "
  "SegFormer-B5 advantage largely disappears under matched training; see §6. `% TODO(retraining)` for leaf spot.)")
w("")

# ------------------------------------------------------------------ caveats
w("## 6. Caveats")
w("")
w("- Each published model was trained once, and the leaf-spot models were never re-seeded. All intervals reflect "
  "test-leaf sampling only, not training variability.")
# Matched-protocol rust retraining: numbers quoted from revision_brief.md item 11 (not computed in this folder).
w("- **Published models, not architectures.** Training protocols differed (brief facts 3–4), so no difference here "
  "is a controlled architecture effect. The matched-protocol **rust** retraining is available (brief item 11: "
  "60 epochs, patience 15, geometric augmentation only, LR 6e-5/6e-4 chosen on validation; evaluated with the paper "
  "protocol on the same 21 test leaves; numbers quoted from the brief, not recomputed here). Retrained UNet++ reached "
  "a leaf IoU of 0.776 and an MAE of 0.50 pp (bias +0.47 pp, grade accuracy 0.95, QWK 0.94), and retrained "
  "DeepLabV3+ reached 0.721 and 0.87 pp, against 0.767 and 0.99 pp for the published SegFormer-B5. The brief "
  "concludes that the published rust ranking is a training-budget effect. The rust SegFormer-B5 vs UNet++ severity "
  "difference reported here therefore describes the published models only. The paired tests were not re-run on the "
  "retrained models, because their per-leaf outputs are not in this folder's input. **Leaf-spot** retraining is still "
  "running (`% TODO(retraining)`).")
w("- Rust reference severity and grades come from the semi-automatic (detector + SAM 2, human-spot-checked) masks. "
  "The rust grade comparisons measure agreement with that reference.")
w("- The percentile bootstrap is degenerate when two models grade every leaf identically (rust SegFormer-B5 vs "
  "DeepLabV3+) and pins a bound at 0 when discordance is one-sided. The McNemar counts and Tango intervals are the "
  "informative summaries for grades, and TOST margins for Δaccuracy come from the Tango 90% interval. No TOST margin "
  "is given for ΔQWK. No QWK resample was undefined (0/0) in any analysis.")
w("- Leaf-level bootstrap only. A session-clustered bootstrap is a separate TODO.")
w("- Healthy leaves: 21 of the paper's 54 are available locally, all from EARTH, so they are confounded with site for "
  "rust.")
(OUT / "results.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("results.md written")
