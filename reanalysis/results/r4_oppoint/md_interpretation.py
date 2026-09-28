# Interpretation sections of results.md; executed by make_md.py in its namespace (R, M, PUB, MAT, OPS, n, ci, ft, w, VER).
# Every number is read from results.json.

S = M["pub_segformer"]; Sa, Sb, Sm, S1, S2 = (S["test"][k] for k in ("argmax", "thr_bias", "thr_mae", "ero1", "ero2"))
Sp = S["test_paired_vs_argmax"]; Sv = S["validation"]; Bd = S["boundary_test"]; H = S["healthy_test_abaxial"]
ALL = PUB + MAT


def ci_incl0(c): return c[0] <= 0 <= c[1]


nb = sum(ci_incl0(M[m]["test"]["thr_bias"]["bias_ci95_bootstrap"]) for m in ALL)
nbm = sum(ci_incl0(M[m]["test"]["thr_mae"]["bias_ci95_bootstrap"]) for m in ALL)
ns = sum(ci_incl0(M[m]["test"]["thr_bias"]["slope_ci95_bootstrap"]) for m in ALL)
nsm = sum(ci_incl0(M[m]["test"]["thr_mae"]["slope_ci95_bootstrap"]) for m in ALL)
sig_arg_bias = [m for m in ALL if not ci_incl0(M[m]["test"]["argmax"]["bias_ci95_bootstrap"])]
sig_arg_slope = [m for m in ALL if not ci_incl0(M[m]["test"]["argmax"]["slope_ci95_bootstrap"])]
thr_all = [M[m]["thresholds_chosen_on_validation"][k] for m in ALL for k in ("min_abs_bias", "min_mae")]
red = lambda m, op: 100 * (1 - M[m]["test"][op]["slope"] / M[m]["test"]["argmax"]["slope"])
ag = R["matched_seed_aggregate"]


def seedline(arch, op, k, d=2, sg=False):
    a = ag[arch][op][k]
    return f"{n(a['mean'], d, sg)} ± {n(a['sd'], d)} ({n(a['min'], d, sg)} to {n(a['max'], d, sg)})"


w("## 9. Answer: does the rust over-estimation disappear, shrink or persist?\n")
w("### SegFormer-B5 (primary; published model, 21 test leaves)\n")
w(f"- **Validation-calibrated threshold.** Validation chose t = {ft(S['thresholds_chosen_on_validation']['min_abs_bias'])} (min |bias|) and "
  f"t = {ft(S['thresholds_chosen_on_validation']['min_mae'])} (min MAE) on p(lesion); on validation the bias fell from "
  f"{n(Sv['argmax']['bias'], 2, True)} to {n(Sv['thr_bias']['bias'], 2, True)} pp. Applied once to test (t = {ft(Sb['threshold'])}): "
  f"bias {n(Sa['bias'], 2, True)} {ci(Sa['bias_ci95_bootstrap'], 2, True)} → **{n(Sb['bias'], 2, True)} {ci(Sb['bias_ci95_bootstrap'], 2, True)} pp** "
  f"(paired change {n(Sp['thr_bias']['bias']['diff'], 2, True)} {ci(Sp['thr_bias']['bias']['ci95_bootstrap'], 2, True)}); "
  f"MAE {n(Sa['mae'])} → {n(Sb['mae'])} {ci(Sb['mae_ci95_bootstrap'])} pp; leaf IoU {n(Sa['leaf_iou'], 3)} → {n(Sb['leaf_iou'], 3)} "
  f"{ci(Sb['leaf_iou_ci95_bootstrap'], 3)} (paired {n(Sp['thr_bias']['leaf_iou']['diff'], 3, True)} {ci(Sp['thr_bias']['leaf_iou']['ci95_bootstrap'], 3, True)}); "
  f"proportional-bias slope {n(Sa['slope'], 3)} {ci(Sa['slope_ci95_t'], 3)} → **{n(Sb['slope'], 3)} {ci(Sb['slope_ci95_t'], 3)}** "
  f"(bootstrap {ci(Sb['slope_ci95_bootstrap'], 3)}; p {'< 0.001' if Sb['slope_p'] < 0.001 else '= ' + n(Sb['slope_p'], 3)}; paired change "
  f"{n(Sp['thr_bias']['slope']['diff'], 3, True)} {ci(Sp['thr_bias']['slope']['ci95_bootstrap'], 3, True)}, a {red('pub_segformer', 'thr_bias'):.0f}% reduction); "
  f"grade accuracy {n(Sa['grade_acc'], 3)} → {n(Sb['grade_acc'], 3)}, QWK {n(Sa['qwk'], 3)} → {n(Sb['qwk'], 3)}, over-graded leaves "
  f"{Sa['n_over_graded']} → {Sb['n_over_graded']} (none under-graded), leaves over-estimated {Sa['n_over_estimated']}/21 → {Sb['n_over_estimated']}/21. "
  f"The MAE-chosen threshold (t = {ft(Sm['threshold'])}) gives the same picture: bias {n(Sm['bias'], 2, True)} {ci(Sm['bias_ci95_bootstrap'], 2, True)}, "
  f"MAE {n(Sm['mae'])}, IoU {n(Sm['leaf_iou'], 3)}, slope {n(Sm['slope'], 3)} {ci(Sm['slope_ci95_t'], 3)}.")
w(f"  **Verdict: the constant bias disappears; the proportional over-estimation shrinks by {red('pub_segformer', 'thr_bias'):.0f}% but persists** "
  f"(slope CI still excludes 0).")
w(f"- **Erosion.** 1-px erosion: bias {n(S1['bias'], 2, True)} {ci(S1['bias_ci95_bootstrap'], 2, True)} pp (now significant under-estimation), "
  f"slope {n(S1['slope'], 3)} {ci(S1['slope_ci95_t'], 3)} (proportional component gone), leaf IoU {n(S1['leaf_iou'], 3)} "
  f"(paired {n(Sp['ero1']['leaf_iou']['diff'], 3, True)} {ci(Sp['ero1']['leaf_iou']['ci95_bootstrap'], 3, True)}), MAE {n(S1['mae'])}, "
  f"{S1['n_under_graded']} leaves under-graded. 2-px erosion over-corrects: bias {n(S2['bias'], 2, True)} {ci(S2['bias_ci95_bootstrap'], 2, True)}, "
  f"slope {n(S2['slope'], 3)} {ci(S2['slope_ci95_t'], 3)}, IoU {n(S2['leaf_iou'], 3)}. Removing one boundary pixel ring removes the "
  f"proportional component entirely but also a fixed amount of area from every lesion, so small lesions become under-estimated; "
  f"the excess predicted area is therefore less than one pixel ring wide on average, which a probability threshold can trim more finely.")
a0, t0, r0 = Bd["argmax"]["pred_all"], Bd["argmax"]["pred_att"], Bd["argmax"]["rev"]
a1, t1, r1 = Bd["thr_bias"]["pred_all"], Bd["thr_bias"]["pred_att"], Bd["thr_bias"]["rev"]
w(f"- **Signed boundary distance (argmax).** Of the predicted lesion contour pixels (n = {a0['n_contour_px']}, 21 leaves), "
  f"{100 * a0['share_positive']:.0f}% lie outside the reference lesion, {100 * a0['share_zero']:.0f}% on its contour and only "
  f"{100 * a0['share_le_minus2px']:.0f}% ≥ 2 px inside; {100 * a0['share_ge_2px']:.0f}% are ≥ 2 px outside. Because just over half are "
  f"outside, the pooled median is {n(a0['median'])} px, but it sits on the 0/1 step of a discrete distribution (bootstrap CI "
  f"{ci(a0['median_ci95_bootstrap'])}; IQR {n(a0['q25'])} to {n(a0['q75'])} px; median of per-leaf medians {n(a0['per_leaf_median_of_medians'])}), "
  f"so the shares and the mean are the more informative summaries. The mean "
  f"({n(a0['mean'], 2, True)} px {ci(a0['mean_ci95_bootstrap'], 2, True)}) is inflated by spurious blobs; for contours of components that touch a "
  f"reference lesion it is {n(t0['mean'], 2, True)} px {ci(t0['mean_ci95_bootstrap'], 2, True)} (median {n(t0['median'])}, IQR {n(t0['q25'])} to {n(t0['q75'])}). "
  f"Reverse direction: reference contour pixels lie a median {n(r0['median'])} px inside the prediction (IQR {n(r0['q25'])} to {n(r0['q75'])}). "
  f"So the argmax prediction is displaced outward by a fraction of a pixel to one pixel, systematically. After the validation-calibrated threshold "
  f"the median is {n(a1['median'])} {ci(a1['median_ci95_bootstrap'])} (IQR {n(a1['q25'])} to {n(a1['q75'])}), {100 * a1['share_positive']:.0f}% of contour "
  f"pixels lie outside, the attached-contour mean is {n(t1['mean'], 2, True)} px {ci(t1['mean_ci95_bootstrap'], 2, True)} and the reverse median is "
  f"{n(r1['median'])} (IQR {n(r1['q25'])} to {n(r1['q75'])}): the systematic outward shift is removed.")
h_any05 = sum(M[m]["healthy_test_abaxial"][op]["ge_0.5pct"] for m in ALL for op in OPS)
w(f"- **Healthy test leaves (21, abaxial).** Leaves with ≥ 1 predicted lesion pixel: argmax {H['argmax']['any_pixel']}/21 → threshold "
  f"{H['thr_bias']['any_pixel']}/21 (max {n(H['argmax']['max_pct'], 2)}% → {n(H['thr_bias']['max_pct'], 2)}%). "
  + ("No healthy leaf reaches 0.5% for any of the 9 models under any operating point, so the calibrated threshold costs no specificity on these leaves.\n"
     if h_any05 == 0 else f"Healthy leaves ≥ 0.5% summed over all models and operating points: {h_any05} (see Section 6).\n"))

w("### Published CNNs and matched-protocol CNNs\n")
for m in ("pub_unetpp", "pub_deeplab"):
    a, b = M[m]["test"]["argmax"], M[m]["test"]["thr_bias"]; v = VER[(m, "thr_bias")]
    w(f"- **{M[m]['label']}** (t = {ft(b['threshold'])}): bias {n(a['bias'], 2, True)} {ci(a['bias_ci95_bootstrap'], 2, True)} → {n(b['bias'], 2, True)} "
      f"{ci(b['bias_ci95_bootstrap'], 2, True)}; slope {n(a['slope'], 3)} → {n(b['slope'], 3)} {ci(b['slope_ci95_t'], 3)} (bootstrap "
      f"{ci(b['slope_ci95_bootstrap'], 3)}); MAE {n(a['mae'])} → {n(b['mae'])}; IoU {n(a['leaf_iou'], 3)} → {n(b['leaf_iou'], 3)}; QWK "
      f"{n(a['qwk'], 3)} → {n(b['qwk'], 3)}. Bias: {v['bias']}; slope: {v['slope']}.")
for arch, lab in (("unetpp", "Matched UNet++"), ("deeplab", "Matched DeepLabV3+")):
    w(f"- **{lab}, 3 seeds** (thresholds {', '.join(ft(x) for x in ag[arch]['thr_bias']['thresholds'])}): bias {seedline(arch, 'argmax', 'bias', 2, True)} → "
      f"{seedline(arch, 'thr_bias', 'bias', 2, True)} pp; slope {seedline(arch, 'argmax', 'slope', 3)} → {seedline(arch, 'thr_bias', 'slope', 3)}; "
      f"MAE {seedline(arch, 'argmax', 'mae')} → {seedline(arch, 'thr_bias', 'mae')}; leaf IoU {seedline(arch, 'argmax', 'leaf_iou', 3)} → "
      f"{seedline(arch, 'thr_bias', 'leaf_iou', 3)}. Per-seed verdicts (bias; slope): "
      + "; ".join(f"seed {i}: {VER[(f'm_{arch}_s{i}', 'thr_bias')]['bias']} / {VER[(f'm_{arch}_s{i}', 'thr_bias')]['slope']}" for i in range(3)) + ".")
w("")
w("### Overall\n")
w(f"At argmax, {len(sig_arg_bias)} of the 9 rust models have a bias whose bootstrap CI excludes 0 and {len(sig_arg_slope)} of 9 a positive "
  f"proportional slope whose CI excludes 0. With the validation-calibrated threshold (min |bias|), the test bias CI includes 0 for "
  f"{nb} of 9 models ({nbm} of 9 with the MAE-chosen threshold) and the slope CI includes 0 for {ns} of 9 ({nsm} of 9). The chosen "
  f"thresholds are all far above 0.5 (range {ft(min(thr_all))}–{ft(max(thr_all))}).\n")
not_incl = [m for m in ALL if not ci_incl0(M[m]["test"]["thr_bias"]["bias_ci95_bootstrap"])]
exc = "; ".join(f"{M[m]['label']}: {n(M[m]['test']['thr_bias']['bias'], 2, True)} {ci(M[m]['test']['thr_bias']['bias_ci95_bootstrap'], 2, True)} pp"
                f" (argmax {n(M[m]['test']['argmax']['bias'], 2, True)} {ci(M[m]['test']['argmax']['bias_ci95_bootstrap'], 2, True)})" for m in not_incl)
w("**Conclusion.** The rust over-estimation is to a large extent an operating-point artifact, not a fixed property of the networks: "
  "the argmax lesion border lies systematically slightly (on average less than one pixel) outside the reference border, and raising the "
  f"lesion-probability threshold, calibrated once on validation leaves, brings the test bias to a value compatible with 0 for {nb} of 9 models"
  + (f" (exception: {exc}, i.e. a model without significant argmax bias that the threshold slightly over-corrected)" if not_incl else "")
  + " and reduces the proportional (severity-dependent) over-estimation substantially. It does not remove the proportional component "
  f"for every model: for SegFormer-B5 the slope falls from {n(Sa['slope'], 3)} to {n(Sb['slope'], 3)} but stays above 0 (CI {ci(Sb['slope_ci95_t'], 3)}), "
  f"and for published UNet++ from {n(M['pub_unetpp']['test']['argmax']['slope'], 3)} to {n(M['pub_unetpp']['test']['thr_bias']['slope'], 3)} "
  f"(CI {ci(M['pub_unetpp']['test']['thr_bias']['slope_ci95_t'], 3)}). The honest summary for the primary model is therefore *bias disappears, "
  f"proportional over-estimation shrinks by {red('pub_segformer', 'thr_bias'):.0f}% but persists*. "
  "A fixed 1-px erosion removes the slope but over-corrects into under-estimation and costs IoU, so it is not a better operating point.\n")
w("**Why the thresholds are so high (mechanism, consistent with but not proven by these data).** Both the published SegFormer-B5 kernel "
  "(`modal-recovery/kernels/lesion_roya_segformer_realpack_kernel.py`, lines 142–158) and the matched rust CNN kernels "
  "(`reanalysis/kaggle_runs/pg26038-mb*-roya-*-lrB*/…py`, `focal_tversky` and `ce_w`) train with 0.7 × Focal Tversky (α = 0.3 on false "
  "positives, β = 0.7 on false negatives, γ = 0.75) + 0.3 × cross-entropy with class weights 0.25/0.5/3.0 (background/leaf/lesion). "
  "Both terms penalize missed lesion pixels more than extra ones, which pushes lesion probabilities up at lesion borders; a threshold "
  "above 0.5 undoes this. The published CNN kernels were not re-checked here (the brief's fact 3 covers them for augmentation only). "
  "**The Tversky β ablation (retraining with α = β = 0.5, or with an unweighted cross-entropy) that would test this causally could not "
  "be run: round 4 permits no training of any kind.**\n")

w("## 10. Caveats\n")
w("- *Relative to a semi-automatic reference.* The rust reference masks are detector + SAM 2 output, spot-checked but not edited "
  "(revision brief, fact 1). \"Over-estimation\" and the boundary offset are measured against that reference's border convention; a "
  "calibrated threshold aligns the models to it and does not show which border is botanically correct.")
w("- *Calibration data.* The threshold was chosen on the same 21 validation leaves that were used for checkpoint selection "
  "(SegFormer-B5: validation lesion IoU; rust CNNs: IoU minus healthy false-positive rate), and validation and test leaves come from "
  "one site and one collection day (Sarchí, 2026-08-19). How well a fixed threshold transfers to other scanners, sites or seasons is untested.")
w("- *Grid extension (disclosed).* The threshold grid was widened above 0.99 after a preview run placed one model's validation "
  "optimum on the former edge (see Method); no chosen threshold lies on the final grid's upper edge "
  f"(flag `at_grid_upper_edge` = {any(M[m]['thresholds_chosen_on_validation']['at_grid_upper_edge'] for m in ALL)} for all models).")
w("- *Small n.* 21 test leaves; only 3 exceed 10% reference severity, so the slope estimates lean on those leaves.")
w("- *Pixel scale.* Distances are in letterbox pixels = native scan pixels for rust (scale 1.0); the physical pixel size was not recorded.")
w("- *Healthy leaves.* Only 21 of the 54 healthy rust-test leaves are available locally (fact 8); counts use the abaxial face only.")
w("- *Seeds.* SegFormer-B5 exists as a single run; its seed-to-seed variability is unknown. Matched CNN results are given per seed and as mean ± SD.")
w("- *Erosion depth* (1 and 2 px) was fixed a priori; its validation values are reported but it was not tuned.\n")

w("## 11. Suggested manuscript text (Results or Discussion; numbers from this file)\n")
w(f"> To test whether the rust over-estimation reflects the decision rule rather than the networks themselves, we kept each model's "
  f"softmax lesion probability and, without retraining, chose a lesion-probability threshold on the 21 validation leaves that "
  f"minimized the absolute bias, then applied it once to the test leaves. For SegFormer-B5 (threshold {ft(Sb['threshold'])}), the test bias "
  f"fell from {n(Sa['bias'], 2, True)} to {n(Sb['bias'], 2, True)} percentage points (95% CI {n(Sb['bias_ci95_bootstrap'][0], 2, True)} to "
  f"{n(Sb['bias_ci95_bootstrap'][1], 2, True)}), the MAE from {n(Sa['mae'])} to {n(Sb['mae'])}, and leaf-level IoU rose from {n(Sa['leaf_iou'], 3)} to "
  f"{n(Sb['leaf_iou'], 3)}; the proportional-bias slope decreased from {n(Sa['slope'], 2)} to {n(Sb['slope'], 3)} (95% CI {n(Sb['slope_ci95_t'][0], 3)} to "
  f"{n(Sb['slope_ci95_t'][1], 3)}) but did not vanish. At the default argmax rule, {100 * a0['share_positive']:.0f}% of the predicted lesion contour "
  f"pixels lay outside the reference lesions (mean offset of contours attached to a reference lesion {n(t0['mean'], 2, True)} pixels, 95% CI "
  f"{n(t0['mean_ci95_bootstrap'][0], 2, True)} to {n(t0['mean_ci95_bootstrap'][1], 2, True)}), against {100 * a1['share_positive']:.0f}% "
  f"({n(t1['mean'], 2, True)} pixels, {n(t1['mean_ci95_bootstrap'][0], 2, True)} to {n(t1['mean_ci95_bootstrap'][1], 2, True)}) after calibration. "
  f"Similar reductions in bias and slope were seen for the published and matched-protocol CNNs. The over-estimation is "
  f"therefore largely an operating-point effect, consistent with a training loss that penalized missed lesion pixels more than extra ones "
  f"(Focal Tversky β = 0.7 > α = 0.3; lesion class weight 3.0); a loss ablation was not performed. Because the rust reference masks are "
  f"semi-automatic, this calibration aligns the predictions with the reference border convention rather than establishing the true lesion border.\n")

w("## 12. Scope note: expert-panel identities (author request of 28 Sep 2026)\n")
w("The author asked that each expert's name, occupation and years of experience be added to the paper. That belongs in the Methods "
  "expert-panel table (`tab:experts`) and the Acknowledgments (round-4 plan, section \"Expert panel identities\") and is outside this "
  "operating-point analysis. The identities are deliberately **not** written into this folder: results folders may become part of the "
  "public code release, which must not contain names or institutions.\n")

w("## Files\n")
w("- `infer_prob.py`: CPU inference (onnxruntime) keeping the softmax; `work/prob/<model>/<split>_<grade>/*.npz` (argmax map, "
  "background-vs-leaf map, lesion probability × 65535; about 87 MB, regenerable with `infer_prob.py`, so it can be left out of the public code release); `work/infer_prob.log` (567 argmax reproductions, 0 mismatches).")
w("- `oppoint.py`: all analyses → `results.json`; `work/leaf_level.csv` (per test leaf × model × operating point); `work/val_curves.csv` "
  "(validation IoU/MAE/bias/slope for every threshold); `work/boundary_counts.npz` (signed-distance histograms per model × operating point); "
  "`work/verdicts.json`.")
w("- `make_md.py` + `md_interpretation.py`: this file.")
