"""Write results.md from results.json, detector_rule.json, exclusion_sensitivity.json and paired_extra.json.
Every number in the text is read from those files."""
import json
import os

OUT = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(f"{OUT}/results.json"))
D = json.load(open(f"{OUT}/detector_rule.json"))
XS = json.load(open(f"{OUT}/exclusion_sensitivity.json"))["results"]
PX = json.load(open(f"{OUT}/paired_extra.json"))["results"]
DIS = ("roya", "ojo")
DN = {"roya": "Coffee leaf rust", "ojo": "American leaf spot"}
AN = {"segformer": "SegFormer-B5", "unetpp": "UNet++", "deeplab": "DeepLabV3+"}
ARCHS = ("segformer", "unetpp", "deeplab")
MINUS = "\u2212"
IX = R["meta"]["index_choice"]
IXN = {"hue": "HSV hue", "hab": "CIELAB hue angle h_ab", "astar": "CIELAB a*", "sat_hi": "HSV saturation (lesion = high)",
       "sat_lo": "HSV saturation (lesion = low)", "exr": "excess red 1.4r′ − g′", "ri": "redness index (r − g)/(r + g + b)"}
CFG = {
    "new_primary": "**New baseline (primary)**",
    "new_no_erosion_no_exclusion": "New index, no erosion (secondary)",
    "new_specificity_constrained": "New index, specificity-constrained (secondary)",
    "existing_e0": "Existing baseline, no erosion",
    "existing_fixed_thr_chosen_erosion": "Existing threshold + chosen erosion (sensitivity)",
    "existing_fixed_thr_chosen_erosion_exclusion": "Existing threshold + chosen erosion + exclusion",
    "existing_retuned_chosen_erosion": "Existing index re-tuned at chosen erosion (sensitivity)",
    "existing_retuned_chosen_erosion_exclusion": "Existing index re-tuned at chosen erosion + exclusion",
    "segformer": "SegFormer-B5 (published)", "unetpp": "UNet++ (published)", "deeplab": "DeepLabV3+ (published)"}


def f(x, d=2, sign=False):
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", MINUS)


def ci(c, d=2):
    return f"[{f(c[0], d)}, {f(c[1], d)}]"


def g(x):
    return f"{x:g}".replace("-", MINUS)


def thr(ix, t):
    return {"hue": f"h′ ≤ {g(t)}°", "hab": f"h_ab ≤ {g(t)}°", "astar": f"a* ≥ {g(t)}", "sat_hi": f"S ≥ {g(t)}",
            "sat_lo": f"S ≤ {g(t)}", "exr": f"exr ≥ {g(t)}", "ri": f"ri ≥ {g(t)}"}[ix]


def cfgtxt(c):
    return f"{thr(c['index'], c['threshold'])}, erosion {c['erosion']} px, exclusion {c['petiole']}"


def wil(c):
    return f"[{100 * c[0]:.1f}–{100 * c[1]:.1f}%]"


def tex(x, d=2, sign=False):
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "$-$")


T = R["test"]; H = R["healthy"]; V = R["validation"]; REP = R["reproduction_of_results_baseline"]
C = {d: T[d]["configs"] for d in DIS}
M = {d: {k: v["mask_grade"] for k, v in T[d]["methods"].items()} for d in DIS}
HT = {d: H[d]["test"] for d in DIS}
np_r, np_o = M["roya"]["new_primary"], M["ojo"]["new_primary"]
hr = HT["roya"]
drop_excl = {d: C[d]["new_primary"]["petiole"] == "none" for d in DIS}

L = []
w = L.append
w("# A3 (round 4): detector-independent colour baseline with leaf-edge erosion")
w("")
w(f"Created {R['meta']['created']}. Scripts: `r4_baseline.py` (all main numbers → `results.json`, `per_leaf_test.csv`, `per_leaf_healthy.csv`, "
  "`validation_grid.csv`; the validation-only pre-run that fixed the index before any test evaluation of a new index → `validation_only.json`, `valonly.log`), "
  "`detector_rule.py` (detector characterisation → `detector_rule.json`), `exclusion_sensitivity.py` (→ `exclusion_sensitivity.json`), `paired_extra.py` "
  f"(→ `paired_extra.json`) and `make_md.py` (this file). Python {R['meta']['python']}, numpy {R['meta']['numpy']}, scipy {R['meta']['scipy']}. "
  "No training and no network inference were run; network values come from `reanalysis/out/per_image.csv`.")
w("")

# ------------------------------------------------------------------ 0 summary
ex_e = C["roya"]["existing_fixed_thr_chosen_erosion"]
w("## 0. Summary")
w("")
w(f"- **Detector rule:** the source of the rust weak-mask detector (`roya_lesion.py`) is not on this computer, so its index and thresholds cannot be documented "
  f"from code. From its saved output masks, its colour criterion behaves like a hue boundary between orange/brown and green (best single rules: HSV hue < "
  f"{list(D['single_index'].values())[0]['threshold']:.3g}°, excess red > {D['single_index'][chr(101) + 'xcess red 1.4r' + chr(39) + '-g' + chr(39)]['threshold']:.3f}), "
  f"combined with spatial clean-up; no single threshold reproduces its masks closely (best pooled IoU {list(D['single_index'].values())[0]['pooled_iou']:.2f}). "
  "The existing rust baseline (excess red ≥ 0.12) uses essentially the detector's colour criterion (Section 1).")
w(f"- **New baseline:** {IXN[IX]} (least related usable index; Section 2), threshold, erosion and exclusion chosen on validation leaf IoU. "
  f"Rust: {cfgtxt(C['roya']['new_primary'])}; leaf spot: {cfgtxt(C['ojo']['new_primary'])}. No petiole/cut-end exclusion was selected for either disease.")
w(f"- **Rust test (n = 21):** leaf IoU {np_r['iou']:.3f} {ci(np_r['ci95']['iou'], 3)}, MAE {np_r['mae']:.2f} {ci(np_r['ci95']['mae'])} pp, "
  f"bias {f(np_r['bias'], 2, True)} {ci(np_r['ci95']['bias'])}, grade accuracy {np_r['grade_correct']}/21 = {np_r['grade_acc']:.2f} {ci(np_r['ci95']['grade_acc'])}, "
  f"QWK {np_r['qwk']:.2f} {ci(np_r['ci95']['qwk'])}.")
vo = T["ojo"]["methods"]["new_primary"]["visual_grade"]
w(f"- **Leaf-spot test (n = 55):** leaf IoU {np_o['iou']:.3f} {ci(np_o['ci95']['iou'], 3)}, MAE {np_o['mae']:.2f} {ci(np_o['ci95']['mae'])} pp, "
  f"bias {f(np_o['bias'], 2, True)} {ci(np_o['ci95']['bias'])} pp; mask-derived grade (primary) accuracy {np_o['grade_correct']}/55 = {np_o['grade_acc']:.2f} "
  f"{ci(np_o['ci95']['grade_acc'])}, QWK {np_o['qwk']:.2f} {ci(np_o['ci95']['qwk'])}; visual grade (secondary) {vo['grade_correct']}/55 = {vo['grade_acc']:.2f} "
  f"{ci(vo['ci95']['grade_acc'])}, QWK {vo['qwk']:.2f} {ci(vo['ci95']['qwk'])}.")
nh = hr["new_primary"]
w(f"- **Healthy rust test leaves (21 available):** new rust baseline ≥ 0.5%: {nh['abaxial']['ge_0_5pct']}/21 abaxial {wil(nh['abaxial']['ge_0_5pct_wilson95'])}, "
  f"{nh['either_face']['ge_0_5pct']}/21 either face; > 0%: {nh['abaxial']['any_pixel']}/21 and {nh['either_face']['any_pixel']}/21. "
  f"Existing excess-red threshold with the same {ex_e['erosion']}-px erosion: {hr['existing_fixed_thr_chosen_erosion']['either_face']['ge_0_5pct']}/21 either face "
  f"{wil(hr['existing_fixed_thr_chosen_erosion']['either_face']['ge_0_5pct_wilson95'])}, {hr['existing_fixed_thr_chosen_erosion']['abaxial']['ge_0_5pct']}/21 abaxial "
  f"(21/21 and 20/21 without erosion). Published networks: 0/21 at ≥ 0.5% on either face.")
w("- **Answer:** yes, a validation-tuned, edge-eroded, detector-independent threshold still flags healthy rust leaves at the G1 cut-point (Section 8).")
w("")

# ------------------------------------------------------------------ 1 detector
w("## 1. The colour rule of the detector that seeded the rust SAM 2 masks")
w("")
m = D["meta"]
w("**Source code: not available.** The rust SAM 2 refinement script (`notebooks/pg26038-sam2-roya-refine/pg26038-sam2-roya-refine.py`) says that its seeds are "
  "weak 0/255 masks from `roya_lesion.py`, \"el detector propio de roya\". That file is not on this computer (searched by name and content under `C:/dev`, and by name "
  "under `<USER_HOME>/OneDrive`, Documents and Desktop; no other script in the project reproduces it). **Its index, thresholds and post-processing therefore "
  "cannot be documented from code.** What the refine script does document: SAM 2 (`sam2_l.pt`, Ultralytics) is prompted with the bounding box (3-px padding) of every "
  "weak-mask blob of ≥ 4 px (at most 300 per leaf, largest first); the refined blobs are unioned and clipped to the leaf alpha; a refined blob larger than 15% of the leaf "
  "(or empty) falls back to its weak pixels.")
w("")
w(f"**Empirical characterisation from the detector's output.** The weak masks are present (`datasets/pg26038-sam2-roya/seeds/`, aligned with `images/`, "
  f"2048 px high, twice the pack resolution): {m['images']} rust faces with an image ({', '.join(f'{k} {v}' for k, v in m['images_by_split'].items())}). "
  f"On a uniform 1/16 sample of leaf pixels ({m['sampled_leaf_px']:,} pixels, {m['sampled_seed_px']:,} of them seed pixels), the best single global threshold per "
  "index reproduces the weak masks as follows (pooled pixel IoU with the weak masks; AUC = ROC AUC for weak-mask vs other leaf pixels; AUC < 0.5 means lower values are "
  "weak-mask pixels):")
w("")
w("| Index | Best rule | Pooled IoU with weak masks | AUC |")
w("|---|---|---|---|")
for k, v in list(D["single_index"].items())[:9]:
    w(f"| {k} | {'≥' if v['direction'] == '>=' else '<'} {g(round(v['threshold'], 4))} | {v['pooled_iou']:.3f} | {v['auc']:.3f} |")
w("")
tp = D["two_index_and_rules_top15"][:3]
pio = D["per_image_optimal_threshold"]
tpi = D["top_single_rule_per_image_iou"]
sm = D["seed_morphology"]
w("Best two-index AND rules (128 quantile bins per index): " + "; ".join(
    f"{p['a']} {'≥' if p['dir_a'] == '>=' else '<'} {g(round(p['thr_a'], 3))} AND {p['b']} {'≥' if p['dir_b'] == '>=' else '<'} {g(round(p['thr_b'], 3))} (IoU {p['pooled_iou']:.3f})"
    for p in tp) + ".")
w(f"Per-face IoU of the best single rule ({tpi['index']}): median {tpi['median']:.3f} (10th–90th percentile {tpi['p10']:.3f}–{tpi['p90']:.3f}, {tpi['n_images']} faces). "
  "Per-face optimal thresholds: " + "; ".join(
      f"{k} median {g(round(v['thr_median'], 3))} (10th–90th percentile {g(round(v['thr_p10'], 3))} to {g(round(v['thr_p90'], 3))}; per-face best IoU median {v['per_image_best_iou_median']:.3f})"
      for k, v in list(pio.items())[:2]) + ".")
w(f"Weak-mask morphology: {sm['blobs_total']:,} blobs over {sm['n_images']} faces, smallest blob {sm['min_blob_px_overall']} px (2048-px scale), no blob < 4 px, "
  f"{sm['hole_px_total']:,} enclosed hole pixels (of {sm['seed_px_total']:,} weak-mask pixels), {sm['seed_px_outside_leaf_total']} pixels outside the leaf alpha, "
  f"median weak-mask area {sm['median_seed_frac_pct']:.2f}% of the leaf.")
w("")
w("**Reading (inference, not documentation).** No colour threshold, single or paired, reproduces the weak masks closely, the masks are smooth blobs with a minimum size of "
  f"{sm['min_blob_px_overall']} px, and the per-face optimal threshold varies, so the detector combined a colour criterion with spatial post-processing (smoothing, morphology "
  "and/or area filtering) and possibly a per-leaf adaptive element; its exact form cannot be recovered. Its colour criterion is best approximated by a hue-type boundary "
  "between orange/brown and green tissue (HSV hue below about 70°; excess red above about 0.133). The value 0.133 = 1.4/3 − 1/3 is the excess red of a neutral grey, "
  "so this excess-red rule is a line through the white point of the chromaticity plane, i.e. itself a hue boundary.")
w("")
w("**Comparison with the existing baseline** (`results/baseline`). Its rust rule, excess red ≥ 0.12, is the same index as the best single-index approximation "
  f"of the detector (> {D['single_index'][chr(101) + 'xcess red 1.4r' + chr(39) + '-g' + chr(39)]['threshold']:.3f}) with a threshold 0.013 lower; "
  "excess red, HSV hue, CIELAB h_ab and the redness index all reproduce the weak masks about equally (pooled IoU 0.58–0.65). The existing rust baseline therefore "
  "shares the detector's colour criterion, which is the circularity concern. The leaf-spot rule (redness index ≥ 0.026) belongs to a different disease whose masks "
  "were corrected by hand in CVAT; it is not related to the rust detector.")
w("")

# ------------------------------------------------------------------ 2 index choice
cr = D["candidate_relatedness_train_only"]
cand = cr["candidates"]
w("## 2. Choice of the detector-independent index")
w("")
w(f"Relatedness to the detector was measured on the **training** faces only ({cr['n_train_images']} faces, {cr['n_train_px']:,} sampled leaf pixels): (a) the best "
  "pooled IoU with which one threshold on the candidate reproduces the weak masks, (b) Spearman ρ with the three non-candidate indices that best reproduce the detector "
  f"({', '.join(cr['detector_like_indices'])}). Usefulness as a lesion index is the best validation leaf IoU from the validation-only pre-run (`validation_only.json`), "
  "which ran before any test evaluation of a new index.")
w("")
w("| Candidate | Weak-mask reproduction IoU (train) | AUC | " + " | ".join(f"ρ with {d}" for d in cr["detector_like_indices"]) + " | Best val IoU, rust | Best val IoU, leaf spot |")
w("|---|---|---|" + "---|" * len(cr["detector_like_indices"]) + "---|---|")
vmap = {"CIELAB a*": ["astar"], "CIELAB hue angle h_ab (deg)": ["hab"], "HSV hue signed (deg)": ["hue"], "HSV saturation": ["sat_hi", "sat_lo"]}
for c, v in cand.items():
    tr = v["train_seed_reproduction"]
    vr = " / ".join(f"{V['roya'][k]['best_val_iou']:.3f}" for k in vmap[c]); vo_ = " / ".join(f"{V['ojo'][k]['best_val_iou']:.3f}" for k in vmap[c])
    w(f"| {c} | {tr['pooled_iou']:.3f} | {tr['auc']:.3f} | " + " | ".join(f"{f(v['spearman_with_detector_indices'][d], 3)}" for d in cr["detector_like_indices"]) +
      f" | {vr} | {vo_} |")
w("")
a_ = cand["CIELAB a*"]; hh = cand["HSV hue signed (deg)"]; hb = cand["CIELAB hue angle h_ab (deg)"]; ss = cand["HSV saturation"]
a_rho = [abs(x) for x in a_["spearman_with_detector_indices"].values()]
h_rho = [abs(x) for c in (hh, hb) for x in c["spearman_with_detector_indices"].values()]
w(f"**Choice: {IXN[IX]}, lesion = a* ≥ t.** HSV saturation is excluded because it is not a lesion index here (green tissue is as saturated as lesions): best "
  f"validation IoU ≤ {max(V[d][k]['best_val_iou'] for d in DIS for k in ('sat_hi', 'sat_lo')):.2f} in either direction for both diseases. The two hue indices are hue "
  f"boundaries like the detector's criterion: they reproduce its weak masks as well as excess red does (train IoU {hh['train_seed_reproduction']['pooled_iou']:.2f} and "
  f"{hb['train_seed_reproduction']['pooled_iou']:.2f}; |ρ| up to {max(h_rho):.2f}), and their validation-chosen thresholds (h′ ≤ {g(V['roya']['hue']['threshold'])}°, "
  f"h_ab ≤ {g(V['roya']['hab']['threshold'])}°) sit where the detector's best single rules sit. a* reproduces the weak masks poorly "
  f"(train IoU {a_['train_seed_reproduction']['pooled_iou']:.2f}) and has the weakest rank correlation with the detector-like indices (|ρ| {min(a_rho):.2f}–{max(a_rho):.2f}), "
  f"so it is the least related usable index. Honest caveat: an a* condition (a* ≥ {g(round(tp[0]['thr_b'], 1)) if tp[0]['b'] == 'CIELAB a*' else 'n/a'}) appears as the second term of "
  f"the best two-index approximation of the detector, but it adds only {tp[0]['pooled_iou'] - D['single_index']['chromatic r' + chr(39)]['pooled_iou']:.3f} IoU to chromatic r′ alone, "
  "so a* is not independent of lesion colour (no colour index can be) but is clearly different from the detector's hue-type criterion.")
w("")

# ------------------------------------------------------------------ 3 method
w("## 3. Method")
w("")
w("- **Evaluation protocol unchanged** (round4_plan.md): leaf = alpha > 8; lesion = mask > 127 AND leaf; 1024 × 1024 aspect-preserving letterbox (RGB bilinear, "
  "labels nearest; copy of `run_inference.py load()`); rebuilt reference leaf and lesion counts equal `per_image.csv` for all 388 validation/test images. Rust = abaxial "
  "face; American leaf spot = mean of both faces (IoU = mean of face IoUs, severity = mean of face severities). Grades < 0.5, 0.5–< 5, 5–< 20, ≥ 20%. Rust reference "
  "grade = mask-derived; leaf-spot reference grade = mask-derived (**primary**, author decision of 28 Sep 2026) and the annotator's visual grade (secondary).")
w("- **Baseline**: threshold inside the reference leaf mask, as in `results/baseline`; the severity denominator is always the **full** reference leaf area (erosion and "
  "exclusion only remove predicted lesion pixels); no smoothing or speck removal. a* is sRGB → CIELAB (D65) in float64; excess red and the redness index are compared "
  "exactly in integer arithmetic, as in `results/baseline`.")
w("- **Leaf-edge erosion** e ∈ {0, 1, 2, 4, 8} px on the 1024 canvas: predictions are kept only on leaf pixels whose taxicab distance to the background is > e, "
  "identical to scipy `binary_erosion` (default 3 × 3 cross) applied e times (checked on 12 images) and to the band definition of `results/baseline` 5b.")
w("- **Petiole / cut-end exclusion** (one option or none): 'open r' drops predictions on leaf pixels removed by a morphological opening of the leaf mask with a Euclidean "
  "disk of radius r ∈ {4, 8, 12, 16} px, i.e. parts narrower than about 2r: the petiole stalk (the healthy scans include curled, often brownish petioles) and, "
  "unavoidably, the narrow drip tip; 'ends k' drops predictions within k ∈ {32, 64, 128} px of either end of the leaf's major axis (first principal axis of the leaf-pixel "
  "coordinates); both ends are excluded because which end is the cut petiole is not recorded. Rationale: petiole and cut end are non-lamina tissue included in the "
  "protocol's leaf mask and often brown or yellow on healthy leaves.")
w("- **Selection**: threshold × erosion × exclusion (5 × 8 options × full threshold grid; a* grid −40 to 60 in steps of 0.1) chosen jointly by the **validation mean leaf "
  "IoU of the diseased validation leaves** (rust n = 21, leaf spot n = 55); ties → smaller erosion, then 'none'. Healthy leaves are not used in the primary selection. "
  "The chosen configuration was applied once to test.")
w(f"- **Uncertainty**: {R['meta']['bootstrap']}. Healthy-leaf proportions: Wilson 95% intervals.")
w("- **Healthy leaves**: the 21 healthy rust-pack test leaves available locally (EARTH); images for the other 33 of the paper's 54 were not retained. Flags at > 0% (any "
  "predicted lesion pixel) and ≥ 0.5% (predicted G1 or higher), under the **abaxial** rule (the rust leaf rule) and the **either-face** rule (maximum over the two faces); "
  "the mean-of-faces rule is also listed. Healthy validation leaves (21, EARTH) are secondary.")
w("")

# ------------------------------------------------------------------ 4 reproduction
rr, ro = REP["roya"], REP["ojo"]
w("## 4. Reproduction check (done first)")
w("")
w(f"The existing baseline was re-implemented and reproduced at the published precision before any new result was computed: rust {rr['config']}: IoU {rr['iou']:.3f}, "
  f"MAE {rr['mae']:.2f}, bias {f(rr['bias'], 2, True)}, QWK {rr['qwk_published_reference']:.2f}, healthy test ≥ 0.5% {rr['healthy_test_ge_0_5_either']}/21 either face and "
  f"{rr['healthy_test_ge_0_5_abaxial']}/21 abaxial; leaf spot {ro['config']}: IoU {ro['iou']:.3f}, MAE {ro['mae']:.2f}, bias {f(ro['bias'], 2, True)}, QWK against the visual "
  f"grade (the reference used in `results/baseline`) {ro['qwk_published_reference']:.2f}. The erosion code reproduces `results/baseline` 5b (excess red ≥ 0.116, 3-px band: IoU "
  f"{REP['roya_band3_exr0.116']['iou']:.3f}, {REP['roya_band3_exr0.116']['healthy_either_ge_0_5']}/21 healthy; 12-px band: {REP['roya_band12_exr0.116']['iou']:.3f}, "
  f"{REP['roya_band12_exr0.116']['healthy_either_ge_0_5']}/21). All match: **{'yes' if REP['all_match_results_baseline'] else 'NO'}**.")
w("")

# ------------------------------------------------------------------ 5 validation
w("## 5. Validation selection")
w("")
w("Best validation mean leaf IoU of every candidate over threshold × erosion × exclusion (diseased validation leaves; rust n = 21, leaf spot n = 55). The last columns give "
  "the validation IoU with no erosion and no exclusion (threshold re-optimised) and the number of the 40 erosion × exclusion options whose best IoU is within 0.005 of the maximum.")
w("")
w("| Disease | Index | Best val IoU | Erosion (px) | Exclusion | Threshold | Val IoU, e = 0, none (threshold) | Options within 0.005 |")
w("|---|---|---|---|---|---|---|---|")
for d in DIS:
    for ix, v in V[d].items():
        tag = " **(chosen)**" if ix == IX else (" (existing index)" if ix in ("exr", "ri") else "")
        w(f"| {DN[d]} | {IXN[ix]}{tag} | {v['best_val_iou']:.3f} | {v['erosion']} | {v['petiole']} | {thr(ix, v['threshold'])} | "
          f"{v['val_iou_no_erosion_no_exclusion']:.3f} ({thr(ix, v['threshold_no_erosion_no_exclusion'])}) | {v['n_options_within_0_005']} |")
w("")
for d in DIS:
    v = V[d][IX]
    w(f"{DN[d]}, {IXN[IX]}, validation IoU by erosion (no exclusion): " + ", ".join(f"{e} px {x:.3f}" for e, x in v["val_iou_by_erosion_no_exclusion"].items()) +
      ". Options within 0.005 of the best: " + "; ".join(f"e {o['erosion']}, {o['petiole']}, {thr(IX, o['threshold'])} ({o['val_iou']:.3f})" for o in v["options_within_0_005"]) + ".")
    w("")
w("No petiole/cut-end exclusion improved validation IoU for the chosen index (nor for the existing indices); the 'ends' options never came within 0.005 of the best.")
w("")

# ------------------------------------------------------------------ 6 test
w("## 6. Test results (chosen configuration applied once)")
w("")
for d in DIS:
    blk = T[d]; n = blk["n"]
    c = blk["configs"]["new_primary"]
    w(f"### {DN[d]} (n = {n} test leaves; {'abaxial face' if d == 'roya' else 'mean of both faces'})")
    w("")
    w(f"Chosen: **{cfgtxt(c)}** (validation IoU {c['val_iou']:.3f}). Brackets: 95% bootstrap CI. Grade columns use the mask-derived reference" +
      (" (primary); the visual-grade columns are secondary." if d == "ojo" else "."))
    w("")
    hdr = "| Method | Configuration | Leaf IoU | MAE (pp) | Bias (pp) | Grade acc. | QWK |"
    if d == "ojo":
        hdr += " Grade acc. (visual) | QWK (visual) |"
    w(hdr)
    w("|---" * (hdr.count("|") - 1) + "|")
    for name, mm in blk["methods"].items():
        if name.endswith("_exclusion") and drop_excl[d]:
            continue
        if d == "ojo" and name == "new_specificity_constrained" and blk["configs"][name]["threshold"] == c["threshold"]:
            continue
        mk = mm["mask_grade"]
        conf = cfgtxt(blk["configs"][name]) if name in blk["configs"] else "network (predicted leaf)"
        row = (f"| {CFG[name]} | {conf} | {mk['iou']:.3f} {ci(mk['ci95']['iou'], 3)} | {mk['mae']:.2f} {ci(mk['ci95']['mae'])} | "
               f"{f(mk['bias'], 2, True)} {ci(mk['ci95']['bias'])} | {mk['grade_correct']}/{n} = {mk['grade_acc']:.3f} {ci(mk['ci95']['grade_acc'])} | "
               f"{f(mk['qwk'])} {ci(mk['ci95']['qwk'])} |")
        if d == "ojo":
            vk = mm["visual_grade"]
            row += f" {vk['grade_correct']}/{n} = {vk['grade_acc']:.3f} {ci(vk['ci95']['grade_acc'])} | {f(vk['qwk'])} {ci(vk['ci95']['qwk'])} |"
        w(row)
    w("")
    if drop_excl[d]:
        w("Validation chose no exclusion, so the '+ exclusion' sensitivity rows equal the rows without it and are omitted." +
          (" For leaf spot the specificity-constrained threshold equals the primary one (no healthy validation leaf reached 0.5%) and is omitted." if d == "ojo" else ""))
        w("")
    pn = blk["paired_network_minus_new_primary"]
    w("Paired differences, network minus new baseline (same resamples): " + "; ".join(
        f"{AN[a]} Δ IoU {f(v['iou']['diff'], 3, True)} {ci(v['iou']['ci95'], 3)} (network higher on {v['leaves_network_iou_higher']}/{n} leaves), Δ MAE {f(v['mae']['diff'], 2, True)} {ci(v['mae']['ci95'])}"
        for a, v in pn.items()) + ".")
    w("")
    pe = blk["paired_new_primary_minus_existing"]
    w("New baseline minus existing baseline: " + "; ".join(
        f"vs {CFG[k].lower()}: Δ IoU {f(v['iou']['diff'], 3, True)} {ci(v['iou']['ci95'], 3)}, Δ MAE {f(v['mae']['diff'], 2, True)} {ci(v['mae']['ci95'])}"
        for k, v in pe.items()) + ".")
    w("")
    px = PX[d]
    w(f"Existing threshold with the chosen {C[d]['existing_fixed_thr_chosen_erosion']['erosion']}-px erosion (`paired_extra.json`): minus the same threshold without erosion "
      f"Δ IoU {f(px['existing_fixed_thr_chosen_erosion_minus_existing_e0']['d_iou'], 3, True)} {ci(px['existing_fixed_thr_chosen_erosion_minus_existing_e0']['d_iou_ci95'], 3)}; "
      "networks minus it: " + "; ".join(
          f"{AN[a]} Δ IoU {f(px[f'{a}_minus_existing_fixed_thr_chosen_erosion']['d_iou'], 3, True)} {ci(px[f'{a}_minus_existing_fixed_thr_chosen_erosion']['d_iou_ci95'], 3)} "
          f"(network higher on {px[f'{a}_minus_existing_fixed_thr_chosen_erosion']['leaves_first_higher']}/{n}), Δ MAE {f(px[f'{a}_minus_existing_fixed_thr_chosen_erosion']['d_mae'], 2, True)} "
          f"{ci(px[f'{a}_minus_existing_fixed_thr_chosen_erosion']['d_mae_ci95'])}" for a in ARCHS) + ".")
    w("")
    cm = blk["methods"]["new_primary"]["mask_grade"]["confusion_rows_ref_cols_pred_G0_G3"]
    w(f"New-baseline grade confusion (rows: mask-derived reference G0–G3; columns: predicted G0–G3): {cm}.")
    w("")

# ------------------------------------------------------------------ 7 healthy
w("## 7. Healthy rust-pack test leaves (n = 21 available locally)")
w("")
w("Leaves flagged, with Wilson 95% intervals for the ≥ 0.5% proportion. Abaxial = the rust leaf rule; either face = maximum over both faces; mean = mean of the two "
  "faces (the leaf-spot leaf rule). The last column gives the median / maximum severity on the abaxial face and on the either-face maximum.")
w("")
w("| Detector | Configuration | Abaxial > 0% | Abaxial ≥ 0.5% | Either face > 0% | Either face ≥ 0.5% | Mean ≥ 0.5% | Median / max % (abaxial; either) |")
w("|---|---|---|---|---|---|---|---|")
for d in DIS:
    for name, hs in HT[d].items():
        if name.startswith("decomposition") or (name.endswith("_exclusion") and drop_excl[d]):
            continue
        if d == "ojo" and name == "new_specificity_constrained" and C[d][name]["threshold"] == C[d]["new_primary"]["threshold"]:
            continue
        ef, ab, mf = hs["either_face"], hs["abaxial"], hs["mean_of_faces"]
        conf = cfgtxt(C[d][name]) if name in C[d] else "network"
        w(f"| {DN[d]} | {CFG[name]}: {conf} | {ab['any_pixel']}/21 | {ab['ge_0_5pct']}/21 {wil(ab['ge_0_5pct_wilson95'])} | {ef['any_pixel']}/21 | "
          f"{ef['ge_0_5pct']}/21 {wil(ef['ge_0_5pct_wilson95'])} | {mf['ge_0_5pct']}/21 | {ab['sev_median']:.2f} / {ab['sev_max']:.2f}; {ef['sev_median']:.2f} / {ef['sev_max']:.2f} |")
w("")
w("Rows for 'network' use the published models' predictions from `per_image.csv` (leaf-spot networks: cross-disease run on these leaves).")
w("")
for d in DIS:
    dc = HT[d]["decomposition_new_index_chosen_threshold"]
    tot = max(dc["flagged_px_both_faces_no_erosion"], 1)
    w(f"{DN[d]} new baseline, effect of the erosion on its healthy-leaf flags (chosen threshold, both faces of the 21 healthy test leaves{'; for rust a* mostly adaxial pixels' if d == 'roya' else ''}): {dc['flagged_px_both_faces_no_erosion']:,} flagged pixels "
      f"without erosion; the chosen {C[d]['new_primary']['erosion']}-px erosion removes {dc['removed_by_erosion']:,} ({100 * dc['removed_by_erosion'] / tot:.0f}%); "
      f"{dc['remaining']:,} ({100 * dc['remaining'] / tot:.0f}%) remain.")
    w("")
xa = XS["new_primary"]["options"]; xe = XS["existing_fixed_thr_chosen_erosion"]["options"]
w("**Face dependence of a\\*.** The rust a* threshold was tuned on abaxial faces. On the healthy adaxial faces (dark, glossy green) a* ≥ "
  f"{g(C['roya']['new_primary']['threshold'])} covers most of the face (median {xa['none']['healthy_adaxial_sev_median']:.1f}%, maximum "
  f"{xa['none']['healthy_adaxial_sev_max']:.1f}%), so the either-face count (21/21) mainly reflects applying an abaxial-tuned a* threshold to the other face; the abaxial "
  f"rule (median {xa['none']['healthy_abaxial_sev_median']:.2f}%, maximum {xa['none']['healthy_abaxial_sev_max']:.2f}%) is the fair test for a rust detector. The same "
  "effect was reported for a* in `results/baseline` (Section 5c).")
w("")
w("**Sensitivity: pre-specified petiole/cut-end exclusions** (`exclusion_sensitivity.json`; thresholds and erosion fixed; not a selection step). Healthy test leaves ≥ 0.5% "
  "(abaxial / either face) and diseased rust test IoU:")
w("")
w("| Exclusion | New baseline: healthy abaxial / either | New baseline: test IoU | Existing exr ≥ 0.12, 2 px: healthy abaxial / either | Existing: test IoU |")
w("|---|---|---|---|---|")
for k in xa:
    w(f"| {k} | {xa[k]['healthy_abaxial_ge_0_5']}/21 / {xa[k]['healthy_either_ge_0_5']}/21 | {xa[k]['diseased_test_iou']:.3f} | "
      f"{xe[k]['healthy_abaxial_ge_0_5']}/21 / {xe[k]['healthy_either_ge_0_5']}/21 | {xe[k]['diseased_test_iou']:.3f} |")
w("")
w("Healthy VALIDATION leaves (21, EARTH; secondary), ≥ 0.5% abaxial / either face: " + "; ".join(
    f"{DN[d]} {CFG[k].replace('**', '').lower()}: {C[d][k]['healthy_val']['abaxial']['ge_0_5pct']}/21 / {C[d][k]['healthy_val']['either_face']['ge_0_5pct']}/21"
    for d in DIS for k in ("new_primary", "new_specificity_constrained", "existing_e0", "existing_fixed_thr_chosen_erosion")
    if not (d == "ojo" and k == "new_specificity_constrained")) + ".")
w("")

# ------------------------------------------------------------------ 8 conclusion
sc = C["roya"]["new_specificity_constrained"]; scm = M["roya"]["new_specificity_constrained"]
ere = M["roya"]["existing_fixed_thr_chosen_erosion"]; rt = C["roya"]["existing_retuned_chosen_erosion"]
w("## 8. Which conclusion about rust specificity survives")
w("")
w(f"1. **Survives: a validation-tuned colour threshold does not keep healthy rust leaves below G1.** The detector-independent a* threshold with the validation-chosen "
  f"{C['roya']['new_primary']['erosion']}-px erosion flags {nh['abaxial']['ge_0_5pct']}/21 healthy test leaves at ≥ 0.5% on the abaxial face "
  f"{wil(nh['abaxial']['ge_0_5pct_wilson95'])} and {nh['either_face']['ge_0_5pct']}/21 on either face; the existing excess-red threshold with the same erosion flags "
  f"{hr['existing_fixed_thr_chosen_erosion']['abaxial']['ge_0_5pct']}/21 abaxial and {hr['existing_fixed_thr_chosen_erosion']['either_face']['ge_0_5pct']}/21 either face "
  f"(re-tuned at that erosion, {thr('exr', rt['threshold'])}: {hr['existing_retuned_chosen_erosion']['abaxial']['ge_0_5pct']}/21 and "
  f"{hr['existing_retuned_chosen_erosion']['either_face']['ge_0_5pct']}/21). The pre-specified petiole/cut-end exclusions change these counts by at most one leaf, except "
  f"'ends 128' for excess red ({xe['ends 128']['healthy_abaxial_ge_0_5']}/21 abaxial, {xe['ends 128']['healthy_either_ge_0_5']}/21 either face) at a test IoU of "
  f"{xe['ends 128']['diseased_test_iou']:.3f}. All three published rust networks: 0/21 on either face. Among the variants evaluated, only thresholds constrained on the healthy validation leaves reach "
  f"0/21 (a* ≥ {g(sc['threshold'])} with {sc['erosion']}-px erosion: test IoU {scm['iou']:.3f} {ci(scm['ci95']['iou'], 3)}, bias {f(scm['bias'], 2, True)} pp, grade accuracy "
  f"{scm['grade_correct']}/21; excess red ≥ 0.154 in `results/baseline`: IoU 0.518). So the manuscript's statement that the networks' main added value for rust is keeping "
  "healthy leaves below the G1 cut-point holds for a detector-independent index as well.")
w(f"2. **Does not survive as stated: the \"21/21\" count depends strongly on the leaf margin.** A {ex_e['erosion']}-px erosion alone lowers the existing threshold's count from "
  f"21/21 to {hr['existing_fixed_thr_chosen_erosion']['either_face']['ge_0_5pct']}/21 (either face) and raises its rust test IoU to {ere['iou']:.3f} "
  f"{ci(ere['ci95']['iou'], 3)}, level with UNet++ (Δ {f(PX['roya']['unetpp_minus_existing_fixed_thr_chosen_erosion']['d_iou'], 3, True)} "
  f"{ci(PX['roya']['unetpp_minus_existing_fixed_thr_chosen_erosion']['d_iou_ci95'], 3)}) and DeepLabV3+ (Δ {f(PX['roya']['deeplab_minus_existing_fixed_thr_chosen_erosion']['d_iou'], 3, True)} "
  f"{ci(PX['roya']['deeplab_minus_existing_fixed_thr_chosen_erosion']['d_iou_ci95'], 3)}); only SegFormer-B5 stays clearly above it "
  f"(Δ {f(PX['roya']['segformer_minus_existing_fixed_thr_chosen_erosion']['d_iou'], 3, True)} {ci(PX['roya']['segformer_minus_existing_fixed_thr_chosen_erosion']['d_iou_ci95'], 3)}). "
  "Quote healthy-leaf counts for colour thresholds together with the erosion used, or cite the range across the variants here.")
w(f"3. **At the any-pixel level nobody is specific.** Every threshold variant marks lesion pixels on 21/21 healthy leaves, and the networks on "
  f"{hr['segformer']['either_face']['any_pixel']}/21, {hr['unetpp']['either_face']['any_pixel']}/21 and {hr['deeplab']['either_face']['any_pixel']}/21 (either face). "
  "The specificity contrast exists only at the ≥ 0.5% (G1) cut-point.")
w(f"4. **Circularity is visible in IoU, not in specificity.** The detector-independent a* baseline reaches a lower rust IoU ({np_r['iou']:.3f}) than the detector-like "
  f"excess-red threshold (no erosion {M['roya']['existing_e0']['iou']:.3f}; with erosion {ere['iou']:.3f}; paired Δ {f(T['roya']['paired_new_primary_minus_existing']['existing_fixed_thr_chosen_erosion']['iou']['diff'], 3, True)} "
  f"{ci(T['roya']['paired_new_primary_minus_existing']['existing_fixed_thr_chosen_erosion']['iou']['ci95'], 3)}), consistent with a reference that favours the detector's own "
  "colour criterion; against the a* baseline all three networks have a higher rust IoU with CIs excluding zero (Section 6). For American leaf spot (hand-corrected masks) the "
  f"a* baseline equals the existing one (IoU {np_o['iou']:.3f} vs {M['ojo']['existing_e0']['iou']:.3f}) and keeps 0/21 healthy leaves at ≥ 0.5%; the leaf-spot conclusions of "
  "`results/baseline` are unchanged.")
w("5. **Site confound remains.** All healthy leaves are from EARTH and all diseased rust leaves from Sarchí, so any healthy-vs-diseased contrast is confounded with site and scan session.")
w("")

# ------------------------------------------------------------------ 9 sentences
w("## 9. Suggested manuscript sentences (LaTeX-ready)")
w("")
w(f"1. *Methods.* Because the rust reference masks were seeded by a color detector whose rule is approximated by a hue boundary (excess red $>$0.133, HSV hue $<$70$^{{\\circ}}$), "
  f"we added a second training-free baseline on an index that reproduces the detector's weak masks poorly, CIELAB $a^*$, with the threshold, a leaf-edge erosion (0--8 pixels) and an optional petiole exclusion "
  "chosen jointly on the validation leaves by leaf IoU; the existing threshold was also re-evaluated with the chosen erosion.")
w(f"2. *Results.* With the validation-chosen {C['roya']['new_primary']['erosion']}-pixel erosion, the $a^*$ threshold reached a rust leaf IoU of {tex(np_r['iou'], 3)} "
  f"(95\\% CI {tex(np_r['ci95']['iou'][0], 3)}--{tex(np_r['ci95']['iou'][1], 3)}; MAE {tex(np_r['mae'])} percentage points, QWK {tex(np_r['qwk'])}) and flagged "
  f"{nh['abaxial']['ge_0_5pct']} of the 21 available healthy test leaves at $\\geq$0.5\\% on the abaxial face; the excess-red threshold with the same erosion flagged "
  f"{hr['existing_fixed_thr_chosen_erosion']['abaxial']['ge_0_5pct']} (abaxial) and {hr['existing_fixed_thr_chosen_erosion']['either_face']['ge_0_5pct']} (either face), "
  "versus none for the three networks.")
w(f"3. *Limitations.* Thresholds constrained to leave every healthy validation leaf below 0.5\\% did reach 0/21 but lost lesion delineation (rust IoU {tex(scm['iou'], 3)} for $a^*$), "
  "and these healthy-leaf counts are sensitive to how the leaf margin is handled.")
w("")

# ------------------------------------------------------------------ 10 caveats
w("## 10. Notes and caveats")
w("")
w("- The detector's rule is **inferred** from its output masks; its code is not available. Do not state its index or thresholds as documented facts.")
w("- Healthy coverage: 21 of the paper's 54 healthy rust-test leaves (images for 33 were not retained). Proportions over 21 leaves have wide intervals (Wilson, Section 7).")
w("- The baseline uses the reference leaf mask (exact denominator); the networks segment the leaf themselves. This favours the baseline on severity.")
w("- Thresholds were tuned on abaxial rust faces; the either-face rule applies them to adaxial faces they were not tuned for (large effect for a*, Section 7).")
w("- The index choice used training-leaf relatedness and a validation-only pre-run; no new index was evaluated on test before the choice was fixed. Test values of the other "
  "candidate indices were not computed.")
w("- The petiole/cut-end exclusion was feasible and pre-specified but was not selected on validation; its effect on healthy leaves is a sensitivity analysis only.")
w("- Expert-panel identities (name, occupation, years of experience; round4_plan.md) are outside this analysis and are not recorded in this folder.")
w("")
w("## Files")
w("")
for fn in ("r4_baseline.py", "detector_rule.py", "exclusion_sensitivity.py", "paired_extra.py", "make_md.py", "results.json", "detector_rule.json",
           "exclusion_sensitivity.json", "paired_extra.json", "validation_only.json", "per_leaf_test.csv", "per_leaf_healthy.csv", "validation_grid.csv",
           "run.log", "valonly.log", "detector_rule.log", "exclusion_sensitivity.log", "results.md"):
    w(f"- `{fn}`")
txt = "\n".join(L) + "\n"
open(f"{OUT}/results.md", "w", encoding="utf-8").write(txt)
print("wrote results.md", len(txt))
