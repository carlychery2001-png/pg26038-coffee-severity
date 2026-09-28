"""Write results.md from results.json. Every number in the text is read from results.json, except the values of the
previous version quoted in the revision log (Section 10), which are fixed historical values (PREVIOUS below)."""
import json
import os

OUT = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(f"{OUT}/results.json"))
DIS = ("roya", "ojo")
DN = {"roya": "Coffee leaf rust", "ojo": "American leaf spot"}
DS = {"roya": "rust", "ojo": "leaf spot"}
DSA = {"roya": "rust", "ojo": "leaf-spot"}          # adjective form
ARCHS = ("segformer", "unetpp", "deeplab")
AN = {"segformer": "SegFormer-B5", "unetpp": "UNet++", "deeplab": "DeepLabV3+", "baseline": "Color threshold",
      "baseline_constrained": "Color threshold, specificity-constrained (secondary)"}
MINUS = "−"
# previous version of this folder (2026-09-27 23:17), quoted only in the revision log
PREVIOUS = dict(rust_bias="+0.16", rust_loa_hi="+4.12", rust_healthy_test="20/21", rust_healthy_val="18/21",
                rust_net_any_px="17/21, 15/21, 7/21", constrained_thr="0.154", constrained_iou="0.518",
                constrained_bias="−2.12", share_rust="58%", share_ojo="84%",
                otsu_ojo_val="ri 0.648, a* 0.616, exr 0.655",
                rust_r="0.923", rust_ccc="0.922", rust_ccc_lo="0.749", rust_mae_hi="1.84", rust_bias_hi="1.10",
                d_iou_unetpp="+0.046", d_iou_deeplab="+0.031", d_iou_deeplab_hi="0.065", deeplab_higher="13/21", deeplab_lower="8/21",
                g0_iou="0.480", m12_bias="\u22120.63", healthy_med_max="0.959 / 2.325")


def f(x, d=2, sign=False):
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", MINUS)


def fmt_g(x):
    return f"{x:g}".replace("-", MINUS)


def ci(c, d=2):
    return f"{f(c[0], d)} to {f(c[1], d)}" if c[0] < 0 else f"{f(c[0], d)}–{f(c[1], d)}"


def tex(x, d=2, sign=False):
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "$-$")


def texci(c, d=2):
    return f"{tex(c[0], d)} to {tex(c[1], d)}" if c[0] < 0 else f"{tex(c[0], d)}--{tex(c[1], d)}"


def names(keys):
    n = [AN[k] for k in keys]
    return n[0] if len(n) == 1 else " and ".join([", ".join(n[:-1]), n[-1]])


T = R["test"]; S = R["specificity"]; V = R["validation_configs"]; PH = R["posthoc_margin_excluded"]["results"]
M = {d: T[d]["methods"] for d in DIS}
P = {d: T[d]["paired_network_minus_baseline"] for d in DIS}
PC = {d: T[d]["paired_network_minus_baseline_constrained"] for d in DIS}
SEL = {d: T[d]["selected_config"] for d in DIS}
CON = {d: T[d]["secondary_constrained_config"] for d in DIS}
CONL = {d: T[d]["secondary_constrained_config_leaf_rule"] for d in DIS}
B10 = R["reproduction"]["brief_item10_healthy_counts"]


def SP(split, d, k, rule="ge_0.5pct", face="either_face"):
    e = S[split]["methods"].get(f"{d}/{k}")
    return e[face][rule] if isinstance(e, dict) else None


def SE(split, d, k, face="either_face"):
    e = S[split]["methods"].get(f"{d}/{k}")
    return e[face] if isinstance(e, dict) else None


def same_cfg(a, b):
    return a["index"] == b["index"] and a["threshold"] == b["threshold"]


IDXN = {"ri": "redness index (r − g)/(r + g + b)", "astar": "CIELAB a*", "exr": "excess red 1.4r′ − g′"}
RULEN = {"otsu": "per-leaf Otsu", "global": "global (validation-tuned)", "otsu_face": "per-face Otsu (secondary)"}
TEXIDX = {"ri": "redness index", "astar": "CIELAB $a^*$", "exr": "excess red"}

L = []
w = L.append
w("# Training-free color-threshold baseline (PG-26038 major revision)\n")
w(f"Created {R['meta']['created']} by `baseline.py` (numbers), `make_figures.py` (figures) and `make_md.py` (this file). "
  f"Python {R['meta']['python']}, numpy {R['meta']['numpy']}, scipy {R['meta']['scipy']}. All numbers are in `results.json`. "
  "Section 10 lists what changed from the previous version of this folder.\n")

# ---------------------------------------------------------------- reproduction
w("## 1. Reproduction of the published network values (done first)\n")
w(f"All published values reproduced from `per_image.csv` at the published precision: **{'yes' if R['reproduction']['all_match'] else 'NO'}**. "
  f"The rebuilt reference labels also match `per_image.csv` pixel for pixel ({R['meta']['reference_rebuild_check']}).\n")
w("| Disease | Model | n | Leaf IoU | MAE | Bias | QWK | Bland–Altman LoA | Match |")
w("|---|---|---|---|---|---|---|---|---|")
for d in DIS:
    for a in ARCHS:
        e = R["reproduction"]["models"][f"{d}/{a}"]; c = e["computed"]
        w(f"| {DN[d]} | {AN[a]} | {e['n']} | {c['iou']:.3f} | {f(c['mae'])} | {f(c['bias'], sign=True)} | {c['qwk']:.2f} | "
          f"{f(c['loa'][0])} to {f(c['loa'][1], sign=True)} | {'all' if all(e['match'].values()) else e['match']} |")
w("")
w("The healthy-leaf counts of revision brief item 10 are also reproduced (rust networks, 21 local healthy test leaves, maximum over both faces): "
  "any predicted lesion pixel " + ", ".join(f"{AN[a]} {B10[a]['any_pixel_either_face']}/21" for a in ARCHS) +
  "; maximum predicted severity " + ", ".join(f"{B10[a]['max_sev_either_face']:.2f}%" for a in ARCHS) + ".\n")

# ---------------------------------------------------------------- method
w("## 2. Baseline definition\n")
w("- **Space and leaf mask.** The indices are thresholded inside the **reference leaf mask** (alpha > 8), in the same 1024 × 1024 letterbox "
  "space (`run_inference.py load()`: RGB bilinear, labels nearest) in which the networks were evaluated. The networks do not use this mask: "
  "they segment the leaf themselves, and their severity denominator is their predicted leaf. The baseline's severity denominator is therefore "
  "the exact reference leaf area, which favors it slightly on severity.")
w("- **Indices** (lesion = index above threshold; lesions are redder/browner than green tissue for both diseases): "
  "(i) redness index (r − g)/(r + g + b); (ii) CIELAB a* (sRGB, D65); (iii) excess red 1.4r′ − g′ on chromatic coordinates.")
w("- **Threshold rules:** (a) **per-leaf** Otsu: one threshold per leaf from the pooled leaf pixels of the faces that the leaf rule uses "
  "(rust: abaxial; leaf spot: both faces), 256 bins over the pooled range (as `skimage.filters.threshold_otsu`); a per-face variant is reported "
  "as a secondary check. (b) One global threshold per disease, grid search on the **validation diseased leaves** to maximize mean leaf lesion IoU "
  f"(grids: {', '.join(f'{k} {fmt_g(v[0])} to {fmt_g(v[1])} step {fmt_g(v[2])}' for k, v in R['meta']['grids'].items())}).")
td = R["meta"]["tie_diagnostics_selected_thresholds"]


def tie_txt(d):
    t = td[d]; n = t["tie_pixels_total"]
    head = f"{DS[d]} ({t['index']} \u2265 {t['threshold']:g}): {n} leaf pixel{'s' if n != 1 else ''} exactly on the threshold over {t['images']} validation and test images"
    if n < t["images"]:
        return head + f", {t['tie_pixels_classified_below_by_float64']} of them put below the threshold by a float64 evaluation"
    fm = t["tie_pixels_mean_per_image_by_face"]
    return (head + f" ({fm['abaxial']:.1f} per abaxial and {fm['adaxial']:.1f} per adaxial image on average), "
            f"{t['tie_pixels_classified_below_by_float64']} of which a float64 evaluation of the index put below the threshold")
w("- **Tie handling.** The letterboxed RGB values are 8-bit integers, so the redness index and excess red are ratios of integers and many pixels "
  "sit exactly on a grid threshold. Global thresholds for these two indices are therefore compared in exact integer arithmetic, and pixels exactly on "
  "the threshold count as lesion (e.g. excess red ≥ 0.12 ⇔ 32r − 28g − 3b ≥ 0). "
  + "Ties at the selected thresholds: " + "; ".join(tie_txt(d) for d in DIS if d in td)
  + ". CIELAB a* is compared in float64.")
w("- **No post-processing** (no smoothing, morphology or speck removal), matching the raw-argmax evaluation of the networks.")
w("- **Selection:** per disease, the configuration with the highest validation mean leaf IoU; evaluated **once** on test with the paper's leaf-level rules "
  "(rust: abaxial face, reference grade from reference severity with cut-points 0.5/5/20%; leaf spot: mean of the two faces, reference grade = annotator's visual grade).")
w("- **Healthy leaves: two face rules, both reported.** *Either face* (primary; the rule of revision brief item 10): a healthy leaf is flagged when either "
  "face meets the criterion (leaf value = maximum over the two faces). *Leaf rule* (secondary): the paper's diseased-leaf rule of the detector's disease "
  "(rust: abaxial face only; leaf spot: mean of the two faces). Flag criteria: any lesion pixel (> 0%), ≥ 0.5% (predicted G1 or higher) and ≥ 5%.")
w(f"- **Uncertainty:** {R['meta']['bootstrap']}. Grade accuracy also has a Wilson interval; healthy-leaf proportions use Wilson intervals.\n")

w("### Validation selection (mean leaf IoU on the validation diseased leaves)\n")
w("| Disease | Index | Rule | Threshold | Validation IoU | Healthy validation leaves ≥ 0.5%: either face (leaf rule) |")
w("|---|---|---|---|---|---|")
for d in DIS:
    for k, c in V[d]["configs"].items():
        star = " **(selected)**" if k == V[d]["selected"] else ""
        thr = "per leaf" if c["threshold"] is None else fmt_g(c["threshold"])
        hv = c.get("healthy_val_flagged_ge_0_5pct")
        hvs = "" if hv is None else f"{hv}/21 ({c['healthy_val_flagged_ge_0_5pct_leaf_rule']}/21)"
        w(f"| {DN[d]} | {IDXN[k.split('_')[0]]} | {RULEN[c['rule']]}{star} | {thr} | {c['val_iou']:.3f} | {hvs} |")
w("")
w(f"Selected: rust = {IDXN[SEL['roya']['index']]} ≥ {SEL['roya']['threshold']:g} "
  f"(validation IoU plateau within 0.005 of the maximum: {V['roya']['configs'][SEL['roya']['index'] + '_global']['plateau_within_0_005_of_max'][0]:g} to "
  f"{V['roya']['configs'][SEL['roya']['index'] + '_global']['plateau_within_0_005_of_max'][1]:g}); "
  f"American leaf spot = {IDXN[SEL['ojo']['index']]} ≥ {SEL['ojo']['threshold']:g} "
  f"(plateau {V['ojo']['configs'][SEL['ojo']['index'] + '_global']['plateau_within_0_005_of_max'][0]:g} to "
  f"{V['ojo']['configs'][SEL['ojo']['index'] + '_global']['plateau_within_0_005_of_max'][1]:g}). "
  "Otsu was the weakest rule for every index: it always splits the leaf into two classes, so on leaves with few lesions it labels a large share of healthy tissue as lesion. "
  "Per-face Otsu (secondary) gave validation IoU " +
  "; ".join(f"{DS[d]} " + ", ".join(f"{k.split('_')[0]} {v['val_iou']:.3f}" for k, v in V[d]["otsu_per_face_secondary"].items()) for d in DIS) +
  " (identical to per-leaf Otsu for rust, where only the abaxial face is used).\n")

# ---------------------------------------------------------------- main table
w("## 3. Test-set comparison (selected baseline vs the three networks)\n")
w("Diseased test leaves: rust n = 21, American leaf spot n = 55. Healthy leaves: the 21 healthy rust-pack TEST leaves available locally "
  "(EARTH site; **the paper used 54, the other 33 are not in the local pack**) and the 21 healthy VALIDATION leaves (EARTH). "
  "Healthy columns: leaves with predicted severity ≥ 0.5% (predicted G1 or higher) on **either face** (primary; revision brief item 10), "
  "with the count under the paper's leaf rule (rust: abaxial; leaf spot: mean of faces) in parentheses. Brackets: 95% CI.\n")
w("| Disease | Method | Leaf IoU | MAE (pp) | Bias (pp) | Pearson r | CCC | Grade accuracy | QWK | Healthy test ≥ 0.5% | Healthy val ≥ 0.5% |")
w("|---|---|---|---|---|---|---|---|---|---|---|")
for d in DIS:
    for k in ("baseline", *ARCHS):
        m = M[d][k]
        ht, htl = SP("test", d, k), SP("test", d, k, face="leaf_rule")
        hv, hvl = SP("val", d, k), SP("val", d, k, face="leaf_rule")
        hts = f"{ht['flagged']}/{ht['n']} ({htl['flagged']}/{htl['n']})" if ht else "n/a"
        hvs = f"{hv['flagged']}/{hv['n']} ({hvl['flagged']}/{hvl['n']})" if hv else "not run"
        name = f"**{AN[k]}** ({SEL[d]['index']} ≥ {SEL[d]['threshold']:g})" if k == "baseline" else AN[k]
        w(f"| {DN[d]} | {name} | {m['iou']:.3f} [{ci(m['ci95']['iou'], 3)}] | {f(m['mae'])} [{ci(m['ci95']['mae'])}] | "
          f"{f(m['bias'], sign=True)} [{ci(m['ci95']['bias'])}] | {m['pearson_r']:.3f} | {m['ccc']:.3f} [{ci(m['ci95']['ccc'], 3)}] | "
          f"{m['grade_correct']}/{m['n']} = {m['grade_acc']:.3f} | {m['qwk']:.2f} [{ci(m['ci95']['qwk'])}] | {hts} | {hvs} |")
w("")
w("Bland–Altman limits of agreement of the baseline (bias ± 1.96 SD): "
  f"rust {f(M['roya']['baseline']['loa'][0])} to {f(M['roya']['baseline']['loa'][1], sign=True)} pp; "
  f"American leaf spot {f(M['ojo']['baseline']['loa'][0])} to {f(M['ojo']['baseline']['loa'][1], sign=True)} pp "
  f"(networks: rust {', '.join(f(M['roya'][a]['loa'][0]) + ' to ' + f(M['roya'][a]['loa'][1], sign=True) for a in ARCHS)}; "
  f"leaf spot {', '.join(f(M['ojo'][a]['loa'][0]) + ' to ' + f(M['ojo'][a]['loa'][1], sign=True) for a in ARCHS)}, for SegFormer-B5, UNet++, DeepLabV3+). "
  "'not run' = no network predictions exist for the healthy validation leaves with the leaf-spot models (inference was not re-run).\n")

w("### Paired differences, network minus baseline (same bootstrap resamples; 95% CI)\n")
w("| Disease | Network | Δ leaf IoU | Leaves with higher network / baseline IoU | Δ MAE (pp) | Δ |bias| (pp) | Δ CCC | Δ QWK |")
w("|---|---|---|---|---|---|---|---|")
for d in DIS:
    for a in ARCHS:
        p = P[d][a]
        star = lambda s: "*" if p[s]["ci_excludes_zero"] else ""
        w(f"| {DN[d]} | {AN[a]} | {f(p['iou']['diff'], 3, True)} [{ci(p['iou']['ci95'], 3)}]{star('iou')} | "
          f"{p['leaves_network_iou_higher']} / {p['leaves_baseline_iou_higher']} of {M[d]['baseline']['n']} | "
          f"{f(p['mae']['diff'], 2, True)} [{ci(p['mae']['ci95'])}]{star('mae')} | {f(p['abs_bias']['diff'], 2, True)} [{ci(p['abs_bias']['ci95'])}]{star('abs_bias')} | "
          f"{f(p['ccc']['diff'], 3, True)} [{ci(p['ccc']['ci95'], 3)}]{star('ccc')} | {f(p['qwk']['diff'], 2, True)} [{ci(p['qwk']['ci95'])}]{star('qwk')} |")
w("")
clear = {d: [a for a in ARCHS if P[d][a]["iou"]["ci_excludes_zero"] and P[d][a]["iou"]["diff"] > 0] for d in DIS}
unclear = {d: [a for a in ARCHS if a not in clear[d]] for d in DIS}
w("\\* = 95% CI excludes zero. Negative Δ MAE or Δ |bias| = the network is closer to the reference. A QWK CI of exactly [0, 0] means the two methods "
  "assigned the same grade to every test leaf. "
  + " ".join(f"{DN[d]}: the IoU difference is clearly positive for {names(clear[d])}"
             + (f"; for {names(unclear[d])} the CI includes zero (" + "; ".join(
                 f"{AN[a]} {f(P[d][a]['iou']['diff'], 3, True)} [{ci(P[d][a]['iou']['ci95'], 3)}], network higher on {P[d][a]['leaves_network_iou_higher']}/{M[d]['baseline']['n']} "
                 f"and baseline higher on {P[d][a]['leaves_baseline_iou_higher']}/{M[d]['baseline']['n']} leaves" for a in unclear[d]) + "), so the threshold is not distinguishable from it in IoU."
                if unclear[d] else ".") for d in DIS) + "\n")

w("### Leaf IoU by reference grade (descriptive)\n")
w("| Disease | Grade | n | Baseline | SegFormer-B5 | UNet++ | DeepLabV3+ |")
w("|---|---|---|---|---|---|---|")
for d in DIS:
    for g, s in T[d]["by_reference_grade"].items():
        w(f"| {DN[d]} | {g} | {s['n']} | {s['baseline']:.3f} | {s['segformer']:.3f} | {s['unetpp']:.3f} | {s['deeplab']:.3f} |")
w("")
w(f"Baseline grade confusion (rows reference G0–G3, columns predicted G0–G3): rust {T['roya']['baseline_confusion_rows_ref_cols_pred_G0_G3']}; "
  f"American leaf spot {T['ojo']['baseline_confusion_rows_ref_cols_pred_G0_G3']}.\n")

# ---------------------------------------------------------------- specificity
w("## 4. Specificity on healthy leaves (both face rules, all flag criteria)\n")
w("Either face = primary (revision brief item 10: maximum over the two faces). Leaf rule = the paper's diseased-leaf rule of the detector's disease "
  "(rust: abaxial face only; leaf spot: mean of faces).\n")
w("| Split (n) | Detector | Either face: any pixel | Either face: ≥ 0.5% (G1+) [Wilson 95% CI of FP rate] | Either face: ≥ 5% | Either face: median / max severity (%) | Leaf rule: any pixel | Leaf rule: ≥ 0.5% |")
w("|---|---|---|---|---|---|---|---|")
for split in ("test", "val"):
    for d in DIS:
        for k in ("baseline", "baseline_constrained", *ARCHS):
            e = SE(split, d, k); el = SE(split, d, k, "leaf_rule")
            if e is None:
                continue
            if k == "baseline_constrained" and same_cfg(CON[d], SEL[d]):
                continue
            g = e["ge_0.5pct"]
            w(f"| {split} ({g['n']}) | {DN[d]} {AN[k]} | {e['any_pixel']['flagged']}/{g['n']} | {g['flagged']}/{g['n']} "
              f"[{100 * g['fp_rate_wilson95'][0]:.1f}–{100 * g['fp_rate_wilson95'][1]:.1f}%] | {e['ge_5pct']['flagged']}/{g['n']} | "
              f"{e['pred_sev_median']:.3f} / {e['pred_sev_max']:.3f} | {el['any_pixel']['flagged']}/{g['n']} | {el['ge_0.5pct']['flagged']}/{g['n']} |")
w("")
loc = S["test"]["baseline_flagged_pixel_location"]


def ge05_list(d):
    return [a for a in ARCHS if SP("test", d, a)["flagged"] > 0]


_parts = []
for d in DIS:
    hit = ge05_list(d)
    if not hit:
        _parts.append(f"No {DSA[d]} network reaches 0.5% on either face.")
    else:
        _parts.append(f"Among the {DSA[d]} networks, " + "; ".join(
            f"{AN[a]} reaches 0.5% on one face of {SP('test', d, a)['flagged']}/21 leaves (maximum {SE('test', d, a)['pred_sev_max']:.2f}%; "
            f"{SP('test', d, a, face='leaf_rule')['flagged']}/21 under the {'abaxial' if d == 'roya' else 'mean-of-faces'} rule)" for a in hit)
            + f"; the other {DSA[d]} networks stay below 0.5% on both faces.")
ge05_txt = " ".join(_parts)
w(f"Where the baseline's lesion pixels fall on the 21 healthy test leaves: {100 * loc['roya']['both_faces']['share_in_margin_band']:.0f}% (rust detector) and "
  f"{100 * loc['ojo']['both_faces']['share_in_margin_band']:.0f}% (leaf-spot detector) of the flagged pixels on both faces lie in the "
  f"{loc['roya']['both_faces']['margin_band_px']}-px leaf-margin band, which is {100 * loc['roya']['both_faces']['margin_band_share_of_leaf_area']:.1f}% of the leaf area "
  f"(rust detector, abaxial faces only: {100 * loc['roya']['leaf_rule_faces']['share_in_margin_band']:.0f}%). The location of the remaining pixels was not quantified.\n")
w("**Specificity definition (consistent with revision brief item 10; not a new finding of this analysis).** The brief already established two points: "
  "the published raw-argmax models predict a few lesion pixels on most healthy leaves, so the manuscript's statement that the zero-false-positive models "
  "left every healthy leaf without any predicted lesion is wrong; and the published 0/54 and 1/54 counts cannot mean 'any lesion pixel' "
  "(`% TODO(author)`: confirm the threshold used). This file uses the brief's either-face rule as the primary healthy-leaf rule for every detector: "
  "rust networks mark lesion pixels on "
  + ", ".join(f"{SP('test', 'roya', a, 'any_pixel')['flagged']}/21" for a in ARCHS) + " local healthy test leaves and leaf-spot networks on "
  + ", ".join(f"{SP('test', 'ojo', a, 'any_pixel')['flagged']}/21" for a in ARCHS)
  + " (SegFormer-B5, UNet++, DeepLabV3+). " + ge05_txt + " "
  "**Face rule matters for the counts:** under the paper's diseased-leaf rule (rust abaxial face only) the rust networks' any-pixel counts are "
  + ", ".join(f"{SP('test', 'roya', a, 'any_pixel', 'leaf_rule')['flagged']}/21" for a in ARCHS)
  + f", and the rust color threshold reaches ≥ 0.5% on {SP('test', 'roya', 'baseline', face='leaf_rule')['flagged']}/21 test and "
  f"{SP('val', 'roya', 'baseline', face='leaf_rule')['flagged']}/21 validation leaves instead of {SP('test', 'roya', 'baseline')['flagged']}/21 and "
  f"{SP('val', 'roya', 'baseline')['flagged']}/21. Any count quoted next to brief-based manuscript text must use the either-face rule, or state the face rule explicitly.\n")

# ---------------------------------------------------------------- secondary
w("## 5. Secondary and sensitivity analyses (not the primary baseline)\n")
w("### 5a. Specificity-constrained threshold (secondary; added after the primary test evaluation, threshold chosen on validation only)\n")
for d in DIS:
    c = V[d]["secondary_specificity_constrained"]
    if same_cfg(CON[d], SEL[d]):
        w(f"- {DN[d]}: the IoU-optimal threshold already leaves all 21 healthy validation leaves below 0.5% on both faces, so the constrained threshold is identical to the primary one.")
    else:
        m = M[d]["baseline_constrained"]
        w(f"- {DN[d]}: highest validation IoU with 0/21 healthy validation leaves ≥ 0.5% on either face = {IDXN[CON[d]['index']]} ≥ {CON[d]['threshold']:g} "
          f"(validation IoU {c['candidates'][CON[d]['index']]['val_iou']:.3f}). Test: IoU {m['iou']:.3f} [{ci(m['ci95']['iou'], 3)}], "
          f"MAE {f(m['mae'])} [{ci(m['ci95']['mae'])}], bias {f(m['bias'], sign=True)} [{ci(m['ci95']['bias'])}], CCC {m['ccc']:.3f}, "
          f"grade accuracy {m['grade_correct']}/{m['n']}, QWK {m['qwk']:.2f} [{ci(m['ci95']['qwk'])}]; healthy test ≥ 0.5% on either face: "
          f"{SP('test', d, 'baseline_constrained')['flagged']}/21 (leaf rule {SP('test', d, 'baseline_constrained', face='leaf_rule')['flagged']}/21). "
          f"SegFormer-B5 minus constrained baseline: Δ IoU {f(PC[d]['segformer']['iou']['diff'], 3, True)} [{ci(PC[d]['segformer']['iou']['ci95'], 3)}], "
          f"Δ MAE {f(PC[d]['segformer']['mae']['diff'], 2, True)} [{ci(PC[d]['segformer']['mae']['ci95'])}].")
        if not same_cfg(CONL[d], CON[d]):
            ml = M[d]["baseline_constrained_leaf_rule"]
            w(f"  - With the paper's leaf rule for the constraint (abaxial face only) the threshold would be {IDXN[CONL[d]['index']]} ≥ {CONL[d]['threshold']:g}: "
              f"test IoU {ml['iou']:.3f} [{ci(ml['ci95']['iou'], 3)}], MAE {f(ml['mae'])}, bias {f(ml['bias'], sign=True)}, QWK {ml['qwk']:.2f}; "
              f"healthy test ≥ 0.5%: {SP('test', d, 'baseline_constrained_leaf_rule', face='leaf_rule')['flagged']}/21 abaxial, "
              f"{SP('test', d, 'baseline_constrained_leaf_rule')['flagged']}/21 on either face.")
w("")
w("### 5b. POST HOC: leaf-margin band excluded from the lesion prediction\n")
w(R["posthoc_margin_excluded"]["note"] + ". Band = " + R["posthoc_margin_excluded"]["band_definition"] + ". 12 px is the leaf erosion of the deployed application at a 1024-px canvas. Threshold re-tuned on validation for each band. Healthy columns: either face (leaf rule).\n")
w("| Disease | Band | Index ≥ threshold | Validation IoU | Test IoU [95% CI] | MAE | Bias | QWK | Healthy test ≥ 0.5% | Healthy val ≥ 0.5% |")
w("|---|---|---|---|---|---|---|---|---|---|")
for k, e in PH.items():
    d = k.split("/")[0]; t = e["test"]
    w(f"| {DN[d]} | {e['band_px']} px | {e['index']} ≥ {fmt_g(e['threshold'])} | {e['val_iou']:.3f} | {t['iou']:.3f} [{ci(t['iou_ci95'], 3)}] | "
      f"{f(t['mae'])} | {f(t['bias'], sign=True)} | {t['qwk']:.2f} | {e['healthy_test']['flagged_ge_0_5pct']}/21 ({e['healthy_test']['flagged_ge_0_5pct_leaf_rule']}/21) | "
      f"{e['healthy_val']['flagged_ge_0_5pct']}/21 ({e['healthy_val']['flagged_ge_0_5pct_leaf_rule']}/21) |")
w("")
w("### 5c. All configurations on test (transparency only; selection used validation alone)\n")
w("| Disease | Configuration | Validation IoU | Test IoU | Test MAE | Test bias | Test QWK | Healthy test ≥ 0.5%: either face (leaf rule) | Healthy test median severity (%): either face (leaf rule) |")
w("|---|---|---|---|---|---|---|---|---|")
for k, e in R["sensitivity_all_configs_on_test"]["results"].items():
    d, cfg = k.split("/")
    w(f"| {DN[d]} | {cfg} | {e['val_iou']:.3f} | {e['test_iou']:.3f} | {f(e['test_mae'])} | {f(e['test_bias'], sign=True)} | {f(e['test_qwk'])} | "
      f"{e['healthy_test_flagged_ge_0_5pct']}/21 ({e['healthy_test_flagged_ge_0_5pct_leaf_rule']}/21) | {e['healthy_test_pred_sev_median']:.2f} ({e['healthy_test_pred_sev_median_leaf_rule']:.2f}) |")
w("")
SA = R["sensitivity_all_configs_on_test"]["results"]
w("`otsu` = per-leaf Otsu (the specified rule); `otsu_face` = per-face Otsu (secondary). The either-face healthy values of the Otsu rows are identical for the two detectors "
  "because Otsu has no disease-specific parameter and, under the either-face rule, is applied to both faces for both detectors. "
  "**Abaxial tuning caveat:** the rust thresholds were tuned on abaxial faces (the rust leaf rule), so the either-face rule also applies them to adaxial faces, "
  "which they were not tuned for; this raises the rust detectors' healthy-leaf values, markedly for a* "
  f"(a* \u2265 {fmt_g(SA['roya/astar_global']['threshold'])}: median {SA['roya/astar_global']['healthy_test_pred_sev_median']:.2f}% on either face vs "
  f"{SA['roya/astar_global']['healthy_test_pred_sev_median_leaf_rule']:.2f}% abaxial) and modestly for the selected excess-red threshold "
  f"({SA['roya/exr_global']['healthy_test_pred_sev_median']:.2f}% vs {SA['roya/exr_global']['healthy_test_pred_sev_median_leaf_rule']:.2f}%).\n")

# ---------------------------------------------------------------- interpretation
mr, mo = M["roya"], M["ojo"]
pr, po = P["roya"], P["ojo"]
hr = SP("test", "roya", "baseline"); hr_l = SP("test", "roya", "baseline", face="leaf_rule")
ho = SP("test", "ojo", "baseline")
m3 = PH["roya/margin3px"]
cnn_max_rust = max(mr[a]["iou"] for a in ("unetpp", "deeplab"))
variants_ok = [(mr["baseline_constrained"]["iou"], SP("test", "roya", "baseline_constrained")["flagged"]),
               (m3["test"]["iou"], m3["healthy_test"]["flagged_ge_0_5pct"]),
               (PH["roya/margin12px"]["test"]["iou"], PH["roya/margin12px"]["healthy_test"]["flagged_ge_0_5pct"])]
none_combines = not any(iou >= min(mr[a]["iou"] for a in ("unetpp", "deeplab")) and fl == 0 for iou, fl in variants_ok)
rust_nets_zero = all(SP("test", "roya", a)["flagged"] == 0 for a in ARCHS)
assert rust_nets_zero and none_combines


def clear_txt(d, P_):
    return " and ".join(f"{f(P_[a]['iou']['diff'], 3, True)}" for a in clear[d])


def unclear_txt(d, P_, texmode=False):
    fm = (lambda c: texci(c, 3)) if texmode else (lambda c: ci(c, 3))
    return "; ".join(f"{f(P_[a]['iou']['diff'], 3, True) if not texmode else tex(P_[a]['iou']['diff'], 3, True)} [{fm(P_[a]['iou']['ci95'])}]" if not texmode else
                     f"{tex(P_[a]['iou']['diff'], 3, True)}, {fm(P_[a]['iou']['ci95'])}" for a in unclear[d])


assert clear["roya"] == clear["ojo"] == ["segformer", "unetpp"] and unclear["roya"] == unclear["ojo"] == ["deeplab"], (clear, unclear)
w("## 6. Where the networks add value (2–3 sentences)\n")
w(f"The networks' advantage is in lesion delineation and, for rust, in keeping healthy leaves below the G1 cut-point. "
  f"SegFormer-B5 and UNet++ had a higher leaf IoU than the validation-tuned color threshold in both diseases, with paired 95% CIs excluding zero "
  f"(rust {f(pr['segformer']['iou']['diff'], 3, True)} and {f(pr['unetpp']['iou']['diff'], 3, True)}; leaf spot {f(po['segformer']['iou']['diff'], 3, True)} and "
  f"{f(po['unetpp']['iou']['diff'], 3, True)}; SegFormer-B5 higher on {pr['segformer']['leaves_network_iou_higher']}/21 and {po['segformer']['leaves_network_iou_higher']}/55 leaves), "
  f"whereas DeepLabV3+ was not distinguishable from it (rust {f(pr['deeplab']['iou']['diff'], 3, True)} [{ci(pr['deeplab']['iou']['ci95'], 3)}], higher on "
  f"{pr['deeplab']['leaves_network_iou_higher']}/21 leaves; leaf spot {f(po['deeplab']['iou']['diff'], 3, True)} [{ci(po['deeplab']['iou']['ci95'], 3)}], with the threshold higher on "
  f"{po['deeplab']['leaves_baseline_iou_higher']}/55 leaves); on the 21 local healthy test leaves no rust network reached 0.5% predicted severity on either face, "
  f"while the IoU-tuned rust threshold did on {hr['flagged']}/21 ({hr_l['flagged']}/21 on the abaxial face alone), and no color-threshold variant evaluated combined "
  f"convolutional-model-level rust IoU with zero such leaves (specificity-constrained: IoU {mr['baseline_constrained']['iou']:.3f}, "
  f"{SP('test', 'roya', 'baseline_constrained')['flagged']}/21; margin-excluded, post hoc: IoU {m3['test']['iou']:.3f}, {m3['healthy_test']['flagged_ge_0_5pct']}/21). "
  f"For severity and grade alone the tuned threshold was competitive (rust MAE {f(mr['baseline']['mae'])} vs {f(min(mr[a]['mae'] for a in ARCHS))}–{f(max(mr[a]['mae'] for a in ARCHS))} pp, "
  f"QWK {mr['baseline']['qwk']:.2f} vs {min(mr[a]['qwk'] for a in ARCHS):.2f}–{max(mr[a]['qwk'] for a in ARCHS):.2f}; leaf spot MAE {f(mo['baseline']['mae'])} vs "
  f"{f(min(mo[a]['mae'] for a in ARCHS))}–{f(max(mo[a]['mae'] for a in ARCHS))} pp, QWK {mo['baseline']['qwk']:.2f} vs {min(mo[a]['qwk'] for a in ARCHS):.2f}–{max(mo[a]['qwk'] for a in ARCHS):.2f}), "
  "which is partly expected because the rust reference masks were seeded by a color detector before SAM 2 refinement and because the baseline's severity "
  "denominator is the exact reference leaf area.\n")

# count MAE/QWK comparisons whose paired CI excludes zero
excl = [(d, a, s) for d in DIS for a in ARCHS for s in ("mae", "qwk") if P[d][a][s]["ci_excludes_zero"]]
w("## 7. Suggested manuscript sentences (LaTeX-ready, precise, non-overclaiming)\n")
w("1. *Methods (Section \\ref{sec:evaluation}).* As a training-free reference, three per-pixel color indices (the redness index $(r-g)/(r+g+b)$, CIELAB $a^*$ and excess red "
  "$1.4r'-g'$ on chromatic coordinates) were thresholded inside the reference leaf mask (alpha $>$ 8), in the same 1024 $\\times$ 1024 letterbox space used to evaluate the networks, "
  "with either a per-leaf Otsu threshold or one global threshold per disease chosen on the validation diseased leaves to maximize mean leaf lesion IoU; the configuration "
  f"with the highest validation IoU (rust: excess red $\\geq$ {SEL['roya']['threshold']:g}; American leaf spot: redness index $\\geq$ {SEL['ojo']['threshold']:g}) was applied once "
  "to the test leaves, without post-processing, and scored with the same leaf-level rules and bootstrap procedure. Because it used the reference leaf mask, the baseline's "
  "severity denominator was the exact leaf area, whereas the networks segment the leaf themselves.\n")
w(f"2. *Results.* The color threshold reached a test leaf IoU of {mr['baseline']['iou']:.3f} (95\\% CI {texci(mr['baseline']['ci95']['iou'], 3)}) for rust and "
  f"{mo['baseline']['iou']:.3f} ({texci(mo['baseline']['ci95']['iou'], 3)}) for American leaf spot; SegFormer-B5 and UNet++ exceeded it for both diseases "
  f"(paired differences {tex(pr['segformer']['iou']['diff'], 3, True)} and {tex(pr['unetpp']['iou']['diff'], 3, True)} for rust and {tex(po['segformer']['iou']['diff'], 3, True)} and "
  f"{tex(po['unetpp']['iou']['diff'], 3, True)} for American leaf spot, all 95\\% CIs excluding zero), whereas DeepLabV3+ was not distinguishable from it "
  f"(rust {tex(pr['deeplab']['iou']['diff'], 3, True)}, 95\\% CI {texci(pr['deeplab']['iou']['ci95'], 3)}; American leaf spot {tex(po['deeplab']['iou']['diff'], 3, True)}, {texci(po['deeplab']['iou']['ci95'], 3)}) (Table~X).\n")
if len(excl) == 0:
    excl_txt = "that included zero in every comparison"
elif len(excl) == 1:
    d, a, s = excl[0]
    excl_txt = (f"that included zero in every comparison but one ({'American leaf spot' if d == 'ojo' else 'rust'} {AN[a]}, "
                + (f"MAE {'lower' if P[d][a]['mae']['diff'] < 0 else 'higher'} by {abs(P[d][a]['mae']['diff']):.2f} percentage points)" if s == "mae" else
                   f"QWK {'higher' if P[d][a]['qwk']['diff'] > 0 else 'lower'} by {abs(P[d][a]['qwk']['diff']):.2f})"))
else:
    excl_txt = f"that excluded zero in {len(excl)} of 12 comparisons"
w(f"3. *Results.* Its severity and grade agreement were similar to those of the networks (rust: MAE {tex(mr['baseline']['mae'])} vs {tex(min(mr[a]['mae'] for a in ARCHS))}--{tex(max(mr[a]['mae'] for a in ARCHS))} percentage points, "
  f"QWK {mr['baseline']['qwk']:.2f} vs {min(mr[a]['qwk'] for a in ARCHS):.2f}--{max(mr[a]['qwk'] for a in ARCHS):.2f}; American leaf spot: MAE {tex(mo['baseline']['mae'])} vs "
  f"{tex(min(mo[a]['mae'] for a in ARCHS))}--{tex(max(mo[a]['mae'] for a in ARCHS))}, QWK {mo['baseline']['qwk']:.2f} vs {min(mo[a]['qwk'] for a in ARCHS):.2f}--{max(mo[a]['qwk'] for a in ARCHS):.2f}), "
  f"with paired bootstrap intervals for the MAE and QWK differences {excl_txt}, although with 21 rust test leaves these intervals are wide and do not establish equivalence.\n")
ls_txt = ("while the leaf-spot color threshold kept all 21 healthy leaves below 0.5\\% on both faces (0/21)" if ho["flagged"] == 0 else
          f"while the leaf-spot color threshold reached 0.5\\% on either face of {ho['flagged']}/21 healthy leaves")
w(f"4. *Results.* On the 21 healthy rust-pack test leaves available locally (the full test partition held 54), the rust color threshold predicted a severity of at least 0.5\\% "
  f"(grade G1 or higher) on at least one face of {'all 21' if hr['flagged'] == 21 else str(hr['flagged'])} leaves (median of the per-leaf maximum "
  f"{SE('test', 'roya', 'baseline')['pred_sev_median']:.2f}\\%), whereas none of the three rust networks did on either face; in a secondary analysis, raising the threshold on validation "
  f"until no healthy validation leaf reached 0.5\\% on either face removed these flags ({SP('test', 'roya', 'baseline_constrained')['flagged']}/21) but lowered rust IoU to "
  f"{mr['baseline_constrained']['iou']:.3f} with a severity bias of {tex(mr['baseline_constrained']['bias'], sign=True)} percentage points, {ls_txt}.\n")
w("5. *Discussion/Limitations.* The networks' advantage over a validation-tuned color threshold thus lies mainly in lesion delineation, clearly so for SegFormer-B5 and UNet++, "
  "and, for rust, in keeping healthy leaves below the G1 cut-point, rather than in severity or grade agreement; the comparison favors the color baseline, because the rust "
  "reference masks were seeded by a color detector before SAM~2 refinement and the baseline used the reference leaf mask, and the rust healthy-leaf contrast is confounded "
  "with site (EARTH vs Sarch\\'i).\n")
w("Face rule for sentence 4: counts use the either-face rule of revision brief item 10 (a leaf is flagged if either face reaches the criterion), so they can sit next to "
  f"the brief-based specificity text; under the abaxial-only rule the rust threshold count is {hr_l['flagged']}/21.\n")

w("## 8. Suggested table and figure captions\n")
w("```latex")
w("% Table: training-free color threshold vs networks (test leaves). Healthy = 21 local healthy test leaves (paper: 54); >= 0.5% predicted severity on either face.")
w("\\begin{tabular}{llccccccc}")
w("\\toprule")
w("Disease & Method & Leaf IoU & MAE (pp) & Bias (pp) & CCC & Grade acc. & QWK & Healthy $\\geq$G1 \\\\")
w("\\midrule")
for d in DIS:
    for k in ("baseline", *ARCHS):
        m = M[d][k]; ht = SP("test", d, k)
        nm = "Color threshold" if k == "baseline" else AN[k]
        w(f"{DN[d] if k == 'baseline' else ''} & {nm} & {m['iou']:.3f} ({texci(m['ci95']['iou'], 3)}) & {tex(m['mae'])} ({texci(m['ci95']['mae'])}) & "
          f"{tex(m['bias'], sign=True)} & {m['ccc']:.3f} & {m['grade_correct']}/{m['n']} & {m['qwk']:.2f} ({texci(m['ci95']['qwk'])}) & {ht['flagged']}/{ht['n']} \\\\")
    w("\\midrule" if d == "roya" else "\\bottomrule")
w("\\end{tabular}")
w("```\n")
ov = json.load(open(f"{OUT}/overlay_examples.json")) if os.path.exists(f"{OUT}/overlay_examples.json") else {"examples": {}}
mags = sorted({v["zoom_inset"]["magnification_vs_panel"] for v in ov["examples"].values() if "zoom_inset" in v})
win_px = sorted({v["zoom_inset"]["window_px"] for v in ov["examples"].values() if "zoom_inset" in v})[0]
mag_txt = f"about {round(min(mags))}--{round(max(mags))}$\\times$" if mags and round(min(mags)) != round(max(mags)) else (f"about {round(mags[0])}$\\times$" if mags else "")
w("**Figure (figure_baseline_comparison).** Training-free color threshold versus the three networks on the test leaves. Points are the leaf-level mean lesion IoU (top), "
  "severity MAE in percentage points (middle) and the percentage of the 21 healthy rust-pack test leaves available locally with a predicted severity of at least 0.5\\% "
  "(grade G1 or higher) on either face (bottom). Bars are 95\\% percentile bootstrap intervals over leaves (10,000 resamples) for IoU and MAE and 95\\% Wilson intervals for the healthy-leaf rate. "
  f"Filled grey diamonds: color threshold selected on validation (rust: excess red $\\geq$ {SEL['roya']['threshold']:g}; American leaf spot: redness index $\\geq$ {SEL['ojo']['threshold']:g}). "
  f"Open diamond (rust only): threshold raised on validation until no healthy validation leaf reached 0.5\\% on either face (excess red $\\geq$ {CON['roya']['threshold']:g}); "
  "for American leaf spot this constraint did not change the threshold. Rust: abaxial face; American leaf spot: mean of both faces. Created by the authors.\n")
w("**Figure (figure_baseline_overlays).** Abaxial faces of three test leaves per disease: the leaf with the median color-threshold IoU, the leaf with the largest absolute IoU "
  "difference between the color threshold and SegFormer-B5, and the healthy leaf with the median color-threshold predicted severity (paper's leaf rule). Top: scan with the reference lesion outline (magenta). "
  "Middle and bottom: color-threshold and SegFormer-B5 predictions on the grey leaf (yellow, overlap with the reference; pink, predicted only; blue, reference only). "
  f"Healthy leaves: the inset enlarges ({mag_txt}) the boxed {win_px} $\\times$ {win_px}-pixel window that contains the most color-threshold lesion pixels on that face. "
  "Values are face-level IoU and predicted severity (the leaf-spot leaf-level values average both faces). Created by the authors.\n")

w("## 9. Notes and caveats\n")
w("- **Healthy coverage:** 21 of the paper's 54 healthy rust-test leaves are present locally; all healthy-leaf numbers here refer to those 21 (plus 21 validation leaves). All healthy leaves are from EARTH (collected 12, 15 and 22 June 2026); diseased rust leaves are from Sarchí, so rust healthy-vs-diseased contrasts are confounded with site and scan session.")
w("- **Healthy-leaf face rule:** the primary rule is the brief's either-face rule (item 10); the paper's diseased-leaf rule (rust abaxial only) gives lower counts for the rust detectors (Section 4). Quote counts with their face rule. The rust color thresholds were tuned on abaxial faces only, so the either-face rule also applies them to adaxial faces (Section 5c).")
w("- **Circularity (rust):** every rust reference mask was produced by a color/redness detector followed by SAM 2 box refinement, unedited. A color-threshold baseline therefore shares part of the reference-generation rule; its rust IoU, MAE and QWK are likely optimistic relative to an independent hand-drawn reference. Leaf-spot masks were detector-seeded but manually corrected in CVAT.")
w("- **Tuning set:** thresholds were tuned on validation diseased leaves only (leaf IoU is undefined on healthy leaves). The rust CNNs, by contrast, were selected with a healthy-leaf false-positive penalty on validation.")
w("- **Leaf mask:** the baseline used the reference leaf mask (exact denominator); the networks segment the leaf themselves. This favors the baseline slightly on severity. The manuscript sentences state this once in Methods (sentence 1) and once in Limitations (sentence 5).")
w("- **Ties:** redness index and excess red thresholds are compared exactly in integer arithmetic (Section 2); a float64 evaluation of 1.4r/s − g/s puts some exact ties on either side of the threshold depending on rounding.")
w("- **Otsu:** per-leaf Otsu (as specified) and per-face Otsu both failed for both diseases (Otsu always splits the leaf), with gross over-estimation of severity on test and on every healthy leaf (Section 5c).")
w("- **Post-hoc analysis:** the margin-exclusion variant (5b) was added after seeing where the primary baseline's healthy-leaf false positives fell and is labeled as post hoc.")
w("")

# ---------------------------------------------------------------- revision log
cr = M["roya"]["baseline"]
w("## 10. Changes from the previous version of this folder\n")
w(f"1. **DeepLabV3+ wording (Sections 6 and 7).** The text no longer says that both convolutional models exceeded the threshold: only SegFormer-B5 and UNet++ have paired IoU CIs above zero; DeepLabV3+ is not distinguishable from the threshold (CIs include zero for both diseases).")
w(f"2. **Healthy-leaf face rule reconciled with revision brief item 10.** Primary rule is now either face (maximum over both faces) for every detector; the paper's leaf rule is reported alongside. "
  f"Rust color threshold ≥ 0.5%: {SP('test', 'roya', 'baseline')['flagged']}/21 test and {SP('val', 'roya', 'baseline')['flagged']}/21 validation (previously {PREVIOUS['rust_healthy_test']} and {PREVIOUS['rust_healthy_val']}, abaxial). "
  f"Rust networks, any pixel: " + ", ".join(f"{SP('test', 'roya', a, 'any_pixel')['flagged']}/21" for a in ARCHS) + f" (previously {PREVIOUS['rust_net_any_px']}, abaxial). "
  "The specificity note now cites the brief instead of presenting the any-pixel result as new. Under the either-face rule, leaf-spot "
  + (", ".join(f"{AN[a]} reaches 0.5% on one face of {SP('test', 'ojo', a)['flagged']}/21 healthy leaves" for a in ge05_list("ojo")) or "no network reaches 0.5% on either face")
  + " (0/21 with the mean of faces, as before).")
if same_cfg(CON["roya"], CONL["roya"]) and f"{CON['roya']['threshold']:g}" == PREVIOUS["constrained_thr"]:
    w(f"3. **Specificity-constrained threshold** re-chosen on validation under the either-face rule: the result is the same threshold as before (excess red \u2265 {CON['roya']['threshold']:g}), "
      f"so its test results are unchanged (IoU {mr['baseline_constrained']['iou']:.3f}, bias {f(mr['baseline_constrained']['bias'], sign=True)}, "
      f"{SP('test', 'roya', 'baseline_constrained')['flagged']}/21 healthy test leaves \u2265 0.5% on either face).")
else:
    w(f"3. **Specificity-constrained threshold** re-chosen on validation under the either-face rule: excess red \u2265 {CON['roya']['threshold']:g} "
      f"(previously {PREVIOUS['constrained_thr']} under the abaxial rule); test IoU {mr['baseline_constrained']['iou']:.3f}, bias {f(mr['baseline_constrained']['bias'], sign=True)} "
      f"(previously {PREVIOUS['constrained_iou']}, {PREVIOUS['constrained_bias']}). The abaxial-rule result is kept in 5a.")
w("4. **Sentence 5 / Section 6 wording:** 'rejecting healthy tissue' replaced by 'keeping healthy leaves below the G1 cut-point' (under the any-pixel rule the rust networks also mark lesion pixels on most healthy leaves).")
w("5. **Methods sentence 1:** now says the indices were thresholded inside the reference leaf mask, in the same letterbox space, and states the exact-denominator advantage; sentence 5 repeats it once as a limitation.")
pd_ = P["roya"]["deeplab"]; pu_ = P["roya"]["unetpp"]
w(f"6. **Exact tie handling** for the redness index and excess red (Section 2). Selected thresholds, validation IoUs of the global rules, test IoU, grade accuracy, QWK and the confusion matrices "
  "are unchanged. Values that changed at the reported precision (all rust; the leaf-spot threshold has a single tie pixel): "
  f"baseline bias {f(cr['bias'], sign=True)} [upper CI {f(cr['ci95']['bias'][1])}] (previously {PREVIOUS['rust_bias']} [{PREVIOUS['rust_bias_hi']}]), "
  f"upper LoA {f(cr['loa'][1], sign=True)} (previously {PREVIOUS['rust_loa_hi']}), Pearson r {cr['pearson_r']:.3f} (previously {PREVIOUS['rust_r']}), "
  f"CCC {cr['ccc']:.3f} [lower CI {cr['ci95']['ccc'][0]:.3f}] (previously {PREVIOUS['rust_ccc']} [{PREVIOUS['rust_ccc_lo']}]), MAE upper CI {f(cr['ci95']['mae'][1])} (previously {PREVIOUS['rust_mae_hi']}); "
  f"paired \u0394 IoU UNet++ {f(pu_['iou']['diff'], 3, True)} (previously {PREVIOUS['d_iou_unetpp']}) and DeepLabV3+ {f(pd_['iou']['diff'], 3, True)} [{ci(pd_['iou']['ci95'], 3)}] "
  f"(previously {PREVIOUS['d_iou_deeplab']} [upper {PREVIOUS['d_iou_deeplab_hi']}]), with DeepLabV3+ higher on {pd_['leaves_network_iou_higher']}/21 and the threshold on "
  f"{pd_['leaves_baseline_iou_higher']}/21 leaves (previously {PREVIOUS['deeplab_higher']} and {PREVIOUS['deeplab_lower']}; the leaf that changed side, GX_009, is a near-tie: "
  "DeepLabV3+ 0.7584 vs threshold 0.7588, previously 0.7583); second-decimal shifts in the rust \u0394 MAE, \u0394 |bias| and \u0394 CCC "
  f"cells of the paired table; baseline IoU of the single G0 rust leaf {T['roya']['by_reference_grade']['G0']['baseline']:.3f} (previously {PREVIOUS['g0_iou']}); "
  f"post-hoc 12-px rust bias {f(PH['roya/margin12px']['test']['bias'], sign=True)} (previously {PREVIOUS['m12_bias']}); rust healthy-test median / max severity under the abaxial rule "
  f"{SE('test', 'roya', 'baseline', 'leaf_rule')['pred_sev_median']:.3f} / {SE('test', 'roya', 'baseline', 'leaf_rule')['pred_sev_max']:.3f} (previously {PREVIOUS['healthy_med_max']}).")
w(f"7. **Per-leaf Otsu** (as specified) replaces per-face Otsu, which is kept as a secondary row; leaf-spot per-leaf Otsu validation IoU: "
  + ", ".join(f"{k.split('_')[0]} {V['ojo']['configs'][k]['val_iou']:.3f}" for k in ('ri_otsu', 'astar_otsu', 'exr_otsu')) + f" (per-face, previous version: {PREVIOUS['otsu_ojo_val']}). Otsu remains by far the weakest rule.")
w(f"8. **Figures.** Comparison: marker edges now match the marker fill (no white edge cutting the CI bars), the color threshold is drawn in neutral dark grey (no longer close to SegFormer-B5 blue), "
  "and the bottom row uses the either-face rule. Overlays: the healthy-leaf panels carry a zoom inset so the sparse false-positive pixels are visible at print size.")
w(f"9. **Margin-band location sentence:** now computed on both faces ({100 * loc['roya']['both_faces']['share_in_margin_band']:.0f}% rust detector, "
  f"{100 * loc['ojo']['both_faces']['share_in_margin_band']:.0f}% leaf-spot detector; previously {PREVIOUS['share_rust']} rust abaxial-only and {PREVIOUS['share_ojo']}); the unquantified "
  "'cut petiole and midrib' claim and the '(overlay figure)' reference were removed.")
w("")
w("## Files\n")
for fn in ("baseline.py", "make_figures.py", "make_md.py", "results.json", "results.md", "per_leaf_test.csv", "per_leaf_healthy.csv",
           "validation_curves.csv", "overlay_examples.json", "figure_baseline_comparison.pdf", "figure_baseline_comparison.png",
           "figure_baseline_overlays.pdf", "figure_baseline_overlays.png"):
    w(f"- `{fn}`")
open(f"{OUT}/results.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("wrote results.md")
