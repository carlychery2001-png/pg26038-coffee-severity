"""Write results.md from results.json (A5 rust operating point). All numbers come from results.json."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results.json")))
M = R["models"]
PUB = ["pub_segformer", "pub_unetpp", "pub_deeplab"]
MAT = ["m_unetpp_s0", "m_unetpp_s1", "m_unetpp_s2", "m_deeplab_s0", "m_deeplab_s1", "m_deeplab_s2"]
OPS = ["argmax", "thr_bias", "thr_mae", "ero1", "ero2"]
OPN = {"argmax": "argmax (paper rule)", "thr_bias": "threshold, val min abs(bias)", "thr_mae": "threshold, val min MAE",
       "ero1": "argmax + 1-px erosion", "ero2": "argmax + 2-px erosion"}


def n(x, d=2, sign=False):
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "−")


def ft(t): return f"{t:g}"


def ci(c, d=2, sign=False): return f"[{n(c[0], d, sign)}, {n(c[1], d, sign)}]"


def excl0(c): return c[0] > 0 or c[1] < 0


def verdict(m, op):
    """disappears / shrinks / persists, separately for bias and slope (test, bootstrap CIs)."""
    t = M[m]["test"]; a = t["argmax"]; o = t[op]; p = M[m]["test_paired_vs_argmax"][op]
    out = {}
    for k in ("bias", "slope"):
        if not excl0(a[f"{k}_ci95_bootstrap"]):
            out[k] = "not significant at argmax already; " + ("now significantly negative" if excl0(o[f"{k}_ci95_bootstrap"]) and o[k] < 0
                                                              else "now significantly positive" if excl0(o[f"{k}_ci95_bootstrap"]) else "still not significant")
            continue
        if not excl0(o[f"{k}_ci95_bootstrap"]):
            v = "disappears (CI includes 0)"
            if o[k] < 0 and a[k] > 0 and abs(o[k]) > 0.25 * abs(a[k]):
                v = "disappears (CI includes 0; point estimate reversed sign)"
        elif o[k] < 0 < a[k]:
            v = "reversed (now significant under-estimation)"
        elif excl0(p[k]["ci95_bootstrap"]) and p[k]["diff"] < 0:
            v = "shrinks (still > 0)"
        else:
            v = "persists"
        out[k] = v
    return out


L = []
w = L.append
w("# A5: Rust operating point (is the rust over-estimation an operating-point artifact?)\n")
w("PG-26038 major revision, round 4, 28 September 2026. Scripts: `infer_prob.py` (CPU inference with onnxruntime that keeps the "
  "softmax lesion probability), `oppoint.py` (all analyses, writes `results.json` and `work/*`), `make_md.py` (this file). "
  "**No model was trained or fine-tuned.** Everything below re-uses the existing ONNX models and changes only the decision rule "
  "applied to their output.\n")
w("## Method\n")
w("- **Models.** Published rust SegFormer-B5, UNet++ and DeepLabV3+ (`modal-recovery/serving/*roya*.onnx`) and the matched-protocol "
  "rust CNNs (UNet++ and DeepLabV3+, lrB = 6e-5/6e-4, seeds 0, 1, 2; `reanalysis/kaggle_runs/*roya*lrB*/output/*.onnx`): 9 models.")
w("- **Images.** Rust pack, abaxial face only (the rust leaf-level rule): 21 diseased validation leaves, 21 diseased test leaves, "
  "and the 21 locally available healthy test leaves (specificity check only).")
w("- **Protocol.** Identical to `run_inference.py`: leaf = alpha > 8; lesion = mask > 127 AND leaf; aspect-preserving letterbox to "
  "1024 × 1024 with black padding; ImageNet normalization; SegFormer logits upsampled bilinearly (PIL) to 1024 before any decision; "
  "metrics in letterbox space. Rust scans are 791 × 1024 px, so the letterbox scale is 1.0 and 1 px = 1 native scan pixel "
  "(the physical pixel size was not recorded, so no distance in mm is given). The softmax over the three classes "
  "(background, leaf, lesion) was stored per pixel (lesion probability quantized to 1/65535).")
w("- **Operating points.** (i) *argmax*: the published rule. (ii) *Validation-calibrated threshold*: a pixel is lesion if "
  "p(lesion) ≥ t, otherwise it takes the argmax of background vs leaf; t was searched on the grid 0.01–0.99 (step 0.01), extended by "
  "0.991–0.999 (step 0.001), 0.9995 and 0.9999, on the 21 **validation** leaves (the extension was added after a preview run put "
  "the validation optimum of matched UNet++ seed 1 on the former upper edge 0.99 with validation bias still positive; the reason "
  "is the validation curve, but the preview's test results for the eight CNNs had already been computed, which we disclose), once to minimize |bias| and once to minimize MAE, and each chosen t was then applied **once** to the test "
  "leaves. (iii) *Erosion*: the argmax lesion mask eroded by 1 px or 2 px (3 × 3 structuring element, 1 or 2 iterations); eroded "
  "pixels become leaf, so the leaf denominator is unchanged. The erosion depth was fixed a priori (not tuned); validation values "
  "are reported for reference.")
w("- **Endpoints (21 test leaves).** Mean leaf lesion IoU, MAE, bias (D = predicted − reference severity, percentage points, pp), "
  "grade accuracy and QWK (mask-derived rust grades, cut-points 0.5/5/20%), and the proportional-bias slope: OLS regression of D on "
  "A = (predicted + reference)/2, as in `results/agreement` Section 4, with t-based and bootstrap 95% CIs. Paired bootstrap "
  "differences (operating point − argmax) test whether a change is larger than resampling noise.")
w("- **Signed boundary distance (test leaves).** For every predicted lesion contour pixel (lesion mask minus its 3 × 3 erosion): "
  "the Euclidean distance to the reference lesion contour, **positive when the pixel lies outside the reference lesion** "
  "(sd = EDT to the reference lesion if outside; sd = 1 − EDT to the non-lesion region if inside). A pixel on the reference "
  "contour has sd = 0, the first pixel ring outside +1, the second ring inside −1. Distances are pooled over all contour pixels of "
  "the 21 leaves (\"all\"); also given for predicted components that touch a reference lesion only (\"attached\", excludes "
  "spurious blobs), and in the reverse direction (each reference contour pixel relative to the prediction; + = the prediction "
  "extends beyond the reference there). Distances take discrete values (0, 1, 1.41, 2, …), so medians and their CIs are coarse; "
  "the mean is also reported.")
w("- **Bootstrap.** 10,000 resamples of the 21 test leaves, seed 20260928, percentile 95% CIs; the same resamples for every model "
  "and operating point (paired). Bootstrap CIs here therefore differ slightly from `results/agreement` (seed 20260927); point "
  "estimates are identical.\n")

w("## 1. Reproduction of the baseline (argmax)\n")
rep = R["reproduction_argmax"]
w(f"All argmax pixel counts (leaf, lesion, intersection, union for every model × image, 567 pairs) were checked against "
  f"`out/per_image.csv` / `out/per_image_retrained.csv` during inference and matched exactly (the script stops on any mismatch). "
  f"Leaf-level reproduction: all match = **{R['reproduction_all_match']}**.\n")
w("| Model | Bias here | Bias `results/agreement` | Slope here [t CI] | Slope `results/agreement` [t CI] |")
w("|---|---|---|---|---|")
for m in PUB:
    r = rep[m]
    w(f"| {M[m]['label']} | {n(r['bias_here'], 3, True)} | {n(r['bias_agreement'], 3, True)} | {n(r['slope_here'], 3)} {ci(r['slope_ci_t_here'], 3)} | "
      f"{n(r['slope_agreement'], 3)} {ci(r['slope_ci_t_agreement'], 3)} |")
w("")
w("| Matched model | Test IoU here / `retrained_summary.json` | Test bias here / summary | Val IoU here / summary |")
w("|---|---|---|---|")
for m in MAT:
    r = rep[m]
    w(f"| {M[m]['label']} | {n(r['test_iou_here'], 3)} / {n(r['test_iou_summary'], 3)} | {n(r['test_bias_here'], 2, True)} / "
      f"{n(r['test_bias_summary'], 2, True)} | {n(r['val_iou_here'], 3)} / {n(r['val_iou_summary'], 3)} |")
w("")

w("## 2. Thresholds chosen on validation (21 leaves)\n")
w("| Model | t (min \\|bias\\|) | t (min MAE) | Val bias: argmax → t_bias | Val MAE: argmax → t_MAE | Val slope: argmax → t_bias | "
  "Val bias at t = 0.50 | Val bias: 1-px / 2-px erosion |")
w("|---|---|---|---|---|---|---|---|")
for m in PUB + MAT:
    v = M[m]["validation"]; t = M[m]["thresholds_chosen_on_validation"]
    w(f"| {M[m]['label']} | {ft(t['min_abs_bias'])} | {ft(t['min_mae'])} | {n(v['argmax']['bias'], 2, True)} → {n(v['thr_bias']['bias'], 2, True)} | "
      f"{n(v['argmax']['mae'])} → {n(v['thr_mae']['mae'])} | {n(v['argmax']['slope'], 3)} → {n(v['thr_bias']['slope'], 3)} | "
      f"{n(v['thr_0.50']['bias'], 2, True)} | {n(v['ero1']['bias'], 2, True)} / {n(v['ero2']['bias'], 2, True)} |")
w("\nt = 0.50 is not the argmax rule (argmax calls lesion when p(lesion) exceeds both other classes), so it is listed separately. "
  "Full validation curves (IoU, MAE, bias, slope for every t): `work/val_curves.csv`.\n")

w("## 3. Test endpoints (21 test leaves, abaxial face)\n")
w("Bias, MAE and IoU with bootstrap 95% CIs; slope with t-based CI and bootstrap CI. Over = leaves over-graded (none under-graded "
  "unless stated in the last column).\n")
w("| Model | Operating point | t | Leaf IoU [CI] | MAE (pp) [CI] | Bias (pp) [CI] | Slope [t CI]; bootstrap CI | p (slope) | Grade acc | QWK | Over / under-graded | Leaves over-estimated |")
w("|---|---|---|---|---|---|---|---|---|---|---|---|")
for m in PUB + MAT:
    for op in OPS:
        s = M[m]["test"][op]
        tt = ft(s["threshold"]) if "threshold" in s else "—"
        pv = "<0.001" if s["slope_p"] < 0.001 else n(s["slope_p"], 3)
        w(f"| {M[m]['label']} | {OPN[op]} | {tt} | {n(s['leaf_iou'], 3)} {ci(s['leaf_iou_ci95_bootstrap'], 3)} | {n(s['mae'])} {ci(s['mae_ci95_bootstrap'])} | "
          f"{n(s['bias'], 2, True)} {ci(s['bias_ci95_bootstrap'], 2, True)} | {n(s['slope'], 3)} {ci(s['slope_ci95_t'], 3)}; {ci(s['slope_ci95_bootstrap'], 3)} | {pv} | "
          f"{n(s['grade_acc'], 3)} | {n(s['qwk'], 3)} | {s['n_over_graded']} / {s['n_under_graded']} | {s['n_over_estimated']}/21 |")
w("")

w("## 4. Paired change versus argmax (test; operating point − argmax; bootstrap 95% CI)\n")
w("| Model | Operating point | Δ bias (pp) [CI] | Δ MAE (pp) [CI] | Δ slope [CI] | Δ leaf IoU [CI] | Bias verdict | Slope verdict |")
w("|---|---|---|---|---|---|---|---|")
VER = {}
for m in PUB + MAT:
    for op in OPS[1:]:
        p = M[m]["test_paired_vs_argmax"][op]; v = verdict(m, op); VER[(m, op)] = v
        w(f"| {M[m]['label']} | {OPN[op]} | {n(p['bias']['diff'], 2, True)} {ci(p['bias']['ci95_bootstrap'], 2, True)} | "
          f"{n(p['mae']['diff'], 2, True)} {ci(p['mae']['ci95_bootstrap'], 2, True)} | {n(p['slope']['diff'], 3, True)} {ci(p['slope']['ci95_bootstrap'], 3, True)} | "
          f"{n(p['leaf_iou']['diff'], 3, True)} {ci(p['leaf_iou']['ci95_bootstrap'], 3, True)} | {v['bias']} | {v['slope']} |")
w("\nVerdict rule (fixed before reading the numbers): *disappears* = the operating point's bootstrap CI includes 0; *shrinks* = its "
  "CI still excludes 0 but the paired change is negative with a CI that excludes 0; *persists* = otherwise; *reversed* = the CI "
  "excludes 0 on the negative side (significant under-estimation). Where the argmax CI already includes 0, the verdict says so "
  "(matched DeepLabV3+ seed 2 has no significant argmax bias or slope).\n")

w("## 5. Matched CNNs: mean ± SD over 3 seeds (range)\n")
w("| Model | Operating point | Thresholds (s0, s1, s2) | Leaf IoU | MAE (pp) | Bias (pp) | Slope | QWK |")
w("|---|---|---|---|---|---|---|---|")
for arch, lab in (("unetpp", "UNet++ matched"), ("deeplab", "DeepLabV3+ matched")):
    for op in OPS:
        a = R["matched_seed_aggregate"][arch][op]
        f = lambda k, d=2, sg=False: f"{n(a[k]['mean'], d, sg)} ± {n(a[k]['sd'], d)} ({n(a[k]['min'], d, sg)} to {n(a[k]['max'], d, sg)})"
        th = ", ".join(ft(x) for x in a["thresholds"]) if "thresholds" in a else "—"
        w(f"| {lab} | {OPN[op]} | {th} | {f('leaf_iou', 3)} | {f('mae')} | {f('bias', 2, True)} | {f('slope', 3)} | {f('qwk', 3)} |")
w("")

w("## 6. Healthy test leaves (21 local, abaxial face)\n")
w("| Model | Operating point | Leaves with ≥ 1 predicted lesion px | Leaves ≥ 0.5% | Leaves ≥ 5% | Max (%) |")
w("|---|---|---|---|---|---|")
for m in PUB + MAT:
    for op in OPS:
        h = M[m]["healthy_test_abaxial"][op]
        w(f"| {M[m]['label']} | {OPN[op]} | {h['any_pixel']}/21 | {h['ge_0.5pct']}/21 | {h['ge_5pct']}/21 | {n(h['max_pct'], 3)} |")
w("\nOnly 21 of the 54 healthy rust-test leaves of the paper are available locally (revision brief, fact 8); counts are abaxial face "
  "only (the rust leaf-level rule), which is why the argmax any-pixel counts equal `results/agreement` Section 10 (abaxial) rather "
  "than the both-face numbers of fact 10.\n")

w("## 7. Signed boundary distance, predicted vs reference lesion contours (test, 21 leaves; px; + = prediction outside reference)\n")
w("| Model | Operating point | Contour px | Median [CI] | IQR (Q1 to Q3) | Mean [CI] | Share > 0 | Share ≥ 2 px outside | Share ≥ 2 px inside | Attached only: median; IQR; mean | Reverse (ref contour): median; IQR; mean |")
w("|---|---|---|---|---|---|---|---|---|---|---|")
for m in PUB + MAT:
    for op in OPS:
        b = M[m]["boundary_test"][op]; a = b["pred_all"]; t = b["pred_att"]; r = b["rev"]
        w(f"| {M[m]['label']} | {OPN[op]} | {a['n_contour_px']} | {n(a['median'])} {ci(a['median_ci95_bootstrap'])} | {n(a['q25'])} to {n(a['q75'])} | "
          f"{n(a['mean'], 2, True)} {ci(a['mean_ci95_bootstrap'], 2, True)} | {n(a['share_positive'], 3)} | {n(a['share_ge_2px'], 3)} | {n(a['share_le_minus2px'], 3)} | "
          f"{n(t['median'])}; {n(t['q25'])} to {n(t['q75'])}; {n(t['mean'], 2, True)} | {n(r['median'])}; {n(r['q25'])} to {n(r['q75'])}; {n(r['mean'], 2, True)} |")
w("\nThe mean is pulled up by spurious predicted blobs far from any reference lesion (compare \"all\" with \"attached\"), and the "
  "reverse mean is pulled down by missed reference lesions far from any prediction; the median is the robust summary. Per-leaf "
  "medians and the share of distances beyond ±10 px are in `results.json`.\n")

w("## 8. Context only: thresholds chosen on the test set (NOT a valid estimate)\n")
w("| Model | Test-optimal t (min \\|bias\\|); bias; slope | Test-optimal t (min MAE); MAE; slope | Slope over the whole grid (min to max) | Slope at t = 0.5 / 0.7 / 0.9 / 0.95 / 0.99 / 0.999 |")
w("|---|---|---|---|---|")
for m in PUB + MAT:
    o = M[m]["test_oracle_context_only"]; st = o["slope_at_t"]
    w(f"| {M[m]['label']} | {ft(o['min_abs_bias']['t'])}; {n(o['min_abs_bias']['bias'], 2, True)}; {n(o['min_abs_bias']['slope'], 3)} | "
      f"{ft(o['min_mae']['t'])}; {n(o['min_mae']['mae'])}; {n(o['min_mae']['slope'], 3)} | {n(o['slope_range_over_grid'][0], 3)} to {n(o['slope_range_over_grid'][1], 3)} | "
      f"{n(st['0.5'], 3)} / {n(st['0.7'], 3)} / {n(st['0.9'], 3)} / {n(st['0.95'], 3)} / {n(st['0.99'], 3)} / {n(st['0.999'], 3)} |")
w("\nThis shows how close the validation-chosen thresholds came to the best possible single threshold on these 21 test leaves; "
  "it is optimistic by construction and must not be reported as performance.\n")

exec(open(os.path.join(HERE, "md_interpretation.py"), encoding="utf-8").read())
open(os.path.join(HERE, "results.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
json.dump({f"{m}|{op}": v for (m, op), v in VER.items()}, open(os.path.join(HERE, "work", "verdicts.json"), "w"), indent=1)
print("tables written; verdicts:")
for (m, op), v in VER.items(): print(m, op, v)
