"""Merge the outputs of the three analysis scripts into results.json and write results.md.

Run order (all in this folder, Python 3.13):
  python make_quant_figures.py        -> figure4.*, figure12.*, quant_results.json (stops if published values differ)
  python make_qualitative_figures.py  -> figure13.*, figure14.*, qualitative_results.json
  python error_location_summary.py    -> error_location.json
  python write_results.py             -> results.json, results.md
"""
from __future__ import annotations
import os as _os  # (release) all paths below are relative to ROOT (repository root by default; see README)
ROOT = (_os.environ.get("PG26038_ROOT") or _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))).replace(chr(92), "/")

import json

import common as C

Q = json.loads((C.OUT / "quant_results.json").read_text(encoding="utf-8"))
L = json.loads((C.OUT / "qualitative_results.json").read_text(encoding="utf-8"))
E = json.loads((C.OUT / "error_location.json").read_text(encoding="utf-8"))
M = C.MODEL_NAME
BANDKEYS = [b["key"] for b in Q["figure12"]["bands"]]


def f3(x):
    return f"{x:.3f}"


def ci3(ci):
    return f"{ci[0]:.3f}–{ci[1]:.3f}"


def sgn(x, nd=2):
    s = f"{x:+.{nd}f}"
    return s.replace("-", "−")


# ---- derived numbers used in the text (all stored in results.json)
_er = E["diseases"]["roya"]["summary"]
_eo = E["diseases"]["ojo"]["summary"]
_w13 = L["figures"]["figure13"]["rows"][0]
_w13_face = [f for f in E["diseases"]["roya"]["per_face"] if f["file"] == _w13["file"]]
assert len(_w13_face) == 1
_w13_face = _w13_face[0]
assert (_w13_face["tp"], _w13_face["fp"], _w13_face["fn"]) == tuple(_w13["counts_plotted"][k] for k in ("tp", "fp", "fn"))
_w14 = L["figures"]["figure14"]["rows"][0]
_sev_sorted = sorted(Q["leaf_tables"]["ojo"], key=lambda r: r["ref_sev_pct"])
_w14_sev_rank = 1 + [r["leaf_id"] for r in _sev_sorted].index(_w14["leaf_id"])
assert _sev_sorted[_w14_sev_rank - 1]["ref_sev_pct"] == _w14["leaf_ref_severity_pct"]
DERIVED = {
    "rust_fp_share_beyond_2px_of_reference_lesion": 1 - _er["fp_le2"] / _er["fp"],
    "rust_fp_share_beyond_5px_of_reference_lesion": 1 - _er["fp_le5"] / _er["fp"],
    "rust_fp_px_beyond_2px": _er["fp"] - _er["fp_le2"],
    "rust_fp_px_beyond_5px": _er["fp"] - _er["fp_le5"],
    "leaf_spot_fp_share_beyond_2px_of_reference_lesion": 1 - _eo["fp_le2"] / _eo["fp"],
    "leaf_spot_fp_share_beyond_5px_of_reference_lesion": 1 - _eo["fp_le5"] / _eo["fp"],
    "rust_worst_leaf": dict(leaf_id=_w13["leaf_id"], fp=_w13_face["fp"],
                            fp_share_within_2px=_w13_face["fp_le2"] / _w13_face["fp"],
                            fp_share_beyond_2px=1 - _w13_face["fp_le2"] / _w13_face["fp"],
                            fp_share_beyond_5px=1 - _w13_face["fp_le5"] / _w13_face["fp"],
                            fp_px_beyond_5px=_w13_face["fp"] - _w13_face["fp_le5"]),
    "leaf_spot_worst_leaf": dict(leaf_id=_w14["leaf_id"], leaf_ref_severity_pct=_w14["leaf_ref_severity_pct"],
                                 abaxial_face_ref_severity_pct=_w14["ref_severity_pct_shown_face"],
                                 rank_of_leaf_ref_severity_ascending=_w14_sev_rank, n_leaves=len(_sev_sorted),
                                 union_lesion_px_abaxial=sum(_w14["counts_plotted"].values()),
                                 union_lesion_px_adaxial=sum(_w14["other_face"]["counts"].values())),
}

notes = {
    "reproduction": "All published leaf-level values (IoU, MAE, bias, QWK, Bland–Altman LoA) reproduced exactly at the "
                    "printed precision for all six models before any figure was drawn (the script stops otherwise).",
    "figure4_label_fix": "Rust UNet++ leaf IoU is 0.646541 (exact mean of 21 abaxial-face IoUs), which rounds to 0.647; "
                         "the current figure4.png label 0.646 does not match the leaf-level mean (cause not traced).",
    "figure4_annotation": "The 'n = ... test leaves' note of each Figure 4 panel is right-aligned to the right end of the "
                          "drawn x-axis (checked in rendered coordinates), so it no longer extends past the axis.",
    "severity_labels": "Figure 14 prints two severities per leaf, each labelled: the leaf severity (mean of the two faces; "
                       "the value that sets the leaf's Figure 12 band and the value quoted in sentence 5) and the "
                       "abaxial-face severity of the face shown. They differ, e.g. "
                       + "; ".join(f"{r['leaf_id'].split('_', 2)[-1]} reference {r['leaf_ref_severity_pct']:.2f}% (leaf) "
                                   f"vs {r['ref_severity_pct_shown_face']:.2f}% (abaxial face)"
                                   for r in L["figures"]["figure14"]["rows"])
                       + ". Figure 13 prints the leaf severity, which for rust is by definition the abaxial-face "
                       "severity (asserted equal). The printed leaf severities are asserted equal to the Figure 12 leaf "
                       "table.",
    "healthy_leaves": f"Only {C.N_HEALTHY_RUST_TEST_LOCAL} of the {C.N_HEALTHY_RUST_TEST_PAPER} healthy rust-pack test "
                      "leaves (EARTH site) are available locally. None of Figures 4, 12, 13 or 14 uses healthy leaves: "
                      "leaf IoU, as in the paper, is computed on the test leaves with disease (rust 21, leaf spot 55), so "
                      "this local gap does not affect these figures.",
    "leaf_spot_face_rule": "The requested rule 'face whose IoU is closest to the leaf mean' is degenerate: the leaf IoU is "
                           "the mean of the two face IoUs, so both faces are exactly equidistant from it (asserted with "
                           "exact fractions for the three selected leaves). A pre-specified, IoU-independent tie-break "
                           "was used: the abaxial face (as for rust and for paper Figure 2). Both face IoUs are printed "
                           "in Figure 14. Under an alternative reading (face closest to the disease-level mean "
                           f"{Q['figure4']['ojo']['segformer']['mean']:.3f}) the faces would be "
                           + ", ".join(
                               min(("abaxial", "adaxial"), key=lambda f: abs(
                                   r[f"face_iou_{f}"] - Q["figure4"]["ojo"]["segformer"]["mean"]))
                               for r in L["figures"]["figure14"]["rows"])
                           + " (worst, median, best).",
    "extra_band": "Bands 0.5–<2, 2–<5, 5–<20 and ≥20% were used as requested. One rust and six leaf-spot test leaves have "
                  "reference severity <0.5%; they are shown as an additional '<0.5' band (the G0 range) so that every "
                  "test leaf is counted (n sums to 21 and 55). Dropping that band leaves the other four bands unchanged. "
                  "The six leaf-spot leaves below 0.5% all carry annotator grade G1 (the leaf-spot grade is visual).",
    "small_n": "CIs are omitted for n < 3 (rust <0.5%: n = 1; leaf spot ≥20%: n = 2); no rust test leaf reached 20%. "
               "Percentile bootstrap CIs for bands with n = 5–6 rest on few distinct resamples (126 multisets for n = 5, "
               "462 for n = 6) and are likely too narrow; treat per-band trends as descriptive.",
    "rust_reference": "The rust reference masks are semi-automatic (colour detector + SAM 2, spot-checked, not edited), "
                      "so rust FP/FN in Figure 13 and in the error-location summary are disagreements with that reference.",
    "cnn_protocol": "Model contrasts compare the published single-run models, whose training protocols differ (CNNs: "
                    "photometric jitter p = 0.8, 30 epochs, FP-penalised selection for rust). Matched retraining is in "
                    "progress; Figures 4 and 12 will need regenerating with those models (% TODO(retraining)).",
    "test_leaves_shown": "Figures 13 and 14 show three test leaves per disease, as the current Figures 9 and 10 already do. "
                         "The data-availability statement says the held-out test set is withheld pending peer review; the "
                         "author may wish to confirm that showing these six test scans is compatible with that.",
    "selection_unchanged": "The selected leaves have leaf IoUs 0.638/0.767/0.857 (rust) and 0.732/0.904/0.961 (leaf spot), "
                           "matching the values printed in the current figure9/figure10 to two decimals, so the leaf "
                           "selection appears unchanged; the overlays are what was wrong (they are now registered).",
    "rendering": "Qualitative panels are drawn in the 1024 x 1024 letterbox space; the PDF embeds the native pixels "
                 "(interpolation 'none'), the 600 dpi PNG upsamples them with nearest neighbour. Outlines are vector "
                 "0.5-level contours of the binary masks with a thin dark halo.",
    "colours": "Model colours/markers as specified. The disease colours of figure5 (rendered ≈ #CC7951 rust, ≈ #448BB0 "
               "leaf spot) are not used, because model is the encoded variable in these figures.",
    "file_mapping": {"figure4": "replaces manuscript figures/figure4.png (label fig4)",
                     "figure12": "replaces manuscript figures/figure8.png (label fig8)",
                     "figure13": "replaces manuscript figures/figure9.png (label fig9)",
                     "figure14": "replaces manuscript figures/figure10.png (label fig10)"},
}

results = {
    "description": "PG-26038 major revision: regenerated paper Figures 4, 12, 13, 14 (no re-inference)",
    "date": "2026-09-27",
    "scripts": ["common.py", "make_quant_figures.py", "make_qualitative_figures.py", "error_location_summary.py",
                "write_results.py"],
    "inputs": {"per_image": str(C.PER_IMAGE).replace("\\", "/"), "preds": str(C.PREDS).replace("\\", "/"),
               "packs": [(ROOT + "/datasets/pg26038-roya-real-pack"), (ROOT + "/datasets/pg26038-ojo-real-pack")],
               "reference_rebuild": "run_inference.py resize_pad()/load(), executed verbatim via ast"},
    "bootstrap": {"resamples": C.B, "seed": C.SEED, "type": "leaf resampling with replacement, percentile 95% CI",
                  "streams": "np.random.default_rng(20260927) created fresh per stratum (Fig. 4: disease; Fig. 12: "
                             "disease x band); the same indices are used for the three models (paired)"},
    "reproduction": Q["reproduction"],
    "figure4": Q["figure4"],
    "figure12": Q["figure12"],
    "figure13": L["figures"]["figure13"],
    "figure14": L["figures"]["figure14"],
    "qualitative_rendering": {k: v for k, v in L.items() if k != "figures"},
    "error_location_segformer": dict(E, derived=DERIVED),
    "leaf_tables": Q["leaf_tables"],
    "notes": notes,
}
(C.OUT / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")

# ------------------------------------------------------------------------------------------------ markdown
R, F4, F12 = Q["reproduction"], Q["figure4"], Q["figure12"]
md = []
md.append("# Regenerated Figures 4, 12, 13 and 14 (PG-26038 major revision)\n")
md.append("Folder: `<ROOT>/reanalysis/results/figures/`. No model inference was re-run. Run order: "
          "`make_quant_figures.py`, `make_qualitative_figures.py`, `error_location_summary.py`, `write_results.py` "
          "(shared code in `common.py`). All numbers are in `results.json`.\n")
md.append("| Paper figure | Files | Replaces manuscript file |\n|---|---|---|")
for stem, old in (("figure4", "figures/figure4.png"), ("figure12", "figures/figure8.png"),
                  ("figure13", "figures/figure9.png"), ("figure14", "figures/figure10.png")):
    md.append(f"| {stem.replace('figure', 'Figure ')} | `{stem}.pdf`, `{stem}.png` (600 dpi) | `{old}` |")
md.append("")

md.append("## 1. Reproduction of the published values (done first)\n")
md.append(f"All reproduced: **{R['all_reproduced']}** (leaf IoU, MAE, bias, QWK and Bland–Altman LoA, at the printed "
          "precision, all six models). Rust = abaxial face, 21 test leaves with disease; American leaf spot = mean of both "
          "faces, 55 test leaves.\n")
md.append("| Disease | Model | Leaf IoU | MAE (pp) | Bias (pp) | QWK | LoA (pp) |\n|---|---|---|---|---|---|---|")
for d in C.DISEASES:
    for a in C.ARCHS:
        m = R[d][a]
        md.append(f"| {C.DISEASE_NAME[d]} | {M[a]} | {m['leaf_iou']:.4f} | {m['mae_pp']:.3f} | {sgn(m['bias_pp'], 3)} | "
                  f"{m['qwk']:.3f} | {sgn(m['loa_pp'][0])} to {sgn(m['loa_pp'][1])} |")
md.append("")

md.append("## 2. Figure 4: leaf-level lesion IoU, 95% bootstrap CI\n")
md.append("10,000 leaf resamples, percentile CI, seed 20260927 (fresh generator per disease; same indices for the three "
          "models). Printed labels equal the leaf-level means (asserted).\n")
md.append("| Disease | Model | n | Leaf IoU | 95% CI |\n|---|---|---|---|---|")
for d in C.DISEASES:
    for a in C.ARCHS:
        f = F4[d][a]
        md.append(f"| {C.DISEASE_NAME[d]} | {M[a]} | {f['n_leaves']} | **{f['label']}** | {ci3(f['ci95'])} |")
md.append("\nPaired differences (same resamples):\n")
md.append("| Disease | Contrast | Mean difference | 95% CI | Leaves where first is higher |\n|---|---|---|---|---|")
for d in C.DISEASES:
    for k, v in F4[d]["paired_differences"].items():
        md.append(f"| {C.DISEASE_NAME[d]} | {k} | {sgn(v['mean'], 3)} | {sgn(v['ci95'][0], 3)} to {sgn(v['ci95'][1], 3)} | "
                  f"{v['leaves_where_first_higher']}/{v['n_leaves']} |")
md.append("")

md.append("## 3. Figure 12: leaf-level IoU by reference severity band\n")
md.append("Bands aligned with the grade cut-points 0.5 / 5 / 20% (G1 split at 2%), plus a `<0.5` band so that every test "
          "leaf is counted. Reference severity: rust abaxial face; leaf spot mean of both faces. CI omitted when n < 3 "
          "(open symbols in the figure).\n")
md.append("| Disease | Band (%) | n | SegFormer-B5 | UNet++ | DeepLabV3+ |\n|---|---|---|---|---|---|")
for d in C.DISEASES:
    for k in BANDKEYS:
        e = F12[d][k]
        cells = []
        for a in C.ARCHS:
            if not e["n_leaves"]:
                cells.append("–")
                continue
            m = e["models"][a]
            cells.append(f"{m['label']}" + (f" ({ci3(m['ci95'])})" if m["ci95"] else " (no CI, n < 3)"))
        md.append(f"| {C.DISEASE_NAME[d]} | {e['label']} | {e['n_leaves']} | " + " | ".join(cells) + " |")
md.append("\nSupplementary trend statistic (not plotted): Spearman ρ between leaf reference severity and leaf IoU, "
          "paired bootstrap 95% CI.\n")
md.append("| Disease | SegFormer-B5 | UNet++ | DeepLabV3+ |\n|---|---|---|---|")
for d in C.DISEASES:
    sp = F12[d]["spearman_severity_vs_iou"]
    md.append(f"| {C.DISEASE_NAME[d]} | " + " | ".join(
        f"{sgn(sp[a]['rho'])} ({sgn(sp[a]['ci95'][0])} to {sgn(sp[a]['ci95'][1])})" for a in C.ARCHS) + " |")
md.append("")

md.append("## 4. Figures 13 and 14: SegFormer-B5 worst / median / best test leaves\n")
md.append("Rendered in 1024 × 1024 letterbox space (image, reference and prediction share one pixel grid; no inverse "
          "transform), cropped to the leaf and all lesion pixels (+10 px), common pixel scale per figure. Columns: "
          "input | reference outline (green) | prediction outline (magenta) | error map (TP white, FP magenta, FN green, "
          "other leaf grey).\n")
md.append("| Figure | Row | Rank | Leaf | Face shown | Leaf IoU | Abaxial IoU | Adaxial IoU | TP / FP / FN px (shown face) | Shown-face IoU recomputed from plotted masks | Leaf severity ref. / pred. (%) | Shown-face severity ref. / pred. (%) |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
for stem in ("figure13", "figure14"):
    for r in L["figures"][stem]["rows"]:
        c = r["counts_plotted"]
        md.append(f"| {stem[-2:]} | {r['role']} | {r['rank']}/{L['figures'][stem]['n_test_leaves']} | {r['leaf_id']} | "
                  f"{r['face_shown']} | **{r['leaf_iou_3dp']}** | {r['face_iou_abaxial']:.3f} | {r['face_iou_adaxial']:.3f} | "
                  f"{c['tp']:,} / {c['fp']:,} / {c['fn']:,} | {r['iou_recomputed_from_plotted_masks']:.6f} (= {r['iou_recomputed_exact']}) | "
                  f"**{r['leaf_ref_severity_pct']:.2f} / {r['leaf_pred_severity_pct']:.2f}** | "
                  + ("= leaf (rust leaf = abaxial face)" if stem == "figure13" else
                     f"{r['ref_severity_pct_shown_face']:.2f} / {r['pred_severity_pct_shown_face']:.2f}") + " |")
md.append("\nVerification (all asserted, all passed): (1) rebuilt reference and saved prediction reproduce the "
          "per_image.csv pixel counts; (2) IoU from the plotted crop arrays = IoU from the error-map raster = CSV value, "
          "as exact integer ratios, and no lesion pixel lies outside the crop; (3) pixel centres inside the drawn outline "
          "paths reproduce each mask exactly (0 mismatched pixels); (4) the error map and input crop embedded in each PDF "
          "are pixel-identical to the computed arrays, and a MuPDF render of the PDF (anti-aliasing off) sampled at every "
          "image-pixel centre reproduces them exactly, from which TP/FP/FN and the IoU were recounted; (5) leaf spot: leaf "
          "IoU = mean of the shown-face IoU (from plotted masks) and the other face's IoU (recomputed from its masks); (6) the "
          "printed leaf severity equals the leaf-table value that sets the Figure 12 band (rust: abaxial face; leaf spot: "
          "mean of both faces), and the leaf-spot shown-face severity is printed under its own 'abaxial-face severity' "
          "label. Only the shown-face IoU and TP/FP/FN counts are functions of the plotted masks alone; severities also use "
          "the predicted leaf mask (not plotted), and the leaf-spot leaf IoU and leaf severity also use the adaxial face.\n")
md.append("Leaf-spot face: the requested rule (face IoU closest to the leaf mean) ties by construction, because the leaf "
          "IoU is the mean of exactly two face IoUs. The tie was broken by a pre-specified, IoU-independent rule: the "
          "**abaxial face** for all three leaves (stated in the figure, which also prints both face IoUs).\n")

er, eo = E["diseases"]["roya"]["summary"], E["diseases"]["ojo"]["summary"]
md.append("### Error location over all SegFormer-B5 test faces (supports the qualitative description)\n")
md.append("| Disease | Faces | TP px | FP px | FN px | FP/FN | Faces with FP > FN | FP within 2 px (5 px) of a reference lesion | FP more than 2 px (5 px) from any reference lesion | FN within 2 px (5 px) of a predicted lesion |\n|---|---|---|---|---|---|---|---|---|---|")
for d, s in (("roya", er), ("ojo", eo)):
    md.append(f"| {C.DISEASE_NAME[d]} | {s['n_faces']} | {s['tp']:,} | {s['fp']:,} | {s['fn']:,} | {s['fp_over_fn_ratio']:.2f} | "
              f"{s['faces_with_fp_gt_fn']}/{s['n_faces']} | {100 * s['share_fp_within_2px_of_reference_lesion']:.0f}% "
              f"({100 * s['share_fp_within_5px_of_reference_lesion']:.0f}%) | "
              f"{100 * (1 - s['share_fp_within_2px_of_reference_lesion']):.0f}% "
              f"({100 * (1 - s['share_fp_within_5px_of_reference_lesion']):.0f}%) | {100 * s['share_fn_within_2px_of_predicted_lesion']:.0f}% "
              f"({100 * s['share_fn_within_5px_of_predicted_lesion']:.0f}%) |")
wl = DERIVED["rust_worst_leaf"]
md.append(f"\nWorst rust leaf ({wl['leaf_id']}, Figure 13 top row): {wl['fp']:,} FP px, of which "
          f"{100 * wl['fp_share_beyond_2px']:.0f}% lie more than 2 px and {100 * wl['fp_share_beyond_5px']:.0f}% more than "
          "5 px from any reference lesion (the FP clusters away from reference lesions visible in its error map).")
md.append("")

md.append("## 5. Notes and caveats\n")
for k, v in notes.items():
    if isinstance(v, dict):
        v = "; ".join(f"{kk}: {vv}" for kk, vv in v.items())
    md.append(f"- **{k.replace('_', ' ')}.** {v}")
md.append("")

# ---- suggested manuscript sentences (numbers pulled from the results)
fr = {a: F4["roya"][a] for a in C.ARCHS}
fo = {a: F4["ojo"][a] for a in C.ARCHS}
pdr = F4["roya"]["paired_differences"]["SegFormer-B5 - UNet++"]
pdo = F4["ojo"]["paired_differences"]["SegFormer-B5 - UNet++"]
b_lo, b_mid = F12["ojo"]["lt0.5"], F12["ojo"]["5-20"]
sp_o = F12["ojo"]["spearman_severity_vs_iou"]
sp_r = F12["roya"]["spearman_severity_vs_iou"]
seg_r = [F12["roya"][k]["models"]["segformer"]["mean"] for k in ("0.5-2", "2-5", "5-20")]
w14 = L["figures"]["figure14"]["rows"][0]
w14c = w14["counts_plotted"]


def tex_ci(ci):
    return f"{ci[0]:.3f}--{ci[1]:.3f}"


md.append("## 6. Suggested manuscript sentences (LaTeX-ready)\n")
md.append(
    "1. Leaf-level lesion IoU (Figure~\\ref{fig4}) was "
    f"{fr['segformer']['label']} (95\\% CI {tex_ci(fr['segformer']['ci95'])}) for SegFormer-B5, "
    f"{fr['unetpp']['label']} ({tex_ci(fr['unetpp']['ci95'])}) for UNet++ and "
    f"{fr['deeplab']['label']} ({tex_ci(fr['deeplab']['ci95'])}) for DeepLabV3+ on the 21 rust test leaves, and "
    f"{fo['segformer']['label']} ({tex_ci(fo['segformer']['ci95'])}), {fo['unetpp']['label']} ({tex_ci(fo['unetpp']['ci95'])}) "
    f"and {fo['deeplab']['label']} ({tex_ci(fo['deeplab']['ci95'])}) on the 55 American leaf spot test leaves.\n")
md.append(
    f"2. In the paired leaf bootstrap, SegFormer-B5 exceeded UNet++ by {pdr['mean']:.3f} ({tex_ci(pdr['ci95'])}) on rust "
    f"and by {pdo['mean']:.3f} ({tex_ci(pdo['ci95'])}) on American leaf spot; these are contrasts between the published "
    "single-run models, which were trained under different protocols, not between architectures as such.\n")
md.append(
    "3. With severity bands aligned to the grade cut-points (Figure~\\ref{fig8}), American leaf spot IoU rose with "
    f"reference severity for all three architectures, from {b_lo['models']['segformer']['mean']:.2f}, "
    f"{b_lo['models']['unetpp']['mean']:.2f} and {b_lo['models']['deeplab']['mean']:.2f} (SegFormer-B5, UNet++, "
    f"DeepLabV3+) below 0.5\\% ($n = {b_lo['n_leaves']}$) to {b_mid['models']['segformer']['mean']:.2f}, "
    f"{b_mid['models']['unetpp']['mean']:.2f} and {b_mid['models']['deeplab']['mean']:.2f} in the 5 to $<$20\\% band "
    f"($n = {b_mid['n_leaves']}$; Spearman $\\rho$ between leaf severity and IoU "
    f"{min(v['rho'] for v in sp_o.values()):.2f}--{max(v['rho'] for v in sp_o.values()):.2f}), whereas for rust no "
    f"severity trend was evident (SegFormer-B5 {min(seg_r):.2f}--{max(seg_r):.2f} across the three bands with five or "
    "more leaves; bootstrap intervals of Spearman $\\rho$ included zero for all three models).\n")
md.append(
    "4. Bands with fewer than three leaves (rust "
    f"$<$0.5\\%, $n = {F12['roya']['lt0.5']['n_leaves']}$; American leaf spot $\\geq$20\\%, "
    f"$n = {F12['ojo']['ge20']['n_leaves']}$) are shown without intervals, no rust test leaf reached 20\\%, and the "
    "intervals of the other bands rest on 5 to 21 leaves each, so the per-band pattern is descriptive.\n")
ww = DERIVED["leaf_spot_worst_leaf"]
ordinal = {1: "lowest", 2: "second lowest", 3: "third lowest"}[ww["rank_of_leaf_ref_severity_ascending"]]
md.append(
    "5. In the qualitative panels (Figures~\\ref{fig9} and~\\ref{fig10}), the worst American leaf spot leaf was also "
    f"one of the mildest (leaf reference severity {ww['leaf_ref_severity_pct']:.2f}\\%, the {ordinal} of the "
    f"{ww['n_leaves']} test leaves), so that its IoU rests on a few hundred lesion pixels per face "
    f"({ww['union_lesion_px_abaxial']} and {ww['union_lesion_px_adaxial']} pixels in the lesion union of the abaxial and "
    "adaxial faces). For rust, SegFormer-B5 predicted more lesion area than the semi-automatic reference in "
    f"{er['faces_with_fp_gt_fn']} of the 21 test leaves; the false-positive pixels lay mostly at lesion margins "
    f"({100 * er['share_fp_within_2px_of_reference_lesion']:.0f}\\% within 2 pixels of a reference lesion), with the "
    "remainder forming additional detections away from reference lesions "
    f"({100 * DERIVED['rust_fp_share_beyond_5px_of_reference_lesion']:.0f}\\% more than 5 pixels away overall, and "
    f"{100 * wl['fp_share_beyond_5px']:.0f}\\% on the worst rust leaf in Figure~\\ref{{fig9}}), in line with the positive "
    f"rust severity bias ({sgn(R['roya']['segformer']['bias_pp']).replace('−', '$-$')} percentage points).\n")

md.append("## 7. Suggested captions\n")
md.append("- **Figure 4 (fig4).** Leaf-level lesion IoU by architecture and disease. Points are means over the test leaves "
          "with disease (coffee leaf rust, $n = 21$, abaxial face; American leaf spot, $n = 55$, mean of both faces); bars "
          "are 95\\% percentile bootstrap confidence intervals (10,000 resamples of leaves).")
md.append("- **Figure 12 (fig8).** Leaf-level lesion IoU by reference severity band. Bands follow the grade cut-points "
          "(0.5\\%, 5\\% and 20\\%; the G1 range is split at 2\\%); $n$ is the number of test leaves per band. Reference "
          "severity is that of the abaxial face for rust and the mean of both faces for American leaf spot, whose reference "
          "grade is the annotator's visual grade, so a leaf's band need not match its grade. Bars are 95\\% percentile "
          "bootstrap confidence intervals (10,000 resamples of leaves within the band); open symbols mark bands with fewer "
          "than three leaves, shown without an interval. No rust test leaf had a reference severity of 20\\% or more. "
          "Symbols are offset horizontally within each band for legibility.")
md.append("- **Figure 13 (fig9).** Qualitative segmentation of coffee leaf rust by SegFormer-B5 for the worst, median and "
          "best of the 21 rust test leaves ranked by leaf-level lesion IoU (abaxial face). Columns: input scan; "
          "semi-automatic reference lesions (green outline); SegFormer-B5 predicted lesions (magenta outline); error map "
          "(white, lesion in both; magenta, predicted only; green, reference only; grey, leaf without lesion in either). "
          "Panels are shown in the $1024 \\times 1024$ letterbox space used for evaluation, cropped to the leaf, at a common "
          "pixel scale, so image, reference and prediction share one pixel grid; the printed IoU and pixel counts were "
          "recomputed from the plotted masks and match the tabulated values. The printed reference and predicted "
          "severities are those of the abaxial face, which defines the rust leaf-level values.")
md.append("- **Figure 14 (fig10).** Qualitative segmentation of American leaf spot by SegFormer-B5 for the worst, median "
          "and best of the 55 American leaf spot test leaves ranked by leaf-level lesion IoU (mean of the abaxial- and "
          "adaxial-face IoUs). Columns, colours, letterbox space, cropping and common pixel scale as in Figure~\\ref{fig9}. "
          "The abaxial face is shown for every leaf: the two face IoUs are, by construction, equally distant from their "
          "mean, so the face closest to the leaf mean is not defined, and the abaxial face was used as a pre-specified "
          "tie-break. For the shown face, the IoU and the TP/FP/FN pixel counts were recomputed from the plotted masks; "
          "the printed leaf IoU is the mean of that value and the adaxial-face IoU, recomputed from the adaxial masks, "
          "which are not plotted. The printed leaf severity is the mean of the two face severities (the value that places "
          "the leaf in its Figure~\\ref{fig8} band); the abaxial-face severity of the shown face is printed separately.")
md.append("")
(C.OUT / "results.md").write_text("\n".join(md), encoding="utf-8")
print("wrote results.json and results.md")
