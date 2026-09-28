"""Write results.md from results.json (structure_analysis.py). All numbers come from results.json."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent
R = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
MN = {"segformer": "SegFormer-B5", "unetpp": "UNet++", "deeplab": "DeepLabV3+"}
DN = {"roya": "Rust", "ojo": "Leaf spot"}
ARCHS = ("segformer", "unetpp", "deeplab")
MINUS = "−"


def f(x, d=2, sign=False):
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", MINUS)


def c(ci, d=2):
    return f"[{f(ci[0], d)}, {f(ci[1], d)}]"


def g4(cnt):
    return "/".join(str(cnt[f"G{g}"]) for g in range(4))


LEAF = list(csv.DictReader(open(OUT / "work" / "leaf_tables.csv", newline="")))


def top_error(dis, arch, split, rule):
    x = [r for r in LEAF if r["disease"] == dis and r["arch"] == arch and r["split"] == split and r["rule"] == rule]
    r = max(x, key=lambda r: abs(float(r["pred_sev"]) - float(r["ref_sev"])))
    return float(r["ref_sev"]), float(r["pred_sev"]) - float(r["ref_sev"])


def mae_without_max_ref(dis, arch, split, rule="both"):
    x = [r for r in LEAF if r["disease"] == dis and r["arch"] == arch and r["split"] == split and r["rule"] == rule]
    top = max(x, key=lambda r: float(r["ref_sev"]))
    e = [abs(float(r["pred_sev"]) - float(r["ref_sev"])) for r in x if r is not top]
    return sum(e) / len(e)


def walk(o, key):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == key:
                yield v
            else:
                yield from walk(v, key)
    elif isinstance(o, list):
        for v in o:
            yield from walk(v, key)


L = []
w = L.append
w("# Validation-set results and data structure")
w("")
w("PG-26038 major revision, reanalysis for the reviewers' questions on the rust reference and protocol, clustering "
  "by collection date, grade distributions and class weights. Scripts: `structure_analysis.py` (all numbers, "
  "`results.json`), `make_figures.py` (figures), `make_md.py` (this file). No model inference was re-run; "
  "everything is computed from `reanalysis/out/per_image.csv`, the saved class maps and the two packs.")
w("")
w("**Leaf-level rules (as in the paper).** Rust = abaxial face; American leaf spot = mean of the two faces "
  "(IoU = mean of face IoUs, severity = mean of face severities). Reference grade: rust = cut-points 0.5/5/20% on "
  "the reference severity; leaf spot = annotator's visual grade. MAE and bias in percentage points (pp); "
  "LoA = bias ± 1.96 SD. 95% CIs: 10,000 leaf resamples, percentile, seed 20260927. Diseased leaves only unless "
  "stated.")
w("")

# ---- reproduction ----
rep = R["reproduction"]
w("## 0. Reproduction of the published test values")
w("")
w(f"All published test values reproduced exactly: **{'yes' if rep['all_match'] else 'NO'}** "
  f"(test diseased leaves: rust {rep['n_test_diseased_leaves']['roya']}, leaf spot {rep['n_test_diseased_leaves']['ojo']}). "
  f"Integrity: reference labels rebuilt from the packs match `per_image.csv` for all "
  f"{R['integrity']['n_distinct_images']} own-disease validation/test images (rows matched: "
  f"{R['integrity']['reference_rebuilt_matches_per_image']}, i.e. {R['integrity']['n_distinct_images']} images × 3 models); "
  f"predicted counts recomputed from the saved class maps match for {R['integrity']['predicted_maps_match_per_image']} rows.")
w("")
w("| Disease | Model | IoU | MAE | Bias | QWK | LoA | Published (IoU / MAE / bias / QWK / LoA) |")
w("|---|---|---|---|---|---|---|---|")
for r in rep["rows"]:
    h, p = r["reproduced"], r["published"]
    w(f"| {DN[r['disease']]} | {r['model']} | {f(h['iou'], 3)} | {f(h['mae'])} | {f(h['bias'], 2, True)} | {f(h['qwk'])} | "
      f"{f(h['loa'][0])} to {f(h['loa'][1], 2, True)} | {f(p['iou'], 3)} / {f(p['mae'])} / {f(p['bias'], 2, True)} / "
      f"{f(p['qwk'])} / {f(p['loa'][0])} to {f(p['loa'][1], 2, True)} |")
w("")

# ---- (1) validation vs test ----
vt = R["val_vs_test"]["results"]
w("## 1. Validation vs test, six models")
w("")
w("Diseased leaves: rust 21 validation / 21 test; leaf spot 55 / 55. The validation split was used for checkpoint "
  "selection (and, for the rust CNNs, for the healthy-leaf false-positive penalty), so it is not an independent "
  "replication; it is shown to document consistency. All rust masks (train, validation, test) come from the same "
  "semi-automatic detector + SAM 2 protocol, so rust validation and test share one reference protocol.")
w("")
w("**Table 1a. Leaf-level metrics on validation and test (95% leaf-bootstrap CI).**")
w("")
w("| Disease | Model | Split | IoU [95% CI] | MAE, pp [95% CI] | Bias, pp [95% CI] | QWK [95% CI] | LoA, pp | Mean ref. severity, % (max) | MAE / mean ref. severity |")
w("|---|---|---|---|---|---|---|---|---|---|")
for dis in ("roya", "ojo"):
    for a in ARCHS:
        for s in ("val", "test"):
            e = vt[dis][a][s]
            w(f"| {DN[dis]} | {MN[a]} | {'Validation' if s == 'val' else 'Test'} | {f(e['iou'], 3)} {c(e['iou_ci95'], 3)} | "
              f"{f(e['mae'])} {c(e['mae_ci95'])} | {f(e['bias'], 2, True)} {c(e['bias_ci95'])} | {f(e['qwk'])} {c(e['qwk_ci95'])} | "
              f"{f(e['loa'][0])} to {f(e['loa'][1], 2, True)} | {f(e['mean_ref_sev'])} ({f(e['max_ref_sev'], 1)}) | "
              f"{f(100 * e['mae_relative_to_mean_ref_sev'], 0)}% |")
w("")
w("**Table 1b. Test minus validation (point difference, 95% CI from independent leaf resamples of the two sets).**")
w("")
w("| Disease | Model | ΔIoU | ΔMAE, pp | ΔBias, pp | ΔQWK |")
w("|---|---|---|---|---|---|")
for dis in ("roya", "ojo"):
    for a in ARCHS:
        d = vt[dis][a]["test_minus_val"]
        w(f"| {DN[dis]} | {MN[a]} | {f(d['iou']['point'], 3, True)} {c(d['iou']['ci95'], 3)} | "
          f"{f(d['mae']['point'], 2, True)} {c(d['mae']['ci95'])} | {f(d['bias']['point'], 2, True)} {c(d['bias']['ci95'])} | "
          f"{f(d['qwk']['point'], 2, True)} {c(d['qwk']['ci95'])} |")
w("")
rv = {a: vt["roya"][a] for a in ARCHS}
gv = R["grade_distributions"]["results"]
w("Reading:")
w(f"- Rust IoU was consistent between validation and test for all three models (differences "
  f"{', '.join(f(rv[a]['test_minus_val']['iou']['point'], 3, True) for a in ARCHS)}; every 95% CI includes 0), "
  f"with the same ordering of the published single-run models (SegFormer-B5 > UNet++ > DeepLabV3+). This ordering "
  f"describes these trained models, not the architectures: the matched-protocol rust retraining (revision brief, items 9 "
  f"and 11) suggests that it reflects the unequal training protocols of the published models. That retraining changed "
  f"several settings at once (epochs and early-stopping patience, geometric augmentation only, learning rate and model "
  f"selection on validation lesion IoU), so it does not isolate the training budget, and it is a single run with the "
  f"seed repeats still pending.")
w(f"- Rust validation MAE and bias were lower than on test ({', '.join(f(rv[a]['val']['mae']) for a in ARCHS)} vs "
  f"{', '.join(f(rv[a]['test']['mae']) for a in ARCHS)} pp), although the differences were within bootstrap "
  f"uncertainty (test minus validation MAE: {'; '.join(MN[a] + ' ' + f(rv[a]['test_minus_val']['mae']['point'], 2, True) + ' ' + c(rv[a]['test_minus_val']['mae']['ci95']) for a in ARCHS)} pp; the CIs of the bias differences also include 0). A "
  f"possible contributor, not an established effect, is that the validation leaves are less severe (mean reference "
  f"severity {f(rv['segformer']['val']['mean_ref_sev'])}% vs "
  f"{f(rv['segformer']['test']['mean_ref_sev'])}%; maximum {f(rv['segformer']['val']['max_ref_sev'], 1)}% vs "
  f"{f(rv['segformer']['test']['max_ref_sev'], 1)}%) while the absolute error grows with severity (Figure 5); MAE relative "
  f"to the mean reference severity is similar in the two splits "
  f"({', '.join(f(100 * rv[a]['val']['mae_relative_to_mean_ref_sev'], 0) + '%' for a in ARCHS)} validation vs "
  f"{', '.join(f(100 * rv[a]['test']['mae_relative_to_mean_ref_sev'], 0) + '%' for a in ARCHS)} test).")
sv = rv["segformer"]["val"]
w(f"- Rust validation reference grades are only G1 ({gv['roya']['val']['mask_grade_counts']['G1']}) and G2 "
  f"({gv['roya']['val']['mask_grade_counts']['G2']}), so rust validation QWK "
  f"({', '.join(f(rv[a]['val']['qwk']) for a in ARCHS)}) rests on one grade boundary. Its CIs are wide for the two CNNs "
  f"(UNet++ {c(rv['unetpp']['val']['qwk_ci95'])}, DeepLabV3+ {c(rv['deeplab']['val']['qwk_ci95'])}) and degenerate for "
  f"SegFormer-B5 ({c(sv['qwk_ci95'])}): all {sv['n']} of its validation grades are correct "
  f"(exact agreement {f(100 * sv['exact_grade_agreement'], 0)}%), so every resample with a defined kappa gives 1.00 and "
  f"the interval conveys no sampling uncertainty.")
ov = {a: vt["ojo"][a] for a in ARCHS}
w(f"- Leaf spot IoU was slightly lower on validation than on test (differences "
  f"{', '.join(f(ov[a]['test_minus_val']['iou']['point'], 3, True) for a in ARCHS)}; CIs include 0). Leaf-spot "
  f"validation MAE ({', '.join(f(ov[a]['val']['mae']) for a in ARCHS)} pp) and LoA are wider than on test, driven "
  f"in part by one validation leaf with {f(ov['segformer']['val']['max_ref_sev'], 1)}% reference severity (SegFormer-B5 "
  f"error {f(top_error('ojo', 'segformer', 'val', 'both')[1], 1, True)} pp on that leaf; the test maximum is "
  f"{f(ov['segformer']['test']['max_ref_sev'], 1)}%). Without that leaf, validation MAE is "
  f"{', '.join(f(mae_without_max_ref('ojo', a, 'val')) for a in ARCHS)} pp. Leaf-spot QWK "
  f"was similar ({', '.join(f(ov[a]['val']['qwk']) for a in ARCHS)} validation vs "
  f"{', '.join(f(ov[a]['test']['qwk']) for a in ARCHS)} test).")
mge = max(vt[d][a][s]["max_grade_error"] for d in ("roya", "ojo") for a in ARCHS for s in ("val", "test"))
w(f"- In both splits the largest grade error of any model was {mge} grade (all errors adjacent-grade).")
w("")

# ---- (2) grade distributions ----
w("## 2. Grade distributions per split")
w("")
nl = R["grade_distributions"]["n_leaves_local"]
w(f"Leaves in the local packs: rust pack {nl['roya']['train']}/{nl['roya']['val']}/{nl['roya']['test']} "
  f"(train/validation/test; half diseased, half healthy), leaf-spot pack {nl['ojo']['train']}/{nl['ojo']['val']}/"
  f"{nl['ojo']['test']}. Severities in letterbox space (as in the evaluation); grades recomputed at native resolution "
  f"are identical for every leaf (rust images are already 1024 px on the long side; leaf-spot severities differ by at "
  f"most {f(max(gv['ojo'][s]['max_abs_sev_diff_native_vs_letterbox'] for s in ('train', 'val', 'test')), 3)} pp).")
w("")
w("**Table 2a. Reference grade counts per split (G0/G1/G2/G3, leaves).**")
w("")
w("| Disease | Grade source | Train | Validation | Test | All |")
w("|---|---|---|---|---|---|")
sp = ("train", "val", "test", "all")
w("| Rust (diseased) | Mask-derived, abaxial (paper rule) | " + " | ".join(g4(gv['roya'][s]['mask_grade_counts']) for s in sp) + " |")
w("| Rust (diseased) | Mask-derived, adaxial | " + " | ".join(g4(gv['roya'][s]['mask_grade_counts_adaxial']) for s in sp) + " |")
w("| Rust (diseased) | Mask-derived, mean of both faces | " + " | ".join(g4(gv['roya'][s]['mask_grade_counts_both_faces']) for s in sp) + " |")
w("| Rust (healthy controls) | Mask-derived (all G0) | " + " | ".join(g4(gv['roya'][s]['healthy_mask_grade_counts']) for s in sp) + " |")
w("| Leaf spot | Visual (annotator; paper reference) | " + " | ".join(g4(gv['ojo'][s]['visual_grade_counts']) for s in sp) + " |")
w("| Leaf spot | Mask-derived, mean of both faces | " + " | ".join(g4(gv['ojo'][s]['mask_grade_counts']) for s in sp) + " |")
w("")
w("**Table 2b. Reference severity (%) of diseased leaves, median [IQR] (range).**")
w("")
w("| Disease | Train | Validation | Test |")
w("|---|---|---|---|")
for dis in ("roya", "ojo"):
    cells = []
    for s in ("train", "val", "test"):
        q = gv[dis][s]["severity_pct"]
        cells.append(f"{f(q['median'])} [{f(q['q1'])}, {f(q['q3'])}] ({f(q['min'])}–{f(q['max'], 1)})")
    w(f"| {DN[dis]} | " + " | ".join(cells) + " |")
w("")
w("**Table 2c. Leaf spot: visual grade vs mask-derived grade.**")
w("")
w("| Split | Exact agreement | QWK | Mask grade below visual | Mask grade above visual |")
w("|---|---|---|---|---|")
for s in sp:
    e = gv["ojo"][s]
    w(f"| {s.capitalize() if s != 'val' else 'Validation'} | {f(100 * e['visual_vs_mask_exact_agreement'], 1)}% | {f(e['visual_vs_mask_qwk'])} | "
      f"{e['mask_below_visual']} | {e['mask_above_visual']} |")
ct = gv["ojo"]["all"]["crosstab_visual_rows_mask_cols"]
w("")
w(f"All splits pooled (rows visual G1–G3, columns mask G0–G3): G1 → {ct[1][0]}/{ct[1][1]}/{ct[1][2]}/{ct[1][3]}; "
  f"G2 → {ct[2][0]}/{ct[2][1]}/{ct[2][2]}/{ct[2][3]}; G3 → {ct[3][0]}/{ct[3][1]}/{ct[3][2]}/{ct[3][3]}. The mask-derived "
  f"grade never exceeded the visual grade; disagreements were always one grade lower (mostly visual G1 leaves whose "
  f"mask severity is below 0.5%).")
w("")
t = gv["split_independence_tests"]
rg = {s_: gv['roya'][s_]['mask_grade_counts'] for s_ in ('val', 'test')}
w(f"Grade mix did not differ detectably between splits (permutation chi-square tests, 10,000 permutations): rust "
  f"mask-derived p = {f(t['roya_mask_grade']['p_perm'])}; leaf spot visual p = {f(t['ojo_visual_grade']['p_perm'])} "
  f"(visual-grade counts are identical in validation and test and proportional in training, consistent with a "
  f"grade-stratified split); leaf spot mask-derived p = {f(t['ojo_mask_grade']['p_perm'])}. With 21 rust leaves per "
  f"held-out split these tests have little power, and the rust validation leaves are milder than the test leaves "
  f"(G2 {rg['val']['G2']}/21 = {f(100 * rg['val']['G2'] / 21, 0)}% vs {rg['test']['G2']}/21 = {f(100 * rg['test']['G2'] / 21, 0)}%; "
  f"maximum reference severity {f(gv['roya']['val']['severity_pct']['max'], 1)}% vs {f(gv['roya']['test']['severity_pct']['max'], 1)}%; "
  f"neither held-out split has a G3 leaf, the training split has {gv['roya']['train']['mask_grade_counts']['G3']}). "
  f"Rust healthy leaves: {gv['roya']['train']['n_healthy_leaves']}/{gv['roya']['val']['n_healthy_leaves']}/"
  f"{gv['roya']['test']['n_healthy_leaves']} (no reference lesion pixels inside the leaf on either face; reference severity 0%). "
  f"**Only 21 healthy rust-pack test leaves are present locally; the paper's specificity analysis used 54.**")
w("")
w(f"Note: the local rust pack holds {R['grade_distributions']['n_rust_diseased_leaves_local']} diseased rust leaves "
  f"(99 train + 21 validation + 21 test, both faces each); the revision brief says 142. The difference was not "
  f"resolved here and should be checked against the manuscript's leaf counts.")
w("")

# ---- (3) dates ----
cd = R["collection_dates"]["leaves_per_date_per_split"]
w("## 3. Collection dates and date-clustered bootstrap")
w("")
w("**Table 3a. Leaves per collection date (from the leaf_id prefix) and split.**")
w("")
w("| Set | Site | Date | Train | Validation | Test |")
w("|---|---|---|---|---|---|")
site = {"roya_diseased": "Sarchí", "roya_healthy": "EARTH (Guácimo)", "ojo_diseased": "EARTH (Guácimo)"}
lab = {"roya_diseased": "Rust (diseased)", "roya_healthy": "Rust pack, healthy controls", "ojo_diseased": "Leaf spot"}
for k in ("roya_diseased", "roya_healthy", "ojo_diseased"):
    ds = sorted({d for s in cd[k].values() for d in s})
    for d in ds:
        w(f"| {lab[k]} | {site[k]} | {d[:4]}-{d[4:6]}-{d[6:]} | " + " | ".join(str(cd[k][s].get(d, 0)) for s in ("train", "val", "test")) + " |")
w("")
comp = R["collection_dates"]["grade_composition_by_split_date"]["ojo"]
w("Leaf-spot grade mix differs strongly by date (visual G1/G2/G3): "
  + "; ".join(f"{k.split('_')[0].replace('val', 'validation')} {k.split('_')[1][4:6]}-{k.split('_')[1][6:]}: "
              f"{v['G1']}/{v['G2']}/{v['G3']}" for k, v in comp.items())
  + ". Date and severity are therefore confounded for leaf spot.")
w("")
cb = R["date_clustered_bootstrap"]
w(f"**Rust:** {cb['roya']['note']}")
w("")
w("**Table 3b. Leaf spot, test (55 leaves: 36 on 2026-06-21, 19 on 2026-06-28): 95% CIs under three bootstrap schemes.**")
w("")
w("| Metric | Model | Estimate | Per date (06-21 / 06-28) | Leaf bootstrap | Date-stratified | Date-clustered (two-stage) | Width ratio, clustered / leaf |")
w("|---|---|---|---|---|---|---|---|")
for key, nm, d in (("iou", "IoU", 3), ("mae", "MAE, pp", 2)):
    for a in ARCHS:
        m = cb["ojo"]["models"][a][key]
        pdv = m["per_date"]
        w(f"| {nm} | {MN[a]} | {f(m['point'], d)} | {f(pdv['20260621'], d)} / {f(pdv['20260628'], d)} | "
          f"{c(m['leaf_ci95'], d)} (width {f(m['leaf_ci_width'], d)}) | {c(m['date_stratified_ci95'], d)} (width {f(m['date_stratified_ci_width'], d)}) | "
          f"{c(m['date_cluster_ci95'], d)} (width {f(m['date_cluster_ci_width'], d)}) | {f(m['width_ratio_cluster_to_leaf'], 1)} |")
w("")
w("Schemes: leaf = resample the 55 leaves; date-stratified = keep both dates and resample leaves within each date; "
  "date-clustered = resample the two dates with replacement, then leaves within each drawn date. "
  f"{cb['ojo']['caveat']} Part of the between-date difference reflects the different grade mix of the two dates "
  "(the 06-28 leaves are more severe, which raises MAE and IoU), not only a date effect.")
w("")
w("**Table 3c. Rust, test (21 leaves, one date): leaf-bootstrap CIs only.**")
w("")
w("| Model | IoU [95% CI] (width) | MAE, pp [95% CI] (width) |")
w("|---|---|---|")
for a in ARCHS:
    m = cb["roya"]["models"][a]
    w(f"| {MN[a]} | {f(m['iou']['point'], 3)} {c(m['iou']['leaf_ci95'], 3)} ({f(m['iou']['leaf_ci_width'], 3)}) | "
      f"{f(m['mae']['point'])} {c(m['mae']['leaf_ci95'])} ({f(m['mae']['leaf_ci_width'])}) |")
w("")

# ---- (4) rust faces ----
rf = R["rust_faces"]["results"]
w("## 4. Rust: abaxial, adaxial and both-face results (Supplement)")
w("")
w("For each face rule the reference severity and reference grade use the same rule (adaxial face; or mean of the two "
  "faces, with IoU = mean of the face IoUs). The last column compares each rule's predicted grade with the paper's "
  "abaxial reference grade. 21 diseased leaves per split; 95% leaf-bootstrap CIs from the same resamples as Table 1a, so "
  "the abaxial rows repeat the Table 1a values and CIs exactly.")
w("")
for s, nm in (("test", "Test"), ("val", "Validation")):
    w(f"**Table 4{'a' if s == 'test' else 'b'}. Rust, {nm.lower()} split.**")
    w("")
    w("| Model | Face rule | IoU [95% CI] | MAE, pp [95% CI] | Bias, pp [95% CI] | QWK [95% CI] | LoA, pp | QWK vs abaxial reference grade |")
    w("|---|---|---|---|---|---|---|---|")
    for a in ARCHS:
        for rule, rn in (("abaxial", "Abaxial (paper)"), ("adaxial", "Adaxial"), ("both", "Both faces (mean)")):
            e = rf[s][a][rule]
            w(f"| {MN[a]} | {rn} | {f(e['iou'], 3)} {c(e['iou_ci95'], 3)} | {f(e['mae'])} {c(e['mae_ci95'])} | "
              f"{f(e['bias'], 2, True)} {c(e['bias_ci95'])} | {f(e['qwk'])} {c(e['qwk_ci95'])} | {f(e['loa'][0])} to {f(e['loa'][1], 2, True)} | "
              f"{f(e['qwk_vs_abaxial_reference_grade'])} |")
    w("")
w(f"Reference grade mix (test): abaxial {g4(rf['test']['segformer']['abaxial']['ref_grade_counts'])}, adaxial "
  f"{g4(rf['test']['segformer']['adaxial']['ref_grade_counts'])}, both faces {g4(rf['test']['segformer']['both']['ref_grade_counts'])} "
  f"(G0/G1/G2/G3). The adaxial rust masks were produced by the same detector + SAM 2 protocol as the abaxial ones; the "
  f"manuscript describes the adaxial face as carrying chlorosis rather than sporulation, so adaxial agreement measures a "
  f"different visual target. On validation, adaxial and both-face QWK vary widely between models "
  f"({f(min(rf['val'][a][r]['qwk'] for a in ARCHS for r in ('adaxial', 'both')))}–"
  f"{f(max(rf['val'][a][r]['qwk'] for a in ARCHS for r in ('adaxial', 'both')))}) and most of their CIs are wide. A QWK CI of "
  f"[1.00, 1.00] (e.g. SegFormer-B5 both faces on validation, DeepLabV3+ both faces on test) means that every grade was "
  f"correct; the percentile interval is then degenerate and conveys no sampling uncertainty.")
w("")
w("Reading: for all three models the adaxial face gives the lowest IoU and the widest limits of agreement on test and "
  "validation; the mean of the two faces is intermediate in IoU. The abaxial rule used in the paper gives the highest "
  "IoU for every model and split.")
w("")

# ---- (5) class frequencies ----
cf = R["class_frequencies"]["results"]
w("## 5. Class pixel frequencies in the training split and implied inverse-frequency weights")
w("")
w("Pixel counts summed over training images after letterboxing to 1024 × 1024 (the training input; the "
  "letterbox padding is background, class 0). Flip/rotation augmentation does not change class counts.")
w("")
w("**Table 5. Class frequencies (%) and cross-entropy weights (background / leaf / lesion).**")
w("")
w("| Set | Images | Background | Leaf | Lesion | Lesion share of leaf tissue | Inverse-frequency weights (scaled to bg = 0.25) | Ratio leaf:bg, lesion:bg | Paper weights (ratios) |")
w("|---|---|---|---|---|---|---|---|---|")
rows = [("Rust, train (all images)", cf["roya"]["train_all_images"]),
        ("Rust, train (diseased leaves only)", cf["roya"]["train_diseased_only"]),
        ("Leaf spot, train", cf["ojo"]["train_all_images"]),
        ("Both diseases, train", cf["both_diseases_train"]["train_all_images"]),
        ("Rust, all splits", cf["roya"]["all_splits"]),
        ("Leaf spot, all splits", cf["ojo"]["all_splits"])]
for nm, e in rows:
    fr = e["frequency"]; iw = e["inverse_frequency_weights_scaled_bg_0p25"]
    ra = e["inverse_frequency_ratios_leaf_to_bg_lesion_to_bg_lesion_to_leaf"]
    w(f"| {nm} | {e['n_images']} | {f(100 * fr['background'], 1)} | {f(100 * fr['leaf'], 2)} | {f(100 * fr['lesion'], 3)} | "
      f"{f(100 * e['lesion_share_of_leaf_tissue'], 2)}% | {f(iw[0])} / {f(iw[1])} / {f(iw[2], 0)} | "
      f"{f(ra[0], 1)}, {f(ra[1], 0)} | 0.25 / 0.5 / 3.0 (2, 12) |")
w("")
er, eo = cf["roya"]["train_all_images"], cf["ojo"]["train_all_images"]
w("Alternative weightings, training split, scaled to background = 0.25 (rust; leaf spot): "
  f"square-root inverse frequency {' / '.join(f(v) for v in er['sqrt_inverse_weights_scaled_bg_0p25'])}; "
  f"{' / '.join(f(v) for v in eo['sqrt_inverse_weights_scaled_bg_0p25'])}. "
  f"ENet log weighting 1/ln(1.02 + f) {' / '.join(f(v) for v in er['enet_log_weights_scaled_bg_0p25'])}; "
  f"{' / '.join(f(v) for v in eo['enet_log_weights_scaled_bg_0p25'])}. "
  f"Excluding the letterbox padding, background/leaf/lesion = "
  f"{' / '.join(f(100 * er['frequency_excluding_padding'][k], 2) for k in ('background', 'leaf', 'lesion'))}% (rust) and "
  f"{' / '.join(f(100 * eo['frequency_excluding_padding'][k], 2) for k in ('background', 'leaf', 'lesion'))}% (leaf spot).")
w("")
w("Reading: the paper's weights 0.25/0.5/3.0 increase with class rarity, but they do **not** follow inverse class "
  f"frequency. Inverse frequency implies leaf:background ≈ {f(er['inverse_frequency_ratios_leaf_to_bg_lesion_to_bg_lesion_to_leaf'][0], 0)}:1 "
  f"and lesion:background ≈ {f(er['inverse_frequency_ratios_leaf_to_bg_lesion_to_bg_lesion_to_leaf'][1], 0)}:1 for rust "
  f"({f(eo['inverse_frequency_ratios_leaf_to_bg_lesion_to_bg_lesion_to_leaf'][0], 0)}:1 and "
  f"{f(eo['inverse_frequency_ratios_leaf_to_bg_lesion_to_bg_lesion_to_leaf'][1], 0)}:1 for leaf spot), against 2:1 and 12:1 in the "
  f"paper; the leaf/background ratio (2) is even below the square-root inverse-frequency ratio. No single power "
  f"w ∝ f^(−p) reproduces all three paper weights (the three pairwise ratios imply p ≈ "
  f"{', '.join(f(math.log(pr) / math.log(fr_), 2) for pr, fr_ in zip((2, 12, 6), er['inverse_frequency_ratios_leaf_to_bg_lesion_to_bg_lesion_to_leaf']))} for rust). "
  f"Under the paper's weights, lesion pixels carry {f(100 * er['ce_weight_mass_share_under_paper_weights']['lesion'], 1)}% "
  f"(rust) and {f(100 * eo['ce_weight_mass_share_under_paper_weights']['lesion'], 1)}% (leaf spot) of the total "
  f"cross-entropy weight, versus {f(100 * er['frequency']['lesion'], 2)}% and {f(100 * eo['frequency']['lesion'], 2)}% "
  f"unweighted. In the training kernels the Focal Tversky term (weight 0.7) is computed on the lesion class alone, so "
  f"lesion rarity is additionally addressed by that term; how much of the correction each term contributes was not "
  f"measured here.")
w("")

# ---- sentences ----
w("## Suggested manuscript sentences")
w("")
w(f"1. (Results or Supplement, validation.) \"On the validation split, whose rust masks were produced with the same "
  f"semi-automatic protocol as the test masks, leaf-level rust IoU was {f(rv['segformer']['val']['iou'], 3)} (SegFormer-B5), "
  f"{f(rv['unetpp']['val']['iou'], 3)} (UNet++) and {f(rv['deeplab']['val']['iou'], 3)} (DeepLabV3+), within 0.03 of the test "
  f"values, and leaf-spot IoU was {f(ov['segformer']['val']['iou'], 3)}, {f(ov['unetpp']['val']['iou'], 3)} and "
  f"{f(ov['deeplab']['val']['iou'], 3)} (Table Sx). Rust validation MAE was lower than on test "
  f"({f(rv['segformer']['val']['mae'])}–{f(max(rv[a]['val']['mae'] for a in ARCHS))} vs "
  f"{f(rv['segformer']['test']['mae'])}–{f(max(rv[a]['test']['mae'] for a in ARCHS))} percentage points), although the "
  f"differences were within bootstrap uncertainty (e.g. SegFormer-B5, test minus validation "
  f"{f(rv['segformer']['test_minus_val']['mae']['point'], 2, True)} percentage points, 95% CI "
  f"{f(rv['segformer']['test_minus_val']['mae']['ci95'][0])} to {f(rv['segformer']['test_minus_val']['mae']['ci95'][1], 2, True)}); "
  f"the validation leaves were also less severe (maximum reference severity "
  f"{f(rv['segformer']['val']['max_ref_sev'], 1)}% vs {f(rv['segformer']['test']['max_ref_sev'], 1)}%). Because the validation split was used for model selection, these values "
  f"document consistency rather than an independent replication.\"")
w(f"2. (Methods, data.) \"All rust leaves were collected on a single day, so between-date variation could not be "
  f"estimated for rust. American leaf spot leaves came from two dates that also differed in grade mix; resampling dates "
  f"and then leaves widened the 95% CI of SegFormer-B5 test IoU from "
  f"{f(cb['ojo']['models']['segformer']['iou']['leaf_ci95'][0], 3)}–{f(cb['ojo']['models']['segformer']['iou']['leaf_ci95'][1], 3)} to "
  f"{f(cb['ojo']['models']['segformer']['iou']['date_cluster_ci95'][0], 3)}–{f(cb['ojo']['models']['segformer']['iou']['date_cluster_ci95'][1], 3)} "
  f"and of MAE from {f(cb['ojo']['models']['segformer']['mae']['leaf_ci95'][0])}–{f(cb['ojo']['models']['segformer']['mae']['leaf_ci95'][1])} to "
  f"{f(cb['ojo']['models']['segformer']['mae']['date_cluster_ci95'][0])}–{f(cb['ojo']['models']['segformer']['mae']['date_cluster_ci95'][1])} "
  f"percentage points; with only two dates this indicates, but cannot reliably estimate, date-level uncertainty.\"")
w(f"3. (Supplement, rust faces.) \"On the adaxial face, where rust shows chlorosis rather than sporulation, test IoU was "
  f"lower ({f(rf['test']['segformer']['adaxial']['iou'], 3)}, {f(rf['test']['unetpp']['adaxial']['iou'], 3)} and "
  f"{f(rf['test']['deeplab']['adaxial']['iou'], 3)} for SegFormer-B5, UNet++ and DeepLabV3+) than on the abaxial face "
  f"({f(rf['test']['segformer']['abaxial']['iou'], 3)}, {f(rf['test']['unetpp']['abaxial']['iou'], 3)} and "
  f"{f(rf['test']['deeplab']['abaxial']['iou'], 3)}), and averaging the two faces gave intermediate values "
  f"({f(rf['test']['segformer']['both']['iou'], 3)}, {f(rf['test']['unetpp']['both']['iou'], 3)} and "
  f"{f(rf['test']['deeplab']['both']['iou'], 3)}; Table Sx).\"")
w(f"4. (Methods, loss; replaces the inverse-frequency claim.) \"In the letterboxed training images, background, healthy "
  f"leaf tissue and lesion made up {f(100 * er['frequency']['background'], 1)}%, {f(100 * er['frequency']['leaf'], 1)}% and "
  f"{f(100 * er['frequency']['lesion'], 2)}% of rust pixels and {f(100 * eo['frequency']['background'], 1)}%, "
  f"{f(100 * eo['frequency']['leaf'], 1)}% and {f(100 * eo['frequency']['lesion'], 2)}% of leaf-spot pixels. The cross-entropy "
  f"weights of 0.25, 0.5 and 3.0 were set heuristically to increase with class rarity and are much flatter than "
  f"inverse-frequency weights, which would be about 0.25, {f(er['inverse_frequency_weights_scaled_bg_0p25'][1], 1)} and "
  f"{f(er['inverse_frequency_weights_scaled_bg_0p25'][2], 0)} for rust. Lesion rarity is additionally addressed by the "
  f"Focal Tversky term (weight 0.7), which is computed on the lesion class alone.\"")
w(f"5. (Methods, data.) \"The reference grade mix did not differ detectably between the training, validation and test "
  f"splits (permutation tests: rust, p = {f(t['roya_mask_grade']['p_perm'])}; American leaf spot, p = "
  f"{f(t['ojo_visual_grade']['p_perm'])} for visual and {f(t['ojo_mask_grade']['p_perm'])} for mask-derived grades; Figure Sx), "
  f"although the rust validation leaves were milder than the test leaves ({rg['val']['G2']} vs {rg['test']['G2']} of 21 "
  f"leaves in G2; maximum reference severity {f(gv['roya']['val']['severity_pct']['max'], 1)}% vs "
  f"{f(gv['roya']['test']['severity_pct']['max'], 1)}%). The mask-derived leaf-spot grade never exceeded the annotator's visual grade and was one "
  f"grade lower for {gv['ojo']['all']['mask_below_visual']} of {gv['ojo']['all']['n_diseased_leaves']} leaves.\"")
w("")

# ---- figures ----
w("## Figures")
w("")
w("- `figure_val_test.pdf` / `.png`: leaf lesion IoU (top) and severity MAE (bottom) on validation (open markers) and "
  "test (filled markers) for the six models, 95% leaf-bootstrap CIs. Suggested caption: \"Validation and test results "
  "under the paper's leaf-level rules (rust: abaxial face; American leaf spot: mean of the two faces). Open markers, "
  "validation (21 rust and 55 leaf-spot leaves); filled markers, test (21 and 55 leaves). Bars are 95% bootstrap "
  "confidence intervals (10,000 leaf resamples). The validation split was used for model selection.\"")
w("- `figure_grade_distribution.pdf` / `.png`: reference grade mix per split. Suggested caption: \"Reference grade "
  "distribution of the diseased leaves in the training, validation and test splits. (a) Coffee leaf rust, grade derived "
  "from the abaxial reference mask (cut-points 0.5, 5 and 20%); the healthy control leaves of the rust pack (99, 21 and "
  "21 locally; all G0) are not shown. (b) American leaf spot, the annotator's visual grade and the grade derived from "
  "the mean reference severity of the two faces. Numbers in the bars are leaf counts.\"")
w("")

# ---- notes ----
w("## Notes and caveats")
w("")
w("- Only 21 of the 54 healthy rust-pack test leaves reported in the paper are present locally; no specificity number "
  "was recomputed here.")
w("- The validation split informed checkpoint selection (all models) and the rust CNN false-positive penalty, so "
  "validation metrics may be slightly optimistic and are not an independent test.")
w("- Rust IoU/MAE measure agreement with the semi-automatic (detector + SAM 2, spot-checked, unedited) reference, in "
  "validation as in test.")
w("- Leaf spot has two collection dates and rust one; a date-clustered bootstrap with two clusters is illustrative only, "
  "and the two leaf-spot dates differ in grade mix.")
w("- QWK bootstrap resamples in which kappa is undefined (a single grade in both ratings) were dropped; counts are in "
  f"results.json (`qwk_n_valid_resamples`; at most {10000 - min(walk(R, 'qwk_n_valid_resamples'))} of 10,000 dropped).")
(OUT / "results.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("results.md written", len(L), "lines")
