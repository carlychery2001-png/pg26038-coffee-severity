"""Round 4, analysis A4 part 1: American leaf spot grade agreement with the MASK-DERIVED grade as the PRIMARY
reference and the annotator's visual grade as the secondary reference (PG-26038 major revision).

Author decision (round4_plan.md, items 9-10): the mask-derived grade becomes the primary leaf-spot reference.

Inputs (no inference is re-run):
  reanalysis/out/per_image.csv              published models (SegFormer-B5, UNet++, DeepLabV3+), leaf-spot test
  reanalysis/out/per_image_retrained.csv    matched-protocol CNNs, LR 6e-5/6e-4 (lrB), seeds 0 (phase-1 run), 1, 2
  reanalysis/results/baseline/per_leaf_test.csv   colour-threshold baseline (redness index >= 0.026), per leaf

Leaf rule (evaluation protocol, unchanged): leaf spot = mean of the two faces (severity = mean of face severities).
Grades: G0 < 0.5 <= G1 < 5 <= G2 < 20 <= G3 (%).
  mask-derived reference (PRIMARY)  = cut-points applied to the leaf-level reference severity (mean of both faces)
  visual reference (secondary)      = annotator's visual grade (grade_label G1/G2/G3)
Predicted grade = cut-points applied to the predicted leaf severity.

Statistics (helpers imported unchanged from results/paired/paired_analysis.py):
  * Accuracy with Wilson 95% interval; QWK (quadratic weights, grades 0..3) with percentile bootstrap 95% CI.
  * Bootstrap: 10,000 resamples of the 55 leaves, numpy default_rng(20260928), ONE index matrix shared by every
    method, reference and pair (paired design).
  * Per-grade recall with Wilson 95% intervals; confusion matrices (rows reference, columns predicted).
  * Paired comparisons (first minus second): exact McNemar on per-leaf grade correctness, Tango score 95% CI for
    d(accuracy) (primary interval, as in results/paired), percentile bootstrap CIs for d(accuracy) and d(QWK),
    Holm within each family and reference standard.
      Family P: the three published models (pairs as in results/paired / manuscript Table 3).
      Family B: each published network vs the colour baseline (network first).
      Family M: each matched CNN seed vs the published SegFormer-B5 (matched first); 6 comparisons.
  * Matched CNNs summarised as mean +- SD (ddof = 1) and range over the three seeds.
"""
from __future__ import annotations
import os as _os  # (release) all paths below are relative to ROOT (repository root by default; see README)
ROOT = (_os.environ.get("PG26038_ROOT") or _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))).replace(chr(92), "/")

import csv
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
RE = Path((ROOT + "/reanalysis"))
CSV_PUB = RE / "out/per_image.csv"
CSV_RET = RE / "out/per_image_retrained.csv"
CSV_BASE = RE / "results/baseline/per_leaf_test.csv"
PAIRED_PY = RE / "results/paired/paired_analysis.py"
PAIRED_JSON = RE / "results/paired/results.json"
BASE_JSON = RE / "results/baseline/results.json"
SEED_JSON = RE / "pass3/seed_summary.json"
SEED = 20260928
B = 10_000

spec = importlib.util.spec_from_file_location("paired_analysis", PAIRED_PY)
PA = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PA)  # main() is guarded by __name__ == "__main__"; only helpers/constants load
grade, qwk, qwk_matrix, wilson, pct_ci, holm = PA.grade, PA.qwk, PA.qwk_matrix, PA.wilson, PA.pct_ci, PA.holm
tango_ci, newcombe_ci, mcnemar_exact = PA.tango_ci, PA.newcombe_ci, PA.mcnemar_exact

RETRAINED_RUNS = {
    "matched_unetpp_s0": "pg26038-mb-ojo-unetpp-lrB",
    "matched_unetpp_s1": "pg26038-mbs-ojo-unetpp-lrB-seed1",
    "matched_unetpp_s2": "pg26038-mbs-ojo-unetpp-lrB-seed2",
    "matched_deeplab_s0": "pg26038-mb-ojo-deeplab-lrB",
    "matched_deeplab_s1": "pg26038-mbs-ojo-deeplab-lrB-seed1",
    "matched_deeplab_s2": "pg26038-mbs-ojo-deeplab-lrB-seed2",
}
NAME = {
    "pub_segformer": "SegFormer-B5 (published)",
    "pub_unetpp": "UNet++ (published)",
    "pub_deeplab": "DeepLabV3+ (published)",
    "matched_unetpp_s0": "UNet++ matched, seed 0",
    "matched_unetpp_s1": "UNet++ matched, seed 1",
    "matched_unetpp_s2": "UNet++ matched, seed 2",
    "matched_deeplab_s0": "DeepLabV3+ matched, seed 0",
    "matched_deeplab_s1": "DeepLabV3+ matched, seed 1",
    "matched_deeplab_s2": "DeepLabV3+ matched, seed 2",
    "baseline": "Colour threshold (redness index >= 0.026)",
}
METHODS = list(NAME)
REFS = ["mask", "visual"]
REF_NAME = {"mask": "mask-derived (PRIMARY)", "visual": "annotator visual (secondary)"}

FAMILIES = {
    "P_published": [("pub_segformer", "pub_deeplab"), ("pub_segformer", "pub_unetpp"), ("pub_unetpp", "pub_deeplab")],
    "B_network_vs_baseline": [("pub_segformer", "baseline"), ("pub_unetpp", "baseline"), ("pub_deeplab", "baseline")],
    "M_matched_vs_segformer": [(m, "pub_segformer") for m in RETRAINED_RUNS],
}


def read(p):
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def leaves_from_rows(rows):
    """Collapse per-face rows (both faces required) to leaf-level mean severities."""
    faces = defaultdict(list)
    for r in rows:
        faces[r["leaf_id"]].append(r)
    out = {}
    for lid, rs in sorted(faces.items()):
        assert sorted(x["face"] for x in rs) == ["abaxial", "adaxial"], lid
        assert len({x["grade_label"] for x in rs}) == 1, lid
        out[lid] = dict(ref_sev=float(np.mean([float(x["ref_sev"]) for x in rs])),
                        pred_sev=float(np.mean([float(x["pred_sev"]) for x in rs])),
                        iou=float(np.mean([float(x["lesion_iou"]) for x in rs])),
                        label=rs[0]["grade_label"])
    return out


def main():
    pub = read(CSV_PUB)
    ret = read(CSV_RET)
    base = read(CSV_BASE)

    L = {}
    for a in ["segformer", "unetpp", "deeplab"]:
        L["pub_" + a] = leaves_from_rows([r for r in pub if r["model_disease"] == "ojo" and r["arch"] == a
                                          and r["image_pack"] == "ojo" and r["split"] == "test"])
    for k, run in RETRAINED_RUNS.items():
        L[k] = leaves_from_rows([r for r in ret if r["run"] == run and r["split"] == "test"])
    L["baseline"] = {r["leaf_id"]: dict(ref_sev=float(r["ref_sev"]), pred_sev=float(r["baseline_sev"]),
                                        iou=float(r["baseline_iou"]), ref_grade_file=int(r["ref_grade"]),
                                        grade_file=int(r["baseline_grade"]))
                     for r in base if r["disease"] == "ojo"}

    ids = list(L["pub_segformer"])
    n = len(ids)
    assert n == 55
    checks = {}
    ref_sev = np.array([L["pub_segformer"][i]["ref_sev"] for i in ids])
    for m in METHODS:
        assert sorted(L[m]) == ids, m
        d = float(np.max(np.abs(np.array([L[m][i]["ref_sev"] for i in ids]) - ref_sev)))
        checks[f"max_abs_ref_sev_diff_vs_published_{m}"] = d
        assert d < 1e-4, (m, d)  # the baseline file stores ref_sev at 6 decimals
    g_mask = grade(ref_sev)
    g_vis = np.array([int(L["pub_segformer"][i]["label"][1]) for i in ids])
    REFG = {"mask": g_mask, "visual": g_vis}
    # baseline file's own grades must agree with our recomputation (visual reference; predicted grade)
    assert all(L["baseline"][i]["ref_grade_file"] == g_vis[k] for k, i in enumerate(ids))
    assert all(L["baseline"][i]["grade_file"] == int(grade(L["baseline"][i]["pred_sev"])) for i in ids)

    P = {m: np.array([L[m][i]["pred_sev"] for i in ids]) for m in METHODS}
    G = {m: grade(P[m]) for m in METHODS}

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(B, n))

    res = dict(
        description="Leaf-spot grade agreement; mask-derived grade = PRIMARY reference, visual grade = secondary. "
                    "Test partition, 55 leaves, leaf = mean of both faces.",
        sources=dict(published=str(CSV_PUB), retrained=str(CSV_RET), baseline=str(CSV_BASE),
                     helpers=str(PAIRED_PY)),
        retrained_runs=RETRAINED_RUNS,
        bootstrap=dict(resamples=B, seed=SEED, generator="numpy.random.default_rng", unit="leaf",
                       interval="percentile", pairing="one index matrix shared by all methods, references, pairs"),
        grade_cut_points_pct=[0.5, 5.0, 20.0],
        integrity_checks=checks,
        reference_grade_counts={r: {f"G{g}": int((REFG[r] == g).sum()) for g in range(4)} for r in REFS},
        per_method={}, matched_seed_summary={}, paired={}, cross_checks={},
    )

    # reference vs reference
    cmv = np.zeros((4, 4), int)
    for i, j in zip(g_mask, g_vis):
        cmv[i, j] += 1
    res["reference_agreement"] = dict(
        n=n, equal=int((g_mask == g_vis).sum()), qwk=float(qwk(g_vis, g_mask)),
        qwk_ci95=pct_ci(qwk(g_vis[idx], g_mask[idx])),
        confusion_rows_mask_cols_visual=cmv.tolist(),
        disagreements=[dict(leaf_id=ids[k], mask=f"G{g_mask[k]}", visual=f"G{g_vis[k]}",
                            ref_sev_pct=round(float(ref_sev[k]), 4)) for k in range(n) if g_mask[k] != g_vis[k]])

    # ------------------------------------------------------------------ per method
    for m in METHODS:
        e = dict(name=NAME[m], n_leaves=n,
                 mae_pp=float(np.abs(P[m] - ref_sev).mean()), bias_pp=float((P[m] - ref_sev).mean()),
                 leaf_iou=float(np.mean([L[m][i]["iou"] for i in ids])), grades={})
        for r in REFS:
            gr, gp = REFG[r], G[m]
            corr = gp == gr
            k = int(corr.sum())
            q = float(qwk(gp, gr))
            assert abs(q - qwk_matrix(gp, gr)) < 1e-12
            qb = qwk(gp[idx], gr[idx])
            cm = np.zeros((4, 4), int)
            for i, j in zip(gr, gp):
                cm[i, j] += 1
            rec = {}
            for g in range(4):
                ng = int((gr == g).sum())
                kg = int(((gr == g) & corr).sum())
                rec[f"G{g}"] = dict(n=ng, correct=kg, recall=(kg / ng if ng else None), wilson95=wilson(kg, ng),
                                    over_graded=int(((gr == g) & (gp > gr)).sum()),
                                    under_graded=int(((gr == g) & (gp < gr)).sum()))
            e["grades"][r] = dict(
                accuracy=k / n, correct=k, n=n, accuracy_wilson95=wilson(k, n),
                accuracy_bootstrap_ci95=pct_ci(corr[idx].mean(1)),
                qwk=q, qwk_ci95=pct_ci(qb), qwk_undefined_resamples=int(np.isnan(qb).sum()),
                confusion_rows_reference_cols_predicted_G0_to_G3=cm.tolist(), per_grade_recall=rec,
                over_graded=int((gp > gr).sum()), under_graded=int((gp < gr).sum()),
                errors_adjacent_only=bool(np.all(np.abs(gp - gr) <= 1)),
                wrong_leaves=[dict(leaf_id=ids[t], ref=f"G{gr[t]}", pred=f"G{gp[t]}",
                                   ref_sev=round(float(ref_sev[t]), 4), pred_sev=round(float(P[m][t]), 4))
                              for t in range(n) if not corr[t]])
        res["per_method"][m] = e

    # ------------------------------------------------------------------ matched seeds: mean +- SD
    for arch in ["unetpp", "deeplab"]:
        ks = [f"matched_{arch}_s{s}" for s in range(3)]
        s = {}
        for r in REFS:
            acc = np.array([res["per_method"][k]["grades"][r]["accuracy"] for k in ks])
            qq = np.array([res["per_method"][k]["grades"][r]["qwk"] for k in ks])
            rec = {}
            for g in range(4):
                vals = [res["per_method"][k]["grades"][r]["per_grade_recall"][f"G{g}"]["recall"] for k in ks]
                rec[f"G{g}"] = None if vals[0] is None else dict(per_seed=vals, mean=float(np.mean(vals)),
                                                                   sd=float(np.std(vals, ddof=1)),
                                                                   min=float(min(vals)), max=float(max(vals)))
            s[r] = dict(accuracy=dict(per_seed=acc.tolist(),
                                      correct_per_seed=[res["per_method"][k]["grades"][r]["correct"] for k in ks],
                                      mean=float(acc.mean()), sd=float(acc.std(ddof=1)),
                                      min=float(acc.min()), max=float(acc.max())),
                        qwk=dict(per_seed=qq.tolist(), mean=float(qq.mean()), sd=float(qq.std(ddof=1)),
                                 min=float(qq.min()), max=float(qq.max())),
                        per_grade_recall=rec)
        res["matched_seed_summary"][arch] = s

    # ------------------------------------------------------------------ paired comparisons
    for fam, pairs in FAMILIES.items():
        res["paired"][fam] = {}
        pv = {r: [] for r in REFS}
        for a, b in pairs:
            key = f"{a}_vs_{b}"
            out = dict(first=NAME[a], second=NAME[b], n_leaves=n)
            for r in REFS:
                gr = REFG[r]
                ca, cb = G[a] == gr, G[b] == gr
                mc = mcnemar_exact(ca, cb)
                dq_boot = qwk(G[a][idx], gr[idx]) - qwk(G[b][idx], gr[idx])
                da_boot = (ca.astype(float) - cb)[idx].mean(1)
                out[r] = dict(
                    acc_first=float(ca.mean()), acc_second=float(cb.mean()),
                    d_accuracy=float(ca.mean() - cb.mean()),
                    d_accuracy_tango95=tango_ci(mc["first_correct_second_wrong"], mc["first_wrong_second_correct"], n),
                    d_accuracy_newcombe95=newcombe_ci(mc["both_correct"], mc["first_correct_second_wrong"],
                                                      mc["first_wrong_second_correct"], mc["both_wrong"]),
                    d_accuracy_boot95=pct_ci(da_boot),
                    d_qwk=float(qwk(G[a], gr) - qwk(G[b], gr)), d_qwk_boot95=pct_ci(dq_boot),
                    d_qwk_undefined_resamples=int(np.isnan(dq_boot).sum()),
                    d_qwk_p_boot=float(min(1.0, 2 * min(np.nanmean(dq_boot <= 0), np.nanmean(dq_boot >= 0)))),
                    mcnemar=mc, grade_identical_leaves=int((G[a] == G[b]).sum()))
                pv[r].append(mc["p_exact"])
            res["paired"][fam][key] = out
        for r in REFS:
            adj = holm(pv[r])
            qadj = holm([res["paired"][fam][f"{a}_vs_{b}"][r]["d_qwk_p_boot"] for a, b in pairs])
            for (a, b), p1, p2 in zip(pairs, adj, qadj):
                res["paired"][fam][f"{a}_vs_{b}"][r]["mcnemar"]["p_holm"] = p1
                res["paired"][fam][f"{a}_vs_{b}"][r]["d_qwk_p_boot_holm"] = p2

    # ------------------------------------------------------------------ conclusions that change between references
    changes = []
    for fam, pairs in FAMILIES.items():
        for a, b in pairs:
            o = res["paired"][fam][f"{a}_vs_{b}"]
            sig = {r: (o[r]["mcnemar"]["p_holm"] < 0.05) for r in REFS}
            tango_ex = {r: (o[r]["d_accuracy_tango95"][0] > 0 or o[r]["d_accuracy_tango95"][1] < 0) for r in REFS}
            qwk_ex = {r: (o[r]["d_qwk_boot95"][0] > 0 or o[r]["d_qwk_boot95"][1] < 0) for r in REFS}
            sign = {r: int(np.sign(o[r]["d_accuracy"])) for r in REFS}
            if sig["mask"] != sig["visual"] or tango_ex["mask"] != tango_ex["visual"] or \
                    qwk_ex["mask"] != qwk_ex["visual"] or sign["mask"] != sign["visual"]:
                changes.append(dict(family=fam, pair=f"{a}_vs_{b}", mcnemar_holm_sig=sig,
                                    tango_excludes_0=tango_ex, qwk_boot_excludes_0=qwk_ex, sign_d_accuracy=sign,
                                    d_accuracy={r: o[r]["d_accuracy"] for r in REFS}))
    res["pairs_whose_reading_changes"] = changes
    res["accuracy_ranking"] = {r: sorted(METHODS, key=lambda m: -res["per_method"][m]["grades"][r]["accuracy"])
                               for r in REFS}

    # ------------------------------------------------------------------ cross-checks
    cc = {}
    pj = json.loads(PAIRED_JSON.read_text(encoding="utf-8"))
    for a in ["segformer", "unetpp", "deeplab"]:
        mine_m = res["per_method"]["pub_" + a]["grades"]["mask"]
        mine_v = res["per_method"]["pub_" + a]["grades"]["visual"]
        th_m = pj["per_model"]["ojo"][a]["grades"]["mask_derived"]
        th_v = pj["per_model"]["ojo"][a]["grades"]["reference"]
        cc[f"paired_{a}"] = dict(
            mask_correct=(mine_m["correct"], th_m["correct"]), mask_qwk=(mine_m["qwk"], th_m["qwk"]),
            mask_confusion_equal=mine_m["confusion_rows_reference_cols_predicted_G0_to_G3"]
            == th_m["confusion_rows_reference_cols_predicted_G0_to_G3"],
            mask_wilson=(mine_m["accuracy_wilson95"], th_m["accuracy_wilson95"]),
            mask_qwk_ci95_this_seed_vs_paired_seed=(mine_m["qwk_ci95"], th_m["qwk_ci95"]),
            visual_correct=(mine_v["correct"], th_v["correct"]), visual_qwk=(mine_v["qwk"], th_v["qwk"]),
            visual_confusion_equal=mine_v["confusion_rows_reference_cols_predicted_G0_to_G3"]
            == th_v["confusion_rows_reference_cols_predicted_G0_to_G3"])
    for a, b in FAMILIES["P_published"]:
        k1 = f"{a}_vs_{b}"
        k2 = f"{a[4:]}_vs_{b[4:]}"
        mine = res["paired"]["P_published"][k1]
        th = pj["paired"]["ojo"][k2]
        cc[f"paired_mcnemar_{k2}"] = dict(
            mask_bc=((mine["mask"]["mcnemar"]["first_correct_second_wrong"],
                      mine["mask"]["mcnemar"]["first_wrong_second_correct"]),
                     (th["mcnemar_mask_derived"]["first_correct_second_wrong"],
                      th["mcnemar_mask_derived"]["first_wrong_second_correct"])),
            mask_tango=(mine["mask"]["d_accuracy_tango95"], th["d_accuracy_mask_derived"]["tango_score_ci95"]),
            visual_bc=((mine["visual"]["mcnemar"]["first_correct_second_wrong"],
                        mine["visual"]["mcnemar"]["first_wrong_second_correct"]),
                       (th["mcnemar"]["first_correct_second_wrong"], th["mcnemar"]["first_wrong_second_correct"])))
    ra = pj["leafspot_reference_agreement"]
    cc["paired_reference_agreement"] = dict(equal=(res["reference_agreement"]["equal"], ra["visual_eq_mask_derived"]),
                                            qwk=(res["reference_agreement"]["qwk"], ra["qwk_visual_vs_mask"]))
    # baseline: visual-reference values published in results/baseline (48/55, QWK 0.85)
    bm = res["per_method"]["baseline"]
    cc["baseline_visual"] = dict(correct=(bm["grades"]["visual"]["correct"], 48),
                                 qwk_2dp=(round(bm["grades"]["visual"]["qwk"], 2), 0.85),
                                 mae_2dp=(round(bm["mae_pp"], 2), 0.23), iou_3dp=(round(bm["leaf_iou"], 3), 0.807))
    # matched: seed_summary.json (visual reference)
    sj = json.loads(SEED_JSON.read_text(encoding="utf-8"))
    for arch in ["unetpp", "deeplab"]:
        s = sj[f"ojo-{arch}"]
        mine_acc = [round(x, 3) for x in res["matched_seed_summary"][arch]["visual"]["accuracy"]["per_seed"]]
        mine_q = [round(x, 3) for x in res["matched_seed_summary"][arch]["visual"]["qwk"]["per_seed"]]
        mine_iou = [round(res["per_method"][f"matched_{arch}_s{k}"]["leaf_iou"], 3) for k in range(3)]
        cc[f"seed_summary_{arch}"] = dict(acc=(mine_acc, s["grade_acc"]["per_seed"]),
                                          qwk=(mine_q, s["qwk"]["per_seed"]),
                                          iou=(mine_iou, s["leaf_iou"]["per_seed"]))
    ok = True
    for k, v in cc.items():
        for kk, vv in v.items():
            if kk.endswith("_equal"):
                ok &= bool(vv)
            elif kk in ("mask_qwk_ci95_this_seed_vs_paired_seed",):
                continue
            elif isinstance(vv, tuple) and len(vv) == 2:
                x, y = vv
                if isinstance(x, (int, float)):
                    ok &= abs(x - y) < 1e-9
                else:
                    ok &= json.dumps(x) == json.dumps(y) if not isinstance(x, list) or not x or not isinstance(x[0], float) \
                        else all(abs(p - q) < 1e-9 for p, q in zip(x, y))
    res["cross_checks"] = cc
    res["cross_checks_all_match"] = bool(ok)

    # per-leaf record
    res["per_leaf"] = [dict(leaf_id=ids[t], ref_sev=round(float(ref_sev[t]), 5), mask_grade=int(g_mask[t]),
                            visual_grade=int(g_vis[t]),
                            **{f"{m}_sev": round(float(P[m][t]), 5) for m in METHODS},
                            **{f"{m}_grade": int(G[m][t]) for m in METHODS}) for t in range(n)]
    with open(OUT / "leafspot_per_leaf_grades.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res["per_leaf"][0]))
        w.writeheader()
        w.writerows(res["per_leaf"])

    (OUT / "grades_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("cross-checks all match:", ok)
    for m in METHODS:
        gm, gv = res["per_method"][m]["grades"]["mask"], res["per_method"][m]["grades"]["visual"]
        print(f"{m:20s} mask {gm['correct']}/55 QWK {gm['qwk']:.3f} {np.round(gm['qwk_ci95'], 3)} over/under "
              f"{gm['over_graded']}/{gm['under_graded']} | visual {gv['correct']}/55 QWK {gv['qwk']:.3f} "
              f"over/under {gv['over_graded']}/{gv['under_graded']}")
    print(json.dumps(changes, indent=1))
    return res


if __name__ == "__main__":
    main()
