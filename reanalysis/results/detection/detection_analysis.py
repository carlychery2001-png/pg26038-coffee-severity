"""Detection, specificity and cross-disease behavior of the six published models (major revision, PG-26038).

Input (reproduced exactly; no model inference is run here):
  reanalysis/out/per_image.csv  one row per (model disease, arch, image pack, split, file) with pixel counts
  (letterbox space) and face-level reference/predicted severity (%). image_pack != model_disease = cross-disease run.

Leaf-level rules (as in the paper):
  rust (roya)        -> abaxial face only
  American leaf spot -> mean of the two faces (IoU = mean of face IoUs; severity = mean of face severities)
  Reference grade: rust from reference severity with cut-points 0.5 / 5 / 20 %; leaf spot = annotator's visual grade.
  Predicted grade: predicted leaf severity with the same cut-points (G0 < 0.5 <= G1 < 5 <= G2 < 20 <= G3).

Primary leaf-level rule for a model's output = the rule of the disease the MODEL was trained for (what the deployed
tool reports for a leaf): rust models read the abaxial face, leaf-spot models the mean of both faces. The same
models' outputs under the other rule and per face are reported as sensitivity analyses.

Cross-disease lesion location (share of predicted lesion pixels inside the reference lesion mask, lesion IoU,
off-lesion severity) is reported on two face sets: the faces the model reads (matching its severity / grade / >=0.5%)
and the faces of the leaf's own-disease rule. They differ for every cross-disease run.
JSON values are stored at full float precision; make_md.py rounds once for display.

Outputs (this folder): results.json, leaf_level.csv.
"""
import os as _os  # (release) all paths below are relative to ROOT (repository root by default; see README)
ROOT = (_os.environ.get("PG26038_ROOT") or _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))).replace(chr(92), "/")
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest, norm, spearmanr

BASE = ROOT
OUT = os.path.dirname(os.path.abspath(__file__))
SEED, B = 20260927, 10_000
Z = float(norm.ppf(0.975))
ARCHS = [("segformer", "SegFormer-B5"), ("unetpp", "UNet++"), ("deeplab", "DeepLabV3+")]
DISEASE = {"roya": "rust", "ojo": "leaf_spot"}
MODEL_RULE = {"roya": "abaxial", "ojo": "mean"}
PUBLISHED = {  # Tables 2/4 and Bland-Altman figure of the manuscript
    ("roya", "segformer"): dict(iou=0.767, mae=0.99, bias=0.99, qwk=0.88, loa=(-1.68, 3.66)),
    ("roya", "unetpp"): dict(iou=0.647, mae=1.77, bias=1.75, qwk=0.79, loa=(-2.68, 6.17)),
    ("roya", "deeplab"): dict(iou=0.632, mae=1.47, bias=1.41, qwk=0.88, loa=(-3.07, 5.89)),
    ("ojo", "segformer"): dict(iou=0.884, mae=0.22, bias=0.16, qwk=0.82, loa=(-0.47, 0.79)),
    ("ojo", "unetpp"): dict(iou=0.853, mae=0.15, bias=-0.03, qwk=0.85, loa=(-0.50, 0.44)),
    ("ojo", "deeplab"): dict(iou=0.815, mae=0.25, bias=0.08, qwk=0.84, loa=(-0.66, 0.82)),
}
DECIMALS = dict(iou=3, mae=2, bias=2, qwk=2, loa=2)


# ----------------------------------------------------------------------------------------------------------- helpers
def grade(s):
    return 0 if s < 0.5 else 1 if s < 5 else 2 if s < 20 else 3


def wilson(k, n):
    p = k / n; d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d; h = (Z / d) * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    lo, hi = max(0.0, c - h), min(1.0, c + h)
    ci = binomtest(k, n).proportion_ci(0.95, method="wilson")          # independent check
    assert abs(ci.low - lo) < 1e-9 and abs(ci.high - hi) < 1e-9
    return lo, hi


def prop(k, n):
    lo, hi = wilson(k, n)
    return {"k": int(k), "n": int(n), "p": k / n, "wilson95": [lo, hi],
            "pct_1dp": round(100 * k / n, 1), "wilson95_pct_1dp": [round(100 * lo, 1), round(100 * hi, 1)]}


def qwk(a, b, K=4):
    a, b = np.asarray(a), np.asarray(b)
    O = np.zeros((K, K));
    for i, j in zip(a, b): O[i, j] += 1
    W = np.array([[(i - j) ** 2 for j in range(K)] for i in range(K)], float)
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    return 1 - (W * O).sum() / (W * E).sum()


_IDX = {}
def boot_idx(n):
    if n not in _IDX:
        _IDX[n] = np.random.default_rng(SEED).integers(0, n, size=(B, n))
    return _IDX[n]


def pct_ci(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def boot(stat, *arrays):
    """Percentile bootstrap over leaves (paired: all arrays resampled with the same leaf indices)."""
    n = len(arrays[0]); idx = boot_idx(n)
    arrs = [np.asarray(a, float) for a in arrays]
    vals = np.array([stat(*[a[i] for a in arrs]) for i in idx])
    return pct_ci(vals)


def boot_two(stat, a, b):
    """Percentile bootstrap for a statistic of two independent leaf sets (each resampled within itself)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ia = boot_idx(len(a)); ib = np.random.default_rng(SEED + 1).integers(0, len(b), size=(B, len(b)))
    return pct_ci([stat(a[i], b[j]) for i, j in zip(ia, ib)])


def r(x):
    """JSON value at full float precision. Display rounding is done once, in make_md.py (no double rounding)."""
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else float(x)


# -------------------------------------------------------------------------------------------------- leaf-level table
rows = list(csv.DictReader(open(f"{BASE}/reanalysis/out/per_image.csv")))
faces = defaultdict(dict)
for row in rows:
    faces[(row["model_disease"], row["arch"], row["image_pack"], row["split"], row["leaf_id"])][row["face"]] = row


def combine(fr, rule):
    use = [fr["abaxial"], fr["adaxial"]] if rule == "mean" else [fr[rule]]
    f = lambda c: [float(u[c]) for u in use]
    ious = [float(u["lesion_iou"]) for u in use if u["lesion_iou"] != ""]
    pred_les, inter = sum(f("pred_lesion_px")), sum(f("inter_lesion"))
    return dict(
        pred_sev=float(np.mean(f("pred_sev"))), ref_sev=float(np.mean(f("ref_sev"))),
        iou=float(np.mean(ious)) if len(ious) == len(use) else float("nan"),
        # predicted lesion pixels and those inside the reference lesion mask (of the leaf's own disease), summed over
        # the faces used by `rule`
        pred_les_px=pred_les, inter_px=inter,
        # predicted severity split into the part OUTSIDE the reference lesion mask (off-lesion) and the part inside it,
        # in % of the predicted leaf area, combined over faces exactly like pred_sev (off + in = pred_sev)
        off_lesion_sev=float(np.mean([100 * (float(u["pred_lesion_px"]) - float(u["inter_lesion"])) / float(u["pred_leaf_px"])
                                      if float(u["pred_leaf_px"]) else 0.0 for u in use])),
        in_lesion_sev=float(np.mean([100 * float(u["inter_lesion"]) / float(u["pred_leaf_px"])
                                     if float(u["pred_leaf_px"]) else 0.0 for u in use])))


leaf = []   # one record per (model, image pack, split, leaf)
for (m, a, p, s, lid), fr in sorted(faces.items()):
    assert set(fr) == {"abaxial", "adaxial"}, (m, a, p, s, lid)
    lab = fr["abaxial"]["grade_label"]
    cls = "healthy" if lab == "healthy" else DISEASE[p]
    own = combine(fr, MODEL_RULE[p])                       # leaf's own-disease rule (for reference severity / grade)
    ref_grade = 0 if cls == "healthy" else (grade(own["ref_sev"]) if p == "roya" else int(lab[1]))
    rec = dict(model_disease=m, arch=a, image_pack=p, split=s, leaf_id=lid, leaf_class=cls, ref_grade=ref_grade,
               ref_sev_leafrule=own["ref_sev"], cross_disease=(cls != "healthy" and p != m),
               sev_abaxial=float(fr["abaxial"]["pred_sev"]), sev_adaxial=float(fr["adaxial"]["pred_sev"]))
    rec["sev_mean"] = (rec["sev_abaxial"] + rec["sev_adaxial"]) / 2
    rec["sev_max"] = max(rec["sev_abaxial"], rec["sev_adaxial"])
    mr = combine(fr, MODEL_RULE[m])                        # model's own deployment rule (primary)
    rec.update(pred_sev=mr["pred_sev"], pred_grade=grade(mr["pred_sev"]))
    # lesion-location measures on two face sets (identical unless the run is cross-disease):
    #   *_modelrule = the faces the MODEL reads (the same faces behind pred_sev / pred_grade / >=0.5%)
    #   *_leafrule  = the faces of the LEAF's own-disease rule (rust leaves: abaxial; leaf-spot leaves: both faces)
    rec.update(iou_modelrule=mr["iou"], pred_les_px_modelrule=mr["pred_les_px"], inter_px_modelrule=mr["inter_px"],
               off_lesion_sev_modelrule=mr["off_lesion_sev"], in_lesion_sev_modelrule=mr["in_lesion_sev"])
    rec.update(iou_leafrule=own["iou"], pred_les_px_leafrule=own["pred_les_px"], inter_px_leafrule=own["inter_px"],
               off_lesion_sev_leafrule=own["off_lesion_sev"], in_lesion_sev_leafrule=own["in_lesion_sev"],
               pred_sev_leafrule=own["pred_sev"], pred_grade_leafrule=grade(own["pred_sev"]))
    leaf.append(rec)

with open(f"{OUT}/leaf_level.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(leaf[0].keys()))
    w.writeheader()
    for rec in leaf:
        w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in rec.items()})


def sel(m, a, p, split, cls):
    out = [x for x in leaf if x["model_disease"] == m and x["arch"] == a and x["image_pack"] == p
           and x["split"] == split and x["leaf_class"] == cls]
    return sorted(out, key=lambda x: x["leaf_id"])


res = {"meta": {
    "input": "reanalysis/out/per_image.csv (reproduced exactly by local re-inference; no inference re-run here)",
    "leaf_rules": "rust = abaxial face only; American leaf spot = mean of the two faces (IoU = mean of face IoUs, "
                  "severity = mean of face severities)",
    "primary_rule_for_model_output": "rule of the disease the model was trained for (rust models: abaxial face; "
                                     "leaf-spot models: mean of both faces); alternatives reported as sensitivity analyses",
    "grades": "G0 < 0.5% <= G1 < 5% <= G2 < 20% <= G3 (predicted grade from predicted leaf severity; rust reference grade "
              "from reference severity; leaf-spot reference grade = annotator's visual grade)",
    "thresholds": {">0%": "any predicted lesion pixel on the leaf-level face(s)", ">=0.5%": "predicted grade G1 or higher"},
    "proportion_ci": "Wilson score, two-sided 95%, no continuity correction",
    "bootstrap": f"{B} resamples of leaves, percentile 95% CI, numpy default_rng seed {SEED}; paired comparisons use the "
                 "same leaf indices for both models",
    "healthy_coverage": "21 healthy rust-pack TEST leaves are available locally (EARTH healthy controls collected 12, 15 "
                        "and 22 June 2026); the paper reported 54. The other 33 are not in the local pack, so the published "
                        "54-leaf counts cannot be reproduced locally beyond these 21.",
    "lesion_location_face_sets": "Cross-disease 'lesion_location' measures (share of predicted lesion pixels inside the "
                                 "reference lesion mask, lesion IoU, off-lesion severity) are given for two face sets: "
                                 "'model_rule_faces' = the faces the model reads (rust models: abaxial; leaf-spot models: "
                                 "both faces), i.e. the same faces behind pred_sev, grade and >=0.5%; and "
                                 "'leaf_rule_faces' = the faces of the leaf's own-disease rule (rust leaves: abaxial; "
                                 "leaf-spot leaves: both faces). The two sets are identical for correct-disease runs and "
                                 "differ for every cross-disease run. 'Off-lesion' = predicted lesion pixels outside the "
                                 "reference lesion mask; this includes lesion margins as well as distant tissue (not "
                                 "separated here).",
    "json_precision": "all values at full float precision; display rounding is applied once, in make_md.py",
}}

# ------------------------------------------------------------------------------------ 1) reproduce published values
repro, ok_all = {}, True
for m in ("roya", "ojo"):
    for a, name in ARCHS:
        L = sel(m, a, m, "test", DISEASE[m])
        pred = np.array([x["pred_sev"] for x in L]); ref = np.array([x["ref_sev_leafrule"] for x in L])
        d = pred - ref
        got = dict(iou=float(np.mean([x["iou_leafrule"] for x in L])), mae=float(np.mean(np.abs(d))),
                   bias=float(np.mean(d)), qwk=float(qwk([x["ref_grade"] for x in L], [x["pred_grade"] for x in L])),
                   loa=(float(d.mean() - 1.96 * d.std(ddof=1)), float(d.mean() + 1.96 * d.std(ddof=1))))
        pub = PUBLISHED[(m, a)]; match = {}
        for k in pub:
            if k == "loa":
                match[k] = all(round(g, 2) == pv for g, pv in zip(got[k], pub[k]))
            else:
                match[k] = round(got[k], DECIMALS[k]) == pub[k]
        ok_all &= all(match.values())
        repro[f"{DISEASE[m]}/{name}"] = {"n_leaves": len(L), "computed": {k: (list(map(lambda v: r(v), g)) if k == "loa"
                                          else r(g)) for k, g in got.items()},
                                          "published": {k: (list(v) if k == "loa" else v) for k, v in pub.items()},
                                          "match_at_published_precision": match}
res["reproduction"] = {"all_match": bool(ok_all), "models": repro}
if not ok_all:
    json.dump(res, open(f"{OUT}/results.json", "w"), indent=2)
    raise SystemExit("Published values NOT reproduced; stopping (see results.json['reproduction']).")

# ------------------------------------------------------------------------- 2) sensitivity on diseased test leaves
sens = {}
for m in ("roya", "ojo"):
    for a, name in ARCHS:
        L = sel(m, a, m, "test", DISEASE[m])
        ps = np.array([x["pred_sev"] for x in L]); refpos = np.array([x["ref_grade"] >= 1 for x in L])
        e = {"n_diseased_leaves": len(L),
             "gt0": prop(int((ps > 0).sum()), len(L)), "ge05": prop(int((ps >= 0.5).sum()), len(L)),
             "min_pred_sev": r(ps.min()), "leaves_below_0p5": [
                 {"leaf_id": x["leaf_id"], "pred_sev": r(x["pred_sev"]), "ref_sev": r(x["ref_sev_leafrule"]),
                  "ref_grade": x["ref_grade"]} for x in L if x["pred_sev"] < 0.5]}
        # restricted to leaves whose REFERENCE-MASK severity is >= 0.5% (rust: one test leaf is reference G0;
        # leaf spot: every test leaf is visually graded G1+, but some reference masks cover < 0.5% of the leaf)
        rm = np.array([x["ref_sev_leafrule"] >= 0.5 for x in L])
        e["ref_mask_sev_ge05_only"] = {"n": int(rm.sum()), "gt0": prop(int((ps[rm] > 0).sum()), int(rm.sum())),
                                       "ge05": prop(int((ps[rm] >= 0.5).sum()), int(rm.sum()))}
        e["n_ref_mask_sev_lt05"] = int((~rm).sum())
        # threshold-free separation of these diseased leaves from the 21 local healthy test leaves (Mann-Whitney AUC);
        # caveat: rust diseased leaves are from Sarchi, healthy leaves from EARTH (site confound)
        hs = np.array([x["pred_sev"] for x in sel(m, a, "roya", "test", "healthy")])
        auc = float(np.mean([(1.0 if d > h else 0.5 if d == h else 0.0) for d in ps for h in hs]))
        e["auc_vs_21_healthy"] = r(auc)
        e["auc_vs_21_healthy_ci"] = [r(v) for v in boot_two(lambda d_, h_: np.mean((d_[:, None] > h_[None, :]) +
                                                            0.5 * (d_[:, None] == h_[None, :])), ps, hs)]
        sens[f"{DISEASE[m]}/{name}"] = e
res["sensitivity_test_diseased"] = sens

# ------------------------------------------------------------------ 3) specificity on the 21 local healthy leaves
spec = {}
for m in ("roya", "ojo"):
    for a, name in ARCHS:
        L = sel(m, a, "roya", "test", "healthy")
        assert len(L) == 21
        variants = {"primary_model_rule": np.array([x["pred_sev"] for x in L]),
                    "abaxial": np.array([x["sev_abaxial"] for x in L]),
                    "adaxial": np.array([x["sev_adaxial"] for x in L]),
                    "either_face_max": np.array([x["sev_max"] for x in L])}
        e = {"rule": MODEL_RULE[m], "n_healthy": len(L)}
        for vn, v in variants.items():
            fp0, fp5 = int((v > 0).sum()), int((v >= 0.5).sum())
            e[vn] = {"fp_gt0": fp0, "fp_ge05": fp5,
                     "specificity_gt0": prop(len(L) - fp0, len(L)), "specificity_ge05": prop(len(L) - fp5, len(L)),
                     "max_pred_sev": r(v.max()), "flagged_gt0": [
                         {"leaf_id": x["leaf_id"], "sev": r(val)} for x, val in zip(L, v) if val > 0]}
        # size of the predicted "lesion" on healthy leaves (letterbox-space pixels on the rule's face(s), summed)
        faces_used = ["abaxial"] if MODEL_RULE[m] == "abaxial" else ["abaxial", "adaxial"]
        px = np.array([sum(int(faces[(m, a, "roya", "test", x["leaf_id"])][f]["pred_lesion_px"]) for f in faces_used) for x in L])
        e["pred_lesion_px_on_rule_faces"] = {"median": float(np.median(px)), "max": int(px.max()),
                                             "median_among_flagged": float(np.median(px[px > 0])) if (px > 0).any() else 0.0}
        e["pred_sev_median_primary"] = r(np.median(variants["primary_model_rule"]))
        e["pred_grade_counts_primary"] = [int((np.array([x["pred_grade"] for x in L]) == g).sum()) for g in range(4)]
        spec[f"{'rust' if m == 'roya' else 'leaf_spot'}/{name}"] = e
res["specificity_local_healthy_test"] = spec
_fp0 = [v["primary_model_rule"]["fp_gt0"] for v in spec.values()]
_maxleaf = max(v["primary_model_rule"]["max_pred_sev"] for v in spec.values())
_maxface = max(v["either_face_max"]["max_pred_sev"] for v in spec.values())
res["definition_check"] = (
    "The manuscript defines a healthy-leaf false positive as ANY predicted lesion pixel and reports 0/54 for four models. "
    "Local re-inference with the exact published protocol (raw argmax in letterbox space; it reproduces Tables 2/4 exactly) "
    f"gives at least one predicted lesion pixel on {min(_fp0)}–{max(_fp0)} of the 21 local healthy test leaves, "
    "depending on the model. These 21 leaves are part of the published 54, so the published counts cannot have come "
    "from the any-pixel rule under this protocol. They are compatible with a severity threshold: the highest "
    f"leaf-level predicted severity on a local healthy leaf is {_maxleaf:.2f}% under the paper's leaf-level rules, and "
    f"the highest single-face value is {_maxface:.2f}%. The training kernels define a healthy-leaf false positive as "
    "predicted severity ≥5% per image. The threshold behind the published 54-leaf counts could not be recovered.")
res["definition_check_numbers"] = {"fp_gt0_range_over_models": [min(_fp0), max(_fp0)],
                                   "max_leaf_level_pred_sev_healthy": _maxleaf, "max_single_face_pred_sev_healthy": _maxface}
res["specificity_published_54"] = {"note": "Published counts (any predicted lesion pixel) on 54 healthy test leaves; not "
                                           "reproducible locally beyond the 21 leaves above.",
                                   "rust": {"SegFormer-B5": "0/54", "UNet++": "1/54", "DeepLabV3+": "0/54"},
                                   "leaf_spot": {"SegFormer-B5": "0/54", "UNet++": "0/54", "DeepLabV3+": "1/54"}}

# Supplementary: rust models on the 21 healthy VALIDATION leaves (not independent: rust CNNs were selected with a
# healthy-validation false-positive penalty; SegFormer-B5 on validation lesion IoU). Reported for completeness only.
sup = {}
for a, name in ARCHS:
    L = sel("roya", a, "roya", "val", "healthy"); v = np.array([x["pred_sev"] for x in L])
    sup[f"rust/{name}"] = {"n": len(L), "fp_gt0": int((v > 0).sum()), "fp_ge05": int((v >= 0.5).sum()),
                           "max_pred_sev": r(v.max())}
res["supplementary_rust_models_on_healthy_validation_leaves_NOT_independent"] = sup

# ---------------------------------------------------------------------------------------------- 4) cross-disease
def lesion_location(m, a, p, L, key, refsev):
    """Where the predicted lesion pixels of model (m, a) fall on the leaves L of pack p, on one face set.
    key = 'modelrule' (faces the model reads; same faces as its pred_sev / grade / >=0.5%) or 'leafrule' (faces of the
    leaf's own-disease rule). refsev = leaf-level reference severity (leaf's own-disease rule), as in the ratio columns."""
    rule = MODEL_RULE[m] if key == "modelrule" else MODEL_RULE[p]
    fu = ["abaxial"] if rule == "abaxial" else ["abaxial", "adaxial"]
    g = lambda c: np.array([x[f"{c}_{key}"] for x in L], float)
    pl, it, off, ins, iou = g("pred_les_px"), g("inter_px"), g("off_lesion_sev"), g("in_lesion_sev"), g("iou")
    ps = np.array([x["pred_sev"] if key == "modelrule" else x["pred_sev_leafrule"] for x in L])
    assert np.allclose(off + ins, ps, atol=1e-3)          # off-lesion + in-lesion severity = predicted severity
    share = it.sum() / pl.sum() if pl.sum() else float("nan")
    top = []
    for i in np.argsort(-off, kind="stable")[:3]:
        fr = faces[(m, a, p, "test", L[i]["leaf_id"])]
        top.append({"leaf_id": L[i]["leaf_id"], "off_lesion_pred_sev": r(off[i]), "pred_sev": r(ps[i]),
                    "ref_sev_leafrule": r(refsev[i]), "per_face": {f: {
                        "pred_lesion_px": int(fr[f]["pred_lesion_px"]),
                        "pred_lesion_px_outside_reference": int(fr[f]["pred_lesion_px"]) - int(fr[f]["inter_lesion"]),
                        "pred_leaf_px": int(fr[f]["pred_leaf_px"]),
                        "off_lesion_pct_of_leaf": r(100 * (int(fr[f]["pred_lesion_px"]) - int(fr[f]["inter_lesion"]))
                                                    / int(fr[f]["pred_leaf_px"]))} for f in fu}})
    return {"faces": "+".join(fu), "rule": rule, "n_leaves": len(L),
            "share_pred_lesion_px_inside_reference_pooled": r(share),
            "share_inside_ci": [r(v) for v in boot(lambda a_, b_: a_.sum() / b_.sum() if b_.sum() else np.nan, it, pl)],
            "share_pred_lesion_px_outside_reference_pooled": r(1 - share),
            "lesion_iou_mean": r(np.nanmean(iou)), "n_leaves_iou_defined": int(np.isfinite(iou).sum()),
            "off_lesion_pred_sev_median": r(np.median(off)),
            "off_lesion_pred_sev_iqr": [r(np.percentile(off, 25)), r(np.percentile(off, 75))],
            "off_lesion_pred_sev_mean": r(off.mean()), "off_lesion_pred_sev_max": r(off.max()),
            "in_lesion_pred_sev_mean": r(ins.mean()), "pred_sev_mean": r(ps.mean()),
            "ratio_mean_pred_sev_over_reference": r(ps.mean() / refsev.mean()),
            "ratio_mean_in_lesion_sev_over_reference": r(ins.mean() / refsev.mean()),
            "n_leaves_pred_sev_ge05": int((ps >= 0.5).sum()),
            "n_leaves_off_lesion_sev_ge05": int((off >= 0.5).sum()),
            "n_leaves_ge05_only_with_off_lesion_px": int(((ps >= 0.5) & (ins < 0.5)).sum()),
            "top3_off_lesion_leaves": top}


cross = {}
for p in ("roya", "ojo"):                       # p = disease on the leaves
    other = "ojo" if p == "roya" else "roya"   # model disease that is WRONG for these leaves
    block = {"leaves": f"{DISEASE[p]} diseased test leaves", "n": None, "reference": {}, "correct_models": {},
             "cross_models": {}, "paired_cross_vs_correct_same_arch": {}}
    ref_L = sel(p, "segformer", p, "test", DISEASE[p])
    refsev = np.array([x["ref_sev_leafrule"] for x in ref_L])
    block["n"] = len(ref_L)
    block["reference"] = {"median_ref_sev": r(np.median(refsev)), "iqr": [r(np.percentile(refsev, 25)), r(np.percentile(refsev, 75))],
                          "grade_counts_G0_G3": [int((np.array([x["ref_grade"] for x in ref_L]) == g).sum()) for g in range(4)]}
    for role, m in (("correct_models", p), ("cross_models", other)):
        for a, name in ARCHS:
            L = sel(m, a, p, "test", DISEASE[p])
            assert [x["leaf_id"] for x in L] == [x["leaf_id"] for x in ref_L]
            ps = np.array([x["pred_sev"] for x in L]); ps_alt = np.array([x["pred_sev_leafrule"] for x in L])
            H = sel(m, a, "roya", "test", "healthy"); hs = np.array([x["pred_sev"] for x in H])
            rho = spearmanr(ps_alt, refsev)
            e = {"model_rule": MODEL_RULE[m],
                 "pred_sev_median": r(np.median(ps)), "pred_sev_median_ci": [r(v) for v in boot(np.median, ps)],
                 "pred_sev_iqr": [r(np.percentile(ps, 25)), r(np.percentile(ps, 75))],
                 "pred_sev_mean": r(ps.mean()), "pred_sev_mean_ci": [r(v) for v in boot(np.mean, ps)],
                 "pred_sev_max": r(ps.max()),
                 "gt0": prop(int((ps > 0).sum()), len(L)), "ge05": prop(int((ps >= 0.5).sum()), len(L)),
                 "pred_grade_counts_G0_G3": [int((np.array([x["pred_grade"] for x in L]) == g).sum()) for g in range(4)],
                 "alt_rule_leaf_disease": {"rule": MODEL_RULE[p], "pred_sev_median": r(np.median(ps_alt)),
                                           "ge05": prop(int((ps_alt >= 0.5).sum()), len(L)),
                                           "pred_grade_counts_G0_G3": [int((np.array([x["pred_grade_leafrule"] for x in L]) == g).sum()) for g in range(4)]},
                 "per_face": {f: {"median": r(np.median([x[f"sev_{f}"] for x in L])),
                                  "ge05": prop(int((np.array([x[f"sev_{f}"] for x in L]) >= 0.5).sum()), len(L))}
                              for f in ("abaxial", "adaxial")},
                 # where do the predicted lesion pixels fall? Two face sets (see meta.lesion_location_face_sets);
                 # model_rule_faces matches the severity / grade / >=0.5% values above
                 "lesion_location": {"model_rule_faces": lesion_location(m, a, p, L, "modelrule", refsev),
                                     "leaf_rule_faces": lesion_location(m, a, p, L, "leafrule", refsev)},
                 "spearman_pred_vs_ref_sev": {"rho": r(rho.statistic), "p": float(rho.pvalue)},
                 "same_model_on_21_healthy_EARTH_leaves": {"pred_sev_median": r(np.median(hs)), "max": r(hs.max()),
                                                          "ge05": f"{int((hs >= 0.5).sum())}/{len(hs)}"}}
            block[role][name] = e
    for a, name in ARCHS:                      # paired comparison on the same leaves
        Lc = sel(p, a, p, "test", DISEASE[p]); Lx = sel(other, a, p, "test", DISEASE[p])
        c = np.array([x["pred_sev"] for x in Lc]); xx = np.array([x["pred_sev"] for x in Lx])
        block["paired_cross_vs_correct_same_arch"][name] = {
            "diff_frac_ge05_cross_minus_correct": r(np.mean(xx >= 0.5) - np.mean(c >= 0.5)),
            "diff_frac_ge05_ci": [r(v) for v in boot(lambda a_, b_: np.mean(a_ >= 0.5) - np.mean(b_ >= 0.5), xx, c)],
            "ratio_mean_pred_sev_cross_over_correct": r(xx.mean() / c.mean()),
            "ratio_mean_ci": [r(v) for v in boot(lambda a_, b_: a_.mean() / b_.mean(), xx, c)],
            "ratio_mean_pred_sev_cross_over_reference": r(xx.mean() / refsev.mean()),
            "ratio_mean_ref_ci": [r(v) for v in boot(lambda a_, b_: a_.mean() / b_.mean(), xx, refsev)],
            "n_leaves_cross_ge05_and_correct_ge05": int(((xx >= 0.5) & (c >= 0.5)).sum()),
            "n_leaves_cross_ge05_and_correct_lt05": int(((xx >= 0.5) & (c < 0.5)).sum()),
            "n_leaves_cross_grade_equal_correct_grade": int(sum(g1 == g2 for g1, g2 in zip(
                [x["pred_grade"] for x in Lx], [x["pred_grade"] for x in Lc])))}
    cross[f"on_{DISEASE[p]}_leaves"] = block
res["cross_disease"] = cross
if os.path.exists(f"{OUT}/verify_per_image.json"):   # written by verify_per_image.py (pixel counts re-derived from maps)
    _v = json.load(open(f"{OUT}/verify_per_image.json"))
    res["input_verification"] = {k: _v[k] for k in ("rows_checked", "rows_matching_all_pixel_counts", "n_mismatches")}

json.dump(res, open(f"{OUT}/results.json", "w"), indent=2)
print("reproduction all_match:", ok_all)
