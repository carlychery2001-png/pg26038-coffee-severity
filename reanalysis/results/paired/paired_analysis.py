"""Paired between-architecture comparisons within disease (PG-26038 major revision).

Reviewer issue addressed: severity and grade differences between architectures were never tested, and the
Conclusions made an implicit non-inferiority claim (grade agreement) without a margin or a test.

Input (reproduced exactly; no model inference is re-run here):
  <ROOT>/reanalysis/out/per_image.csv   one row per (model disease, arch, image pack, split, image)

Leaf-level rules (as in the paper):
  * rust (model_disease = image_pack = 'roya'): abaxial face only; diseased leaves = grade_label 'roya'.
  * American leaf spot ('ojo'): mean of the two faces (IoU = mean of face IoUs, severity = mean of face severities).
  * Reference grade: rust = cut-points 0.5 / 5 / 20 % on reference severity; leaf spot = annotator visual grade
    (grade_label G1/G2/G3). Secondary leaf-spot reference: MASK-DERIVED grade (same cut-points on reference
    severity, leaf-level mean of both faces).
  * Healthy leaves: rust-pack TEST leaves with grade_label 'healthy' (EARTH site). Only 21 of the paper's 54 are
    present locally. Rust models: abaxial face; leaf-spot models (cross-disease runs, image_pack = 'roya'):
    mean of both faces. Used only for a separate G0-recall row; never mixed into accuracy/QWK.

Statistics:
  * Paired bootstrap: 10,000 resamples of leaves (with replacement), one index matrix per disease drawn from
    numpy default_rng(20260927) in the order rust, leaf spot, and reused for every model, pair and metric
    (fully paired design). Percentile intervals (numpy linear interpolation). Resamples in which a statistic is
    undefined (QWK 0/0) are dropped and counted.
  * Differences are first model minus second model, pair order as in the paper's Table 3:
    SegFormer-B5 vs DeepLabV3+, SegFormer-B5 vs UNet++, UNet++ vs DeepLabV3+.
    dMAE < 0 and d|bias| < 0 favour the first model; dQWK > 0 and dAccuracy > 0 favour the first model.
  * Two-sided bootstrap p = min(1, 2 min(P*(d <= 0), P*(d >= 0))) (inverts the percentile interval), Holm across
    the three pairs within disease for each metric; supplementary to the intervals.
  * Exact McNemar test on per-leaf grade correctness: two-sided exact binomial test on the discordant pairs
    (b = first correct & second wrong, c = first wrong & second correct), p = 1 when b + c = 0.
    Holm correction across the three pairs within each disease (and within each reference standard).
  * Paired difference in grade accuracy: in addition to the bootstrap, Tango's (1998) asymptotic score interval
    (primary for accuracy, because the percentile bootstrap collapses to [0, 0] or has a bound pinned at 0 when
    discordant leaves are absent or all favour one model) and Newcombe's (1998) method-10 interval (cross-check).
  * Wilson score 95% intervals (no continuity correction) for grade accuracy and per-grade recall.
  * QWK with quadratic weights on the grade values 0..3 (equivalent to Cohen's weighted kappa with a fixed
    4-level scale; independent of which categories happen to occur).
  * Equivalence framing (descriptive, no margin was pre-specified): the smallest symmetric margin that a TOST at
    alpha = 0.05 would accept is max(|lower|, |upper|) of a 90% interval. For dAccuracy the 90% interval is Tango's
    score interval (the 90% percentile bootstrap is degenerate here: [0, 0] or one bound pinned at 0 whenever
    discordant leaves are absent or all on one side, which would understate the margin). For the other metrics it is
    the 90% percentile interval; when that interval has a bound pinned at exactly 0 (dQWK here) no margin is given
    (tost_margin_needed = null), because the value would be anti-conservative.
  * Comparisons are between the published trained models. Their training protocols differed (revision brief facts
    3-4), so the differences are not controlled architecture effects.

Outputs (this folder): results.json (all numbers), stdout log. Figures: make_figures.py.
"""
from __future__ import annotations
import os as _os  # (release) all paths below are relative to ROOT (repository root by default; see README)
ROOT = (_os.environ.get("PG26038_ROOT") or _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))).replace(chr(92), "/")

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import binomtest, norm

OUT = Path(__file__).resolve().parent
CSV = Path((ROOT + "/reanalysis/out/per_image.csv"))
SEED = 20260927
B = 10_000
CUTS = (0.5, 5.0, 20.0)
Z = float(norm.ppf(0.975))

DISEASES = ["roya", "ojo"]
DIS_NAME = {"roya": "Coffee leaf rust", "ojo": "American leaf spot"}
ARCHS = ["segformer", "unetpp", "deeplab"]
ARCH_NAME = {"segformer": "SegFormer-B5", "unetpp": "UNet++", "deeplab": "DeepLabV3+"}
PAIRS = [("segformer", "deeplab"), ("segformer", "unetpp"), ("unetpp", "deeplab")]

# Published values (manuscript Tables 2-5 and the Bland-Altman paragraph) used as sanity checks.
PUB = {
    "roya": {
        "segformer": dict(iou=0.767, mae=0.99, bias=0.99, qwk=0.88, acc=0.90, loa=(-1.68, 3.66)),
        "unetpp": dict(iou=0.647, mae=1.77, bias=1.75, qwk=0.79, acc=0.81, loa=(-2.68, 6.17)),
        "deeplab": dict(iou=0.632, mae=1.47, bias=1.41, qwk=0.88, acc=0.90, loa=(-3.07, 5.89)),
    },
    "ojo": {
        "segformer": dict(iou=0.884, mae=0.22, bias=0.16, qwk=0.82, acc=0.84, loa=(-0.47, 0.79)),
        "unetpp": dict(iou=0.853, mae=0.15, bias=-0.03, qwk=0.85, acc=0.87, loa=(-0.50, 0.44)),
        "deeplab": dict(iou=0.815, mae=0.25, bias=0.08, qwk=0.84, acc=0.85, loa=(-0.66, 0.82)),
    },
}
PUB_N = {"roya": 21, "ojo": 55}
# Published Table 3 (paired IoU differences), used as a cross-check of the pairing.
PUB_DIOU = {
    ("roya", "segformer", "deeplab"): (0.136, 0.106, 0.169),
    ("roya", "segformer", "unetpp"): (0.121, 0.093, 0.150),
    ("roya", "unetpp", "deeplab"): (0.015, -0.003, 0.034),
    ("ojo", "segformer", "deeplab"): (0.070, 0.061, 0.080),
    ("ojo", "segformer", "unetpp"): (0.031, 0.023, 0.041),
    ("ojo", "unetpp", "deeplab"): (0.038, 0.034, 0.044),
}


# ----------------------------------------------------------------------------------------------- helpers
def grade(sev):
    """G0 < 0.5 <= G1 < 5 <= G2 < 20 <= G3 (severity in %)."""
    return np.digitize(np.asarray(sev, float), CUTS).astype(int)


def qwk(a, b):
    """Quadratic-weighted kappa on integer grades (vectorised over the last axis).

    kappa_w = 1 - sum (a-b)^2 / sum_{i,j} (a_i - b_j)^2 / n  =  2 cov / (var_a + var_b + (mean_a - mean_b)^2)
    (population moments). NaN when the denominator is 0 (both raters constant and equal).
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = a.shape[-1]
    num = ((a - b) ** 2).sum(-1)
    den = (a ** 2).sum(-1) + (b ** 2).sum(-1) - 2.0 * a.sum(-1) * b.sum(-1) / n
    with np.errstate(invalid="ignore", divide="ignore"):
        k = 1.0 - num / den
    return np.where(den > 1e-12, k, np.nan)


def qwk_matrix(a, b, k=4):
    """Reference implementation from the confusion matrix (cross-check of qwk())."""
    O = np.zeros((k, k))
    for i, j in zip(a, b):
        O[i, j] += 1
    n = O.sum()
    E = np.outer(O.sum(1), O.sum(0)) / n
    W = np.subtract.outer(np.arange(k), np.arange(k)) ** 2
    den = (W * E).sum()
    return float("nan") if den == 0 else 1 - (W * O).sum() / den


def wilson(k, n):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = (Z / d) * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    lo, hi = max(0.0, c - h), min(1.0, c + h)
    ci = binomtest(k, n).proportion_ci(0.95, method="wilson")
    assert abs(ci.low - lo) < 1e-12 and abs(ci.high - hi) < 1e-12
    return [lo, hi]


def pct_ci(x, level=0.95):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    a = (1 - level) / 2 * 100
    lo, hi = np.percentile(x, [a, 100 - a])
    return [float(lo), float(hi)]


def holm(ps):
    m = len(ps)
    order = np.argsort(ps)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def tango_ci(b, c, n, level=0.95):
    """Tango (1998) asymptotic score CI (default 95%) for a paired difference of proportions, (b - c) / n.

    b = leaves only the first model grades correctly, c = leaves only the second grades correctly.
    Constrained MLE of p21 in closed form; limits found by root finding. Unlike the percentile bootstrap it
    stays informative when there are no or only one-sided discordant leaves.
    Check: Bentur et al. data (a,b,c,d = 1,1,7,12) gives -0.517 to -0.026 (Fagerland et al. 2014).
    """
    Z = float(norm.ppf(1 - (1 - level) / 2))  # local critical value (shadows the module-level 95% Z)
    D = (b - c) / n

    def T(d):
        W = -b - c + (2 * n - b + c) * d
        p21 = (math.sqrt(W * W + 8 * n * c * d * (1 - d)) - W) / (4 * n)
        v = n * (2 * p21 + d - d * d)
        num = b - c - n * d
        if v <= 1e-300:
            return 0.0 if num == 0 else math.copysign(math.inf, num)
        return num / math.sqrt(v)

    e = 1e-10
    lo = -1.0 if T(-1 + e) < Z else brentq(lambda x: T(x) - Z, -1 + e, D, xtol=1e-13)
    hi = 1.0 if T(1 - e) > -Z else brentq(lambda x: T(x) + Z, D, 1 - e, xtol=1e-13)
    return [float(lo), float(hi)]


def newcombe_ci(a, b, c, d):
    """Newcombe (1998) method 10 hybrid score 95% CI for a paired difference of proportions (cross-check).

    Check: Bentur et al. data (1,1,7,12) gives -0.507 to -0.026.
    """
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n)
    l2, u2 = wilson(a + c, n)
    den = (a + b) * (c + d) * (a + c) * (b + d)
    if den == 0:
        phi = 0.0
    elif a * d > b * c:
        phi = max(a * d - b * c - n / 2, 0) / math.sqrt(den)
    else:
        phi = (a * d - b * c) / math.sqrt(den)
    dl = math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    du = math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return [p1 - p2 - dl, p1 - p2 + du]


def mcnemar_exact(ca, cb):
    ca = np.asarray(ca, bool)
    cb = np.asarray(cb, bool)
    b = int((ca & ~cb).sum())
    c = int((~ca & cb).sum())
    p = 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)
    return dict(first_correct_second_wrong=b, first_wrong_second_correct=c, discordant=b + c,
                both_correct=int((ca & cb).sum()), both_wrong=int((~ca & ~cb).sum()), p_exact=p)


def r2(x, k):
    return None if x is None else round(float(x), k)


# ----------------------------------------------------------------------------------------------- data
def load_rows():
    with CSV.open(newline="") as fh:
        return list(csv.DictReader(fh))


def leaf_table(rows, dis, arch):
    """Diseased test leaves of one within-disease model, leaf-level values, sorted by leaf_id."""
    sel = [r for r in rows if r["model_disease"] == dis and r["arch"] == arch and r["image_pack"] == dis
           and r["split"] == "test"]
    if dis == "roya":
        sel = [r for r in sel if r["grade_label"] == "roya" and r["face"] == "abaxial"]
    faces = defaultdict(list)
    for r in sel:
        faces[r["leaf_id"]].append(r)
    out = {}
    for lid, rs in sorted(faces.items()):
        if dis == "ojo":
            assert sorted(x["face"] for x in rs) == ["abaxial", "adaxial"], lid
            assert len({x["grade_label"] for x in rs}) == 1, lid
        else:
            assert len(rs) == 1, lid
        out[lid] = dict(
            iou=float(np.mean([float(x["lesion_iou"]) for x in rs])),
            ref_sev=float(np.mean([float(x["ref_sev"]) for x in rs])),
            pred_sev=float(np.mean([float(x["pred_sev"]) for x in rs])),
            label=rs[0]["grade_label"],
        )
    return out


def healthy_table(rows, dis, arch):
    """Healthy rust-pack TEST leaves (EARTH) scored by a model of disease `dis`.

    Leaf-level rule: rust models = abaxial face (pred_sev, any_lesion_px); leaf-spot models = mean of both faces
    (any_lesion_px = a lesion pixel on either face). In addition, for every model, the either-face summaries
    (any_lesion_px_either_face, max_face_sev) use both faces, as in revision_brief item 10.
    """
    sel_all = [r for r in rows if r["model_disease"] == dis and r["arch"] == arch and r["image_pack"] == "roya"
               and r["split"] == "test" and r["grade_label"] == "healthy"]
    faces = defaultdict(list)
    for r in sel_all:
        faces[r["leaf_id"]].append(r)
    out = {}
    for lid, rs_all in sorted(faces.items()):
        assert sorted(x["face"] for x in rs_all) == ["abaxial", "adaxial"], lid
        rs = [x for x in rs_all if x["face"] == "abaxial"] if dis == "roya" else rs_all
        out[lid] = dict(pred_sev=float(np.mean([float(x["pred_sev"]) for x in rs])),
                        any_lesion_px=bool(sum(int(x["pred_lesion_px"]) for x in rs) > 0),
                        any_lesion_px_either_face=bool(sum(int(x["pred_lesion_px"]) for x in rs_all) > 0),
                        max_face_sev=float(max(float(x["pred_sev"]) for x in rs_all)),
                        ref_lesion_px=int(sum(int(x["ref_lesion_px"]) for x in rs_all)))
    return out


# ----------------------------------------------------------------------------------------------- analysis
def main():
    t = tango_ci(1, 7, 21)
    nc = newcombe_ci(1, 1, 7, 12)
    assert abs(t[0] + 0.517) < 5e-4 and abs(t[1] + 0.026) < 5e-4, t
    t90 = tango_ci(1, 7, 21, 0.90)
    assert t[0] < t90[0] < -6 / 21 < t90[1] < t[1], t90  # 90% interval nested inside the 95% interval
    assert abs(nc[0] + 0.507) < 5e-4 and abs(nc[1] + 0.026) < 5e-4, nc
    rows = load_rows()
    rng = np.random.default_rng(SEED)
    res = dict(
        description="Paired between-architecture comparisons within disease; leaf-level; test partition.",
        source=str(CSV),
        bootstrap=dict(resamples=B, seed=SEED, generator="numpy.random.default_rng", unit="leaf",
                       interval="percentile (numpy linear interpolation)",
                       pairing="one index matrix per disease (drawn in order rust, leaf spot), shared by all "
                               "models, pairs and metrics",
                       undefined_resamples="dropped and counted (QWK 0/0 only)"),
        grade_cut_points_pct=list(CUTS),
        sign_convention="difference = first model minus second model; dMAE<0 and d|bias|<0 favour the first "
                        "model; dQWK>0 and dAccuracy>0 favour the first model",
        mcnemar="exact two-sided binomial test on discordant leaves (grade correct vs wrong); Holm across the "
                "three pairs within disease and reference standard",
        wilson="Wilson score interval, two-sided 95%, no continuity correction",
        paired_accuracy_interval="Tango (1998) asymptotic score interval (primary for dAccuracy); Newcombe (1998) "
                                 "method 10 as cross-check; percentile bootstrap also given",
        equivalence_note="No equivalence or non-inferiority margin was pre-specified. 'tost_margin_needed' is "
                         "the smallest symmetric margin a TOST at alpha=0.05 (90% CI inside +/-margin) would "
                         "accept; descriptive only. For d_accuracy (and d_accuracy_mask_derived) it comes from the "
                         "Tango score 90% interval ('tango_score_ci90'), because the 90% percentile bootstrap is "
                         "[0, 0] or has a bound pinned at 0 for every pair and would understate the margin. For "
                         "other metrics it comes from the 90% percentile interval and is null when that interval "
                         "has a bound pinned at exactly 0 (all d_qwk here); see 'tost_margin_basis'.",
        comparison_note="All comparisons are between the published trained models. Training protocols differed "
                        "(revision brief facts 3-4: the published CNNs used photometric jitter p=0.8 and, for rust, "
                        "at most 30 epochs, patience 12 and FP-penalised model selection; SegFormer-B5 used "
                        "geometric augmentation only, up to 60 epochs, patience 15). The differences are therefore "
                        "not controlled architecture effects (see revision brief item 11 for matched-protocol rust "
                        "retraining).",
        coverage_note="Healthy rust-pack test leaves: 21 present locally (EARTH site, collected 2026-06-12/15/22); "
                      "the paper used 54. Healthy leaves enter only the separate G0-recall rows.",
        reproduction={}, per_model={}, paired={}, healthy_g0={}, leafspot_reference_agreement={},
    )
    all_ok = True
    log = []

    for dis in DISEASES:
        T = {a: leaf_table(rows, dis, a) for a in ARCHS}
        ids = list(T["segformer"])
        for a in ARCHS:
            assert list(T[a]) == ids, (dis, a)
        n = len(ids)
        assert n == PUB_N[dis], (dis, n)
        ref_sev = np.array([T["segformer"][i]["ref_sev"] for i in ids])
        for a in ARCHS:  # reference severity must be identical across models (same reference masks)
            assert np.allclose([T[a][i]["ref_sev"] for i in ids], ref_sev, atol=1e-9), (dis, a)
        g_mask = grade(ref_sev)
        if dis == "roya":
            g_ref = g_mask
            refs = {"reference": g_ref}
        else:
            g_ref = np.array([int(T["segformer"][i]["label"][1]) for i in ids])
            refs = {"reference": g_ref, "mask_derived": g_mask}

        idx = rng.integers(0, n, size=(B, n))
        V = {}
        for a in ARCHS:
            iou = np.array([T[a][i]["iou"] for i in ids])
            ps = np.array([T[a][i]["pred_sev"] for i in ids])
            d = ps - ref_sev
            gp = grade(ps)
            V[a] = dict(iou=iou, pred_sev=ps, diff=d, absdiff=np.abs(d), g=gp)

        # ---------------- per-model metrics + reproduction checks
        res["per_model"][dis] = {}
        res["reproduction"][dis] = {}
        for a in ARCHS:
            v = V[a]
            sd = float(np.std(v["diff"], ddof=1))
            bias = float(v["diff"].mean())
            m = dict(
                n_leaves=n,
                leaf_iou=float(v["iou"].mean()),
                mae_pp=float(v["absdiff"].mean()),
                bias_pp=bias,
                abs_bias_pp=abs(bias),
                sd_diff_pp=sd,
                loa_pp=[bias - 1.96 * sd, bias + 1.96 * sd],
                rmse_pp=float(np.sqrt((v["diff"] ** 2).mean())),
                pearson_r=float(np.corrcoef(v["pred_sev"], ref_sev)[0, 1]),
                # per-leaf signed errors (predicted - reference severity): sign of the mean bias is not the sign
                # of every leaf's error
                n_leaves_negative_error=int((v["diff"] < 0).sum()),
                n_leaves_positive_error=int((v["diff"] > 0).sum()),
                min_signed_error_pp=float(v["diff"].min()),
                max_signed_error_pp=float(v["diff"].max()),
            )
            # bootstrap CIs of single-model quantities (same resamples as the paired analysis)
            m["leaf_iou_ci95"] = pct_ci(v["iou"][idx].mean(1))
            m["mae_ci95"] = pct_ci(v["absdiff"][idx].mean(1))
            m["bias_ci95"] = pct_ci(v["diff"][idx].mean(1))
            m["grades"] = {}
            for rname, gr in refs.items():
                corr = v["g"] == gr
                k = int(corr.sum())
                qb = qwk(v["g"][idx], gr[idx])
                q = float(qwk(v["g"], gr))
                assert abs(q - qwk_matrix(v["g"], gr)) < 1e-12
                cm = np.zeros((4, 4), int)
                for i, j in zip(gr, v["g"]):
                    cm[i, j] += 1
                recall = {}
                for g in range(4):
                    ng = int((gr == g).sum())
                    kg = int(((gr == g) & corr).sum())
                    recall[f"G{g}"] = dict(n=ng, correct=kg, recall=(kg / ng if ng else None),
                                           wilson95=wilson(kg, ng),
                                           over_graded=int(((gr == g) & (v["g"] > gr)).sum()),
                                           under_graded=int(((gr == g) & (v["g"] < gr)).sum()))
                m["grades"][rname] = dict(
                    accuracy=k / n, correct=k, n=n, accuracy_wilson95=wilson(k, n),
                    accuracy_bootstrap_ci95=pct_ci(corr[idx].mean(1)),
                    qwk=q, qwk_ci95=pct_ci(qb), qwk_undefined_resamples=int(np.isnan(qb).sum()),
                    qwk_resamples_equal_1=int((qb == 1).sum()),
                    confusion_rows_reference_cols_predicted_G0_to_G3=cm.tolist(),
                    per_grade_recall=recall,
                    errors_adjacent_only=bool(np.all(np.abs(v["g"] - gr) <= 1)),
                    over_graded=int((v["g"] > gr).sum()), under_graded=int((v["g"] < gr).sum()),
                )
            res["per_model"][dis][a] = m

            p = PUB[dis][a]
            chk = dict(
                leaf_iou=(round(m["leaf_iou"], 3), p["iou"]),
                mae=(round(m["mae_pp"], 2), p["mae"]),
                bias=(round(bias, 2), p["bias"]),
                qwk=(round(m["grades"]["reference"]["qwk"], 2), p["qwk"]),
                accuracy=(round(m["grades"]["reference"]["accuracy"], 2), p["acc"]),
                loa_low=(round(m["loa_pp"][0], 2), p["loa"][0]),
                loa_high=(round(m["loa_pp"][1], 2), p["loa"][1]),
            )
            ok = all(abs(x - y) < 1e-9 for x, y in chk.values())
            all_ok &= ok
            res["reproduction"][dis][a] = dict(match=ok, reproduced_vs_published={k: list(v2) for k, v2 in chk.items()})
            log.append(f"{dis:4s} {a:9s} IoU {m['leaf_iou']:.4f} MAE {m['mae_pp']:.4f} bias {bias:+.4f} "
                       f"LoA [{m['loa_pp'][0]:+.3f}, {m['loa_pp'][1]:+.3f}] QWK {m['grades']['reference']['qwk']:.4f} "
                       f"acc {m['grades']['reference']['correct']}/{n}  -> {'OK' if ok else 'MISMATCH ' + str(chk)}"
                       f"  | negative errors {m['n_leaves_negative_error']}/{n} (min {m['min_signed_error_pp']:+.3f} pp)")

        # ---------------- paired comparisons
        res["paired"][dis] = {}
        mc_p = {r: [] for r in refs}
        for a, b in PAIRS:
            va, vb = V[a], V[b]
            key = f"{a}_vs_{b}"
            out = dict(first=ARCH_NAME[a], second=ARCH_NAME[b], n_leaves=n)

            def summarise(point, boot, extra=None):
                d = dict(estimate=float(point), ci95=pct_ci(boot), ci90=pct_ci(boot, 0.90),
                         ci98_33_bonferroni3=pct_ci(boot, 1 - 0.05 / 3),
                         undefined_resamples=int(np.isnan(boot).sum()),
                         boot_share_below_0=float(np.nanmean(boot < 0)),
                         boot_share_above_0=float(np.nanmean(boot > 0)))
                # two-sided bootstrap p that inverts the percentile interval (conservative when mass sits at 0)
                d["p_boot"] = float(min(1.0, 2 * min(np.nanmean(boot <= 0), np.nanmean(boot >= 0))))
                lo90, hi90 = d["ci90"]
                d["ci90_bound_pinned_at_0"] = bool(abs(lo90) < 1e-12 or abs(hi90) < 1e-12)
                if d["ci90_bound_pinned_at_0"]:
                    # one-sided or absent discordance: the percentile bound sits exactly at 0 and a margin taken
                    # from it would be anti-conservative
                    d["tost_margin_needed"] = None
                    d["tost_margin_basis"] = ("not given: 90% percentile bootstrap interval has a bound pinned at 0 "
                                              "(degenerate)")
                else:
                    d["tost_margin_needed"] = float(max(abs(lo90), abs(hi90)))
                    d["tost_margin_basis"] = "90% percentile bootstrap interval"
                d["ci95_excludes_0"] = bool(d["ci95"][0] > 0 or d["ci95"][1] < 0)
                if extra:
                    d.update(extra)
                return d

            # IoU (cross-check with Table 3)
            out["d_leaf_iou"] = summarise(va["iou"].mean() - vb["iou"].mean(),
                                          (va["iou"] - vb["iou"])[idx].mean(1))
            pub = PUB_DIOU[(dis, a, b)]
            out["d_leaf_iou"]["published_table3"] = list(pub)
            rep = [out["d_leaf_iou"]["estimate"]] + out["d_leaf_iou"]["ci95"]
            out["d_leaf_iou"]["table3_mismatches_3dp"] = [
                dict(quantity=q, reproduced=float(x), reproduced_3dp=round(float(x), 3), published=y)
                for q, x, y in zip(["estimate", "ci95_low", "ci95_high"], rep, pub) if abs(round(float(x), 3) - y) > 1e-9]
            # MAE
            out["d_mae_pp"] = summarise(va["absdiff"].mean() - vb["absdiff"].mean(),
                                        (va["absdiff"] - vb["absdiff"])[idx].mean(1))
            # |bias|
            ba, bb = va["diff"][idx].mean(1), vb["diff"][idx].mean(1)
            out["d_abs_bias_pp"] = summarise(abs(va["diff"].mean()) - abs(vb["diff"].mean()), np.abs(ba) - np.abs(bb))
            # signed bias difference (for context: shift between the two models' mean errors)
            out["d_bias_pp"] = summarise(va["diff"].mean() - vb["diff"].mean(), ba - bb)
            for rname, gr in refs.items():
                sfx = "" if rname == "reference" else "_mask_derived"
                qa, qb_ = qwk(va["g"][idx], gr[idx]), qwk(vb["g"][idx], gr[idx])
                dq = qa - qb_  # NaN if either undefined
                out["d_qwk" + sfx] = summarise(float(qwk(va["g"], gr) - qwk(vb["g"], gr)), dq)
                ca, cb = va["g"] == gr, vb["g"] == gr
                out["d_accuracy" + sfx] = summarise(ca.mean() - cb.mean(), (ca.astype(float) - cb)[idx].mean(1))
                mc = mcnemar_exact(ca, cb)
                out["mcnemar" + sfx] = mc
                out["d_accuracy" + sfx]["tango_score_ci95"] = tango_ci(mc["first_correct_second_wrong"],
                                                                       mc["first_wrong_second_correct"], n)
                t90 = tango_ci(mc["first_correct_second_wrong"], mc["first_wrong_second_correct"], n, 0.90)
                out["d_accuracy" + sfx]["tango_score_ci90"] = t90
                # TOST margin for dAccuracy from the Tango 90% interval (the bootstrap 90% interval is degenerate)
                out["d_accuracy" + sfx]["tost_margin_needed"] = float(max(abs(t90[0]), abs(t90[1])))
                out["d_accuracy" + sfx]["tost_margin_basis"] = "Tango score 90% interval"
                out["d_accuracy" + sfx]["newcombe_hybrid_ci95"] = newcombe_ci(
                    mc["both_correct"], mc["first_correct_second_wrong"], mc["first_wrong_second_correct"],
                    mc["both_wrong"])
                mc_p[rname].append(mc["p_exact"])
                # Do the two models give the same grade to each leaf?
                out["grade_identical_leaves" + sfx] = int((va["g"] == vb["g"]).sum())
            res["paired"][dis][key] = out
        for m in ["d_leaf_iou", "d_mae_pp", "d_abs_bias_pp", "d_bias_pp", "d_qwk", "d_accuracy"] + (
                ["d_qwk_mask_derived", "d_accuracy_mask_derived"] if dis == "ojo" else []):
            adj = holm([res["paired"][dis][f"{a}_vs_{b}"][m]["p_boot"] for a, b in PAIRS])
            for (a, b), p_adj in zip(PAIRS, adj):
                res["paired"][dis][f"{a}_vs_{b}"][m]["p_boot_holm3"] = p_adj
        for rname in refs:
            sfx = "" if rname == "reference" else "_mask_derived"
            adj = holm(mc_p[rname])
            for (a, b), p_adj in zip(PAIRS, adj):
                res["paired"][dis][f"{a}_vs_{b}"]["mcnemar" + sfx]["p_holm3"] = p_adj

        # ---------------- leaf-spot visual vs mask-derived reference
        if dis == "ojo":
            agree = g_ref == g_mask
            dis_rows = []
            for i, lid in enumerate(ids):
                if not agree[i]:
                    dis_rows.append(dict(leaf_id=lid, visual=f"G{g_ref[i]}", mask_derived=f"G{g_mask[i]}",
                                         ref_sev_pct=round(float(ref_sev[i]), 4)))
            cmv = np.zeros((4, 4), int)
            for i, j in zip(g_ref, g_mask):
                cmv[i, j] += 1
            res["leafspot_reference_agreement"] = dict(
                n=n, visual_eq_mask_derived=int(agree.sum()),
                qwk_visual_vs_mask=float(qwk(g_mask, g_ref)),
                confusion_rows_visual_cols_mask_G0_to_G3=cmv.tolist(),
                disagreements=dis_rows,
                all_disagreements_single_grade=bool(np.all(np.abs(g_ref - g_mask) <= 1)),
                visual_G1_mask_G0=int(((g_ref == 1) & (g_mask == 0)).sum()),
                mask_grade_counts={f"G{g}": int((g_mask == g).sum()) for g in range(4)},
                visual_grade_counts={f"G{g}": int((g_ref == g).sum()) for g in range(4)},
            )
            log.append(f"leaf spot visual == mask-derived grade: {int(agree.sum())}/{n}")
        else:
            res["rust_reference_grade_counts"] = {f"G{g}": int((g_ref == g).sum()) for g in range(4)}

        # ---------------- healthy leaves (G0 recall), separate
        res["healthy_g0"][dis] = {}
        for a in ARCHS:
            H = healthy_table(rows, dis, a)
            ps = np.array([h["pred_sev"] for h in H.values()])
            assert all(h["ref_lesion_px"] == 0 for h in H.values())
            k0 = int((grade(ps) == 0).sum())
            nh = len(ps)
            kany = int(sum(h["any_lesion_px"] for h in H.values()))
            kany2 = int(sum(h["any_lesion_px_either_face"] for h in H.values()))
            mface = float(max(h["max_face_sev"] for h in H.values()))
            res["healthy_g0"][dis][a] = dict(
                n_healthy_local=nh, n_healthy_paper=54, site="EARTH",
                run="within-disease" if dis == "roya" else "cross-disease (leaf-spot model on rust-pack healthy leaves)",
                face_rule="abaxial" if dis == "roya" else "mean of both faces",
                predicted_G0=k0, g0_recall=k0 / nh, g0_recall_wilson95=wilson(k0, nh),
                any_predicted_lesion_pixel=kany,
                any_predicted_lesion_pixel_face_rule=("abaxial face only" if dis == "roya"
                                                      else "either face (leaf = mean of both faces)"),
                max_pred_sev_pct=float(ps.max()),
                any_predicted_lesion_pixel_either_face=kany2,
                max_face_pred_sev_pct_either_face=mface,
                flagged_leaves_any_pixel=[lid for lid, h in H.items() if h["any_lesion_px"]],
                flagged_leaves_any_pixel_either_face=[lid for lid, h in H.items() if h["any_lesion_px_either_face"]],
            )
            log.append(f"healthy {dis:4s} {a:9s}: G0 {k0}/{nh}, any-pixel (face rule) {kany}/{nh}, "
                       f"any-pixel (either face) {kany2}/{nh}, max leaf sev {ps.max():.4f}%, "
                       f"max face sev {mface:.4f}%")

        # per-leaf table for the record
        res.setdefault("per_leaf", {})[dis] = [
            dict(leaf_id=lid, ref_sev=round(float(ref_sev[i]), 5), ref_grade=int(g_ref[i]),
                 mask_grade=int(g_mask[i]),
                 **{f"{a}_pred_sev": round(float(V[a]["pred_sev"][i]), 5) for a in ARCHS},
                 **{f"{a}_grade": int(V[a]["g"][i]) for a in ARCHS},
                 **{f"{a}_iou": round(float(V[a]["iou"][i]), 6) for a in ARCHS})
            for i, lid in enumerate(ids)]

    res["reproduced_published_values"] = bool(all_ok)
    (OUT / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\n".join(log))
    print("ALL PUBLISHED VALUES REPRODUCED" if all_ok else "REPRODUCTION FAILED - STOP")
    return res


if __name__ == "__main__":
    main()
