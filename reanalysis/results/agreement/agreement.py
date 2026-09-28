"""Severity agreement, done properly (PG-26038 major revision; reviewers: Bland-Altman misdescribed, proportional bias).

Unit of analysis: the leaf. Rust = diseased test leaves, abaxial face only (n = 21); American leaf spot = diseased
test leaves, mean of the two faces (severity = mean of face severities, IoU = mean of face IoUs; n = 55).
Severity (%) = lesion px / leaf px in 1024 x 1024 letterbox space, reference and prediction each over its own leaf mask
(exactly as the published evaluation). Difference D = predicted - reference (percentage points, pp);
A = (predicted + reference) / 2.

Per disease x model:
  * reference severity: mean / median / SD / range; grade counts.
  * bias (mean D) with 95% bootstrap CI; constant limits of agreement (LoA = bias +/- 1.96 SD) with bootstrap CIs and
    the Bland & Altman (1986) approximate parametric CIs (SE(LoA) ~ sqrt(3 s^2 / n)) as a cross-check.
  * OLS regression of D on A (slope, intercept, 95% t-CI, p; bootstrap CI as a robustness check because the residuals
    are heteroscedastic); regression-based LoA (Bland & Altman 1999, Stat Methods Med Res 8:135-160, section 3.2):
    D_hat(A) = b0 + b1 A; regress |residuals| on A: |R| = c0 + c1 A; SD(A) = sqrt(pi/2) (c0 + c1 A);
    LoA(A) = D_hat(A) +/- 1.96 SD(A). Evaluated at A = 1, 5, 10, 20 and at the 5% / 20% cut-points on the
    reference scale (the point of the fitted line whose reference value R = A - D/2 equals the cut-point).
  * calibration regression predicted = alpha + beta * reference (secondary; beta - 1 = proportional error).
  * Lin's concordance correlation coefficient (1989; 1/n moments) with bootstrap CI and the z-transform CI
    (Lin 1989, corrected 2000), Pearson r (precision) and the bias-correction factor Cb = CCC / r (accuracy).
  * MAE, relative error = MAE as % of the mean reference severity (and relative bias), RMSE.
  * log-ratio LoA (sensitivity; geometric mean ratio predicted/reference and its 95% LoA).
  * error decomposition: D = FP% - FN% + L, where FP% / FN% = false-positive / false-negative lesion px as % of the
    reference leaf and L = 100 * pred_lesion * (1/pred_leaf - 1/ref_leaf) is the leaf-mask (denominator) term. The
    decomposition is exact per face, hence per leaf and for the mean; OLS slopes on A add up exactly as well.
    FP% is further split into spurious components, a 2-px boundary band around reference lesions, and extension
    beyond 2 px (see pixel_metrics.py).
  * IoU sensitivity: mean leaf IoU (published), median leaf IoU, pooled (micro) IoU = sum(intersection) / sum(union)
    over the images in the leaf-level analysis.
  * boundary F1 (lesion class) at 2 px (also 1, 3, 5 px) in letterbox space, leaf mean and pooled.
Bootstrap: 10,000 resamples of leaves (percentile 95% CIs), numpy default_rng(20260927); one index matrix per
disease (rust drawn first, then leaf spot), shared by the three models so that model CIs are paired.

Inputs: ../../out/per_image.csv and work/pixel_metrics.csv (from pixel_metrics.py; pixel counts verified against
per_image.csv). Output: results.json (+ work/leaf_level.csv).
"""
import os as _os  # (release) all paths below are relative to ROOT (repository root by default; see README)
ROOT = (_os.environ.get("PG26038_ROOT") or _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))).replace(chr(92), "/")
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np
from scipy import stats

BASE = (ROOT + "/reanalysis")
OUT = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(OUT, "work")
B = 10_000
SEED = 20260927
CUTS = (0.5, 5.0, 20.0)
A_EVAL = (1.0, 5.0, 10.0, 20.0)
DISEASES = [("roya", "rust", "Coffee leaf rust"), ("ojo", "leaf_spot", "American leaf spot")]
ARCHS = [("segformer", "SegFormer-B5"), ("unetpp", "UNet++"), ("deeplab", "DeepLabV3+")]
K_SD = math.sqrt(math.pi / 2)
Z975 = float(stats.norm.ppf(0.975))

PUBLISHED = {  # manuscript Tables 2, 4, 5 and the Bland-Altman paragraph
    ("rust", "segformer"): dict(iou=0.767, mae=0.99, bias=0.99, qwk=0.88, loa=(-1.68, 3.66)),
    ("rust", "unetpp"): dict(iou=0.647, mae=1.77, bias=1.75, qwk=0.79, loa=(-2.68, 6.17)),
    ("rust", "deeplab"): dict(iou=0.632, mae=1.47, bias=1.41, qwk=0.88, loa=(-3.07, 5.89)),
    ("leaf_spot", "segformer"): dict(iou=0.884, mae=0.22, bias=0.16, qwk=0.82, loa=(-0.47, 0.79)),
    ("leaf_spot", "unetpp"): dict(iou=0.853, mae=0.15, bias=-0.03, qwk=0.85, loa=(-0.50, 0.44)),
    ("leaf_spot", "deeplab"): dict(iou=0.815, mae=0.25, bias=0.08, qwk=0.84, loa=(-0.66, 0.82)),
}


def grade(s):
    return 0 if s < CUTS[0] else 1 if s < CUTS[1] else 2 if s < CUTS[2] else 3


def qwk(a, b, k=4):
    o = np.bincount(k * np.asarray(a) + np.asarray(b), minlength=k * k).reshape(k, k).astype(float)
    e = np.outer(o.sum(1), o.sum(0)) / o.sum()
    w = (np.subtract.outer(np.arange(k), np.arange(k)) ** 2) / (k - 1) ** 2
    return float(1 - (w * o).sum() / (w * e).sum())


def pct_ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


# ------------------------------------------------------------------------------------------------ vectorised stats
def ols(x, y):
    """Row-wise OLS of y on x for 2-D arrays (B, n) -> intercept, slope, residuals."""
    xm, ym = x.mean(-1, keepdims=True), y.mean(-1, keepdims=True)
    sxx = ((x - xm) ** 2).sum(-1)
    sxy = ((x - xm) * (y - ym)).sum(-1)
    b1 = sxy / sxx
    b0 = ym[..., 0] - b1 * xm[..., 0]
    res = y - (b0[..., None] + b1[..., None] * x)
    return b0, b1, res


def ccc_parts(x, y):
    """Lin's CCC with 1/n moments; x = reference, y = prediction. Returns ccc, r, cb, u (location shift)."""
    mx, my = x.mean(-1), y.mean(-1)
    vx, vy = x.var(-1), y.var(-1)
    sxy = ((x - mx[..., None]) * (y - my[..., None])).mean(-1)
    ccc = 2 * sxy / (vx + vy + (mx - my) ** 2)
    r = sxy / np.sqrt(vx * vy)
    cb = ccc / r
    u = (my - mx) / np.sqrt(np.sqrt(vx * vy))
    return ccc, r, cb, u


def stat_block(ref, pred, extra):
    """All leaf-level statistics for arrays of shape (..., n). extra: dict of per-leaf arrays (same shape)."""
    d = pred - ref
    a = (pred + ref) / 2
    n = d.shape[-1]
    out = {}
    out["bias"] = d.mean(-1)
    sd = d.std(-1, ddof=1)
    out["sd_diff"] = sd
    out["loa_lower"] = out["bias"] - 1.96 * sd
    out["loa_upper"] = out["bias"] + 1.96 * sd
    out["mae"] = np.abs(d).mean(-1)
    out["rmse"] = np.sqrt((d ** 2).mean(-1))
    mref = ref.mean(-1)
    out["mean_ref"] = mref
    out["rel_error_pct"] = 100 * out["mae"] / mref
    out["rel_bias_pct"] = 100 * out["bias"] / mref
    b0, b1, res = ols(a, d)
    out["ba_intercept"], out["ba_slope"] = b0, b1
    c0, c1, _ = ols(a, np.abs(res))
    out["absres_intercept"], out["absres_slope"] = c0, c1
    for av in A_EVAL:
        mu = b0 + b1 * av
        s = K_SD * (c0 + c1 * av)
        out[f"fit_at_A{av:g}"] = mu
        out[f"sd_at_A{av:g}"] = s
        out[f"rloa_lower_at_A{av:g}"] = mu - 1.96 * s
        out[f"rloa_upper_at_A{av:g}"] = mu + 1.96 * s
    for cut in (5.0, 20.0):
        # point on the fitted line with reference R = A - D/2 = cut: D = (b0 + b1 R) / (1 - b1/2)
        dcut = (b0 + b1 * cut) / (1 - b1 / 2)
        acut = cut + dcut / 2
        s = K_SD * (c0 + c1 * acut)
        out[f"fit_at_ref{cut:g}"] = dcut
        out[f"fit_at_ref{cut:g}_relative_pct"] = 100 * dcut / cut
        out[f"rloa_lower_at_ref{cut:g}"] = dcut - 1.96 * s
        out[f"rloa_upper_at_ref{cut:g}"] = dcut + 1.96 * s
        # reference severity at which the fitted prediction R + D(R) reaches the cut-point
        out[f"ref_reaching_cut{cut:g}"] = (cut * (1 - b1 / 2) - b0) / (1 + b1 / 2)
    al, be, _ = ols(ref, pred)
    out["calib_intercept"], out["calib_slope"] = al, be
    ccc, r, cb, u = ccc_parts(ref, pred)
    out["ccc"], out["pearson_r"], out["cb"], out["ccc_u"] = ccc, r, cb, u
    # decomposition (per-leaf arrays in extra); slopes on A add up to the BA slope
    for k in ("fp_pct", "fn_pct", "leafterm_pct", "fp_spurious_pct", "fp_band_pct", "fp_extension_pct",
              "fp_outside_ref_leaf_pct", "fn_missed_pct", "fn_partial_pct"):
        v = extra[k]
        out[f"mean_{k}"] = v.mean(-1)
        out[f"slope_{k}_on_A"] = ols(a, v)[1]
    out["mean_leaf_iou"] = extra["iou"].mean(-1)
    out["median_leaf_iou"] = np.median(extra["iou"], -1)
    out["pooled_iou"] = extra["inter"].sum(-1) / extra["union"].sum(-1)
    for t in (1, 2, 3, 5):
        out[f"mean_leaf_bf1_t{t}"] = extra[f"bf1_t{t}"].mean(-1)
        mp, mr = extra[f"bm_pred_t{t}"].sum(-1), extra[f"bm_ref_t{t}"].sum(-1)
        p, rc = mp / extra["pb"].sum(-1), mr / extra["rb"].sum(-1)
        out[f"pooled_bf1_t{t}"] = 2 * p * rc / (p + rc)
        out[f"pooled_bprec_t{t}"], out[f"pooled_brec_t{t}"] = p, rc
    lr = np.log(pred / ref)
    out["logratio_mean"] = lr.mean(-1)
    out["logratio_sd"] = lr.std(-1, ddof=1)
    out["gmr"] = np.exp(out["logratio_mean"])
    out["gmr_loa_lower"] = np.exp(out["logratio_mean"] - 1.96 * out["logratio_sd"])
    out["gmr_loa_upper"] = np.exp(out["logratio_mean"] + 1.96 * out["logratio_sd"])
    return out


# ------------------------------------------------------------------------------------------------ data
def leaf_table(pixel_rows, dis_key, arch):
    by_leaf = defaultdict(list)
    for r in pixel_rows:
        if r["model_disease"] == dis_key and r["arch"] == arch:
            by_leaf[r["leaf_id"]].append(r)
    leaves = []
    for lid in sorted(by_leaf):
        faces = by_leaf[lid]
        if dis_key == "roya":
            assert len(faces) == 1 and faces[0]["face"] == "abaxial"
        else:
            assert len(faces) == 2 and {f["face"] for f in faces} == {"abaxial", "adaxial"}
        per = defaultdict(list)
        for f in faces:
            g = {k: float(f[k]) for k in f if k not in ("model_disease", "arch", "image_pack", "split", "file",
                                                        "leaf_id", "face", "grade_label")}
            lr, lp = g["ref_leaf_px"], g["pred_leaf_px"]
            ref_sev = 100 * g["ref_lesion_px"] / lr
            pred_sev = 100 * g["pred_lesion_px"] / lp
            per["ref_sev"].append(ref_sev)
            per["pred_sev"].append(pred_sev)
            per["iou"].append(g["inter_lesion"] / g["union_lesion"])
            per["dice"].append(2 * g["inter_lesion"] / (g["ref_lesion_px"] + g["pred_lesion_px"]))
            per["inter"].append(g["inter_lesion"])
            per["union"].append(g["union_lesion"])
            for k in ("fp", "fn", "fp_spurious", "fp_band", "fp_extension", "fp_outside_ref_leaf", "fn_missed",
                      "fn_partial"):
                per[f"{k}_pct"].append(100 * g[f"{k}_px"] / lr)
            per["leafterm_pct"].append(100 * g["pred_lesion_px"] * (1 / lp - 1 / lr))
            # exactness of the decomposition for this face
            assert abs((pred_sev - ref_sev) - (per["fp_pct"][-1] - per["fn_pct"][-1] + per["leafterm_pct"][-1])) < 1e-9
            pb, rb = g["pred_boundary_px"], g["ref_boundary_px"]
            per["pb"].append(pb)
            per["rb"].append(rb)
            for t in (1, 2, 3, 5):
                mp, mr = g[f"bmatch_pred_t{t}"], g[f"bmatch_ref_t{t}"]
                per[f"bm_pred_t{t}"].append(mp)
                per[f"bm_ref_t{t}"].append(mr)
                if pb == 0 and rb == 0:
                    f1 = 1.0
                elif pb == 0 or rb == 0:
                    f1 = 0.0
                else:
                    p, rc = mp / pb, mr / rb
                    f1 = 0.0 if p + rc == 0 else 2 * p * rc / (p + rc)
                per[f"bf1_t{t}"].append(f1)
        rec = {"leaf_id": lid, "grade_label": faces[0]["grade_label"], "n_faces": len(faces)}
        for k, v in per.items():
            # counts are summed over faces (pooled metrics); per-leaf rates / percentages are averaged over faces
            rec[k] = float(np.sum(v)) if k in ("inter", "union", "pb", "rb") or k.startswith("bm_") else float(np.mean(v))
        leaves.append(rec)
    return leaves


def round_floats(o, nd=6):
    if isinstance(o, float):
        return round(o, nd)
    if isinstance(o, dict):
        return {k: round_floats(v, nd) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [round_floats(v, nd) for v in o]
    return o


def main():
    pixel_rows = list(csv.DictReader(open(os.path.join(WORK, "pixel_metrics.csv"))))
    per_image = list(csv.DictReader(open(f"{BASE}/out/per_image.csv")))
    rng = np.random.default_rng(SEED)
    results = {
        "meta": {
            "unit": "leaf; rust = abaxial face of the 21 diseased test leaves; American leaf spot = mean of both faces "
                    "of the 55 diseased test leaves",
            "difference": "predicted - reference severity, percentage points (pp); A = mean of predicted and reference",
            "severity_definition": "lesion px / leaf px in 1024 x 1024 letterbox space (reference and prediction each "
                                   "over its own leaf mask), as in the published evaluation",
            "bootstrap": f"{B} resamples of leaves, percentile 95% CIs, numpy default_rng({SEED}); one index matrix per "
                         "disease (rust drawn first), shared by the three models",
            "regression_loa": "Bland & Altman (1999) section 3.2: D_hat = b0 + b1*A (OLS); |residual| = c0 + c1*A (OLS); "
                              "SD(A) = sqrt(pi/2)*(c0 + c1*A); LoA(A) = D_hat(A) +/- 1.96*SD(A)",
            "ccc": "Lin (1989), 1/n moments; CI: bootstrap percentile (primary) and Fisher-z with Lin's asymptotic "
                   "variance (Lin 1989, corrected 2000); Cb = CCC / Pearson r",
            "decomposition": "D = FP% - FN% + L; FP%, FN% = false-positive / false-negative lesion px as % of the "
                             "reference leaf; L = 100*pred_lesion*(1/pred_leaf - 1/ref_leaf) (leaf-mask term); exact "
                             "per face; FP split into spurious components (no overlap with any reference lesion), a "
                             "2-px boundary band around reference lesions, and extension beyond 2 px",
            "boundary_f1": "lesion boundary = mask minus 3x3 erosion; tolerance Euclidean distance in 1024x1024 "
                           "letterbox px; rust images 791x1024 (letterbox scale 1.0, 2 px = 2 native px); leaf-spot "
                           "images 1583x2048 (scale 0.5, 2 px = 4 native px)",
            "pooled_iou": "sum of lesion intersections / sum of lesion unions over all images in the leaf-level analysis "
                          "(rust: 21 abaxial images; leaf spot: 110 images)",
            "healthy_note": "Only 21 of the 54 healthy rust-test leaves reported in the paper are present locally; "
                            "the healthy-leaf summary below covers those 21 (EARTH site) and does not replace the "
                            "published 54-leaf specificity counts.",
        },
        "reproduction": {},
        "models": {},
    }
    leaf_csv = []
    for dis_key, dis, dis_name in DISEASES:
        tabs = {arch: leaf_table(pixel_rows, dis_key, arch) for arch, _ in ARCHS}
        ids = [l["leaf_id"] for l in tabs["segformer"]]
        for arch, _ in ARCHS:
            assert [l["leaf_id"] for l in tabs[arch]] == ids
            # reference severity identical across models (same reference masks)
            assert np.allclose([l["ref_sev"] for l in tabs[arch]], [l["ref_sev"] for l in tabs["segformer"]])
        n = len(ids)
        idx = rng.integers(0, n, size=(B, n))
        for arch, arch_name in ARCHS:
            L = tabs[arch]
            ref = np.array([l["ref_sev"] for l in L])
            pred = np.array([l["pred_sev"] for l in L])
            keys = [k for k in L[0] if k not in ("leaf_id", "grade_label", "n_faces", "ref_sev", "pred_sev")]
            extra = {k: np.array([l[k] for l in L]) for k in keys}
            pt = {k: float(v) for k, v in stat_block(ref, pred, extra).items()}
            bs = stat_block(ref[idx], pred[idx], {k: v[idx] for k, v in extra.items()})
            ci = {k: pct_ci(v) for k, v in bs.items()}
            d = pred - ref
            a = (pred + ref) / 2
            # OLS inference for D on A
            lr_ba = stats.linregress(a, d)
            tq = float(stats.t.ppf(0.975, n - 2))
            lr_abs = stats.linregress(a, np.abs(d - (lr_ba.intercept + lr_ba.slope * a)))
            lr_cal = stats.linregress(ref, pred)
            # HC3 robust SE of the BA slope (sensitivity to heteroscedasticity)
            X = np.column_stack([np.ones(n), a])
            XtXi = np.linalg.inv(X.T @ X)
            e = d - X @ np.array([lr_ba.intercept, lr_ba.slope])
            h = np.einsum("ij,jk,ik->i", X, XtXi, X)
            V = XtXi @ (X.T * (e / (1 - h)) ** 2) @ X @ XtXi
            se_hc3 = float(math.sqrt(V[1, 1]))
            p_hc3 = float(2 * stats.t.sf(abs(lr_ba.slope / se_hc3), n - 2))
            # Lin z-transform CI
            ccc, r, u = pt["ccc"], pt["pearson_r"], pt["ccc_u"]
            var_z = (1 / (n - 2)) * ((1 - r ** 2) * ccc ** 2 / ((1 - ccc ** 2) * r ** 2)
                                     + 2 * ccc ** 3 * (1 - ccc) * u ** 2 / (r * (1 - ccc ** 2) ** 2)
                                     - ccc ** 4 * u ** 4 / (2 * r ** 2 * (1 - ccc ** 2) ** 2))
            zc = math.atanh(ccc)
            ccc_z_ci = [math.tanh(zc - Z975 * math.sqrt(var_z)), math.tanh(zc + Z975 * math.sqrt(var_z))]
            # parametric (Bland-Altman 1986) CIs for bias and constant LoA
            s = pt["sd_diff"]
            tq1 = float(stats.t.ppf(0.975, n - 1))
            se_b, se_l = s / math.sqrt(n), math.sqrt(3 * s ** 2 / n)
            # grades
            rg = [grade(x) for x in ref] if dis == "rust" else [int(l["grade_label"][1]) for l in L]
            pg = [grade(x) for x in pred]
            # mask_grade = grade derived from the reference MASK severity (cut-points 0.5/5/20%); for rust it equals the
            # reference grade, for leaf spot the reference grade is the annotator's visual grade.
            #   'visual_mask_gap'  : predicted grade == mask-derived grade != visual grade (an exact reproduction of
            #                        the reference mask would give the same error)
            #   'severity_error'   : mask-derived grade == reference grade; the prediction crossed a cut-point
            #   'gap_and_severity' : neither of the above
            def _cat(g_ref, g_mask, g_pred):
                if g_mask == g_ref:
                    return "severity_error"
                return "visual_mask_gap" if g_pred == g_mask else "gap_and_severity"
            disagreements = []
            for l, g0, g1 in zip(L, rg, pg):
                if g0 == g1:
                    continue
                gm = grade(l["ref_sev"])
                disagreements.append(dict(leaf_id=l["leaf_id"], ref_sev=l["ref_sev"], pred_sev=l["pred_sev"],
                                          ref_grade=g0, pred_grade=g1, mask_grade=gm, category=_cat(g0, gm, g1),
                                          direction="over" if g1 > g0 else "under"))
            dis_cat_counts = {c: int(sum(1 for x in disagreements if x["category"] == c))
                              for c in ("visual_mask_gap", "severity_error", "gap_and_severity")}
            # leaves in the band just below each cut-point where the fitted prediction is at or above the cut
            band_leaves = {}
            for cut in (5.0, 20.0):
                lo = pt[f"ref_reaching_cut{cut:g}"]
                band_leaves[cut] = [dict(leaf_id=l["leaf_id"], ref_sev=l["ref_sev"], pred_sev=l["pred_sev"], ref_grade=g0,
                                         pred_grade=g1, over_graded=bool(g1 > g0))
                                    for l, g0, g1 in zip(L, rg, pg) if lo <= l["ref_sev"] < cut]
            m = {
                "disease": dis, "model": arch_name, "n_leaves": n,
                "reference_severity": {
                    "mean": float(ref.mean()), "median": float(np.median(ref)), "sd": float(ref.std(ddof=1)),
                    "min": float(ref.min()), "max": float(ref.max()),
                    "q25": float(np.percentile(ref, 25)), "q75": float(np.percentile(ref, 75)),
                    "n_above_10pct": int((ref > 10).sum()), "n_at_or_above_20pct": int((ref >= 20).sum()),
                    "grade_counts_reference": {f"G{g}": int(sum(1 for x in rg if x == g)) for g in range(4)},
                    "grade_source": "mask-derived (cut-points 0.5/5/20%)" if dis == "rust" else "annotator's visual grade",
                },
                "predicted_severity": {"mean": float(pred.mean()), "median": float(np.median(pred)),
                                       "min": float(pred.min()), "max": float(pred.max())},
                "mean_of_A": {"min": float(a.min()), "max": float(a.max()), "n_A_above_15": int((a > 15).sum())},
                "bias": {"estimate": pt["bias"], "ci95_bootstrap": ci["bias"],
                         "ci95_t": [pt["bias"] - tq1 * se_b, pt["bias"] + tq1 * se_b]},
                "sd_of_differences": {"estimate": s, "ci95_bootstrap": ci["sd_diff"]},
                "constant_loa": {
                    "lower": pt["loa_lower"], "upper": pt["loa_upper"],
                    "lower_ci95_bootstrap": ci["loa_lower"], "upper_ci95_bootstrap": ci["loa_upper"],
                    "lower_ci95_parametric_BA1986": [pt["loa_lower"] - tq1 * se_l, pt["loa_lower"] + tq1 * se_l],
                    "upper_ci95_parametric_BA1986": [pt["loa_upper"] - tq1 * se_l, pt["loa_upper"] + tq1 * se_l],
                },
                "regression_of_differences_on_means": {
                    "slope": lr_ba.slope, "slope_ci95_t": [lr_ba.slope - tq * lr_ba.stderr, lr_ba.slope + tq * lr_ba.stderr],
                    "slope_p": lr_ba.pvalue, "slope_se": lr_ba.stderr,
                    "slope_ci95_bootstrap": ci["ba_slope"],
                    "slope_se_hc3": se_hc3, "slope_p_hc3": p_hc3,
                    "slope_ci95_hc3": [lr_ba.slope - tq * se_hc3, lr_ba.slope + tq * se_hc3],
                    "intercept": lr_ba.intercept,
                    "intercept_ci95_t": [lr_ba.intercept - tq * lr_ba.intercept_stderr,
                                         lr_ba.intercept + tq * lr_ba.intercept_stderr],
                    "intercept_p": float(2 * stats.t.sf(abs(lr_ba.intercept / lr_ba.intercept_stderr), n - 2)),
                    "intercept_ci95_bootstrap": ci["ba_intercept"],
                    "r_squared": lr_ba.rvalue ** 2, "df": n - 2,
                },
                "regression_based_loa": {
                    "abs_residual_intercept_c0": lr_abs.intercept, "abs_residual_slope_c1": lr_abs.slope,
                    "abs_residual_slope_ci95_t": [lr_abs.slope - tq * lr_abs.stderr, lr_abs.slope + tq * lr_abs.stderr],
                    "abs_residual_slope_p": lr_abs.pvalue,
                    "abs_residual_slope_ci95_bootstrap": ci["absres_slope"],
                    "sd_function": f"SD(A) = sqrt(pi/2) * ({lr_abs.intercept:.5f} + {lr_abs.slope:.5f} * A)",
                    "mean_function": f"D(A) = {lr_ba.intercept:.5f} + {lr_ba.slope:.5f} * A",
                    "sd_nonnegative_from_A": (max(0.0, -lr_abs.intercept / lr_abs.slope) if lr_abs.intercept < 0 else 0.0),
                    "at_mean_severity": {
                        f"A={av:g}": {
                            "fitted_difference": pt[f"fit_at_A{av:g}"], "fitted_difference_ci95_bootstrap": ci[f"fit_at_A{av:g}"],
                            "sd": pt[f"sd_at_A{av:g}"],
                            "loa_lower": pt[f"rloa_lower_at_A{av:g}"], "loa_upper": pt[f"rloa_upper_at_A{av:g}"],
                            "loa_lower_ci95_bootstrap": ci[f"rloa_lower_at_A{av:g}"],
                            "loa_upper_ci95_bootstrap": ci[f"rloa_upper_at_A{av:g}"],
                            "within_observed_A_range": bool(a.min() <= av <= a.max()),
                        } for av in A_EVAL},
                    "at_reference_cutpoint": {
                        f"ref={cut:g}%": {
                            "fitted_difference": pt[f"fit_at_ref{cut:g}"],
                            "fitted_difference_ci95_bootstrap": ci[f"fit_at_ref{cut:g}"],
                            "fitted_difference_relative_pct": pt[f"fit_at_ref{cut:g}_relative_pct"],
                            "fitted_difference_relative_pct_ci95_bootstrap": ci[f"fit_at_ref{cut:g}_relative_pct"],
                            "loa_lower": pt[f"rloa_lower_at_ref{cut:g}"], "loa_upper": pt[f"rloa_upper_at_ref{cut:g}"],
                            "loa_lower_ci95_bootstrap": ci[f"rloa_lower_at_ref{cut:g}"],
                            "loa_upper_ci95_bootstrap": ci[f"rloa_upper_at_ref{cut:g}"],
                            "reference_severity_whose_fitted_prediction_reaches_cut": pt[f"ref_reaching_cut{cut:g}"],
                            "reference_severity_whose_fitted_prediction_reaches_cut_ci95_bootstrap": ci[f"ref_reaching_cut{cut:g}"],
                            "n_leaves_with_reference_in_that_band_below_cut": int(((ref >= pt[f"ref_reaching_cut{cut:g}"]) & (ref < cut)).sum()),
                            "leaves_in_that_band_below_cut": band_leaves[cut],
                            "n_over_graded_in_that_band": int(sum(x["over_graded"] for x in band_leaves[cut])),
                            "band_note": "the fitted mean prediction is at or above the cut-point for reference "
                                         "severities in [lower edge, cut); at the lower edge it only just reaches the "
                                         "cut, so leaves there are prone to, not certain of, over-grading",
                            "n_leaves_with_reference_above_cut": int((ref >= cut).sum()),
                        } for cut in (5.0, 20.0)},
                },
                "calibration_pred_on_ref": {
                    "slope": lr_cal.slope, "slope_ci95_t": [lr_cal.slope - tq * lr_cal.stderr, lr_cal.slope + tq * lr_cal.stderr],
                    "slope_ci95_bootstrap": ci["calib_slope"], "p_slope_eq_1": float(2 * stats.t.sf(abs((lr_cal.slope - 1) / lr_cal.stderr), n - 2)),
                    "intercept": lr_cal.intercept, "intercept_ci95_bootstrap": ci["calib_intercept"],
                    "predicted_at_ref5": lr_cal.intercept + 5 * lr_cal.slope,
                    "predicted_at_ref20": lr_cal.intercept + 20 * lr_cal.slope,
                },
                "lin_ccc": {"ccc": ccc, "ci95_bootstrap": ci["ccc"], "ci95_z_transform": ccc_z_ci,
                            "pearson_r": r, "pearson_r_ci95_bootstrap": ci["pearson_r"],
                            "cb": pt["cb"], "cb_ci95_bootstrap": ci["cb"], "location_shift_u": u,
                            # CCC = r * Cb, so -log(CCC) = -log(r) - log(Cb): share of the (log) shortfall from 1
                            "share_of_log_shortfall_precision_r": math.log(r) / math.log(ccc),
                            "share_of_log_shortfall_accuracy_cb": math.log(pt["cb"]) / math.log(ccc)},
                "mae": {"estimate": pt["mae"], "ci95_bootstrap": ci["mae"]},
                "rmse": {"estimate": pt["rmse"], "ci95_bootstrap": ci["rmse"]},
                "relative_error_pct": {"estimate": pt["rel_error_pct"], "ci95_bootstrap": ci["rel_error_pct"],
                                       "definition": "100 * MAE / mean reference severity"},
                "relative_bias_pct": {"estimate": pt["rel_bias_pct"], "ci95_bootstrap": ci["rel_bias_pct"]},
                "log_ratio_loa": {"geometric_mean_ratio": pt["gmr"], "gmr_ci95_bootstrap": ci["gmr"],
                                  "loa_ratio_lower": pt["gmr_loa_lower"], "loa_ratio_upper": pt["gmr_loa_upper"],
                                  "n_leaves_with_ratio_above_1": int((pred > ref).sum())},
                "error_decomposition_pct_of_leaf": {
                    k: {"mean": pt[f"mean_{k}"], "mean_ci95_bootstrap": ci[f"mean_{k}"],
                        "slope_on_A": pt[f"slope_{k}_on_A"], "slope_on_A_ci95_bootstrap": ci[f"slope_{k}_on_A"]}
                    for k in ("fp_pct", "fn_pct", "leafterm_pct", "fp_spurious_pct", "fp_band_pct", "fp_extension_pct",
                              "fp_outside_ref_leaf_pct", "fn_missed_pct", "fn_partial_pct")},
                "iou": {"mean_leaf_iou": pt["mean_leaf_iou"], "mean_leaf_iou_ci95_bootstrap": ci["mean_leaf_iou"],
                        "median_leaf_iou": pt["median_leaf_iou"], "median_leaf_iou_ci95_bootstrap": ci["median_leaf_iou"],
                        "pooled_micro_iou": pt["pooled_iou"], "pooled_micro_iou_ci95_bootstrap": ci["pooled_iou"],
                        "min_leaf_iou": float(extra["iou"].min())},
                "boundary_f1": {f"tol_{t}px": {"mean_leaf": pt[f"mean_leaf_bf1_t{t}"],
                                               "mean_leaf_ci95_bootstrap": ci[f"mean_leaf_bf1_t{t}"],
                                               "pooled": pt[f"pooled_bf1_t{t}"],
                                               "pooled_ci95_bootstrap": ci[f"pooled_bf1_t{t}"],
                                               "pooled_precision": pt[f"pooled_bprec_t{t}"],
                                               "pooled_recall": pt[f"pooled_brec_t{t}"]} for t in (1, 2, 3, 5)},
                "grades": {"qwk": qwk(rg, pg), "accuracy": float(np.mean(np.array(rg) == np.array(pg))),
                           "n_over": int(sum(g1 > g0 for g0, g1 in zip(rg, pg))),
                           "n_under": int(sum(g1 < g0 for g0, g1 in zip(rg, pg))),
                           "disagreements": disagreements,
                           "disagreement_category_counts": dis_cat_counts,
                           "category_definitions": {
                               "visual_mask_gap": "predicted grade equals the grade derived from the reference mask "
                                                  "severity, which differs from the (visual) reference grade",
                               "severity_error": "grade derived from the reference mask equals the reference grade; "
                                                 "the predicted severity crossed a cut-point",
                               "gap_and_severity": "neither of the above"}},
            }
            # decomposition sums (sanity): bias = FP - FN + L ; slope = sum of component slopes
            dsum = pt["mean_fp_pct"] - pt["mean_fn_pct"] + pt["mean_leafterm_pct"]
            ssum = pt["slope_fp_pct_on_A"] - pt["slope_fn_pct_on_A"] + pt["slope_leafterm_pct_on_A"]
            assert abs(dsum - pt["bias"]) < 1e-9 and abs(ssum - lr_ba.slope) < 1e-9
            assert abs(pt["ba_slope"] - lr_ba.slope) < 1e-12
            fp_split = pt["mean_fp_spurious_pct"] + pt["mean_fp_band_pct"] + pt["mean_fp_extension_pct"]
            assert abs(fp_split - pt["mean_fp_pct"]) < 1e-9
            results["models"][f"{dis}/{arch}"] = m
            # reproduction check against the manuscript
            pub = PUBLISHED[(dis, arch)]
            got = dict(iou=round(pt["mean_leaf_iou"], 3), mae=round(pt["mae"], 2), bias=round(pt["bias"], 2),
                       qwk=round(m["grades"]["qwk"], 2),
                       loa=(round(pt["loa_lower"], 2), round(pt["loa_upper"], 2)))
            ok = all(abs(got[k] - pub[k]) < 1e-9 for k in ("iou", "mae", "bias", "qwk")) and \
                all(abs(g - p) < 1e-9 for g, p in zip(got["loa"], pub["loa"]))
            results["reproduction"][f"{dis}/{arch}"] = {"published": pub, "reproduced": got, "match": ok}
            for l, x_ref, x_pred in zip(L, ref, pred):
                leaf_csv.append({"disease": dis, "model": arch_name, "leaf_id": l["leaf_id"],
                                 "grade_label": l["grade_label"], "ref_sev": x_ref, "pred_sev": x_pred,
                                 "diff": x_pred - x_ref, "mean": (x_pred + x_ref) / 2, "leaf_iou": l["iou"],
                                 "fp_pct": l["fp_pct"], "fn_pct": l["fn_pct"], "leafterm_pct": l["leafterm_pct"],
                                 "fp_spurious_pct": l["fp_spurious_pct"], "fp_band_pct": l["fp_band_pct"],
                                 "fp_extension_pct": l["fp_extension_pct"], "bf1_2px": l["bf1_t2"]})
    results["reproduction_all_match"] = all(v["match"] for v in results["reproduction"].values())

    # healthy rust-pack test leaves present locally (abaxial face; rust models); context only
    healthy = {}
    for arch, arch_name in ARCHS:
        hs = [r for r in per_image if r["model_disease"] == "roya" and r["arch"] == arch and r["image_pack"] == "roya"
              and r["split"] == "test" and r["grade_label"] == "healthy" and r["face"] == "abaxial"]
        sev = np.array([float(r["pred_sev"]) for r in hs])
        px = np.array([int(r["pred_lesion_px"]) for r in hs])
        healthy[arch_name] = {"n_leaves": len(hs), "mean_pred_sev": float(sev.mean()), "max_pred_sev": float(sev.max()),
                              "n_any_lesion_px": int((px > 0).sum()), "n_pred_sev_ge_0.5": int((sev >= 0.5).sum()),
                              "n_pred_sev_ge_5": int((sev >= 5).sum())}
    results["healthy_rust_test_leaves_local_abaxial"] = healthy
    # same 21 leaves, leaf value = maximum over BOTH faces (definition used in revision_brief.md, fact 10): a leaf
    # counts as carrying a predicted lesion if either face has >= 1 predicted lesion pixel
    healthy_both = {}
    for arch, arch_name in ARCHS:
        hs = [r for r in per_image if r["model_disease"] == "roya" and r["arch"] == arch and r["image_pack"] == "roya"
              and r["split"] == "test" and r["grade_label"] == "healthy"]
        by = defaultdict(list)
        for r in hs:
            by[r["leaf_id"]].append(r)
        assert all(len(v) == 2 and {x["face"] for x in v} == {"abaxial", "adaxial"} for v in by.values())
        sev = np.array([max(float(x["pred_sev"]) for x in v) for v in by.values()])
        px = np.array([max(int(x["pred_lesion_px"]) for x in v) for v in by.values()])
        healthy_both[arch_name] = {"n_leaves": len(by), "mean_of_leaf_max_pred_sev": float(sev.mean()),
                                   "max_pred_sev": float(sev.max()), "n_any_lesion_px_either_face": int((px > 0).sum()),
                                   "n_pred_sev_ge_0.5_either_face": int((sev >= 0.5).sum()),
                                   "n_pred_sev_ge_5_either_face": int((sev >= 5).sum())}
    results["healthy_rust_test_leaves_local_max_over_both_faces"] = healthy_both
    results["healthy_definitions_note"] = (
        "Two leaf-level definitions give different counts for the same 21 leaves: abaxial face only (the rust leaf-level "
        "rule used everywhere else in this analysis) and maximum over both faces (revision_brief.md fact 10). Use one, "
        "state it, and do not mix the two. The healthy leaves come from the EARTH site, the diseased rust leaves from "
        "Sarchi, so healthy-leaf false alarms are not a site-matched estimate of spurious detections on diseased leaves; "
        "for that see error_decomposition_pct_of_leaf.fp_spurious_pct.")

    results = round_floats(results)
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    with open(os.path.join(WORK, "leaf_level.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(leaf_csv[0].keys()))
        wr.writeheader(); wr.writerows(leaf_csv)
    print("reproduction all match:", results["reproduction_all_match"])
    for k, v in results["reproduction"].items():
        print(k, v["reproduced"], "OK" if v["match"] else "MISMATCH", v["published"])


if __name__ == "__main__":
    main()
