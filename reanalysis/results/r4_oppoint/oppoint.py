"""A5 rust operating point, step 2: operating-point analyses on the saved probabilities (work/prob, from infer_prob.py).

Leaf-level rule for rust: one image per leaf, the abaxial face. Severity = 100 x predicted lesion px / predicted leaf px
(predicted leaf = class 1 or 2), reference severity likewise from the letterbox reference label; all metrics in
1024 x 1024 letterbox space (rust scans are 791 x 1024, letterbox scale 1.0, so 1 px = 1 native scan px).
D = predicted - reference severity (pp); A = mean of the two.

Operating points (all applied to the same softmax output; nothing is retrained):
  argmax      raw 3-class argmax (published operating point; must reproduce results/agreement and seed_facts.md)
  thr_bias    lesion iff p(lesion) >= t, t chosen on the 21 VALIDATION leaves to minimise |bias|; otherwise the pixel
              takes the argmax of background vs leaf. Applied once to test.
  thr_mae     same, t chosen on validation to minimise MAE.
  ero1, ero2  argmax lesion mask eroded by 1 or 2 px (3 x 3 structuring element, iterated; border counts as
              non-lesion); eroded pixels become leaf, so the leaf denominator is unchanged.
Threshold grid: t = 0.01, 0.02, ..., 0.99 on p(lesion) quantised to 1/65535 (lesion iff p2q >= ceil(t x 65535)).

Endpoints on the 21 TEST leaves: mean leaf lesion IoU, MAE, bias, grade accuracy, QWK (mask-derived rust grades,
cut-points 0.5/5/20%), OLS slope of D on A (proportional bias) with t CI and bootstrap CI, and paired bootstrap
differences (operating point - argmax). Healthy test leaves (21 local, abaxial face): leaves with any predicted lesion
pixel and with predicted severity >= 0.5%.
Signed boundary distance (test leaves): for every predicted lesion contour pixel (lesion minus its 3 x 3 erosion) the
Euclidean distance to the reference lesion contour, signed + outside the reference lesion and - inside it:
  sd = EDT(~ref)(p) if p is outside the reference lesion, sd = 1 - EDT(ref)(p) if inside
so a pixel ON the reference contour (inner boundary pixel of the reference lesion) has sd = 0, the first ring outside
has +1, the second ring inside has -1. The reverse direction (each reference contour pixel relative to the predicted
lesion; + = the prediction extends outside the reference there) is also given.
Bootstrap: 10,000 resamples of leaves, seed 20260928, percentile 95% CIs, the same resample indices for every model
and operating point (paired).
Outputs: results.json, work/leaf_level.csv, work/val_curves.csv, work/boundary_counts.npz.
"""
import os as _os  # (release) all paths below are relative to ROOT (repository root by default; see README)
ROOT = (_os.environ.get("PG26038_ROOT") or _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))).replace(chr(92), "/")
import csv, json, math, os
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy import stats

BASE = ROOT
HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "work"); PROB = os.path.join(WORK, "prob")
PACK = f"{BASE}/datasets/pg26038-roya-real-pack"
SIZE = 1024; B = 10000; SEED = 20260928
EIGHT = np.ones((3, 3), bool)
# 0.01..0.99 step 0.01, extended by 0.991..0.999 step 0.001 and 0.9995, 0.9999: a preview run put the validation optimum of one
# model (matched UNet++ seed 1) on the old upper edge 0.99 with validation bias still > 0, so the grid was widened on that basis
GRID = np.concatenate([np.round(np.arange(1, 100) / 100, 2), np.round(np.arange(991, 1000) / 1000, 3), [0.9995, 0.9999]])
MODELS = ["pub_segformer", "pub_unetpp", "pub_deeplab", "m_unetpp_s0", "m_unetpp_s1", "m_unetpp_s2",
          "m_deeplab_s0", "m_deeplab_s1", "m_deeplab_s2"]
LABEL = {"pub_segformer": "SegFormer-B5 (published)", "pub_unetpp": "UNet++ (published)", "pub_deeplab": "DeepLabV3+ (published)",
         "m_unetpp_s0": "UNet++ matched lrB seed 0", "m_unetpp_s1": "UNet++ matched lrB seed 1", "m_unetpp_s2": "UNet++ matched lrB seed 2",
         "m_deeplab_s0": "DeepLabV3+ matched lrB seed 0", "m_deeplab_s1": "DeepLabV3+ matched lrB seed 1",
         "m_deeplab_s2": "DeepLabV3+ matched lrB seed 2"}
OPS = ["argmax", "thr_bias", "thr_mae", "ero1", "ero2"]
ALL_MODELS = list(MODELS)
if os.environ.get("OPP_MODELS"):  # debugging subset only; the final run uses all nine models
    MODELS = os.environ["OPP_MODELS"].split(",")
CLIP = 10.0
MCNT = None


def resize_pad(arr, nearest=False):  # verbatim from run_inference.py
    im = Image.fromarray(arr); w, h = im.size; s = SIZE / max(w, h)
    im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.NEAREST if nearest else Image.BILINEAR)
    canvas = Image.new(im.mode, (SIZE, SIZE), 0); canvas.paste(im, ((SIZE - im.width) // 2, (SIZE - im.height) // 2))
    return np.asarray(canvas)


def load_label(file):
    rgba = np.asarray(Image.open(f"{PACK}/images/{file}").convert("RGBA"))
    leaf = rgba[..., 3] > 8
    les = (np.asarray(Image.open(f"{PACK}/masks/{file}").convert("L")) > 127) & leaf
    lab = np.zeros(leaf.shape, np.uint8); lab[leaf] = 1; lab[les] = 2
    return resize_pad(lab, nearest=True)


def grade(s): return 0 if s < 0.5 else 1 if s < 5 else 2 if s < 20 else 3


def qwk(a, b, K=4):  # as eval_retrained.py
    O = np.zeros((K, K))
    for x, y in zip(a, b): O[x, y] += 1
    E = np.outer(O.sum(1), O.sum(0)) / O.sum(); W = (np.subtract.outer(np.arange(K), np.arange(K)) ** 2) / (K - 1) ** 2
    den = (W * E).sum()
    return float(1 - (W * O).sum() / den) if den > 0 else float("nan")


def load_npz(m, sub, file):
    z = np.load(os.path.join(PROB, m, sub, file.replace(".png", ".npz")))
    return z["argmax"], z["c01"], z["p2"]


def counts_from_mask(les, leaf_nonles, ref_les, ref_leaf_px):
    pred_leaf = les | leaf_nonles
    return dict(pred_les=int(les.sum()), pred_leaf=int(pred_leaf.sum()), inter=int((les & ref_les).sum()),
                union=int((les | ref_les).sum()), ref_les=int(ref_les.sum()), ref_leaf=ref_leaf_px)


def thr_counts(am, c01, p2, ref_les, ref_leaf_px):
    """Counts for every grid threshold via histograms of the quantised lesion probability (exact)."""
    h_all = np.bincount(p2.ravel(), minlength=65536)
    h_ref = np.bincount(p2[ref_les], minlength=65536)
    h_c1 = np.bincount(p2[c01 == 1], minlength=65536)
    ge = lambda h: np.cumsum(h[::-1])[::-1]  # ge(h)[q] = number of pixels with p2q >= q
    g_all, g_ref, g_c1 = ge(h_all), ge(h_ref), ge(h_c1)
    n_c1 = int((c01 == 1).sum()); n_ref = int(ref_les.sum())
    out = []
    for t in GRID:
        q = int(math.ceil(t * 65535))
        les = int(g_all[q]); inter = int(g_ref[q]); c1_below = n_c1 - int(g_c1[q])
        out.append(dict(pred_les=les, pred_leaf=les + c1_below, inter=inter, union=les + n_ref - inter, ref_les=n_ref, ref_leaf=ref_leaf_px))
    return out


def leaf_metrics(cs):
    ref = np.array([100 * c["ref_les"] / c["ref_leaf"] for c in cs])
    pred = np.array([100 * c["pred_les"] / c["pred_leaf"] if c["pred_leaf"] else 0.0 for c in cs])
    iou = np.array([c["inter"] / c["union"] if c["union"] else np.nan for c in cs])
    return ref, pred, iou


def ols(x, y):
    n = len(x); X = np.column_stack([np.ones(n), x]); beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta; s2 = res @ res / (n - 2); cov = s2 * np.linalg.inv(X.T @ X); se = math.sqrt(cov[1, 1])
    tq = stats.t.ppf(0.975, n - 2); p = 2 * stats.t.sf(abs(beta[1] / se), n - 2)
    return float(beta[0]), float(beta[1]), float(se), [float(beta[1] - tq * se), float(beta[1] + tq * se)], float(p)


def summarise(ref, pred, iou):
    D = pred - ref; A = (pred + ref) / 2
    b0, b1, se, ci, p = ols(A, D)
    rg = [grade(s) for s in ref]; pg = [grade(s) for s in pred]
    return dict(n=len(ref), leaf_iou=float(np.nanmean(iou)), mae=float(np.mean(np.abs(D))), bias=float(np.mean(D)),
                slope=b1, slope_ci95_t=ci, slope_p=p, intercept=b0, grade_acc=float(np.mean(np.array(rg) == pg)), qwk=qwk(rg, pg),
                n_over_graded=int(sum(b > a for a, b in zip(rg, pg))), n_under_graded=int(sum(b < a for a, b in zip(rg, pg))),
                n_over_estimated=int((D > 0).sum()))


def slope_vec(A, D):  # rows = bootstrap resamples
    Am = A.mean(1, keepdims=True); Dm = D.mean(1, keepdims=True)
    return ((A - Am) * (D - Dm)).sum(1) / ((A - Am) ** 2).sum(1)


def boot(ref, pred, iou, idx):
    R, P, I = ref[idx], pred[idx], iou[idx]; D = P - R; A = (P + R) / 2
    return dict(leaf_iou=I.mean(1), mae=np.abs(D).mean(1), bias=D.mean(1), slope=slope_vec(A, D))


def pct(v): return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def signed_dists(les, ref_les, d_out_ref, d_in_ref):
    """Signed distance of each predicted contour pixel to the reference contour (+ outside the reference lesion)."""
    bd = les & ~ndi.binary_erosion(les, structure=EIGHT, border_value=0)
    sd = np.where(ref_les[bd], 1.0 - d_in_ref[bd], d_out_ref[bd])
    # components touching a reference lesion ("attached") vs spurious
    lab, n = ndi.label(les, structure=EIGHT)
    touch = np.zeros(n + 1, bool); touch[np.unique(lab[les & ref_les])] = True; touch[0] = False
    att = touch[lab[bd]]
    return sd, att


def rev_dists(les, ref_les):
    """Signed distance of each reference contour pixel to the predicted contour (+ = prediction extends outside it)."""
    br = ref_les & ~ndi.binary_erosion(ref_les, structure=EIGHT, border_value=0)
    if not les.any():
        return -np.full(int(br.sum()), np.inf)
    d_in = ndi.distance_transform_edt(les); d_out = ndi.distance_transform_edt(~les)
    return np.where(les[br], d_in[br] - 1.0, -d_out[br])


def wquant(vals, W, qs):
    """Inverted-CDF quantiles of a pooled distribution given unique sorted vals and count rows W (B x U)."""
    cw = np.cumsum(W, axis=1); tot = cw[:, -1:]
    out = []
    for q in qs:
        k = (cw < q * tot).sum(1)
        out.append(vals[np.minimum(k, len(vals) - 1)])
    return out


def main():
    rows = [r for r in csv.DictReader(open(f"{PACK}/split.csv")) if "abaxial" in r["file"]]
    sets = {"val": sorted([r for r in rows if r["split"] == "val" and r["grade"] == "roya"], key=lambda r: r["file"]),
            "test": sorted([r for r in rows if r["split"] == "test" and r["grade"] == "roya"], key=lambda r: r["file"]),
            "healthy": sorted([r for r in rows if r["split"] == "test" and r["grade"] == "healthy"], key=lambda r: r["file"])}
    sub = {"val": "val_roya", "test": "test_roya", "healthy": "test_healthy"}
    labels = {k: {r["file"]: load_label(r["file"]) for r in v} for k, v in sets.items()}
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(sets["test"]), size=(B, len(sets["test"])))  # shared by all models/OPs (paired)
    global MCNT
    MCNT = np.stack([np.bincount(row, minlength=len(sets["test"])) for row in idx]).astype(np.float64)  # B x n leaf weights

    res = {"meta": {"date": "2026-09-28", "n_val_leaves": len(sets["val"]), "n_test_leaves": len(sets["test"]),
                    "n_healthy_test_leaves": len(sets["healthy"]), "bootstrap": {"B": B, "seed": SEED, "unit": "leaf", "ci": "percentile 95%"},
                    "threshold_grid": "0.01..0.99 step 0.01, 0.991..0.999 step 0.001, 0.9995, 0.9999", "face": "abaxial (rust leaf-level rule)",
                    "erosion": "3x3 structuring element, 1 or 2 iterations, border_value=0; eroded pixels become leaf"},
           "models": {}}
    leaf_rows, curve_rows, bcounts = [], [], {}
    for m in MODELS:
        print("model", m, flush=True)
        # ---------- validation: threshold curves ----------
        val_c = {t: [] for t in GRID}; val_arg = []; val_ops = {"ero1": [], "ero2": []}
        for r in sets["val"]:
            lab = labels["val"][r["file"]]; ref_les = lab == 2; ref_leaf = int((lab >= 1).sum())
            am, c01, p2 = load_npz(m, sub["val"], r["file"])
            for t, c in zip(GRID, thr_counts(am, c01, p2, ref_les, ref_leaf)): val_c[t].append(c)
            les = am == 2; nonles = am == 1
            val_arg.append(counts_from_mask(les, nonles, ref_les, ref_leaf))
            for k, it in (("ero1", 1), ("ero2", 2)):
                e = ndi.binary_erosion(les, structure=EIGHT, iterations=it, border_value=0)
                val_ops[k].append(counts_from_mask(e, nonles | (les & ~e), ref_les, ref_leaf))
        curve = []
        for t in GRID:
            s = summarise(*leaf_metrics(val_c[t])); curve.append((t, s))
            curve_rows.append(dict(model=m, t=t, val_leaf_iou=round(s["leaf_iou"], 6), val_mae=round(s["mae"], 6),
                                   val_bias=round(s["bias"], 6), val_slope=round(s["slope"], 6)))
        t_bias = min(curve, key=lambda z: (abs(z[1]["bias"]), abs(z[0] - 0.5)))[0]
        t_mae = min(curve, key=lambda z: (z[1]["mae"], abs(z[0] - 0.5)))[0]
        t_iou = max(curve, key=lambda z: (z[1]["leaf_iou"], -abs(z[0] - 0.5)))[0]
        val_summ = {"argmax": summarise(*leaf_metrics(val_arg)), "thr_bias": dict(curve)[t_bias], "thr_mae": dict(curve)[t_mae],
                    "thr_0.50": dict(curve)[0.5], "thr_valIoU": dict(curve)[t_iou],
                    "ero1": summarise(*leaf_metrics(val_ops["ero1"])), "ero2": summarise(*leaf_metrics(val_ops["ero2"]))}
        thr = {"thr_bias": float(t_bias), "thr_mae": float(t_mae)}

        # ---------- test and healthy: apply once ----------
        test_c = {op: [] for op in OPS}; test_grid = {t: [] for t in GRID}; healthy = {op: [] for op in OPS}
        bd = {op: {"pred_all": [], "pred_att": [], "rev": []} for op in OPS}
        for kind in ("test", "healthy"):
            for r in sets[kind]:
                lab = labels[kind][r["file"]]; ref_les = lab == 2; ref_leaf = int((lab >= 1).sum())
                am, c01, p2 = load_npz(m, sub[kind], r["file"])
                les0 = am == 2; nonles0 = am == 1; p = p2.astype(np.int64)
                masks = {"argmax": (les0, nonles0)}
                for op in ("thr_bias", "thr_mae"):
                    l = p >= int(math.ceil(thr[op] * 65535)); masks[op] = (l, ~l & (c01 == 1))
                for op, it in (("ero1", 1), ("ero2", 2)):
                    e = ndi.binary_erosion(les0, structure=EIGHT, iterations=it, border_value=0)
                    masks[op] = (e, nonles0 | (les0 & ~e))
                if kind == "healthy":
                    for op in OPS:
                        c = counts_from_mask(*masks[op], ref_les, ref_leaf)
                        healthy[op].append(100 * c["pred_les"] / c["pred_leaf"] if c["pred_leaf"] else 0.0)
                    continue
                for t, c in zip(GRID, thr_counts(am, c01, p2, ref_les, ref_leaf)): test_grid[t].append(c)
                d_out_ref = ndi.distance_transform_edt(~ref_les); d_in_ref = ndi.distance_transform_edt(ref_les)
                for op in OPS:
                    c = counts_from_mask(*masks[op], ref_les, ref_leaf); test_c[op].append(c)
                    sd, att = signed_dists(masks[op][0], ref_les, d_out_ref, d_in_ref)
                    bd[op]["pred_all"].append(sd); bd[op]["pred_att"].append(sd[att]); bd[op]["rev"].append(rev_dists(masks[op][0], ref_les))
                    leaf_rows.append(dict(model=m, op=op, threshold=thr.get(op, ""), file=r["file"], ref_les=c["ref_les"], ref_leaf=c["ref_leaf"],
                                          pred_les=c["pred_les"], pred_leaf=c["pred_leaf"], inter=c["inter"], union=c["union"],
                                          ref_sev=round(100 * c["ref_les"] / c["ref_leaf"], 5),
                                          pred_sev=round(100 * c["pred_les"] / c["pred_leaf"], 5) if c["pred_leaf"] else 0,
                                          iou=round(c["inter"] / c["union"], 6) if c["union"] else ""))
        mres = {"label": LABEL[m], "thresholds_chosen_on_validation": {"min_abs_bias": float(t_bias), "min_mae": float(t_mae), "max_leaf_iou (context)": float(t_iou),
                                                                   "at_grid_upper_edge": bool(max(t_bias, t_mae) >= GRID.max())},
                "validation": val_summ, "test": {}, "test_paired_vs_argmax": {}, "healthy_test_abaxial": {}, "boundary_test": {}}
        L = {op: leaf_metrics(test_c[op]) for op in OPS}
        bt = {op: boot(*L[op], idx) for op in OPS}
        for op in OPS:
            s = summarise(*L[op])
            for k in ("leaf_iou", "mae", "bias", "slope"):
                s[f"{k}_ci95_bootstrap"] = pct(bt[op][k])
            if op in thr: s["threshold"] = thr[op]
            mres["test"][op] = s
            if op != "argmax":
                mres["test_paired_vs_argmax"][op] = {k: {"diff": float(s[k] - mres["test"]["argmax"][k]),
                                                         "ci95_bootstrap": pct(bt[op][k] - bt["argmax"][k])}
                                                     for k in ("leaf_iou", "mae", "bias", "slope")}
            h = np.array(healthy[op])
            mres["healthy_test_abaxial"][op] = {"n": len(h), "any_pixel": int((h > 0).sum()), "ge_0.5pct": int((h >= 0.5).sum()),
                                                "ge_5pct": int((h >= 5).sum()), "max_pct": float(h.max()), "mean_pct": float(h.mean())}
        # test-set oracle thresholds (context only, NOT a valid operating point: chosen on test)
        tc = [(t, summarise(*leaf_metrics(test_grid[t]))) for t in GRID]
        o_b = min(tc, key=lambda z: abs(z[1]["bias"])); o_m = min(tc, key=lambda z: z[1]["mae"])
        mres["test_oracle_context_only"] = {"min_abs_bias": {"t": float(o_b[0]), **{k: o_b[1][k] for k in ("bias", "mae", "leaf_iou", "slope")}},
                                            "min_mae": {"t": float(o_m[0]), **{k: o_m[1][k] for k in ("bias", "mae", "leaf_iou", "slope")}},
                                            "slope_range_over_grid": [float(min(z[1]["slope"] for z in tc)), float(max(z[1]["slope"] for z in tc))],
                                            "slope_at_t": {str(t): float(s["slope"]) for t, s in tc if t in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999)}}
        # ---------- boundary distance summaries ----------
        for op in OPS:
            out = {}
            for kind in ("pred_all", "pred_att", "rev"):
                arrs = bd[op][kind]
                finite = [a[np.isfinite(a)] for a in arrs]
                # quantiles: values beyond +-CLIP px are pooled into sentinel bins (exact as long as the quantile lies inside)
                qv = [np.where(a > CLIP, 999.0, np.where(a < -CLIP, -999.0, np.round(a, 4))) for a in finite]
                vals = np.unique(np.concatenate(qv))
                C = np.stack([np.bincount(np.searchsorted(vals, a), minlength=len(vals)) for a in qv]).astype(np.float64)
                q1, med, q3 = wquant(vals, C.sum(0, keepdims=True), (0.25, 0.5, 0.75))
                Wb = MCNT @ C  # B x U
                bq1, bmed, bq3 = wquant(vals, Wb, (0.25, 0.5, 0.75))
                sums = np.array([a.sum() for a in finite]); ns = np.array([len(a) for a in finite], float)
                bmean = (MCNT @ sums) / (MCNT @ ns)
                allv = np.concatenate(finite)
                per_leaf_med = [float(np.median(a)) if len(a) else float("nan") for a in finite]
                out[kind] = {"n_contour_px": int(ns.sum()), "median": float(med[0]), "q25": float(q1[0]), "q75": float(q3[0]),
                             "median_ci95_bootstrap": pct(bmed), "q25_ci95_bootstrap": pct(bq1), "q75_ci95_bootstrap": pct(bq3),
                             "mean": float(sums.sum() / ns.sum()), "mean_ci95_bootstrap": pct(bmean),
                             "share_positive": float((allv > 0).mean()), "share_zero": float((allv == 0).mean()),
                             "share_ge_2px": float((allv >= 2).mean()), "share_le_minus2px": float((allv <= -2).mean()),
                             "share_beyond_clip": float((np.abs(allv) > CLIP).mean()),
                             "per_leaf_median_of_medians": float(np.nanmedian(per_leaf_med)),
                             "n_nonfinite_excluded": int(sum((~np.isfinite(a)).sum() for a in arrs))}
                bcounts[f"{m}|{op}|{kind}|vals"] = vals; bcounts[f"{m}|{op}|{kind}|counts"] = C
            mres["boundary_test"][op] = out
        res["models"][m] = mres
        a, tb, tm = mres["test"]["argmax"], mres["test"]["thr_bias"], mres["test"]["thr_mae"]
        print(f"  t_bias={t_bias} t_mae={t_mae} | argmax bias {a['bias']:.2f} slope {a['slope']:.3f} | thr_bias bias {tb['bias']:.2f} "
              f"slope {tb['slope']:.3f} | thr_mae bias {tm['bias']:.2f} slope {tm['slope']:.3f}", flush=True)

    if MODELS != ALL_MODELS:  # debugging subset: stop before aggregation, write nothing into the results folder
        json.dump(res, open(os.environ.get("OPP_DEBUG_OUT", os.path.join(WORK, "debug.json")), "w"), indent=1); return
    # ---------- seed aggregation for matched CNNs ----------
    agg = {}
    for arch in ("unetpp", "deeplab"):
        ms = [f"m_{arch}_s{s}" for s in range(3)]
        agg[arch] = {}
        for op in OPS:
            d = {}
            for k in ("leaf_iou", "mae", "bias", "slope", "grade_acc", "qwk"):
                v = np.array([res["models"][m]["test"][op][k] for m in ms])
                d[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "min": float(v.min()), "max": float(v.max()), "per_seed": v.tolist()}
            if op in ("thr_bias", "thr_mae"):
                d["thresholds"] = [res["models"][m]["test"][op]["threshold"] for m in ms]
            d["boundary_pred_all_median"] = [res["models"][m]["boundary_test"][op]["pred_all"]["median"] for m in ms]
            d["boundary_pred_all_mean"] = [res["models"][m]["boundary_test"][op]["pred_all"]["mean"] for m in ms]
            d["healthy_ge_0.5pct"] = [res["models"][m]["healthy_test_abaxial"][op]["ge_0.5pct"] for m in ms]
            agg[arch][op] = d
    res["matched_seed_aggregate"] = agg

    # ---------- reproduction check against earlier verified results ----------
    ag = json.load(open(f"{BASE}/reanalysis/results/agreement/results.json"))["models"]
    rep = {}
    for m, k in (("pub_segformer", "rust/segformer"), ("pub_unetpp", "rust/unetpp"), ("pub_deeplab", "rust/deeplab")):
        a = res["models"][m]["test"]["argmax"]
        rep[m] = {"bias_here": a["bias"], "bias_agreement": ag[k]["bias"]["estimate"],
                  "slope_here": a["slope"], "slope_agreement": ag[k]["regression_of_differences_on_means"]["slope"],
                  "slope_ci_t_here": a["slope_ci95_t"], "slope_ci_t_agreement": ag[k]["regression_of_differences_on_means"]["slope_ci95_t"]}
        rep[m]["match"] = bool(abs(rep[m]["bias_here"] - rep[m]["bias_agreement"]) < 1e-5 and abs(rep[m]["slope_here"] - rep[m]["slope_agreement"]) < 1e-5)
    rs = json.load(open(f"{BASE}/reanalysis/out/retrained_summary.json"))
    runs = {"m_unetpp_s0": "pg26038-mb-roya-unetpp-lrB", "m_unetpp_s1": "pg26038-mbs-roya-unetpp-lrB-seed1",
            "m_unetpp_s2": "pg26038-mbs-roya-unetpp-lrB-seed2", "m_deeplab_s0": "pg26038-mb-roya-deeplab-lrB",
            "m_deeplab_s1": "pg26038-mbs-roya-deeplab-lrB-seed1", "m_deeplab_s2": "pg26038-mbs-roya-deeplab-lrB-seed2"}
    for m, run in runs.items():
        a = res["models"][m]["test"]["argmax"]; b = rs[run]["test"]; v = res["models"][m]["validation"]["argmax"]
        rep[m] = {"test_iou_here": a["leaf_iou"], "test_iou_summary": b["leaf_iou"], "test_bias_here": a["bias"], "test_bias_summary": b["bias"],
                  "val_iou_here": v["leaf_iou"], "val_iou_summary": rs[run]["val"]["leaf_iou"]}
        # tolerance 1e-5: retrained_summary.json is computed from per_image_retrained.csv, whose severities (5 dp) and IoUs (6 dp)
        # are rounded, so leaf means differ by ~1e-6; the pixel counts themselves were checked exactly in infer_prob.py
        rep[m]["match"] = bool(abs(a["leaf_iou"] - b["leaf_iou"]) < 1e-5 and abs(a["bias"] - b["bias"]) < 1e-5
                               and abs(v["leaf_iou"] - rs[run]["val"]["leaf_iou"]) < 1e-5)
    res["reproduction_argmax"] = rep
    res["reproduction_all_match"] = all(v["match"] for v in rep.values())
    print("reproduction all match:", res["reproduction_all_match"], flush=True)

    json.dump(res, open(os.path.join(HERE, "results.json"), "w"), indent=1)
    with open(os.path.join(WORK, "leaf_level.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(leaf_rows[0].keys())); w.writeheader(); w.writerows(leaf_rows)
    with open(os.path.join(WORK, "val_curves.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(curve_rows[0].keys())); w.writeheader(); w.writerows(curve_rows)
    np.savez_compressed(os.path.join(WORK, "boundary_counts.npz"), **bcounts)
    print("written results.json", flush=True)


if __name__ == "__main__":
    main()
