"""A1 parts 2-3 - paired tests and equivalence (TOST) of matched-protocol CNNs vs the published SegFormer-B5,
and matched UNet++ vs matched DeepLabV3+, per disease, test split.

Inputs (no inference is re-run):
  <ROOT>/reanalysis/out/per_image.csv            published models (SegFormer-B5 own-disease pack, test split)
  <ROOT>/reanalysis/out/per_image_retrained.csv  matched runs (lrB = 6e-5/6e-4), seeds 0-2
  <ROOT>/reanalysis/pass3/seed_summary.json      cross-check of the per-seed values

Leaf-level rules (paper protocol; identical to eval_retrained.py and results/paired):
  rust: diseased test leaves (grade_label 'roya'), abaxial face only (n = 21);
  leaf spot: mean of the two faces (IoU = mean of face IoUs, severity = mean of face severities) (n = 55).
  Grades: G0 < 0.5 <= G1 < 5 <= G2 < 20 <= G3 (%). Reference grade: rust = grade of the reference-mask severity;
  leaf spot PRIMARY = mask-derived grade (grade of the leaf-level mean reference severity; round-4 plan),
  SECONDARY = the annotator's visual grade (grade_label G1-G3). QWK: quadratic weights on grades 0..3.

Matched-model summary statistic (method a, primary): seed-mean. For every endpoint the per-seed statistic is
computed on the same (resampled) leaves and averaged over the 3 seeds. For IoU, MAE, bias and accuracy this equals
the statistic of the per-leaf seed-mean (per-leaf seed-mean IoU, seed-mean absolute error, seed-mean signed error,
seed-mean correctness). |bias| = |seed-mean bias|. QWK = mean of the 3 per-seed QWKs.
Method b (sensitivity): seed x leaf bootstrap. Same leaf resamples; in each resample one seed is drawn uniformly
and independently for each matched model and that seed's statistic is used.

Statistics: 10,000 resamples of leaves, numpy default_rng(20260928); leaf index matrices drawn rust then leaf
spot, then the seed draws (rust: UNet++, DeepLabV3+; leaf spot: UNet++, DeepLabV3+). Percentile intervals.
Resamples where a statistic is undefined (QWK 0/0) are dropped and counted. Differences = first minus second.
Two-sided bootstrap p = min(1, 2 min(P*(d<=0), P*(d>=0))). Wilcoxon signed-rank (two-sided) on per-leaf IoU
differences (matched per-leaf seed-mean IoU). Holm within disease across the 3 contrasts, per endpoint.
TOST at alpha 0.05 via the 90% percentile bootstrap interval, margins fixed in the round-4 plan AFTER the primary
results were known (post hoc): leaf IoU +-0.03, MAE +-0.5 pp, QWK +-0.05.
  equivalent      : 90% CI entirely inside (-m, +m)
  not equivalent  : 90% CI entirely outside [-m, +m] (lower >= +m or upper <= -m)
  inconclusive    : otherwise (the 90% CI overlaps a margin)
Bootstrap TOST p = max(P*(d <= -m), P*(d >= +m)).
"""
from __future__ import annotations
import os as _os  # (release) all paths below are relative to ROOT (repository root by default; see README)
ROOT = (_os.environ.get("PG26038_ROOT") or _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))).replace(chr(92), "/")

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
PUB_CSV = Path((ROOT + "/reanalysis/out/per_image.csv"))
RET_CSV = Path((ROOT + "/reanalysis/out/per_image_retrained.csv"))
SEED_JSON = Path((ROOT + "/reanalysis/pass3/seed_summary.json"))
SEED = 20260928
B = 10_000
CUTS = (0.5, 5.0, 20.0)
DISEASES = ["roya", "ojo"]
DIS_NAME = {"roya": "Coffee leaf rust (abaxial face)", "ojo": "American leaf spot (mean of both faces)"}
N_EXPECTED = {"roya": 21, "ojo": 55}
MODELS = ["segformer", "unetpp", "deeplab"]
NAME = {"segformer": "SegFormer-B5 (published)", "unetpp": "UNet++ (matched)", "deeplab": "DeepLabV3+ (matched)"}
CONTRASTS = [("unetpp", "segformer"), ("deeplab", "segformer"), ("unetpp", "deeplab")]
MARGINS = {"leaf_iou": 0.03, "mae": 0.5, "qwk": 0.05, "qwk_visual": 0.05}


def runs(dis, arch):
    return [f"pg26038-mb-{dis}-{arch}-lrB", f"pg26038-mbs-{dis}-{arch}-lrB-seed1", f"pg26038-mbs-{dis}-{arch}-lrB-seed2"]


def grade(sev):
    return np.digitize(np.asarray(sev, float), CUTS).astype(int)


def qwk(a, b):
    """Quadratic-weighted kappa on integer grades, vectorised over the last axis (same as results/paired)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a, b = np.broadcast_arrays(a, b)
    n = a.shape[-1]
    num = ((a - b) ** 2).sum(-1)
    den = (a ** 2).sum(-1) + (b ** 2).sum(-1) - 2.0 * a.sum(-1) * b.sum(-1) / n
    with np.errstate(invalid="ignore", divide="ignore"):
        k = 1.0 - num / den
    return np.where(den > 1e-12, k, np.nan)


def qwk_matrix(a, b, k=4):
    O = np.zeros((k, k))
    for i, j in zip(a, b):
        O[i, j] += 1
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    W = np.subtract.outer(np.arange(k), np.arange(k)) ** 2
    den = (W * E).sum()
    return float("nan") if den == 0 else 1 - (W * O).sum() / den


def pct(x, level):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    a = (1 - level) / 2 * 100
    lo, hi = np.percentile(x, [a, 100 - a])
    return [float(lo), float(hi)]


def holm(ps):
    ps = [1.0 if p is None else p for p in ps]
    m = len(ps)
    order = np.argsort(ps)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


# ----------------------------------------------------------------------------------------------------- data
def leaf_table(rows, dis):
    """rows of one model/run on the test split -> dict leaf_id -> (iou, pred_sev, ref_sev, visual_label)."""
    per = {}
    for r in rows:
        if r["split"] != "test" or r["grade_label"] == "healthy":
            continue
        if dis == "roya" and r["face"] != "abaxial":
            continue
        per.setdefault(r["leaf_id"], []).append(r)
    out = {}
    for lid, v in per.items():
        if dis == "ojo":
            assert len(v) == 2, (lid, len(v))
        else:
            assert len(v) == 1, (lid, len(v))
        out[lid] = (np.mean([float(x["lesion_iou"]) for x in v]), np.mean([float(x["pred_sev"]) for x in v]),
                    np.mean([float(x["ref_sev"]) for x in v]), v[0]["grade_label"])
    return out


def load():
    pub = list(csv.DictReader(open(PUB_CSV)))
    ret = list(csv.DictReader(open(RET_CSV)))
    data = {}
    for dis in DISEASES:
        tabs = {"segformer": [leaf_table([r for r in pub if r["model_disease"] == dis and r["image_pack"] == dis
                                          and r["arch"] == "segformer"], dis)]}
        for arch in ("unetpp", "deeplab"):
            tabs[arch] = []
            for run in runs(dis, arch):
                rr = [r for r in ret if r["run"] == run]
                assert rr, run
                tabs[arch].append(leaf_table(rr, dis))
        leaves = sorted(tabs["segformer"][0])
        for m in MODELS:
            for t in tabs[m]:
                assert sorted(t) == leaves, (dis, m)
        assert len(leaves) == N_EXPECTED[dis], (dis, len(leaves))
        rs = np.array([tabs["segformer"][0][l][2] for l in leaves])
        maxdiff = 0.0
        for m in MODELS:
            for t in tabs[m]:
                maxdiff = max(maxdiff, float(np.max(np.abs(np.array([t[l][2] for l in leaves]) - rs))))
        assert maxdiff < 1e-3, maxdiff
        d = {"leaves": leaves, "rs": rs, "ref_sev_max_abs_diff_across_models": maxdiff,
             "g_mask": grade(rs)}
        if dis == "ojo":
            d["g_vis"] = np.array([int(tabs["segformer"][0][l][3][1]) for l in leaves])
        for m in MODELS:
            d[m] = {"iou": np.array([[t[l][0] for l in leaves] for t in tabs[m]]),
                    "ps": np.array([[t[l][1] for l in leaves] for t in tabs[m]])}
        data[dis] = d
    return data


# ----------------------------------------------------------------------------------------------------- stats
def endpoints(dis):
    e = ["leaf_iou", "mae", "bias", "abs_bias", "grade_acc", "qwk"]
    if dis == "ojo":
        e += ["grade_acc_visual", "qwk_visual"]
    return e


def per_seed_stats(d, m, idx, dis):
    """Per-seed statistics on resampled leaves. idx: (B, n) -> dict endpoint -> (S, B)."""
    iou, ps, rs = d[m]["iou"], d[m]["ps"], d["rs"]
    err = ps - rs[None]
    gp = grade(ps)
    out = {"leaf_iou": iou[:, idx].mean(-1),
           "mae": np.abs(err)[:, idx].mean(-1),
           "bias": err[:, idx].mean(-1)}
    refs = {"": d["g_mask"]}
    if dis == "ojo":
        refs["_visual"] = d["g_vis"]
    for sfx, gr in refs.items():
        out["grade_acc" + sfx] = (gp == gr[None])[:, idx].mean(-1)
        out["qwk" + sfx] = qwk(gr[idx][None], gp[:, idx])
    return out


def summarise_a(ps_stats):
    """Seed-mean summary (method a)."""
    out = {}
    for k, v in ps_stats.items():
        out[k] = v.mean(0)
    out["abs_bias"] = np.abs(ps_stats["bias"].mean(0))
    return out


def summarise_b(ps_stats, choice):
    """One drawn seed per resample (method b). choice: (B,) seed index (all zeros for the single-run model)."""
    cols = np.arange(choice.shape[0])
    out = {k: v[choice, cols] for k, v in ps_stats.items()}
    out["abs_bias"] = np.abs(out["bias"])
    return out


def verdict(ci90, m):
    lo, hi = ci90
    if lo > -m and hi < m:
        return "equivalent"
    if lo >= m or hi <= -m:
        return "not equivalent"
    return "inconclusive"


def main():
    data = load()
    rng = np.random.default_rng(SEED)
    idx = {dis: rng.integers(0, N_EXPECTED[dis], size=(B, N_EXPECTED[dis])) for dis in DISEASES}
    choice = {dis: {"segformer": np.zeros(B, int), "unetpp": rng.integers(0, 3, B), "deeplab": rng.integers(0, 3, B)}
              for dis in DISEASES}
    seedsum = json.load(open(SEED_JSON))
    res = {"design": {
        "bootstrap": {"resamples": B, "seed": SEED, "generator": "numpy.random.default_rng", "unit": "leaf",
                      "interval": "percentile (numpy linear)", "undefined": "QWK 0/0 resamples dropped and counted",
                      "draw_order": "leaf index matrices rust then leaf spot; then seed draws rust (UNet++, DeepLabV3+), leaf spot (UNet++, DeepLabV3+)"},
        "method_a": "seed-mean of the matched model (per-seed statistic on the same leaves, averaged over seeds 0-2); |bias| = |seed-mean bias|; QWK = mean of per-seed QWKs",
        "method_b": "seed x leaf bootstrap: same leaf resamples, one seed drawn per matched model per resample",
        "contrasts": [f"{a} minus {b}" for a, b in CONTRASTS],
        "holm": "within disease across the 3 contrasts, per endpoint (and for the Wilcoxon tests)",
        "tost_margins": MARGINS,
        "tost_margins_note": "fixed in round4_plan.md after the primary results were known (post hoc; not preregistered)",
        "tost_rule": "alpha 0.05 via 90% percentile bootstrap CI; equivalent = CI inside (-m,m); not equivalent = CI entirely outside [-m,m]; inconclusive otherwise",
        "leaf_spot_grade_reference": "primary = mask-derived (round-4 plan), secondary = visual",
        "inputs": [str(PUB_CSV), str(RET_CSV)]}}
    res["diseases"] = {}
    for dis in DISEASES:
        d = data[dis]
        n = len(d["leaves"])
        ident = np.arange(n)[None]
        E = endpoints(dis)
        obs_seed = {m: {k: v[:, 0] for k, v in per_seed_stats(d, m, ident, dis).items()} for m in MODELS}
        boot_seed = {m: per_seed_stats(d, m, idx[dis], dis) for m in MODELS}
        obs_a = {m: {k: float(v[0]) for k, v in summarise_a({k2: v2[:, None] for k2, v2 in obs_seed[m].items()}).items()}
                 for m in MODELS}
        boot_a = {m: summarise_a(boot_seed[m]) for m in MODELS}
        boot_b = {m: summarise_b(boot_seed[m], choice[dis][m]) for m in MODELS}
        R = {"n_leaves": n, "ref_sev_max_abs_diff_across_models": d["ref_sev_max_abs_diff_across_models"]}
        # cross-check qwk implementation
        for m in MODELS:
            for s in range(d[m]["ps"].shape[0]):
                assert abs(qwk_matrix(d["g_mask"], grade(d[m]["ps"][s])) - obs_seed[m]["qwk"][s]) < 1e-9
        # ---- per-model values
        mv = {}
        for m in MODELS:
            S = d[m]["ps"].shape[0]
            entry = {"n_seeds": S, "seed_mean_point": {k: obs_a[m][k] for k in E}}
            per = {k: [float(x) for x in obs_seed[m][k]] for k in obs_seed[m]}
            per["abs_bias"] = [abs(x) for x in per["bias"]]
            entry["per_seed"] = per
            if S > 1:
                entry["mean_sd_range_over_seeds"] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                                                         "min": float(np.min(v)), "max": float(np.max(v))} for k, v in per.items()}
            # 95% CI of the model's own value (method a) for context
            entry["ci95_method_a"] = {k: pct(boot_a[m][k], 0.95) for k in E}
            entry["correct_leaves_seed_mean"] = float(obs_a[m]["grade_acc"] * n)
            mv[m] = entry
        R["models"] = mv
        # ---- cross-check with seed_summary.json
        xc = {}
        for arch in ("unetpp", "deeplab"):
            key = f"{dis}-{arch}"
            js = seedsum[key]
            mapping = {"leaf_iou": "leaf_iou", "mae": "mae", "bias": "bias",
                       "grade_acc": "grade_acc" if dis == "roya" else "grade_acc_visual",
                       "qwk": "qwk" if dis == "roya" else "qwk_visual"}
            rows = {}
            ok = True
            for jk, mk in mapping.items():
                mine = mv[arch]["per_seed"][mk]
                theirs = js[jk]["per_seed"]
                good = all(abs(round(a, 3) - b) < 1e-9 or abs(a - b) <= 0.0005 + 1e-9 for a, b in zip(mine, theirs))
                ok &= good
                rows[jk] = {"seed_summary_per_seed": theirs, "recomputed_per_seed": [round(x, 4) for x in mine],
                            "recomputed_field": mk, "match_3dp": good,
                            "seed_summary_mean_sd": [js[jk]["mean"], js[jk]["sd"]],
                            "recomputed_mean_sd": [round(float(np.mean(mine)), 4), round(float(np.std(mine, ddof=1)), 4)]}
            xc[arch] = {"all_match": ok, "fields": rows}
        R["cross_check_seed_summary"] = xc
        # ---- contrasts
        C = {}
        for a, b in CONTRASTS:
            key = f"{a}_minus_{b}"
            cc = {"first": NAME[a], "second": NAME[b]}
            for k in E:
                pt = obs_a[a][k] - obs_a[b][k]
                entry = {"point": float(pt)}
                for meth, bs in (("a", boot_a), ("b", boot_b)):
                    dd = bs[a][k] - bs[b][k]
                    valid = dd[~np.isnan(dd)]
                    p = float(min(1.0, 2 * min(np.mean(valid <= 0), np.mean(valid >= 0))))
                    e = {"ci95": pct(dd, 0.95), "ci90": pct(dd, 0.90), "p_boot": p,
                         "undefined_resamples": int(np.isnan(dd).sum()),
                         "share_exact_zero": float(np.mean(valid == 0))}
                    if k in MARGINS:
                        mg = MARGINS[k]
                        e["tost_margin"] = mg
                        e["tost_p"] = float(max(np.mean(valid <= -mg), np.mean(valid >= mg)))
                        e["tost_verdict"] = verdict(e["ci90"], mg)
                    # a percentile bound sitting at exactly 0 means >= 5% of resamples had no difference at all
                    # (all discordant leaves favour one model); such intervals understate that tail
                    e["ci90_bound_pinned_at_zero"] = bool(min(abs(e["ci90"][0]), abs(e["ci90"][1])) < 1e-12)
                    e["ci95_bound_pinned_at_zero"] = bool(min(abs(e["ci95"][0]), abs(e["ci95"][1])) < 1e-12)
                    entry[f"method_{meth}"] = e
                # per-seed point differences (descriptive)
                sa, sb = obs_seed[a][k] if k != "abs_bias" else np.abs(obs_seed[a]["bias"]), \
                    obs_seed[b][k] if k != "abs_bias" else np.abs(obs_seed[b]["bias"])
                if len(sb) == 1:
                    entry["per_seed_point_diffs_first_model_seeds"] = [float(x - sb[0]) for x in sa]
                elif len(sa) == 1:
                    entry["per_seed_point_diffs_second_model_seeds"] = [float(sa[0] - x) for x in sb]
                else:
                    entry["per_seed_point_diffs_same_seed_index"] = [float(x - y) for x, y in zip(sa, sb)]
                cc[k] = entry
            # Wilcoxon on per-leaf IoU (matched: per-leaf seed-mean)
            dl = d[a]["iou"].mean(0) - d[b]["iou"].mean(0)
            nz = dl[dl != 0]
            ties = len(np.unique(np.abs(nz))) < len(nz)
            meth = "exact" if (len(nz) == len(dl) and not ties) else "approx"
            w = wilcoxon(dl, alternative="two-sided", method=meth)
            cc["wilcoxon_leaf_iou"] = {"n": int(len(dl)), "n_nonzero": int(len(nz)), "statistic_W": float(w.statistic),
                                       "p": float(w.pvalue), "method": meth, "ties": bool(ties),
                                       "n_first_better": int((dl > 0).sum()), "n_second_better": int((dl < 0).sum()),
                                       "median_diff": float(np.median(dl)), "mean_diff": float(np.mean(dl))}
            C[key] = cc
        # ---- Holm within disease across contrasts
        keys = [f"{a}_minus_{b}" for a, b in CONTRASTS]
        adj = holm([C[k]["wilcoxon_leaf_iou"]["p"] for k in keys])
        for k, p in zip(keys, adj):
            C[k]["wilcoxon_leaf_iou"]["p_holm3"] = p
        for ep in E:
            for meth in ("a", "b"):
                adj = holm([C[k][ep][f"method_{meth}"]["p_boot"] for k in keys])
                for k, p in zip(keys, adj):
                    C[k][ep][f"method_{meth}"]["p_boot_holm3"] = p
                if ep in MARGINS:
                    adj = holm([C[k][ep][f"method_{meth}"]["tost_p"] for k in keys])
                    for k, p in zip(keys, adj):
                        e = C[k][ep][f"method_{meth}"]
                        e["tost_p_holm3"] = p
                        e["equivalent_after_holm"] = bool(e["tost_verdict"] == "equivalent" and p < 0.05)
        R["contrasts"] = C
        res["diseases"][dis] = R
        # per-leaf table
        with open(HERE / f"per_leaf_{dis}.csv", "w", newline="") as fh:
            wr = csv.writer(fh)
            hdr = ["leaf_id", "ref_sev", "ref_grade_mask"] + (["ref_grade_visual"] if dis == "ojo" else [])
            hdr += ["segformer_iou", "segformer_pred_sev"]
            for arch in ("unetpp", "deeplab"):
                for s in range(3):
                    hdr += [f"{arch}_s{s}_iou", f"{arch}_s{s}_pred_sev"]
            wr.writerow(hdr)
            for i, l in enumerate(d["leaves"]):
                row = [l, round(float(d["rs"][i]), 5), int(d["g_mask"][i])] + ([int(d["g_vis"][i])] if dis == "ojo" else [])
                row += [round(float(d["segformer"]["iou"][0, i]), 6), round(float(d["segformer"]["ps"][0, i]), 5)]
                for arch in ("unetpp", "deeplab"):
                    for s in range(3):
                        row += [round(float(d[arch]["iou"][s, i]), 6), round(float(d[arch]["ps"][s, i]), 5)]
                wr.writerow(row)
    json.dump(res, open(HERE / "paired_results.json", "w"), indent=1)
    # console summary
    for dis in DISEASES:
        R = res["diseases"][dis]
        print(f"== {dis} n={R['n_leaves']}  xcheck:", {a: R['cross_check_seed_summary'][a]['all_match'] for a in ('unetpp', 'deeplab')})
        for m in MODELS:
            print("  ", m, {k: round(v, 4) for k, v in R["models"][m]["seed_mean_point"].items()})
        for k, cc in R["contrasts"].items():
            w = cc["wilcoxon_leaf_iou"]
            print(f"  {k}: Wilcoxon W={w['statistic_W']:.1f} p={w['p']:.4g} holm={w['p_holm3']:.4g} ({w['method']}) {w['n_first_better']}/{w['n_second_better']}")
            for ep in endpoints(dis):
                e = cc[ep]
                a, b = e["method_a"], e["method_b"]
                s = f"    {ep:17s} d={e['point']:+.4f} a95=[{a['ci95'][0]:+.4f},{a['ci95'][1]:+.4f}] a90=[{a['ci90'][0]:+.4f},{a['ci90'][1]:+.4f}] p={a['p_boot']:.4f} holm={a['p_boot_holm3']:.4f}"
                s += f" | b95=[{b['ci95'][0]:+.4f},{b['ci95'][1]:+.4f}] b90=[{b['ci90'][0]:+.4f},{b['ci90'][1]:+.4f}] p={b['p_boot']:.4f}"
                if "tost_verdict" in a:
                    s += f" | TOST a:{a['tost_verdict']} (p={a['tost_p']:.4f}) b:{b['tost_verdict']} (p={b['tost_p']:.4f})"
                if a["undefined_resamples"] or b["undefined_resamples"]:
                    s += f" undef a{a['undefined_resamples']} b{b['undefined_resamples']}"
                print(s)


if __name__ == "__main__":
    main()
