"""Validation-set results and data structure (PG-26038 major revision; reviewer questions on the rust
reference/protocol, clustering by collection date, and grade/class distributions).

Reads (never writes):
  <ROOT>/reanalysis/out/per_image.csv          per-image reference/predicted pixel counts (reproduced inference)
  <ROOT>/reanalysis/out/preds/...              predicted class maps (used only for an integrity check)
  <ROOT>/datasets/pg26038-roya-real-pack       images/, masks/, split.csv
  <ROOT>/datasets/pg26038-ojo-real-pack        images/, masks/, meta/split.csv
Writes only to the folder that holds this script:
  work/ref_pixels.csv     reference class pixel counts per image, letterbox (1024 x 1024) and native resolution
  work/leaf_tables.csv    per-leaf metrics for every (disease, model, split, face rule)
  results.json            every number reported in results.md

Definitions (as in the paper and in run_inference.py):
  leaf = alpha > 8; lesion = mask > 127 AND leaf; labels letterboxed to 1024 x 1024 with NEAREST, black (0) padding.
  Leaf-level rules: rust = abaxial face; American leaf spot = mean of the two faces (IoU = mean of face IoUs,
  severity = mean of face severities). Reference grade: rust = cut-points 0.5/5/20% on the reference severity
  (same face rule); leaf spot = annotator's visual grade (G1-G3 -> 1-3). Predicted grade = cut-points on the
  predicted severity. QWK = quadratic-weighted Cohen's kappa over grades 0-3.
  MAE = mean |pred - ref| severity (percentage points); bias = mean (pred - ref); LoA = bias +/- 1.96 SD (ddof = 1).
Bootstrap: 10,000 resamples of leaves with replacement, percentile 95% CIs, numpy default_rng([20260927, tag]) where
  the integer tag identifies the data set (listed in results.json), so that the three models of a disease share the
  same resamples and independent data sets (val vs test) use independent streams. Every table that reports a
  statistic of the same leaves reuses the same resamples (e.g. the rust face-rule tables reuse the rust val/test
  streams, so the abaxial rows there are identical to the validation/test table, and the leaf-bootstrap column of the
  date-clustering table reuses the leaf-spot test stream), so one statistic never carries two different CIs.
"""
from __future__ import annotations
import os as _os  # (release) all paths below are relative to ROOT (repository root by default; see README)
ROOT = (_os.environ.get("PG26038_ROOT") or _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))).replace(chr(92), "/")

import csv
import json
import os
import platform
import time
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
from PIL import Image
from scipy.stats import chi2_contingency

BASE = ROOT
OUT = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(OUT, "work")
os.makedirs(WORK, exist_ok=True)
PER_IMAGE = f"{BASE}/reanalysis/out/per_image.csv"
PREDS = f"{BASE}/reanalysis/out/preds"
PACKS = {"roya": (f"{BASE}/datasets/pg26038-roya-real-pack", f"{BASE}/datasets/pg26038-roya-real-pack/split.csv"),
         "ojo": (f"{BASE}/datasets/pg26038-ojo-real-pack", f"{BASE}/datasets/pg26038-ojo-real-pack/meta/split.csv")}
SIZE = 1024
B = 10000
SEED = 20260927
DIS = ("roya", "ojo")
DIS_NAME = {"roya": "Coffee leaf rust", "ojo": "American leaf spot"}
ARCHS = ("segformer", "unetpp", "deeplab")
ARCH_NAME = {"segformer": "SegFormer-B5", "unetpp": "UNet++", "deeplab": "DeepLabV3+"}
SPLITS = ("train", "val", "test")
CUTS = (0.5, 5.0, 20.0)
K = 4

# Published values (manuscript Tables 2 and 4, Results text); rounding as printed.
PUB = {
    "roya": {"segformer": dict(iou=0.767, mae=0.99, bias=0.99, qwk=0.88, loa=(-1.68, 3.66)),
             "unetpp": dict(iou=0.647, mae=1.77, bias=1.75, qwk=0.79, loa=(-2.68, 6.17)),
             "deeplab": dict(iou=0.632, mae=1.47, bias=1.41, qwk=0.88, loa=(-3.07, 5.89))},
    "ojo": {"segformer": dict(iou=0.884, mae=0.22, bias=0.16, qwk=0.82, loa=(-0.47, 0.79)),
            "unetpp": dict(iou=0.853, mae=0.15, bias=-0.03, qwk=0.85, loa=(-0.50, 0.44)),
            "deeplab": dict(iou=0.815, mae=0.25, bias=0.08, qwk=0.84, loa=(-0.66, 0.82))},
}
PUB_MAE_CI = {"roya": {"segformer": (0.48, 1.61), "unetpp": (0.90, 2.76), "deeplab": (0.64, 2.50)},
              "ojo": {"segformer": (0.15, 0.30), "unetpp": (0.10, 0.20), "deeplab": (0.17, 0.33)}}

# Bootstrap stream tags (default_rng([SEED, tag])). The rust face-rule tables (section 4) reuse tags 1 and 2.
TAG = {("roya", "test"): 1, ("roya", "val"): 2, ("ojo", "test"): 3, ("ojo", "val"): 4,
       ("ojo", "test", "date_cluster"): 5, ("ojo", "test", "date_stratified"): 6}


def grade(s: float) -> int:
    return 0 if s < CUTS[0] else 1 if s < CUTS[1] else 2 if s < CUTS[2] else 3


def qwk(a, b) -> float:
    O = np.zeros((K, K))
    for x, y in zip(a, b):
        O[int(x), int(y)] += 1
    Ex = np.outer(O.sum(1), O.sum(0)) / O.sum()
    W = (np.subtract.outer(np.arange(K), np.arange(K)) ** 2) / (K - 1) ** 2
    den = (W * Ex).sum()
    return float("nan") if den == 0 else float(1 - (W * O).sum() / den)


# ----------------------------------------------------------------------------------------------------------------
# Reference pixels per image (letterbox + native) and integrity check of the predicted maps
# ----------------------------------------------------------------------------------------------------------------
def resize_pad(arr, nearest=False):
    """Verbatim from run_inference.py."""
    im = Image.fromarray(arr); w, h = im.size; s = SIZE / max(w, h)
    im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.NEAREST if nearest else Image.BILINEAR)
    canvas = Image.new(im.mode, (SIZE, SIZE), 0); canvas.paste(im, ((SIZE - im.width) // 2, (SIZE - im.height) // 2))
    return np.asarray(canvas)


def ref_worker(job):
    pack, r = job
    root = PACKS[pack][0]
    rgba = np.asarray(Image.open(f"{root}/images/{r['file']}").convert("RGBA"))
    leaf = rgba[..., 3] > 8
    les = (np.asarray(Image.open(f"{root}/masks/{r['file']}").convert("L")) > 127) & leaf
    lab = np.zeros(leaf.shape, np.uint8); lab[leaf] = 1; lab[les] = 2
    h, w = leaf.shape
    s = SIZE / max(w, h)
    pad_px = SIZE * SIZE - max(1, round(w * s)) * max(1, round(h * s))
    lb = resize_pad(lab, nearest=True)
    out = dict(pack=pack, split=r["split"], file=r["file"], leaf_id=r["leaf_id"], grade_label=r["grade"],
               face="abaxial" if "abaxial" in r["file"] else "adaxial",
               lb_bg_px=int((lb == 0).sum()), lb_leaf_px=int((lb == 1).sum()), lb_lesion_px=int((lb == 2).sum()),
               lb_pad_px=int(pad_px), nat_w=w, nat_h=h, nat_bg_px=int((lab == 0).sum()),
               nat_leaf_px=int((lab == 1).sum()), nat_lesion_px=int((lab == 2).sum()), pred_check="")
    if r["split"] in ("val", "test"):
        checks = []
        for arch in ARCHS:
            p = np.asarray(Image.open(f"{PREDS}/{pack}_{arch}/{pack}_{r['split']}/{r['file']}"))
            checks.append("%s:%d:%d:%d:%d" % (arch, int((p >= 1).sum()), int((p == 2).sum()),
                                              int(((p == 2) & (lb == 2)).sum()), int(((p == 2) | (lb == 2)).sum())))
        out["pred_check"] = ";".join(checks)
    return out


def build_ref_pixels():
    path = os.path.join(WORK, "ref_pixels.csv")
    if os.path.exists(path):
        return list(csv.DictReader(open(path, newline="")))
    jobs = [(pack, r) for pack in DIS for r in csv.DictReader(open(PACKS[pack][1], newline=""))]
    t0 = time.time()
    with Pool(8) as pool:
        rows = pool.map(ref_worker, jobs, chunksize=4)
    print(f"reference pixels for {len(rows)} images in {time.time() - t0:.0f} s", flush=True)
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    return list(csv.DictReader(open(path, newline="")))


# ----------------------------------------------------------------------------------------------------------------
# Leaf tables
# ----------------------------------------------------------------------------------------------------------------
def load_per_image():
    rows = list(csv.DictReader(open(PER_IMAGE, newline="")))
    for r in rows:
        for k in ("ref_leaf_px", "ref_lesion_px", "pred_leaf_px", "pred_lesion_px", "inter_lesion", "union_lesion"):
            r[k] = int(r[k])
        r["ref_sev"] = float(r["ref_sev"]); r["pred_sev"] = float(r["pred_sev"])
        r["lesion_iou"] = float(r["lesion_iou"]) if r["lesion_iou"] != "" else float("nan")
    return rows


def leaf_table(P, dis, arch, split, rule):
    """Diseased leaves of the model's own pack. rule: 'abaxial' | 'adaxial' | 'both'."""
    faces = defaultdict(dict)
    for r in P:
        if (r["model_disease"] == dis and r["arch"] == arch and r["image_pack"] == dis and r["split"] == split
                and r["grade_label"] != "healthy"):
            faces[r["leaf_id"]][r["face"]] = r
    out = []
    for lid in sorted(faces):
        f = faces[lid]
        assert set(f) == {"abaxial", "adaxial"}, lid
        use = [f["abaxial"], f["adaxial"]] if rule == "both" else [f[rule]]
        ious = [u["lesion_iou"] for u in use if np.isfinite(u["lesion_iou"])]
        ref = float(np.mean([u["ref_sev"] for u in use])); pred = float(np.mean([u["pred_sev"] for u in use]))
        rg = int(f["abaxial"]["grade_label"][1]) if dis == "ojo" else grade(ref)
        out.append(dict(disease=dis, arch=arch, split=split, rule=rule, leaf_id=lid, date=lid[:8],
                        iou=float(np.mean(ious)) if ious else float("nan"), n_iou_faces=len(ious),
                        ref_sev=ref, pred_sev=pred, ref_grade=rg, pred_grade=grade(pred),
                        ref_grade_abaxial=grade(f["abaxial"]["ref_sev"]) if dis == "roya" else rg))
    return out


def point_metrics(L, ref_grade_key="ref_grade"):
    iou = np.array([x["iou"] for x in L]); ref = np.array([x["ref_sev"] for x in L])
    pred = np.array([x["pred_sev"] for x in L]); e = pred - ref
    rg = [x[ref_grade_key] for x in L]; pg = [x["pred_grade"] for x in L]
    sd = float(e.std(ddof=1))
    return dict(n=len(L), iou=float(np.nanmean(iou)), n_iou_undefined=int(np.isnan(iou).sum()),
                mae=float(np.abs(e).mean()), bias=float(e.mean()), sd_diff=sd,
                loa=[float(e.mean() - 1.96 * sd), float(e.mean() + 1.96 * sd)],
                rmse=float(np.sqrt((e ** 2).mean())), qwk=qwk(rg, pg),
                mean_ref_sev=float(ref.mean()), max_ref_sev=float(ref.max()),
                mae_relative_to_mean_ref_sev=float(np.abs(e).mean() / ref.mean()),
                exact_grade_agreement=float(np.mean(np.array(rg) == np.array(pg))),
                max_grade_error=int(np.max(np.abs(np.array(rg) - np.array(pg)))),
                ref_grade_counts={f"G{g}": int(sum(1 for v in rg if v == g)) for g in range(K)},
                pred_grade_counts={f"G{g}": int(sum(1 for v in pg if v == g)) for g in range(K)})


def boot_idx(tag, n):
    rng = np.random.default_rng([SEED, tag])
    return rng.integers(0, n, (B, n))


def ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return [float(v) for v in np.percentile(x, [2.5, 97.5])], int(len(x))


def boot_metrics(L, idx, ref_grade_key="ref_grade"):
    iou = np.array([x["iou"] for x in L]); ref = np.array([x["ref_sev"] for x in L])
    pred = np.array([x["pred_sev"] for x in L]); e = pred - ref
    rg = np.array([x[ref_grade_key] for x in L]); pg = np.array([x["pred_grade"] for x in L])
    with np.errstate(all="ignore"):
        s_iou = np.nanmean(iou[idx], axis=1)
    s_mae = np.abs(e)[idx].mean(1); s_bias = e[idx].mean(1)
    s_qwk = np.array([qwk(rg[i], pg[i]) for i in idx])
    return dict(iou=s_iou, mae=s_mae, bias=s_bias, qwk=s_qwk)


def summarize_boot(bs):
    out = {}
    for k, v in bs.items():
        c, nv = ci(v)
        out[f"{k}_ci95"] = c
        if nv < B:
            out[f"{k}_n_valid_resamples"] = nv
    return out


# ----------------------------------------------------------------------------------------------------------------
def main():
    t0 = time.time()
    P = load_per_image()
    R = build_ref_pixels()
    res = dict(meta=dict(script=os.path.abspath(__file__), per_image=PER_IMAGE, packs={k: v[0] for k, v in PACKS.items()},
                         python=platform.python_version(), numpy=np.__version__, B=B, seed=SEED,
                         bootstrap_streams={"/".join(k): v for k, v in TAG.items()},
                         ci="percentile 95% (2.5th and 97.5th percentiles, numpy linear interpolation)",
                         leaf_rules="rust = abaxial face; American leaf spot = mean of the two faces",
                         reference_grade="rust: cut-points 0.5/5/20% on reference severity; leaf spot: annotator's visual grade",
                         loa="bias +/- 1.96 SD of differences (ddof = 1)",
                         qwk="quadratic-weighted Cohen's kappa over grades 0-3",
                         healthy_test_coverage="21 healthy rust-pack test leaves are present locally; the paper used 54 (33 not in the local pack)"))

    # ---- integrity: rebuilt reference labels vs per_image.csv; predicted maps vs per_image.csv ----
    Rk = {(r["pack"], r["file"]): r for r in R}
    ref_ok = ref_n = pred_ok = pred_n = 0; mism = []; imgs = set()
    for r in P:
        if r["image_pack"] != r["model_disease"]:
            continue
        q = Rk[(r["image_pack"], r["file"])]
        imgs.add((r["image_pack"], r["file"]))
        ref_n += 1
        lb_leaf = int(q["lb_leaf_px"]) + int(q["lb_lesion_px"])
        if lb_leaf == r["ref_leaf_px"] and int(q["lb_lesion_px"]) == r["ref_lesion_px"]:
            ref_ok += 1
        else:
            mism.append(r["file"])
        chk = {c.split(":")[0]: [int(v) for v in c.split(":")[1:]] for c in q["pred_check"].split(";")}
        pred_n += 1
        if chk[r["arch"]] == [r["pred_leaf_px"], r["pred_lesion_px"], r["inter_lesion"], r["union_lesion"]]:
            pred_ok += 1
    res["integrity"] = dict(reference_rebuilt_matches_per_image=f"{ref_ok}/{ref_n}",
                            predicted_maps_match_per_image=f"{pred_ok}/{pred_n}", mismatches=mism[:20],
                            n_rows=ref_n, n_distinct_images=len(imgs), n_images_rebuilt_all_splits=len(R),
                            note="Own-disease val/test rows of per_image.csv (one row per image and model: "
                         "n_distinct_images images x 3 models = n_rows); reference labels rebuilt from the packs "
                                 "with run_inference.load() rules; predicted counts recomputed from the saved class maps.")
    assert ref_ok == ref_n and pred_ok == pred_n, res["integrity"]

    # ---- leaf tables ----
    LT = {}
    for dis in DIS:
        for arch in ARCHS:
            for split in ("val", "test"):
                rules = ("abaxial", "adaxial", "both") if dis == "roya" else ("both",)
                for rule in rules:
                    LT[(dis, arch, split, rule)] = leaf_table(P, dis, arch, split, rule)
    main_rule = {"roya": "abaxial", "ojo": "both"}
    with open(os.path.join(WORK, "leaf_tables.csv"), "w", newline="") as f:
        allrows = [x for v in LT.values() for x in v]
        wr = csv.DictWriter(f, fieldnames=list(allrows[0].keys())); wr.writeheader(); wr.writerows(allrows)

    # ---- (0) reproduction of published test values ----
    rep = []; all_ok = True
    for dis in DIS:
        for arch in ARCHS:
            m = point_metrics(LT[(dis, arch, "test", main_rule[dis])])
            p = PUB[dis][arch]
            here = dict(iou=round(m["iou"], 3), mae=round(m["mae"], 2), bias=round(m["bias"], 2), qwk=round(m["qwk"], 2),
                        loa=[round(v, 2) for v in m["loa"]])
            ok = (here["iou"] == p["iou"] and here["mae"] == p["mae"] and here["bias"] == p["bias"]
                  and here["qwk"] == p["qwk"] and tuple(here["loa"]) == p["loa"])
            all_ok &= ok
            rep.append(dict(disease=dis, model=ARCH_NAME[arch], n_leaves=m["n"], reproduced=here, published=p, match=ok))
    res["reproduction"] = dict(all_match=bool(all_ok), rows=rep,
                               n_test_diseased_leaves={d: point_metrics(LT[(d, "segformer", "test", main_rule[d])])["n"] for d in DIS})
    assert all_ok, rep
    print("published test values reproduced", flush=True)

    # ---- (1) validation vs test, six models ----
    vt = {}
    for dis in DIS:
        vt[dis] = {}
        idx = {s: boot_idx(TAG[(dis, s)], len(LT[(dis, "segformer", s, main_rule[dis])])) for s in ("val", "test")}
        for arch in ARCHS:
            e = {}; bs = {}
            for s in ("val", "test"):
                L = LT[(dis, arch, s, main_rule[dis])]
                bs[s] = boot_metrics(L, idx[s])
                e[s] = point_metrics(L) | summarize_boot(bs[s])
            d = {}
            for k in ("iou", "mae", "bias", "qwk"):
                diff = bs["test"][k] - bs["val"][k]
                c, nv = ci(diff)
                d[k] = dict(point=e["test"][k] - e["val"][k], ci95=c, n_valid=nv)
            e["test_minus_val"] = d
            if dis in PUB_MAE_CI:
                e["test"]["published_mae_ci95"] = list(PUB_MAE_CI[dis][arch])
            vt[dis][arch] = e
    res["val_vs_test"] = dict(
        note="Diseased leaves only (rust 21 val / 21 test; leaf spot 55 / 55). test_minus_val CIs: independent "
             "leaf resamples of the two sets. Validation is not a fully independent check: it was used for "
             "checkpoint selection (and, for the rust CNNs, the healthy-leaf false-positive penalty).",
        results=vt)

    # ---- (2) grade distributions per split ----
    faces = defaultdict(dict)
    for r in R:
        faces[(r["pack"], r["leaf_id"])][r["face"]] = r
    leaves = []
    for (pack, lid), f in sorted(faces.items()):
        assert set(f) == {"abaxial", "adaxial"}
        sev = {}
        for sp in ("lb", "nat"):
            fs = {fc: 100 * int(f[fc][f"{sp}_lesion_px"]) / (int(f[fc][f"{sp}_leaf_px"]) + int(f[fc][f"{sp}_lesion_px"]))
                  for fc in f}
            sev[sp] = fs["abaxial"] if pack == "roya" else (fs["abaxial"] + fs["adaxial"]) / 2
            sev[sp + "_ad"] = fs["adaxial"]; sev[sp + "_both"] = (fs["abaxial"] + fs["adaxial"]) / 2
        g = f["abaxial"]["grade_label"]
        leaves.append(dict(pack=pack, leaf_id=lid, split=f["abaxial"]["split"], date=lid[:8], label=g,
                           healthy=(g == "healthy"), sev_lb=sev["lb"], sev_nat=sev["nat"],
                           sev_lb_adaxial=sev["lb_ad"], sev_lb_both=sev["lb_both"],
                           g_mask_lb=grade(sev["lb"]), g_mask_nat=grade(sev["nat"]),
                           g_visual=int(g[1]) if pack == "ojo" else None))
    gd = {}
    for pack in DIS:
        gd[pack] = {}
        for split in SPLITS + ("all",):
            Ls = [x for x in leaves if x["pack"] == pack and (split == "all" or x["split"] == split)]
            dz = [x for x in Ls if not x["healthy"]]; hl = [x for x in Ls if x["healthy"]]
            sv = np.array([x["sev_lb"] for x in dz])
            e = dict(n_diseased_leaves=len(dz),
                     mask_grade_counts={f"G{g}": sum(1 for x in dz if x["g_mask_lb"] == g) for g in range(K)},
                     mask_grade_counts_native={f"G{g}": sum(1 for x in dz if x["g_mask_nat"] == g) for g in range(K)},
                     n_grade_changes_native_vs_letterbox=sum(1 for x in dz if x["g_mask_nat"] != x["g_mask_lb"]),
                     max_abs_sev_diff_native_vs_letterbox=float(max(abs(x["sev_nat"] - x["sev_lb"]) for x in dz)),
                     severity_pct=dict(median=float(np.median(sv)), q1=float(np.percentile(sv, 25)),
                                       q3=float(np.percentile(sv, 75)), min=float(sv.min()), max=float(sv.max()),
                                       mean=float(sv.mean())))
            if pack == "roya":
                e["n_healthy_leaves"] = len(hl)
                e["healthy_max_reference_severity_pct"] = float(max(x["sev_lb"] for x in hl)) if hl else None
                e["healthy_mask_grade_counts"] = {f"G{g}": sum(1 for x in hl if x["g_mask_lb"] == g) for g in range(K)}
                e["mask_grade_counts_adaxial"] = {f"G{g}": sum(1 for x in dz if grade(x["sev_lb_adaxial"]) == g) for g in range(K)}
                e["mask_grade_counts_both_faces"] = {f"G{g}": sum(1 for x in dz if grade(x["sev_lb_both"]) == g) for g in range(K)}
            else:
                e["visual_grade_counts"] = {f"G{g}": sum(1 for x in dz if x["g_visual"] == g) for g in range(K)}
                ct = np.zeros((K, K), int)
                for x in dz:
                    ct[x["g_visual"], x["g_mask_lb"]] += 1
                e["crosstab_visual_rows_mask_cols"] = ct.tolist()
                e["visual_vs_mask_exact_agreement"] = float(np.mean([x["g_visual"] == x["g_mask_lb"] for x in dz]))
                e["visual_vs_mask_qwk"] = qwk([x["g_visual"] for x in dz], [x["g_mask_lb"] for x in dz])
                e["mask_below_visual"] = int(sum(1 for x in dz if x["g_mask_lb"] < x["g_visual"]))
                e["mask_above_visual"] = int(sum(1 for x in dz if x["g_mask_lb"] > x["g_visual"]))
            gd[pack][split] = e

    # Monte Carlo permutation chi-square test of grade distribution vs split (diseased leaves).
    def perm_chi2(labels, groups, tag, n_perm=B):
        labels = np.asarray(labels); groups = np.asarray(groups)
        cats = sorted(set(labels)); grs = sorted(set(groups))

        def stat(lab):
            t = np.array([[np.sum((groups == g) & (lab == c)) for c in cats] for g in grs])
            return chi2_contingency(t, correction=False)[0], t
        s0, t0_ = stat(labels)
        rng = np.random.default_rng([SEED, tag])
        ge = sum(stat(rng.permutation(labels))[0] >= s0 - 1e-12 for _ in range(n_perm))
        return dict(chi2=float(s0), table=t0_.tolist(), categories=[f"G{c}" for c in cats], groups=grs,
                    p_perm=(ge + 1) / (n_perm + 1), n_perm=n_perm)
    gd["split_independence_tests"] = {
        "roya_mask_grade": perm_chi2([x["g_mask_lb"] for x in leaves if x["pack"] == "roya" and not x["healthy"]],
                                     [x["split"] for x in leaves if x["pack"] == "roya" and not x["healthy"]], 11),
        "ojo_visual_grade": perm_chi2([x["g_visual"] for x in leaves if x["pack"] == "ojo"],
                                      [x["split"] for x in leaves if x["pack"] == "ojo"], 12),
        "ojo_mask_grade": perm_chi2([x["g_mask_lb"] for x in leaves if x["pack"] == "ojo"],
                                    [x["split"] for x in leaves if x["pack"] == "ojo"], 13),
        "note": "Monte Carlo permutation test of the Pearson chi-square statistic (labels permuted across leaves), "
                "p = (1 + #{chi2* >= chi2}) / (1 + 10,000). Descriptive check that splits share one grade mix.",
    }
    res["grade_distributions"] = dict(
        definitions=dict(severity="letterbox space (as in the evaluation); native-resolution grades as a check",
                         rust_leaf="abaxial face; healthy rust-pack leaves reported separately",
                         leaf_spot_leaf="mean of the two faces; visual grade = annotator's grade (split CSV)"),
        n_leaves_local={p: {s: sum(1 for x in leaves if x["pack"] == p and x["split"] == s) for s in SPLITS} for p in DIS},
        n_rust_diseased_leaves_local=sum(1 for x in leaves if x["pack"] == "roya" and not x["healthy"]),
        results=gd)

    # ---- (3) collection dates per split + date-clustered bootstrap ----
    dates = {}
    for pack in DIS:
        for cls in (("diseased", "healthy") if pack == "roya" else ("diseased",)):
            key = f"{pack}_{cls}"
            dates[key] = {}
            for split in SPLITS:
                c = Counter(x["date"] for x in leaves if x["pack"] == pack and x["split"] == split
                            and x["healthy"] == (cls == "healthy"))
                dates[key][split] = dict(sorted(c.items()))
    # grade composition by date (leaf spot, visual; rust mask-derived)
    comp = {}
    for pack in DIS:
        comp[pack] = {}
        for split in SPLITS:
            for d in sorted({x["date"] for x in leaves if x["pack"] == pack and not x["healthy"]}):
                Ls = [x for x in leaves if x["pack"] == pack and x["split"] == split and x["date"] == d and not x["healthy"]]
                if not Ls:
                    continue
                key = "g_visual" if pack == "ojo" else "g_mask_lb"
                comp[pack][f"{split}_{d}"] = {f"G{g}": sum(1 for x in Ls if x[key] == g) for g in range(K)}

    clus = {"roya": dict(n_dates_test=len({x["date"] for x in LT[("roya", "segformer", "test", "abaxial")]}),
                         dates_test=sorted({x["date"] for x in LT[("roya", "segformer", "test", "abaxial")]}),
                         note="All rust leaves (train, val, test) were collected on a single date (2026-08-19); "
                              "a date-clustered bootstrap is not possible for rust, and between-date variation "
                              "cannot be estimated. Only the leaf bootstrap is reported.")}
    Lref = LT[("ojo", "segformer", "test", "both")]
    dts = sorted({x["date"] for x in Lref})
    by_date = {d: np.array([i for i, x in enumerate(Lref) if x["date"] == d]) for d in dts}
    rng = np.random.default_rng([SEED, TAG[("ojo", "test", "date_cluster")]])
    cl_idx = []
    for _ in range(B):
        ds = rng.integers(0, len(dts), len(dts))
        cl_idx.append(np.concatenate([rng.choice(by_date[dts[j]], len(by_date[dts[j]]), replace=True) for j in ds]))
    rng = np.random.default_rng([SEED, TAG[("ojo", "test", "date_stratified")]])
    st_idx = [np.concatenate([rng.choice(by_date[d], len(by_date[d]), replace=True) for d in dts]) for _ in range(B)]
    leaf_idx = boot_idx(TAG[("ojo", "test")], len(Lref))
    clus["ojo"] = dict(dates_test=dts, n_leaves_per_date={d: int(len(v)) for d, v in by_date.items()},
                       scheme=dict(leaf="resample 55 leaves with replacement (same resamples as the val/test table)",
                                   date_cluster="two-stage: resample the 2 dates with replacement, then resample "
                                                "leaves with replacement within each drawn date (n_d leaves)",
                                   date_stratified="dates fixed; leaves resampled within each date (n_d leaves)"),
                       caveat="Only two collection dates exist for leaf spot; a cluster bootstrap with two clusters "
                              "estimates between-date variance from two values and is not a reliable basis for "
                              "inference. It is reported as a sensitivity illustration only.",
                       models={})
    for arch in ARCHS:
        L = LT[("ojo", arch, "test", "both")]
        assert [x["leaf_id"] for x in L] == [x["leaf_id"] for x in Lref]
        iou = np.array([x["iou"] for x in L]); ae = np.abs(np.array([x["pred_sev"] - x["ref_sev"] for x in L]))
        m = {}
        for name, v in (("iou", iou), ("mae", ae)):
            ci_leaf, _ = ci(v[leaf_idx].mean(1))
            ci_cl, _ = ci([v[i].mean() for i in cl_idx])
            ci_st, _ = ci([v[i].mean() for i in st_idx])
            m[name] = dict(point=float(v.mean()),
                           per_date={d: float(v[by_date[d]].mean()) for d in dts},
                           leaf_ci95=ci_leaf, leaf_ci_width=ci_leaf[1] - ci_leaf[0],
                           date_cluster_ci95=ci_cl, date_cluster_ci_width=ci_cl[1] - ci_cl[0],
                           date_stratified_ci95=ci_st, date_stratified_ci_width=ci_st[1] - ci_st[0],
                           width_ratio_cluster_to_leaf=(ci_cl[1] - ci_cl[0]) / (ci_leaf[1] - ci_leaf[0]))
        clus["ojo"]["models"][arch] = m
    # rust leaf bootstrap widths for completeness
    clus["roya"]["models"] = {}
    for arch in ARCHS:
        e = vt["roya"][arch]["test"]
        clus["roya"]["models"][arch] = dict(iou=dict(point=e["iou"], leaf_ci95=e["iou_ci95"],
                                                     leaf_ci_width=e["iou_ci95"][1] - e["iou_ci95"][0]),
                                            mae=dict(point=e["mae"], leaf_ci95=e["mae_ci95"],
                                                     leaf_ci_width=e["mae_ci95"][1] - e["mae_ci95"][0]))
    res["collection_dates"] = dict(leaves_per_date_per_split=dates, grade_composition_by_split_date=comp,
                                   site_note="Sarchi: rust (2026-08-19). EARTH (Guacimo): American leaf spot "
                                             "(2026-06-21, 2026-06-28) and the healthy rust-pack leaves "
                                             "(2026-06-12, 2026-06-15, 2026-06-22).")
    res["date_clustered_bootstrap"] = clus

    # ---- (4) rust adaxial and both-face results ----
    rf = {}
    for split in ("test", "val"):
        rf[split] = {}
        # Same resamples as the validation/test table (section 1): leaf order is identical (sorted leaf_id) for every
        # face rule, so the abaxial rows here reproduce the section-1 CIs exactly.
        n_split = len(LT[("roya", "segformer", split, "abaxial")])
        assert all([x["leaf_id"] for x in LT[("roya", a, split, r)]] == [x["leaf_id"] for x in LT[("roya", "segformer", split, "abaxial")]]
                   for a in ARCHS for r in ("abaxial", "adaxial", "both"))
        idx = boot_idx(TAG[("roya", split)], n_split)
        for arch in ARCHS:
            rf[split][arch] = {}
            for rule in ("abaxial", "adaxial", "both"):
                L = LT[("roya", arch, split, rule)]
                e = point_metrics(L) | summarize_boot(boot_metrics(L, idx))
                e["qwk_vs_abaxial_reference_grade"] = qwk([x["ref_grade_abaxial"] for x in L], [x["pred_grade"] for x in L])
                e["mean_reference_severity"] = float(np.mean([x["ref_sev"] for x in L]))
                rf[split][arch][rule] = e
    res["rust_faces"] = dict(
        note="Rust diseased leaves (21 per split). For each face rule the reference severity and the reference grade "
             "use the same rule (adaxial: adaxial face; both: mean of the two faces; IoU = mean of the face IoUs). "
             "qwk_vs_abaxial_reference_grade compares the rule's predicted grade with the paper's (abaxial) reference "
             "grade. Abaxial = the paper's primary rule (repeated for comparison). CIs: leaf bootstrap with the same "
             "resamples as val_vs_test (streams roya/test and roya/val), so the abaxial CIs equal those in val_vs_test.",
        results=rf)

    # ---- (5) class pixel frequencies in the training split (letterbox) ----
    cf = {}

    def freq(rows):
        bg = sum(int(r["lb_bg_px"]) for r in rows); lf = sum(int(r["lb_leaf_px"]) for r in rows)
        ls = sum(int(r["lb_lesion_px"]) for r in rows); pad = sum(int(r["lb_pad_px"]) for r in rows)
        tot = bg + lf + ls
        f = np.array([bg, lf, ls]) / tot
        inv = 1 / f
        paper = np.array([0.25, 0.5, 3.0])
        # exponent p for w ~ f^-p that best fits the paper's weights (least squares on log scale, free scale)
        X = np.vstack([np.ones(3), -np.log(f)]).T
        coef = np.linalg.lstsq(X, np.log(paper), rcond=None)[0]
        enet = 1 / np.log(1.02 + f)
        mass_paper = paper * f / (paper * f).sum()
        return dict(n_images=len(rows), pixels=dict(background=bg, leaf=lf, lesion=ls, total=tot,
                                                    padding_part_of_background=pad),
                    frequency=dict(background=f[0], leaf=f[1], lesion=f[2]),
                    frequency_excluding_padding=dict(background=(bg - pad) / (tot - pad), leaf=lf / (tot - pad),
                                                     lesion=ls / (tot - pad)),
                    lesion_share_of_leaf_tissue=ls / (lf + ls),
                    inverse_frequency_weights_scaled_bg_0p25=(0.25 * inv / inv[0]).tolist(),
                    inverse_frequency_weights_balanced_N_over_K_nc=(1 / (3 * f)).tolist(),
                    inverse_frequency_ratios_leaf_to_bg_lesion_to_bg_lesion_to_leaf=[inv[1] / inv[0], inv[2] / inv[0], inv[2] / inv[1]],
                    sqrt_inverse_weights_scaled_bg_0p25=(0.25 * np.sqrt(inv / inv[0])).tolist(),
                    enet_log_weights_scaled_bg_0p25=(0.25 * enet / enet[0]).tolist(),
                    paper_weights=paper.tolist(), paper_ratios=[2.0, 12.0, 6.0],
                    fitted_exponent_p_w_prop_f_minus_p=float(coef[1]),
                    ce_weight_mass_share_under_paper_weights=dict(zip(("background", "leaf", "lesion"), mass_paper.tolist())),
                    ce_weight_mass_share_unweighted=dict(zip(("background", "leaf", "lesion"), f.tolist())))
    for pack in DIS:
        tr = [r for r in R if r["pack"] == pack and r["split"] == "train"]
        cf[pack] = dict(train_all_images=freq(tr))
        if pack == "roya":
            cf[pack]["train_diseased_only"] = freq([r for r in tr if r["grade_label"] != "healthy"])
        cf[pack]["all_splits"] = freq([r for r in R if r["pack"] == pack])
    cf["both_diseases_train"] = dict(train_all_images=freq([r for r in R if r["split"] == "train"]))
    res["class_frequencies"] = dict(
        note="Pixel counts summed over training images after letterboxing to 1024 x 1024 (the training input: "
             "the kernels letterbox labels with NEAREST and pad with background). Flip/rotation augmentation "
             "does not change class counts. Background includes the letterbox padding.",
        results=cf)

    # ---- per-leaf data for figures ----
    res["figure_data"] = dict(
        leaves=[dict(pack=x["pack"], split=x["split"], date=x["date"], healthy=x["healthy"], sev_lb=x["sev_lb"],
                     g_mask=x["g_mask_lb"], g_visual=x["g_visual"]) for x in leaves])
    res["meta"]["runtime_s"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
