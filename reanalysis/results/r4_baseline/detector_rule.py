"""A3 step 1: characterise the colour rule of the rust weak-mask detector that seeded the SAM 2 masks.

The detector source (`roya_lesion.py`, named in notebooks/pg26038-sam2-roya-refine/pg26038-sam2-roya-refine.py and in
revision_brief.md fact 1) is NOT present on this computer (searched by name and content under C:/dev, and by name
under <USER_HOME>/OneDrive, Documents and Desktop). Its OUTPUT is present: datasets/pg26038-sam2-roya/seeds/<stem>.png
= the weak 0/255 masks that SAM 2 was prompted with, aligned with datasets/pg26038-sam2-roya/images/<stem>.png (RGBA,
2048 px high: the "staged scale" of the refine script, twice the resolution of the 1024-px pack images).

This script infers the rule empirically from those masks:
  1. For every leaf pixel (alpha > 0, as in the refine script) it computes candidate colour indices.
  2. Single-index rules: best global threshold (either direction) by pooled pixel IoU with the seed masks, plus ROC AUC.
  3. Two-index AND rules for every pair of indices (quantile-binned 2-D cumulative histograms, 128 bins per index).
  4. Spread of the per-image optimal threshold of the best indices (a fixed global rule gives a tight spread).
  5. Seed-mask morphology (blobs, holes, pixels outside the leaf).
  6. Relatedness of the candidate detector-independent baseline indices (CIELAB a*, CIELAB hue angle h_ab, HSV hue,
     HSV saturation) to the detector, measured on TRAINING-split leaves only (no validation or test leaf is used to
     choose the index): Spearman rho with the best-matching detector index over leaf pixels and the best pooled IoU
     with which a single threshold on the candidate reproduces the seed masks.
Pixel sampling: a fixed uniform 1-in-16 subsample of leaf pixels per image (numpy default_rng(20260928)); pooled
counts stay unbiased. Outputs: detector_rule.json (this folder); feature cache detector_features.npz (deleted at end).
"""
import os as _os  # (release) all paths below are relative to ROOT (repository root by default; see README)
ROOT = (_os.environ.get("PG26038_ROOT") or _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))).replace(chr(92), "/")
import csv
import glob
import json
import os
import time

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.stats import rankdata, spearmanr

BASE = ROOT
OUT = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = f"{BASE}/datasets/pg26038-sam2-roya/images"
SEED_DIR = f"{BASE}/datasets/pg26038-sam2-roya/seeds"
SPLIT = f"{BASE}/datasets/pg26038-roya-real-pack/split.csv"
RNG = np.random.default_rng(20260928)
SUB = 16
NB = 128
CANDIDATES = ("CIELAB a*", "CIELAB hue angle h_ab (deg)", "HSV hue signed (deg)", "HSV saturation")


def feats(rgb):
    x = rgb.astype(np.float64)
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    s = r + g + b
    ss = np.where(s > 0, s, 1.0)
    rc, gc, bc = r / ss, g / ss, b / ss
    mx = np.max(x, 1); mn = np.min(x, 1); d = mx - mn
    nz = d > 0
    rm = nz & (mx == r); gm = nz & (mx == g) & ~rm; bm = nz & ~rm & ~gm
    hs = np.full_like(r, 300.0)                      # signed HSV hue in [-60, 300); grey pixels -> 300 (never lesion)
    dd_ = np.where(nz, d, 1.0)
    hs[rm] = 60 * (g[rm] - b[rm]) / dd_[rm]
    hs[gm] = 120 + 60 * (b[gm] - r[gm]) / dd_[gm]
    hs[bm] = 240 + 60 * (r[bm] - g[bm]) / dd_[bm]
    sat = np.where(mx > 0, d / np.where(mx > 0, mx, 1), 0)

    def lin(c):
        c = c / 255.0
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    R, G, Bl = lin(r), lin(g), lin(b)
    X = (0.4124564 * R + 0.3575761 * G + 0.1804375 * Bl) / 0.95047
    Y = (0.2126729 * R + 0.7151522 * G + 0.0721750 * Bl)
    Z = (0.0193339 * R + 0.1191920 * G + 0.9503041 * Bl) / 1.08883
    e = 6.0 / 29.0
    f = lambda t: np.where(t > e ** 3, np.cbrt(t), t / (3 * e * e) + 4.0 / 29.0)
    L = 116 * f(Y) - 16; A = 500 * (f(X) - f(Y)); Bb = 200 * (f(Y) - f(Z))
    hab = np.degrees(np.arctan2(Bb, A)) % 360
    return {
        "R": r, "G": g, "B": b,
        "excess red 1.4r'-g'": 1.4 * rc - gc,
        "redness index (r-g)/(r+g+b)": np.where(s > 0, (r - g) / ss, 0),
        "r-g": r - g,
        "g/r": g / np.where(r > 0, r, 1),
        "excess green 2g'-r'-b'": 2 * gc - rc - bc,
        "chromatic r'": rc, "chromatic g'": gc, "chromatic b'": bc,
        "HSV hue signed (deg)": hs, "HSV saturation": sat, "HSV value": mx,
        "CIELAB L*": L, "CIELAB a*": A, "CIELAB b*": Bb, "CIELAB hue angle h_ab (deg)": hab,
        "CIELAB chroma": np.hypot(A, Bb),
    }


def best_threshold(v, y):
    """Best pooled IoU over thresholds in both directions (predict v >= t, or v < t); plus ROC AUC (seed = positive)."""
    o = np.argsort(v, kind="stable"); vs = v[o]; ys = y[o]
    P = int(ys.sum()); n = len(v)
    first = np.r_[True, vs[1:] != vs[:-1]]
    idx = np.nonzero(first)[0]
    cpos = np.concatenate([[0], np.cumsum(ys)])
    tp = P - cpos[idx]; pp = n - idx                     # predict v >= vs[idx]
    iou_ge = tp / (P + pp - tp)
    tp2 = P - tp; pp2 = n - pp                           # predict v < vs[idx]
    den2 = P + pp2 - tp2
    iou_lt = np.where(den2 > 0, tp2 / np.where(den2 > 0, den2, 1), 0)
    i1 = int(np.argmax(iou_ge)); i2 = int(np.argmax(iou_lt))
    rk = rankdata(v)
    auc = float((rk[y].sum() - P * (P + 1) / 2) / (P * (n - P)))
    if iou_ge[i1] >= iou_lt[i2]:
        return dict(direction=">=", threshold=float(vs[idx[i1]]), pooled_iou=float(iou_ge[i1]), auc=auc)
    return dict(direction="<", threshold=float(vs[idx[i2]]), pooled_iou=float(iou_lt[i2]), auc=auc)


def qbins(v):
    edges = np.unique(np.quantile(v, np.linspace(0, 1, NB + 1)))
    b = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, len(edges) - 2)
    return b, edges


def pair_best(va, vb, y):
    """Best AND rule over quantile-bin thresholds, all four direction combinations."""
    ba, ea = qbins(va); bb, eb = qbins(vb)
    na, nb = len(ea) - 1, len(eb) - 1
    tot = np.zeros((na, nb)); pos = np.zeros((na, nb))
    np.add.at(tot, (ba, bb), 1); np.add.at(pos, (ba, bb), y)
    P = y.sum(); best = None
    for da in (">=", "<"):
        for db in (">=", "<"):
            T, Q = tot, pos
            if da == "<": T, Q = T[::-1], Q[::-1]
            if db == "<": T, Q = T[:, ::-1], Q[:, ::-1]
            # suffix sums: predicted positive = bins i' >= i and j' >= j (in the flipped frame)
            PP = T[::-1, ::-1].cumsum(0).cumsum(1)[::-1, ::-1]
            TP = Q[::-1, ::-1].cumsum(0).cumsum(1)[::-1, ::-1]
            iou = TP / (P + PP - TP)
            i, j = np.unravel_index(np.argmax(iou), iou.shape)
            ia = i if da == ">=" else na - 1 - i
            jb = j if db == ">=" else nb - 1 - j
            ta = float(ea[ia]) if da == ">=" else float(ea[ia + 1])
            tb = float(eb[jb]) if db == ">=" else float(eb[jb + 1])
            cand = dict(pooled_iou=float(iou[i, j]), dir_a=da, thr_a=ta, dir_b=db, thr_b=tb)
            if best is None or cand["pooled_iou"] > best["pooled_iou"]:
                best = cand
    return best


def main():
    t0 = time.time()
    split_of = {r["file"][:-4]: r["split"] for r in csv.DictReader(open(SPLIT, newline=""))}
    files = sorted(glob.glob(f"{IMG_DIR}/*.png"))
    V = {}; Y = []; IMGID = []; stems = []; blob_stats = []
    for k, f in enumerate(files):
        stem = os.path.splitext(os.path.basename(f))[0]
        sf = f"{SEED_DIR}/{stem}.png"
        if not os.path.exists(sf):
            continue
        rgba = np.asarray(Image.open(f).convert("RGBA"))
        leaf = rgba[..., 3] > 0
        seed_full = np.asarray(Image.open(sf).convert("L")) > 127
        seed = seed_full & leaf
        lab, nb = ndimage.label(seed, structure=np.ones((3, 3)))
        sizes = np.bincount(lab.ravel())[1:] if nb else np.array([], int)
        filled = ndimage.binary_fill_holes(seed)
        blob_stats.append(dict(stem=stem, split=split_of.get(stem, "not in pack"), leaf_px=int(leaf.sum()),
                               seed_px=int(seed.sum()), seed_outside_leaf_px=int((seed_full & ~leaf).sum()),
                               n_blobs=int(nb), min_blob=int(sizes.min()) if nb else 0,
                               n_blobs_lt4=int((sizes < 4).sum()), hole_px=int((filled & ~seed).sum())))
        ii = np.nonzero(leaf.ravel())[0]
        pick = ii[RNG.random(len(ii)) < 1.0 / SUB]
        F = feats(rgba[..., :3].reshape(-1, 3)[pick])
        for kk, vv in F.items():
            V.setdefault(kk, []).append(vv.astype(np.float64))
        Y.append(seed.ravel()[pick]); IMGID.append(np.full(len(pick), len(stems), np.int32)); stems.append(stem)
        if k % 50 == 0:
            print(k, stem, f"{time.time() - t0:.0f}s", flush=True)
    Y = np.concatenate(Y); IMGID = np.concatenate(IMGID)
    V = {k: np.concatenate(v) for k, v in V.items()}
    splits = np.array([split_of.get(s, "not in pack") for s in stems])
    from collections import Counter
    res = dict(meta=dict(script=os.path.abspath(__file__), created=time.strftime("%Y-%m-%d %H:%M"),
                         images=len(stems), images_by_split=dict(Counter(splits.tolist())),
                         sampled_leaf_px=int(len(Y)), sampled_seed_px=int(Y.sum()),
                         subsample=f"uniform 1/{SUB} of leaf pixels (alpha > 0) per image, default_rng(20260928)",
                         source_status="roya_lesion.py source not found on this computer; rule inferred from its output (weak seed masks)"))
    single = {k: best_threshold(v, Y) for k, v in V.items()}
    res["single_index"] = dict(sorted(single.items(), key=lambda kv: -kv[1]["pooled_iou"]))
    for k, v in res["single_index"].items():
        print(f"{k:34s} {v}", flush=True)
    print(f"single done {time.time() - t0:.0f}s", flush=True)
    names = list(V)
    pairs = []
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            pb = pair_best(V[names[a]], V[names[b]], Y)
            pairs.append(dict(a=names[a], b=names[b], **pb))
    res["two_index_and_rules_top15"] = sorted(pairs, key=lambda d: -d["pooled_iou"])[:15]
    print(json.dumps(res["two_index_and_rules_top15"][:6], indent=0), flush=True)
    per_img = {}
    for k in list(res["single_index"])[:4]:
        d = res["single_index"][k]["direction"]; sg = 1 if d == ">=" else -1
        ts, ious = [], []
        for i in range(len(stems)):
            m = IMGID == i
            if Y[m].sum() < 50:
                continue
            bt = best_threshold(sg * V[k][m], Y[m])
            if bt["direction"] != ">=":
                continue
            ts.append(sg * bt["threshold"]); ious.append(bt["pooled_iou"])
        ts = np.array(ts)
        per_img[k] = dict(direction=d, n_images=len(ts), thr_median=float(np.median(ts)),
                          thr_p10=float(np.percentile(ts, 10)), thr_p90=float(np.percentile(ts, 90)),
                          per_image_best_iou_median=float(np.median(ious)))
    res["per_image_optimal_threshold"] = per_img
    top = list(res["single_index"])[0]
    tk = res["single_index"][top]
    pred = V[top] >= tk["threshold"] if tk["direction"] == ">=" else V[top] < tk["threshold"]
    pi = [float((pred[IMGID == i] & Y[IMGID == i]).sum() / (pred[IMGID == i] | Y[IMGID == i]).sum())
          for i in range(len(stems)) if (pred[IMGID == i] | Y[IMGID == i]).any()]
    res["top_single_rule_per_image_iou"] = dict(index=top, n_images=len(pi), median=float(np.median(pi)),
                                                p10=float(np.percentile(pi, 10)), p90=float(np.percentile(pi, 90)))
    # relatedness of candidate baseline indices to the detector, TRAINING leaves only
    tr = np.isin(IMGID, np.nonzero(splits == "train")[0])
    det_idx = [k for k in res["single_index"] if k not in CANDIDATES][:3]
    rel = {}
    sub = np.nonzero(tr)[0][::4]
    for c in CANDIDATES:
        bt = best_threshold(V[c][tr], Y[tr])
        rel[c] = dict(train_seed_reproduction=bt,
                      spearman_with_detector_indices={d: float(spearmanr(V[c][sub], V[d][sub])[0]) for d in det_idx})
    res["candidate_relatedness_train_only"] = dict(
        n_train_images=int((splits == "train").sum()), n_train_px=int(tr.sum()), detector_like_indices=det_idx,
        note="rho computed on every 4th sampled training leaf pixel", candidates=rel)
    bs = blob_stats
    res["seed_morphology"] = dict(
        n_images=len(bs), seed_px_outside_leaf_total=int(sum(b["seed_outside_leaf_px"] for b in bs)),
        blobs_total=int(sum(b["n_blobs"] for b in bs)), blobs_lt4px_total=int(sum(b["n_blobs_lt4"] for b in bs)),
        min_blob_px_overall=int(min((b["min_blob"] for b in bs if b["n_blobs"]), default=0)),
        hole_px_total=int(sum(b["hole_px"] for b in bs)), seed_px_total=int(sum(b["seed_px"] for b in bs)),
        median_seed_frac_pct=float(np.median([100 * b["seed_px"] / b["leaf_px"] for b in bs])),
        images_with_empty_seed=int(sum(b["seed_px"] == 0 for b in bs)))
    res["per_image_blob_stats"] = bs
    json.dump(res, open(f"{OUT}/detector_rule.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_image_blob_stats", "single_index")}, indent=1))
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
