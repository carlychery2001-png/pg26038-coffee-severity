"""A3 (round 4): detector-independent colour baseline with validation-chosen leaf-edge erosion and petiole/cut-end
exclusion, plus the existing baseline (excess red / redness index) rerun with the same erosion (sensitivity).

Evaluation protocol (round4_plan.md, unchanged): leaf = alpha > 8; lesion = mask > 127 AND leaf; aspect-preserving
letterbox to 1024 x 1024 (RGB bilinear, labels nearest, black padding; copy of run_inference.py load()); rust = abaxial
face; American leaf spot = mean of the two faces (IoU = mean of face IoUs, severity = mean of face severities); grades
<0.5, 0.5-<5, 5-<20, >=20 %. Reference grade: rust = mask-derived; leaf spot = mask-derived (PRIMARY, author decision of
28 Sep 2026) and the annotator's visual grade (secondary). Bootstrap: 10,000 resamples of leaves, numpy
default_rng(20260928) re-seeded per disease, one index matrix per disease shared by all methods (paired differences).

Baseline (as the existing one): thresholding inside the REFERENCE leaf mask; the severity denominator is always the full
reference leaf area; no smoothing or speck removal. New elements:
  * colour index: chosen among CIELAB a*, CIELAB hue angle h_ab, HSV hue and HSV saturation as the one least related to
    the detector that seeded the rust masks (relatedness measured on TRAINING leaves only by detector_rule.py; see
    INDEX_CHOICE below and results.md);
  * leaf-edge erosion e in {0, 1, 2, 4, 8} letterbox px: lesion predictions are allowed only on leaf pixels whose taxicab
    distance to the background is > e (identical to scipy binary_erosion with the default 3x3 cross, e iterations;
    checked below), i.e. predictions inside the e-px margin band are removed;
  * petiole / cut-end exclusion, one of: none; 'open r' = leaf pixels removed by a morphological opening of the leaf
    mask with a Euclidean disk of radius r in {4, 8, 12, 16} px (thin structures narrower than 2r: the petiole stalk,
    and also the narrow drip tip); 'ends k' = leaf pixels within k in {32, 64, 128} px (Euclidean) of either end of the
    leaf's major axis (the two leaf pixels with extreme projection on the first principal axis of the leaf-pixel
    coordinates; both ends are excluded because which end is the cut petiole is not recorded);
  * threshold x erosion x exclusion chosen jointly by the validation mean leaf IoU on the validation DISEASED leaves
    (rust 21, leaf spot 55); healthy leaves are never used for the primary selection. Evaluated once on test.
Secondary (labelled): (i) the new index with the threshold chosen on validation subject to 0/21 healthy validation leaves
at >= 0.5 % on either face (same erosion/exclusion); (ii) the existing baseline with its fixed threshold and the new
baseline's validation-chosen erosion (and erosion + exclusion), and with the threshold re-tuned on validation at that
erosion/exclusion.
Outputs: results.json, per_leaf_test.csv, per_leaf_healthy.csv, validation_grid.csv (this folder).
"""
import os as _os  # (release) all paths below are relative to ROOT (repository root by default; see README)
ROOT = (_os.environ.get("PG26038_ROOT") or _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))).replace(chr(92), "/")
import csv
import json
import os
import platform
import time
from collections import defaultdict

import numpy as np
import scipy
from PIL import Image
from scipy import ndimage
from scipy.stats import binomtest

BASE = ROOT
OUT = os.path.dirname(os.path.abspath(__file__))
PER_IMAGE = f"{BASE}/reanalysis/out/per_image.csv"
PACKS = {"roya": (f"{BASE}/datasets/pg26038-roya-real-pack", f"{BASE}/datasets/pg26038-roya-real-pack/split.csv"),
         "ojo": (f"{BASE}/datasets/pg26038-ojo-real-pack", f"{BASE}/datasets/pg26038-ojo-real-pack/meta/split.csv")}
SIZE = 1024
B = 10_000
SEED = 20260928
K = 4
DIS = ("roya", "ojo")
DIS_NAME = {"roya": "Coffee leaf rust", "ojo": "American leaf spot"}
RULE = {"roya": "abaxial", "ojo": "mean"}
ARCHS = ("segformer", "unetpp", "deeplab")
ARCH_NAME = {"segformer": "SegFormer-B5", "unetpp": "UNet++", "deeplab": "DeepLabV3+"}
EROSIONS = (0, 1, 2, 4, 8)
OPEN_R = (4, 8, 12, 16)
END_K = (32, 64, 128)
PETIOLE = [("none", 0)] + [("open", r) for r in OPEN_R] + [("ends", k) for k in END_K]
OPTIONS = [(e, p) for e in EROSIONS for p in range(len(PETIOLE))]

# Candidate indices. score >= thr <=> lesion. 'exact' scores are integers (exact rational comparisons on 8-bit RGB).
#   hue    signed HSV hue h' in [-60, 300) deg; lesion <=> h' <= t (redder/yellower than green); score = -ceil(2 h'),
#          threshold grid t = -60..300 step 0.5 (score >= -2t <=> h' <= t, exact). Grey pixels (max = min): never lesion.
#   hab    signed CIELAB hue angle in (-90, 270] deg; lesion <=> h_ab <= t; score = -h_ab (float64); grid step 0.1
#   astar  CIELAB a*; lesion <=> a* >= t (float64); grid -40..60 step 0.1
#   sat_hi HSV saturation S = (max-min)/max; lesion <=> S >= t; score = floor(1000 S) exact; grid 0..1 step 0.001
#   sat_lo lesion <=> S <= t; score = -ceil(1000 S) exact
#   exr    excess red 1.4r'-g' (existing rust baseline); score = floor(200 (7r-5g)/s) exact; grid step 0.002
#   ri     redness index (r-g)/(r+g+b) (existing leaf-spot baseline); score = floor(1000 (r-g)/s) exact; step 0.001
INDEX_NAME = {"hue": "HSV hue h' (signed, deg; lesion = h' <= t)", "hab": "CIELAB hue angle h_ab (signed, deg; lesion = h_ab <= t)",
              "astar": "CIELAB a* (lesion = a* >= t)", "sat_hi": "HSV saturation (lesion = S >= t)",
              "sat_lo": "HSV saturation (lesion = S <= t)", "exr": "excess red 1.4r'-g' (lesion = exr >= t)",
              "ri": "redness index (r-g)/(r+g+b) (lesion = ri >= t)"}
CAND = ("hue", "hab", "astar", "sat_hi", "sat_lo")
EXISTING = {"roya": ("exr", 0.12), "ojo": ("ri", 0.026)}
# grid in threshold units t, and the map t -> score threshold
GRID_T = {"hue": np.round(np.arange(-60.0, 300.0001, 0.5), 3), "hab": np.round(np.arange(-90.0, 270.0001, 0.1), 3),
          "astar": np.round(np.arange(-40.0, 60.0001, 0.1), 3), "sat_hi": np.round(np.arange(0.0, 1.0000001, 0.001), 4),
          "sat_lo": np.round(np.arange(0.0, 1.0000001, 0.001), 4), "exr": np.round(np.arange(-0.6, 1.2000001, 0.002), 4),
          "ri": np.round(np.arange(-0.3, 0.6000001, 0.001), 4)}


def t_to_score(index, t):
    t = np.asarray(t, np.float64)
    if index == "hue":
        return -np.round(2 * t).astype(np.int64)
    if index == "hab":
        return -t
    if index == "astar":
        return t
    if index == "sat_hi":
        return np.round(1000 * t).astype(np.int64)
    if index == "sat_lo":
        return -np.round(1000 * t).astype(np.int64)
    if index in ("exr", "ri"):
        return np.round(1000 * t).astype(np.int64)
    raise KeyError(index)


# INDEX_CHOICE: CIELAB a*. Fixed before any TEST evaluation of a new index, from detector_rule.json (relatedness to the
# detector's weak masks) and the validation-only run (validation_only.json): HSV saturation is not a lesion index
# (validation IoU <= 0.09 in both directions and both diseases); HSV hue and h_ab reproduce the detector's weak masks
# about as well as excess red (they are hue boundaries like the detector's criterion); a* is the least related usable
# index. See results.md section 2.
INDEX_CHOICE = "astar"


def grade(s):
    return 0 if s < 0.5 else 1 if s < 5 else 2 if s < 20 else 3


def resize_pad(arr, nearest=False):
    im = Image.fromarray(arr); w, h = im.size; s = SIZE / max(w, h)
    im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.NEAREST if nearest else Image.BILINEAR)
    canvas = Image.new(im.mode, (SIZE, SIZE), 0); canvas.paste(im, ((SIZE - im.width) // 2, (SIZE - im.height) // 2))
    return np.asarray(canvas)


def load(pack, file):
    root = PACKS[pack][0]
    rgba = np.asarray(Image.open(f"{root}/images/{file}").convert("RGBA"))
    leaf = rgba[..., 3] > 8
    les = (np.asarray(Image.open(f"{root}/masks/{file}").convert("L")) > 127) & leaf
    lab = np.zeros(leaf.shape, np.uint8); lab[leaf] = 1; lab[les] = 2
    return resize_pad(rgba[..., :3]), resize_pad(lab, nearest=True)


def scores(rgb_leaf):
    """rgb_leaf: (n, 3) uint8 leaf pixels -> dict of score arrays (score >= score-threshold <=> lesion)."""
    x = rgb_leaf.astype(np.int64); r, g, b = x[:, 0], x[:, 1], x[:, 2]
    s = r + g + b; sp = np.where(s > 0, s, 1)
    out = {}
    out["exr"] = np.where(s > 0, (200 * (7 * r - 5 * g)) // sp, 0)
    out["ri"] = np.where(s > 0, (1000 * (r - g)) // sp, 0)
    mx = x.max(1); mn = x.min(1); d = mx - mn; dp = np.where(d > 0, d, 1)
    rm = (d > 0) & (mx == r); gm = (d > 0) & (mx == g) & ~rm; bm = (d > 0) & ~rm & ~gm
    # 2 h' d = 120 * num + 2 * off * d ; ceil(2h') = -((-(120 num + 2 off d)) // d)
    num = np.zeros_like(r); off = np.zeros_like(r)
    num[rm] = (g - b)[rm]; off[rm] = 0
    num[gm] = (b - r)[gm]; off[gm] = 120
    num[bm] = (r - g)[bm]; off[bm] = 240
    X = 120 * num + 2 * off * d
    ceil2h = -((-X) // dp)
    out["hue"] = np.where(d > 0, -ceil2h, -10 ** 6)          # grey: never lesion
    mxp = np.where(mx > 0, mx, 1)
    out["sat_hi"] = np.where(mx > 0, (1000 * d) // mxp, 0)
    out["sat_lo"] = np.where(mx > 0, (-1000 * d) // mxp, 0)  # = -ceil(1000 S)
    f = x.astype(np.float64)

    def lin(c):
        c = c / 255.0
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    R, G, Bl = lin(f[:, 0]), lin(f[:, 1]), lin(f[:, 2])
    Xc = (0.4124564 * R + 0.3575761 * G + 0.1804375 * Bl) / 0.95047
    Yc = (0.2126729 * R + 0.7151522 * G + 0.0721750 * Bl)
    Zc = (0.0193339 * R + 0.1191920 * G + 0.9503041 * Bl) / 1.08883
    e = 6.0 / 29.0
    fn = lambda t: np.where(t > e ** 3, np.cbrt(t), t / (3 * e * e) + 4.0 / 29.0)
    A = 500 * (fn(Xc) - fn(Yc)); Bb = 200 * (fn(Yc) - fn(Zc))
    h = np.degrees(np.arctan2(Bb, A))                       # (-180, 180]
    h = np.where(h <= -90, h + 360, h)                       # signed in (-90, 270]
    out["astar"] = A
    out["hab"] = -h
    for k in ("exr", "ri", "hue", "sat_hi", "sat_lo"):
        out[k] = out[k].astype(np.int32)
    return out


def geometry(leaf_full):
    """Per-leaf-pixel geometry: taxicab distance to background, opening-removal bits, distance to axis ends.
    Computed on the leaf bounding box padded with PADG background pixels (exact: every distance involved is <= 16 px
    beyond the leaf, and everything outside the box is background); outputs are in leaf-pixel order of leaf_full."""
    PADG = 24
    ys0, xs0 = np.nonzero(leaf_full)
    leaf = np.pad(leaf_full[ys0.min():ys0.max() + 1, xs0.min():xs0.max() + 1], PADG)
    cdt = ndimage.distance_transform_cdt(leaf, metric="taxicab")
    edt = ndimage.distance_transform_edt(leaf)
    bits = np.zeros(leaf.shape, np.uint8)
    for i, r in enumerate(OPEN_R):
        er = edt > r
        opened = (ndimage.distance_transform_edt(~er) <= r) & leaf if er.any() else np.zeros_like(leaf)
        bits |= ((leaf & ~opened).astype(np.uint8) << i)
    ys, xs = np.nonzero(leaf)
    ys, xs = ys + ys0.min() - PADG, xs + xs0.min() - PADG      # full-canvas coordinates (same order as leaf_full)
    assert np.array_equal(ys, ys0) and np.array_equal(xs, xs0)
    c = np.stack([ys, xs], 1).astype(np.float64); c0 = c - c.mean(0)
    w, v = np.linalg.eigh(c0.T @ c0); ax = v[:, np.argmax(w)]
    pr = c0 @ ax
    e1, e2 = c[np.argmin(pr)], c[np.argmax(pr)]
    dend = np.minimum(np.hypot(ys - e1[0], xs - e1[1]), np.hypot(ys - e2[0], xs - e2[1]))
    return cdt[leaf].astype(np.int16), bits[leaf], dend.astype(np.float32), (e1.tolist(), e2.tolist())


def eligible(rec, e, p):
    m = rec["cdt"] > e
    kind, val = PETIOLE[p]
    if kind == "open":
        m = m & ((rec["obits"] >> OPEN_R.index(val)) & 1 == 0)
    elif kind == "ends":
        m = m & (rec["dend"] > val)
    return m


# ---------------------------------------------------------------------------------------------------------------- stats
def qwk(a, b):
    O = np.zeros((K, K))
    for x, y in zip(a, b): O[x, y] += 1
    Ex = np.outer(O.sum(1), O.sum(0)) / O.sum()
    W = (np.subtract.outer(np.arange(K), np.arange(K)) ** 2) / (K - 1) ** 2
    den = (W * Ex).sum()
    return float("nan") if den == 0 else float(1 - (W * O).sum() / den)


def qwk_boot(a, b, IDX):
    Bn, n = IDX.shape
    O = np.zeros((Bn, K, K))
    np.add.at(O, (np.repeat(np.arange(Bn), n), a[IDX].ravel(), b[IDX].ravel()), 1)
    Ex = O.sum(2)[:, :, None] * O.sum(1)[:, None, :] / n
    W = (np.subtract.outer(np.arange(K), np.arange(K)) ** 2) / (K - 1) ** 2
    num = (W * O).sum((1, 2)); den = (W * Ex).sum((1, 2))
    with np.errstate(all="ignore"):
        return np.where(den > 0, 1 - num / den, np.nan)


def pct_ci(x):
    x = np.asarray(x, np.float64); ok = np.isfinite(x)
    lo, hi = np.percentile(x[ok], [2.5, 97.5])
    return [float(lo), float(hi)], int((~ok).sum())


def wilson(k, n):
    ci = binomtest(k, n).proportion_ci(0.95, method="wilson")
    return [float(ci.low), float(ci.high)]


def metrics(m, IDX, ref_grade_key):
    iou, rs, ps = m["iou"], m["ref_sev"], m["pred_sev"]
    rg = m[ref_grade_key]; pg = np.array([grade(p) for p in ps])
    d = ps - rs
    pt = dict(n=int(len(iou)), iou=float(iou.mean()), mae=float(np.abs(d).mean()), bias=float(d.mean()),
              grade_acc=float((rg == pg).mean()), grade_correct=int((rg == pg).sum()), qwk=qwk(rg, pg))
    D = ps[IDX] - rs[IDX]
    bt = dict(iou=iou[IDX].mean(1), mae=np.abs(D).mean(1), bias=D.mean(1), grade_acc=(rg[IDX] == pg[IDX]).mean(1),
              qwk=qwk_boot(rg, pg, IDX))
    ci = {}; dropped = {}
    for k, v in bt.items():
        ci[k], dropped[k] = pct_ci(v)
    pt["ci95"] = ci
    pt["boot_dropped"] = {k: v for k, v in dropped.items() if v}
    pt["grade_acc_wilson95"] = wilson(pt["grade_correct"], pt["n"])
    cm = np.zeros((K, K), int)
    for x, y in zip(rg, pg): cm[x, y] += 1
    pt["confusion_rows_ref_cols_pred_G0_G3"] = cm.tolist()
    return pt, bt


# ---------------------------------------------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    PER = list(csv.DictReader(open(PER_IMAGE, newline="")))
    ref_counts = {(r["image_pack"], r["file"]): (int(r["ref_leaf_px"]), int(r["ref_lesion_px"]))
                  for r in PER if r["arch"] == "segformer" and r["model_disease"] == r["image_pack"]}
    IMG = {}; by_leaf = defaultdict(dict); leaf_meta = {}
    n_cdt_checked = 0
    for pack in DIS:
        for r in csv.DictReader(open(PACKS[pack][1], newline="")):
            if r["split"] not in ("val", "test"):
                continue
            rgb, lab = load(pack, r["file"])
            leaf = lab >= 1; les = lab == 2
            assert (int(leaf.sum()), int(les.sum())) == ref_counts[(pack, r["file"])], r["file"]
            cdt, obits, dend, ends = geometry(leaf)
            if n_cdt_checked < 12:        # erosion definition check against scipy binary_erosion (3x3 cross)
                for e in EROSIONS[1:]:
                    ref_er = ndimage.binary_erosion(leaf, iterations=e)
                    assert np.array_equal(ref_er[leaf], cdt > e), (r["file"], e)
                n_cdt_checked += 1
            face = "abaxial" if "abaxial" in r["file"] else "adaxial"
            rec = dict(pack=pack, split=r["split"], leaf_id=r["leaf_id"], face=face, grade_label=r["grade"],
                       n_leaf=int(leaf.sum()), n_les=int(les.sum()), les=les[leaf], cdt=cdt, obits=obits, dend=dend,
                       ends=ends, **scores(rgb[leaf]))
            IMG[(pack, r["file"])] = rec
            by_leaf[(pack, r["split"], r["leaf_id"])][face] = (pack, r["file"])
            leaf_meta[(pack, r["split"], r["leaf_id"])] = r["grade"]
    print(f"loaded {len(IMG)} images; reference counts match per_image.csv; erosion check on {n_cdt_checked} images "
          f"({time.time() - t0:.0f}s)", flush=True)

    def lids_of(pack, split, healthy):
        return sorted(l for (p, s, l), g in leaf_meta.items() if p == pack and s == split and ((g == "healthy") == healthy))

    DIS_IDS = {(d, s): lids_of(d, s, False) for d in DIS for s in ("val", "test")}
    H_IDS = {s: lids_of("roya", s, True) for s in ("val", "test")}
    assert len(DIS_IDS[("roya", "test")]) == 21 and len(DIS_IDS[("ojo", "test")]) == 55
    assert len(DIS_IDS[("roya", "val")]) == 21 and len(DIS_IDS[("ojo", "val")]) == 55
    assert len(H_IDS["test"]) == 21 and len(H_IDS["val"]) == 21

    def faces_of(pack, split, lid, rule):
        f = by_leaf[(pack, split, lid)]
        return [f["abaxial"]] if rule == "abaxial" else [f["abaxial"], f["adaxial"]]

    # ---------------- validation grid: index x erosion x exclusion x threshold ----------------
    def val_curve(dis, index, e, p):
        grid_s = t_to_score(index, GRID_T[index])
        lids = DIS_IDS[(dis, "val")]
        L = np.zeros((len(lids), len(grid_s)))
        for i, lid in enumerate(lids):
            fi = []
            for key in faces_of(dis, "val", lid, RULE[dis]):
                rec = IMG[key]; el = eligible(rec, e, p); v = rec[index][el]; ls = rec["les"][el]
                sl = np.sort(v[ls]); sn = np.sort(v[~ls])
                tp = len(sl) - np.searchsorted(sl, grid_s, "left")
                fp = len(sn) - np.searchsorted(sn, grid_s, "left")
                fi.append(tp / (rec["n_les"] + fp))
            L[i] = np.mean(fi, axis=0)
        return L.mean(0)

    def healthy_val_flags(dis, index, e, p):
        """Healthy validation leaves (rust pack) with >= 0.5 % on either face, at every grid threshold."""
        grid_s = t_to_score(index, GRID_T[index])
        H = np.zeros((len(H_IDS["val"]), len(grid_s)))
        for i, lid in enumerate(H_IDS["val"]):
            fs = []
            for key in faces_of("roya", "val", lid, "mean"):
                rec = IMG[key]; v = np.sort(rec[index][eligible(rec, e, p)])
                fs.append(100.0 * (len(v) - np.searchsorted(v, grid_s, "left")) / rec["n_leaf"])
            H[i] = np.maximum(*fs)
        return (H >= 0.5).sum(0)

    res = dict(meta=dict(
        script=os.path.abspath(__file__), created=time.strftime("%Y-%m-%d %H:%M"), python=platform.python_version(),
        numpy=np.__version__, scipy=scipy.__version__, bootstrap=f"{B} resamples of leaves, percentile 95% CI, default_rng({SEED}) re-seeded per disease, one index matrix per disease shared by all methods",
        evaluation_space="1024 x 1024 aspect-preserving letterbox (run_inference.py load()); leaf = alpha > 8; lesion = mask > 127 AND leaf",
        leaf_rules="rust: abaxial face; leaf spot: mean of the two faces", severity_denominator="full reference leaf area (unchanged by erosion/exclusion)",
        erosion="lesion predictions allowed only where taxicab distance to background > e (= scipy binary_erosion, 3x3 cross, e iterations; verified on 12 images)",
        petiole_options=[f"{k} {v}" if k != "none" else "none" for k, v in PETIOLE], erosions=list(EROSIONS),
        tuning_set="validation diseased leaves only (rust 21, leaf spot 55); selection = max validation mean leaf IoU over threshold x erosion x exclusion; ties -> smaller erosion, then earlier exclusion option",
        healthy_coverage="21 healthy rust-pack TEST leaves present locally (EARTH); the other 33 of the paper's 54 were not retained. 21 healthy VALIDATION leaves (EARTH) reported as secondary.",
        index_choice=INDEX_CHOICE, index_names=INDEX_NAME, grids={k: [float(v[0]), float(v[-1]), float(v[1] - v[0])] for k, v in GRID_T.items()}))

    grid_rows = []
    val = {}
    for dis in DIS:
        val[dis] = {}
        for index in CAND + (EXISTING[dis][0],):
            best = None; table = {}
            for (e, p) in OPTIONS:
                c = val_curve(dis, index, e, p)
                j = int(np.argmax(c))
                table[(e, p)] = (float(c[j]), float(GRID_T[index][j]), c)
                grid_rows.append([dis, index, e, "none" if PETIOLE[p][0] == "none" else f"{PETIOLE[p][0]} {PETIOLE[p][1]}",
                                  f"{GRID_T[index][j]:g}", f"{c[j]:.6f}"])
                if best is None or c[j] > best[0] + 1e-12:
                    best = (float(c[j]), e, p, float(GRID_T[index][j]))
            val[dis][index] = dict(best=best, table=table)
            print(dis, index, "best val IoU %.4f e=%d %s t=%g" % (best[0], best[1], PETIOLE[best[2]], best[3]),
                  f"({time.time() - t0:.0f}s)", flush=True)
    with open(f"{OUT}/validation_grid.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["disease", "index", "erosion_px", "petiole_exclusion", "best_threshold", "val_mean_leaf_iou"])
        w.writerows(grid_rows)

    # ---------------- evaluation helpers ----------------
    def face_pred(rec, index, t, e, p):
        return (rec[index] >= t_to_score(index, t)) & eligible(rec, e, p)

    def diseased_eval(dis, split, index, t, e, p):
        out = {}
        for lid in DIS_IDS[(dis, split)]:
            fs = []
            for key in faces_of(dis, split, lid, RULE[dis]):
                rec = IMG[key]; pr = face_pred(rec, index, t, e, p)
                inter = int((pr & rec["les"]).sum()); union = int((pr | rec["les"]).sum())
                fs.append((inter / union, 100.0 * pr.sum() / rec["n_leaf"], 100.0 * rec["n_les"] / rec["n_leaf"]))
            gl = leaf_meta[(dis, split, lid)]
            out[lid] = dict(iou=float(np.mean([f[0] for f in fs])), pred_sev=float(np.mean([f[1] for f in fs])),
                            ref_sev=float(np.mean([f[2] for f in fs])), visual_grade=int(gl[1]) if dis == "ojo" else None)
        return out

    def healthy_eval(split, index, t, e, p):
        out = {}
        for lid in H_IDS[split]:
            v = {}
            for key in faces_of("roya", split, lid, "mean"):
                rec = IMG[key]; pr = face_pred(rec, index, t, e, p)
                v[rec["face"]] = (100.0 * pr.sum() / rec["n_leaf"], int(pr.sum()))
            out[lid] = dict(sev_abaxial=v["abaxial"][0], sev_adaxial=v["adaxial"][0],
                            px_abaxial=v["abaxial"][1], px_adaxial=v["adaxial"][1])
        return out

    def healthy_summary(hv, lids):
        ab = np.array([hv[l]["sev_abaxial"] for l in lids]); ad = np.array([hv[l]["sev_adaxial"] for l in lids])
        pab = np.array([hv[l]["px_abaxial"] for l in lids]); pad = np.array([hv[l]["px_adaxial"] for l in lids])
        either = np.maximum(ab, ad); mean = (ab + ad) / 2
        n = len(lids); o = {}
        for name, sev, anyp in (("either_face", either, (pab > 0) | (pad > 0)), ("abaxial", ab, pab > 0),
                                ("mean_of_faces", mean, (pab > 0) | (pad > 0))):
            k0 = int(anyp.sum()); k5 = int((sev >= 0.5).sum())
            o[name] = dict(n=n, any_pixel=k0, any_pixel_wilson95=wilson(k0, n), ge_0_5pct=k5, ge_0_5pct_wilson95=wilson(k5, n),
                           ge_5pct=int((sev >= 5).sum()), sev_median=float(np.median(sev)), sev_max=float(sev.max()))
        return o

    def to_arr(ev, lids):
        rs = np.array([ev[l]["ref_sev"] for l in lids])
        m = dict(iou=np.array([ev[l]["iou"] for l in lids]), ref_sev=rs, pred_sev=np.array([ev[l]["pred_sev"] for l in lids]),
                 mask_grade=np.array([grade(x) for x in rs]))
        if ev[lids[0]].get("visual_grade") is not None:
            m["visual_grade"] = np.array([ev[l]["visual_grade"] for l in lids])
        return m

    # networks from per_image.csv (context and paired comparisons)
    def net_eval(dis, arch, split):
        t = defaultdict(dict)
        for r in PER:
            if (r["model_disease"], r["arch"], r["image_pack"], r["split"]) == (dis, arch, dis, split):
                t[r["leaf_id"]][r["face"]] = r
        out = {}
        for lid in DIS_IDS[(dis, split)]:
            use = [t[lid]["abaxial"]] if RULE[dis] == "abaxial" else [t[lid]["abaxial"], t[lid]["adaxial"]]
            out[lid] = dict(iou=float(np.mean([float(u["lesion_iou"]) for u in use])),
                            pred_sev=float(np.mean([float(u["pred_sev"]) for u in use])),
                            ref_sev=float(np.mean([float(u["ref_sev"]) for u in use])),
                            visual_grade=int(use[0]["grade_label"][1]) if dis == "ojo" else None)
        return out

    def net_healthy(dis, arch, split):
        t = defaultdict(dict)
        for r in PER:
            if (r["model_disease"], r["arch"], r["image_pack"], r["split"]) == (dis, arch, "roya", split):
                t[r["leaf_id"]][r["face"]] = r
        if not t:
            return None
        return {lid: dict(sev_abaxial=float(t[lid]["abaxial"]["pred_sev"]), sev_adaxial=float(t[lid]["adaxial"]["pred_sev"]),
                          px_abaxial=int(t[lid]["abaxial"]["pred_lesion_px"]), px_adaxial=int(t[lid]["adaxial"]["pred_lesion_px"]))
                for lid in H_IDS[split]}

    # ---------------- reproduction checks of the existing baseline (results/baseline) ----------------
    rep = {}
    for dis, (idx_, t_) in EXISTING.items():
        ev = diseased_eval(dis, "test", idx_, t_, 0, 0); m = to_arr(ev, DIS_IDS[(dis, "test")])
        gk = "mask_grade" if dis == "roya" else "visual_grade"
        pg = np.array([grade(x) for x in m["pred_sev"]])
        rep[dis] = dict(config=f"{idx_} >= {t_}, no erosion", iou=float(m["iou"].mean()),
                        mae=float(np.abs(m["pred_sev"] - m["ref_sev"]).mean()), bias=float((m["pred_sev"] - m["ref_sev"]).mean()),
                        qwk_published_reference=qwk(m[gk], pg))
        hs = healthy_summary(healthy_eval("test", idx_, t_, 0, 0), H_IDS["test"])
        rep[dis]["healthy_test_ge_0_5_either"] = hs["either_face"]["ge_0_5pct"]; rep[dis]["healthy_test_ge_0_5_abaxial"] = hs["abaxial"]["ge_0_5pct"]
    # results/baseline section 5b rows (3 px and 12 px bands, exr >= 0.116)
    for e_ in (3, 12):
        lids = DIS_IDS[("roya", "test")]; out = {}
        for lid in lids:
            rec = IMG[faces_of("roya", "test", lid, "abaxial")[0]]
            pr = (rec["exr"] >= 116) & (rec["cdt"] > e_)
            out[lid] = (int((pr & rec["les"]).sum()) / int((pr | rec["les"]).sum()), 100.0 * pr.sum() / rec["n_leaf"],
                        100.0 * rec["n_les"] / rec["n_leaf"])
        hflag = 0
        for lid in H_IDS["test"]:
            sv = [100.0 * ((IMG[k]["exr"] >= 116) & (IMG[k]["cdt"] > e_)).sum() / IMG[k]["n_leaf"] for k in faces_of("roya", "test", lid, "mean")]
            hflag += max(sv) >= 0.5
        rep[f"roya_band{e_}_exr0.116"] = dict(iou=float(np.mean([v[0] for v in out.values()])),
                                              bias=float(np.mean([v[1] - v[2] for v in out.values()])), healthy_either_ge_0_5=int(hflag))
    expected = {"roya": (0.600, 1.02, 0.17, 0.93, 21, 20), "ojo": (0.807, 0.23, -0.11, 0.85, 0, 0)}
    ok = True
    for dis, (i_, m_, b_, q_, h1, h2) in expected.items():
        r_ = rep[dis]
        ok &= (round(r_["iou"], 3) == i_ and round(r_["mae"], 2) == m_ and round(r_["bias"], 2) == b_ and
               round(r_["qwk_published_reference"], 2) == q_ and r_["healthy_test_ge_0_5_either"] == h1 and r_["healthy_test_ge_0_5_abaxial"] == h2)
    ok &= round(rep["roya_band3_exr0.116"]["iou"], 3) == 0.633 and rep["roya_band3_exr0.116"]["healthy_either_ge_0_5"] == 15
    ok &= round(rep["roya_band12_exr0.116"]["iou"], 3) == 0.547 and rep["roya_band12_exr0.116"]["healthy_either_ge_0_5"] == 11
    rep["all_match_results_baseline"] = bool(ok)
    res["reproduction_of_results_baseline"] = rep
    print("reproduction of results/baseline:", ok, json.dumps(rep), flush=True)
    if not ok:
        json.dump(res, open(f"{OUT}/results.json", "w"), indent=1)
        raise SystemExit("existing baseline not reproduced; stopping")

    # ---------------- validation summary ----------------
    res["validation"] = {}
    for dis in DIS:
        blk = {}
        for index, vv in val[dis].items():
            bi, be, bp, bt = vv["best"]
            near = sorted([(k, v[0], v[1]) for k, v in vv["table"].items() if v[0] >= bi - 0.005], key=lambda z: -z[1])
            blk[index] = dict(index_name=INDEX_NAME[index], best_val_iou=bi, erosion=be, petiole=PETIOLE[bp][0] if PETIOLE[bp][0] == "none" else f"{PETIOLE[bp][0]} {PETIOLE[bp][1]}",
                              threshold=bt, val_iou_no_erosion_no_exclusion=vv["table"][(0, 0)][0],
                              threshold_no_erosion_no_exclusion=vv["table"][(0, 0)][1],
                              val_iou_by_erosion_no_exclusion={str(e): vv["table"][(e, 0)][0] for e in EROSIONS},
                              options_within_0_005=[dict(erosion=k[0], petiole="none" if PETIOLE[k[1]][0] == "none" else f"{PETIOLE[k[1]][0]} {PETIOLE[k[1]][1]}",
                                                         val_iou=a, threshold=t) for k, a, t in near[:12]],
                              n_options_within_0_005=len(near))
        res["validation"][dis] = blk
    if os.environ.get("R4_VALONLY"):
        json.dump(res, open(f"{OUT}/validation_only.json", "w"), indent=1)
        print("validation only: stopping before any test evaluation of the new index", flush=True)
        return

    # ---------------- test: chosen index (once), secondary variants, existing baseline sensitivity ----------------
    res["test"] = {}; res["healthy"] = {}
    per_leaf_rows = []; healthy_rows = []
    for dis in DIS:
        lids = DIS_IDS[(dis, "test")]
        rng = np.random.default_rng(SEED)
        IDX = rng.integers(0, len(lids), size=(B, len(lids)))
        _, e_c, p_c, t_c = val[dis][INDEX_CHOICE]["best"]
        # secondary (i): specificity-constrained threshold for the chosen index at the chosen erosion/exclusion
        c = val[dis][INDEX_CHOICE]["table"][(e_c, p_c)][2]
        hf = healthy_val_flags(dis, INDEX_CHOICE, e_c, p_c)
        okc = np.where(hf == 0, c, -np.inf); jc = int(np.argmax(okc))
        t_cons = float(GRID_T[INDEX_CHOICE][jc]); v_cons = float(c[jc])
        j_best = int(np.argmax(c))
        # existing index: fixed threshold with the chosen erosion; + exclusion; re-tuned threshold at chosen e/p
        ex_idx, ex_t = EXISTING[dis]
        ce = val_curve(dis, ex_idx, e_c, 0); cep = val[dis][ex_idx]["table"][(e_c, p_c)][2]
        t_ex_e = float(GRID_T[ex_idx][int(np.argmax(ce))]); t_ex_ep = float(GRID_T[ex_idx][int(np.argmax(cep))])
        j_fixed = int(np.argmin(np.abs(GRID_T[ex_idx] - ex_t)))
        configs = {
            "new_primary": (INDEX_CHOICE, t_c, e_c, p_c, "PRIMARY: new detector-independent index, threshold + erosion + exclusion chosen on validation IoU"),
            "new_no_erosion_no_exclusion": (INDEX_CHOICE, float(val[dis][INDEX_CHOICE]["table"][(0, 0)][1]), 0, 0,
                                            "secondary: new index, threshold chosen on validation with e = 0 and no exclusion"),
            "new_specificity_constrained": (INDEX_CHOICE, t_cons, e_c, p_c, "secondary: new index at the chosen erosion/exclusion, threshold = best validation IoU with 0/21 healthy validation leaves >= 0.5 % on either face"),
            "existing_e0": (ex_idx, ex_t, 0, 0, "existing baseline as published in results/baseline (no erosion)"),
            "existing_fixed_thr_chosen_erosion": (ex_idx, ex_t, e_c, 0, "sensitivity: existing threshold, new baseline's validation-chosen erosion"),
            "existing_fixed_thr_chosen_erosion_exclusion": (ex_idx, ex_t, e_c, p_c, "sensitivity: existing threshold, chosen erosion + chosen exclusion"),
            "existing_retuned_chosen_erosion": (ex_idx, t_ex_e, e_c, 0, "sensitivity: existing index, threshold re-tuned on validation at the chosen erosion"),
            "existing_retuned_chosen_erosion_exclusion": (ex_idx, t_ex_ep, e_c, p_c, "sensitivity: existing index, threshold re-tuned on validation at the chosen erosion + exclusion"),
        }
        methods = {}; cfg_out = {}
        for name, (ix, t, e, p, desc) in configs.items():
            ev = diseased_eval(dis, "test", ix, t, e, p)
            methods[name] = to_arr(ev, lids)
            vc = val_curve(dis, ix, e, p)
            jv = int(np.argmin(np.abs(GRID_T[ix] - t)))
            hvv = healthy_summary(healthy_eval("val", ix, t, e, p), H_IDS["val"])
            cfg_out[name] = dict(description=desc, index=ix, index_name=INDEX_NAME[ix], threshold=t, erosion=e,
                                 petiole="none" if PETIOLE[p][0] == "none" else f"{PETIOLE[p][0]} {PETIOLE[p][1]}",
                                 val_iou=float(vc[jv]), healthy_val=hvv)
        for a in ARCHS:
            methods[a] = to_arr(net_eval(dis, a, "test"), lids)
        gkeys = ["mask_grade"] + (["visual_grade"] if dis == "ojo" else [])
        blk = dict(n=len(lids), configs=cfg_out, methods={}, paired_network_minus_new_primary={},
                   paired_new_primary_minus_existing={})
        boots = {}
        for name, m in methods.items():
            blk["methods"][name] = {}
            for gk in gkeys:
                pt, bt = metrics(m, IDX, gk)
                blk["methods"][name][gk] = pt
                if gk == "mask_grade":
                    boots[name] = bt
        for a in ARCHS:
            dd = {}
            for s in ("iou", "mae"):
                cc, nd = pct_ci(boots[a][s] - boots["new_primary"][s])
                pa = blk["methods"][a]["mask_grade"][s]; pb = blk["methods"]["new_primary"]["mask_grade"][s]
                dd[s] = dict(diff=float(pa - pb), ci95=cc, ci_excludes_zero=bool(cc[0] > 0 or cc[1] < 0))
            dd["leaves_network_iou_higher"] = int((methods[a]["iou"] > methods["new_primary"]["iou"]).sum())
            blk["paired_network_minus_new_primary"][a] = dd
        for ref in ("existing_e0", "existing_fixed_thr_chosen_erosion"):
            dd = {}
            for s in ("iou", "mae"):
                cc, nd = pct_ci(boots["new_primary"][s] - boots[ref][s])
                dd[s] = dict(diff=float(blk["methods"]["new_primary"]["mask_grade"][s] - blk["methods"][ref]["mask_grade"][s]),
                             ci95=cc, ci_excludes_zero=bool(cc[0] > 0 or cc[1] < 0))
            blk["paired_new_primary_minus_existing"][ref] = dd
        res["test"][dis] = blk
        # healthy test leaves (21) for every configuration and the networks
        hb = {}
        for name, (ix, t, e, p, desc) in configs.items():
            hv = healthy_eval("test", ix, t, e, p)
            hb[name] = healthy_summary(hv, H_IDS["test"])
            if name in ("new_primary", "existing_e0", "existing_fixed_thr_chosen_erosion"):
                for lid in H_IDS["test"]:
                    healthy_rows.append(dict(disease=dis, config=name, leaf_id=lid, **{k: round(v, 6) if isinstance(v, float) else v for k, v in hv[lid].items()}))
        for a in ARCHS:
            nh = net_healthy(dis, a, "test")
            if nh is not None:
                hb[a] = healthy_summary(nh, H_IDS["test"])
        # decomposition: of the healthy-test flagged pixels of the chosen index/threshold without erosion/exclusion,
        # the share removed by the chosen erosion and by the chosen exclusion
        tot = rem_e = rem_p = 0
        for lid in H_IDS["test"]:
            for key in faces_of("roya", "test", lid, "mean"):
                rec = IMG[key]; base = rec[INDEX_CHOICE] >= t_to_score(INDEX_CHOICE, t_c)
                el_e = rec["cdt"] > e_c; el_ep = eligible(rec, e_c, p_c)
                tot += int(base.sum()); rem_e += int((base & ~el_e).sum()); rem_p += int((base & el_e & ~el_ep).sum())
        hb["decomposition_new_index_chosen_threshold"] = dict(
            flagged_px_both_faces_no_erosion=tot, removed_by_erosion=rem_e, removed_by_exclusion_after_erosion=rem_p,
            remaining=tot - rem_e - rem_p, note="healthy test leaves, both faces, chosen index and threshold")
        res["healthy"][dis] = dict(test=hb)
        for i, lid in enumerate(lids):
            row = dict(disease=dis, leaf_id=lid, ref_sev=round(methods["new_primary"]["ref_sev"][i], 6),
                       mask_grade=int(methods["new_primary"]["mask_grade"][i]))
            if dis == "ojo":
                row["visual_grade"] = int(methods["new_primary"]["visual_grade"][i])
            for name in ("new_primary", "existing_e0", "existing_fixed_thr_chosen_erosion", "segformer", "unetpp", "deeplab"):
                row[f"{name}_iou"] = round(float(methods[name]["iou"][i]), 6)
                row[f"{name}_sev"] = round(float(methods[name]["pred_sev"][i]), 6)
            per_leaf_rows.append(row)
        print(dis, "test done", f"({time.time() - t0:.0f}s)", flush=True)

    for fn, rows in (("per_leaf_test.csv", per_leaf_rows), ("per_leaf_healthy.csv", healthy_rows)):
        keys = []
        for r in rows:
            for k in r:
                if k not in keys: keys.append(k)
        with open(f"{OUT}/{fn}", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    json.dump(res, open(f"{OUT}/results.json", "w"), indent=1)
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
