"""Training-free color-threshold baseline for the PG-26038 major revision (reviewers: "no classical baseline").

What this script does
  0. Reproduces the published network values (leaf IoU, MAE, bias, QWK, Bland-Altman limits) from
     reanalysis/out/per_image.csv and stops if any value does not match at the published precision. Also checks the
     healthy-leaf counts of revision_brief.md item 10 (rust networks, 21 local healthy test leaves, maximum over both
     faces: any-pixel 19/21, 19/21, 13/21; maximum severity 0.38 / 0.27 / 0.09 %).
  1. Rebuilds every validation and test image of both packs in the evaluation space EXACTLY as
     reanalysis/run_inference.py load() does (leaf = alpha > 8, lesion = mask > 127 & leaf, aspect-preserving
     letterbox to 1024 x 1024, black padding, BILINEAR for RGB, NEAREST for labels) and checks the rebuilt
     reference leaf/lesion pixel counts against per_image.csv.
  2. Computes three per-pixel color indices inside the REFERENCE leaf mask (letterbox label >= 1, i.e. alpha > 8;
     the networks segment the leaf themselves, so the baseline's severity denominator is the exact reference leaf):
        ri    redness index (r - g) / (r + g + b)                  (0 where r + g + b = 0)
        astar CIELAB a* (sRGB, D65 white)
        exr   excess red 1.4 r' - g', r' = r/(r+g+b), g' = g/(r+g+b) (0 where r + g + b = 0)
     Lesion = index above a threshold (higher index = redder/browner than green tissue for both diseases).
     Tie handling: the letterboxed RGB values are 8-bit integers, so ri and exr are ratios of integers and many pixels
     sit exactly on a grid threshold (e.g. exr = 0.12 <=> 32r - 28g - 3b = 0). Global-threshold comparisons for ri and
     exr are therefore done in EXACT integer arithmetic (lesion = index >= t, ties counted as lesion):
        ri  >= k/1000  <=>  floor(1000 (r - g) / s)       >= k
        exr >= m/1000  <=>  floor(200 (7r - 5g) / s)      >= m        (s = r + g + b > 0; index 0 where s = 0)
     a* is compared in float64 (irrational values; no exact ties at the 0.1 grid).
  3. Two threshold rules per index (6 configurations):
        otsu    per-LEAF Otsu threshold (one threshold per leaf, from the pooled leaf pixels of the face(s) that the
                leaf-level rule uses: rust abaxial; leaf spot both faces; 256-bin histogram over the pooled range,
                identical to skimage.filters.threshold_otsu); lesion = index > t. A per-FACE Otsu variant is also
                reported (secondary).
        global  one threshold per disease, chosen on the VALIDATION diseased leaves to maximize mean leaf lesion
                IoU (grid search; lesion = index >= t)
     No smoothing, morphology or speck removal (the networks were also evaluated on raw argmax output).
  4. Picks, per disease, the configuration with the highest VALIDATION mean leaf IoU, then evaluates that single
     configuration once on the test leaves with the paper's leaf-level rules:
        rust       abaxial face only; reference grade from reference severity (cut-points 0.5 / 5 / 20 %)
        leaf spot  mean of the two faces (IoU = mean of face IoUs, severity = mean of face severities);
                   reference grade = annotator's visual grade
     Metrics: leaf IoU, MAE, RMSE, bias, Pearson r, Lin's CCC, Bland-Altman limits, grade accuracy, QWK
     (K = 4, G0-G3), each with a percentile bootstrap CI (10,000 resamples of leaves, numpy default_rng seed
     20260927, re-seeded per disease; all methods share the same resampled leaf indices, so differences are
     paired). Degenerate resamples (undefined r/CCC/QWK) are dropped and counted.
  5. Specificity on healthy leaves (rust pack; EARTH site): the 21 healthy TEST leaves present locally (the paper
     used 54) and the 21 healthy VALIDATION leaves. Two face rules, both reported for every detector:
        either_face  PRIMARY; the rule of revision_brief.md item 10: a healthy leaf is flagged when EITHER face meets
                     the flag criterion (leaf value = maximum over the two faces)
        leaf_rule    the paper's diseased-leaf rule of the detector's disease (rust: abaxial face; leaf spot: mean of
                     the two faces)
     Flag criteria: any lesion pixel (> 0 %), predicted grade G1 or higher (>= 0.5 %), and >= 5 %. Networks are
     scored on the same leaves from per_image.csv where predictions exist (rust networks: test and validation;
     leaf-spot networks: test only, from the cross-disease run; no inference is re-run). The secondary
     specificity-constrained threshold is chosen on validation under the primary (either-face) rule.

Outputs (this folder): results.json (all numbers), per_leaf_test.csv, per_leaf_healthy.csv,
validation_curves.csv. Figures: make_figures.py. Markdown: make_md.py.
"""
import os as _os  # (release) all paths below are relative to ROOT (repository root by default; see README)
ROOT = (_os.environ.get("PG26038_ROOT") or _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))).replace(chr(92), "/")
import csv
import json
import math
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
SEED = 20260927
K = 4
EDGE_PX = 3   # width (letterbox px) of the leaf-margin band used to locate false-positive pixels on healthy leaves
DIS = ("roya", "ojo")
DIS_NAME = {"roya": "Coffee leaf rust", "ojo": "American leaf spot"}
ARCHS = ("segformer", "unetpp", "deeplab")
ARCH_NAME = {"segformer": "SegFormer-B5", "unetpp": "UNet++", "deeplab": "DeepLabV3+"}
INDICES = ("ri", "astar", "exr")
INDEX_NAME = {"ri": "redness index (r-g)/(r+g+b)", "astar": "CIELAB a*", "exr": "excess red 1.4r'-g'"}
EXACT = ("ri", "exr")          # compared with exact integer arithmetic (see docstring)
RULES = ("otsu", "global")
GRID = {"ri": np.round(np.arange(-0.300, 0.6000001, 0.001), 4),
        "astar": np.round(np.arange(-40.0, 60.00001, 0.1), 2),
        "exr": np.round(np.arange(-0.600, 1.2000001, 0.002), 4)}
FLAG_RULES = {"any_pixel": "any predicted lesion pixel (> 0 %; the paper's stated definition)",
              "ge_0.5pct": "predicted severity >= 0.5 % (predicted grade G1 or higher)",
              "ge_5pct": "predicted severity >= 5 % (false-positive rule used to select the rust CNNs)"}
FACE_RULES = {"either_face": "PRIMARY (revision_brief.md item 10): flagged when either face meets the criterion; leaf value = maximum over the two faces",
              "leaf_rule": "paper's diseased-leaf rule of the detector's disease (rust: abaxial face only; leaf spot: mean of the two faces)"}
PUBLISHED = {  # manuscript Tables 2/4 and the Bland-Altman figure (figure5)
    ("roya", "segformer"): dict(iou=0.767, mae=0.99, bias=0.99, qwk=0.88, loa=(-1.68, 3.66)),
    ("roya", "unetpp"): dict(iou=0.647, mae=1.77, bias=1.75, qwk=0.79, loa=(-2.68, 6.17)),
    ("roya", "deeplab"): dict(iou=0.632, mae=1.47, bias=1.41, qwk=0.88, loa=(-3.07, 5.89)),
    ("ojo", "segformer"): dict(iou=0.884, mae=0.22, bias=0.16, qwk=0.82, loa=(-0.47, 0.79)),
    ("ojo", "unetpp"): dict(iou=0.853, mae=0.15, bias=-0.03, qwk=0.85, loa=(-0.50, 0.44)),
    ("ojo", "deeplab"): dict(iou=0.815, mae=0.25, bias=0.08, qwk=0.84, loa=(-0.66, 0.82)),
}
PUB_DP = dict(iou=3, mae=2, bias=2, qwk=2, loa=2)
BRIEF_ITEM10 = {"segformer": (19, 0.38), "unetpp": (19, 0.27), "deeplab": (13, 0.09)}   # any-pixel count, max severity %


def grade(s):
    return 0 if s < 0.5 else 1 if s < 5 else 2 if s < 20 else 3


# ----------------------------------------------------------------------------------------------------------------
# Exact copy of the evaluation-space loader of reanalysis/run_inference.py
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


# ----------------------------------------------------------------------------------------------------------------
# Color indices (inputs: float arrays of r, g, b in 0..255)
def idx_ri(r, g, b):
    s = r + g + b
    return np.where(s > 0, (r - g) / np.where(s > 0, s, 1.0), 0.0)


def idx_exr(r, g, b):
    s = r + g + b
    ss = np.where(s > 0, s, 1.0)
    return np.where(s > 0, 1.4 * r / ss - g / ss, 0.0)


def idx_astar(r, g, b):
    def lin(c):
        c = c / 255.0
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    R, G, Bl = lin(r), lin(g), lin(b)
    X = (0.4124564 * R + 0.3575761 * G + 0.1804375 * Bl) / 0.95047
    Y = (0.2126729 * R + 0.7151522 * G + 0.0721750 * Bl) / 1.00000
    d = 6.0 / 29.0
    f = lambda t: np.where(t > d ** 3, np.cbrt(t), t / (3 * d * d) + 4.0 / 29.0)
    return 500.0 * (f(X) - f(Y))


INDEX_FN = {"ri": idx_ri, "astar": idx_astar, "exr": idx_exr}


def index_image(rgb, name):
    x = rgb.astype(np.float64)
    return INDEX_FN[name](x[..., 0], x[..., 1], x[..., 2])


def index_num_den(rgb, name):
    """Integer numerator/denominator with index >= m/1000 <=> num >= m * den (den > 0). ri: num = 1000 (r - g);
    exr: num = 200 (7r - 5g); den = r + g + b, set to 1 (with num = 0, i.e. index 0) where r + g + b = 0."""
    x = rgb.astype(np.int64); r, g, b = x[..., 0], x[..., 1], x[..., 2]
    s = r + g + b
    num = 1000 * (r - g) if name == "ri" else 200 * (7 * r - 5 * g)
    return np.where(s > 0, num, 0), np.where(s > 0, s, 1)


def index_key(rgb, name):
    """Exact integer key for ri/exr: key >= m <=> index >= m/1000 (floor division, exact for negative values)."""
    num, den = index_num_den(rgb, name)
    return num // den


def thr_key(index, thr):
    """Threshold in key units: integer m = 1000 t for ri/exr (grid values are multiples of 0.001), float t for a*."""
    return int(round(thr * 1000)) if index in EXACT else float(thr)


def grid_keys(index):
    g = GRID[index]
    return np.round(g * 1000).astype(np.int64) if index in EXACT else g


def global_pred_image(rgb, index, thr):
    """Full-image boolean lesion prediction for a global threshold (exact for ri/exr); used by make_figures.py."""
    if index in EXACT:
        return index_key(rgb, index) >= thr_key(index, thr)
    return index_image(rgb, index) >= thr


def otsu(values, nbins=256):
    """skimage.filters.threshold_otsu (histogram over the value range, bin centres)."""
    v = np.asarray(values, np.float64)
    if v.min() == v.max():
        return float(v.min())
    hist, edges = np.histogram(v, bins=nbins, range=(v.min(), v.max()))
    centres = (edges[:-1] + edges[1:]) / 2.0
    hist = hist.astype(np.float64)
    w1 = np.cumsum(hist); w2 = np.cumsum(hist[::-1])[::-1]
    m1 = np.cumsum(hist * centres) / w1
    m2 = (np.cumsum((hist * centres)[::-1]) / w2[::-1])[::-1]
    var12 = w1[:-1] * w2[1:] * (m1[:-1] - m2[1:]) ** 2
    return float(centres[np.argmax(var12)])


# ----------------------------------------------------------------------------------------------------------------
# Statistics
def qwk(a, b):
    O = np.zeros((K, K))
    for x, y in zip(a, b): O[x, y] += 1
    Ex = np.outer(O.sum(1), O.sum(0)) / O.sum()
    W = (np.subtract.outer(np.arange(K), np.arange(K)) ** 2) / (K - 1) ** 2
    den = (W * Ex).sum()
    return float("nan") if den == 0 else float(1 - (W * O).sum() / den)


def qwk_boot(a, b, IDX):
    """Vectorised QWK over bootstrap resamples; NaN where undefined."""
    Bn, n = IDX.shape
    O = np.zeros((Bn, K, K))
    np.add.at(O, (np.repeat(np.arange(Bn), n), a[IDX].ravel(), b[IDX].ravel()), 1)
    Ex = O.sum(2)[:, :, None] * O.sum(1)[:, None, :] / n
    W = (np.subtract.outer(np.arange(K), np.arange(K)) ** 2) / (K - 1) ** 2
    num = (W * O).sum((1, 2)); den = (W * Ex).sum((1, 2))
    with np.errstate(all="ignore"):
        return np.where(den > 0, 1 - num / den, np.nan)


def pearson_ccc(x, y, axis=-1):
    mx, my = x.mean(axis, keepdims=True), y.mean(axis, keepdims=True)
    vx, vy = ((x - mx) ** 2).mean(axis), ((y - my) ** 2).mean(axis)
    cov = ((x - mx) * (y - my)).mean(axis)
    with np.errstate(all="ignore"):
        r = cov / np.sqrt(vx * vy)
        ccc = 2 * cov / (vx + vy + (mx.squeeze(axis) - my.squeeze(axis)) ** 2)   # Lin (1989), 1/n moments
    return r, ccc


def point_metrics(m):
    iou, rs, ps, rg, pg = (m[k] for k in ("iou", "ref_sev", "pred_sev", "ref_grade", "pred_grade"))
    d = ps - rs
    r, ccc = pearson_ccc(rs, ps)
    sd = d.std(ddof=1)
    return dict(n=int(len(iou)), iou=float(iou.mean()), iou_median=float(np.median(iou)),
                mae=float(np.abs(d).mean()), rmse=float(np.sqrt((d ** 2).mean())), bias=float(d.mean()),
                sd_diff=float(sd), loa=[float(d.mean() - 1.96 * sd), float(d.mean() + 1.96 * sd)],
                pearson_r=float(r), ccc=float(ccc), grade_acc=float((rg == pg).mean()),
                grade_correct=int((rg == pg).sum()), qwk=qwk(rg, pg),
                ref_sev_mean=float(rs.mean()), pred_sev_mean=float(ps.mean()))


def boot_metrics(m, IDX):
    iou, rs, ps, rg, pg = (m[k] for k in ("iou", "ref_sev", "pred_sev", "ref_grade", "pred_grade"))
    X, Y = rs[IDX], ps[IDX]; D = Y - X
    r, ccc = pearson_ccc(X, Y, axis=1)
    return dict(iou=iou[IDX].mean(1), mae=np.abs(D).mean(1), rmse=np.sqrt((D ** 2).mean(1)), bias=D.mean(1),
                pearson_r=r, ccc=ccc, grade_acc=(rg[IDX] == pg[IDX]).mean(1), qwk=qwk_boot(rg, pg, IDX))


def pct_ci(x):
    x = np.asarray(x, np.float64); ok = np.isfinite(x)
    lo, hi = np.percentile(x[ok], [2.5, 97.5])
    return [float(lo), float(hi)], int((~ok).sum())


def excludes_zero(c):
    return bool(c[0] > 0 or c[1] < 0)


def wilson(k, n):
    ci = binomtest(k, n).proportion_ci(0.95, method="wilson")
    return [float(ci.low), float(ci.high)]


# ----------------------------------------------------------------------------------------------------------------
# Network results from per_image.csv
PER = list(csv.DictReader(open(PER_IMAGE, newline="")))


def faces(model_dis, arch, image_pack, split):
    t = defaultdict(dict)
    for r in PER:
        if (r["model_disease"], r["arch"], r["image_pack"], r["split"]) == (model_dis, arch, image_pack, split):
            t[r["leaf_id"]][r["face"]] = r
    return t


def leaf_rows(dis, face_dict, rule):
    """Collapse faces to one leaf value with the leaf-level rule ('abaxial' or 'mean')."""
    out = {}
    for lid, f in face_dict.items():
        use = [f["abaxial"]] if rule == "abaxial" else [f["abaxial"], f["adaxial"]]
        out[lid] = dict(
            iou=float(np.mean([float(u["lesion_iou"]) for u in use])) if all(u["lesion_iou"] != "" for u in use) else float("nan"),
            ref_sev=float(np.mean([float(u["ref_sev"]) for u in use])),
            pred_sev=float(np.mean([float(u["pred_sev"]) for u in use])),
            any_px=any(int(u["pred_lesion_px"]) > 0 for u in use),
            grade_label=f["abaxial"]["grade_label"])
    return out


def ref_grade_of(dis, row):
    return grade(row["ref_sev"]) if dis == "roya" else int(row["grade_label"][1])


def to_arrays(dis, leaves, lids):
    iou = np.array([leaves[l]["iou"] for l in lids]); rs = np.array([leaves[l]["ref_sev"] for l in lids])
    ps = np.array([leaves[l]["pred_sev"] for l in lids])
    return dict(iou=iou, ref_sev=rs, pred_sev=ps, ref_grade=np.array([ref_grade_of(dis, leaves[l]) for l in lids]),
                pred_grade=np.array([grade(p) for p in ps]))


RULE = {"roya": "abaxial", "ojo": "mean"}


def diseased_ids(dis, split):
    f = faces(dis, "segformer", dis, split)
    return sorted(l for l, v in f.items() if v["abaxial"]["grade_label"] != "healthy")


def healthy_ids(split):
    f = faces("roya", "segformer", "roya", split)
    return sorted(l for l, v in f.items() if v["abaxial"]["grade_label"] == "healthy")


def net_healthy(dis, arch, split, lids):
    """Healthy-leaf values of a published network under both face rules (None if no predictions exist)."""
    fd = faces(dis, arch, "roya", split)
    if not fd:
        return None
    out = {}
    for lid in lids:
        ab, ad = fd[lid]["abaxial"], fd[lid]["adaxial"]
        s = {"abaxial": float(ab["pred_sev"]), "adaxial": float(ad["pred_sev"])}
        a = {"abaxial": int(ab["pred_lesion_px"]) > 0, "adaxial": int(ad["pred_lesion_px"]) > 0}
        out[lid] = dict(sev_abaxial=s["abaxial"], sev_adaxial=s["adaxial"], sev_either=max(s.values()),
                        any_either=any(a.values()), any_abaxial=a["abaxial"], any_adaxial=a["adaxial"],
                        sev_leaf_rule=s["abaxial"] if RULE[dis] == "abaxial" else float(np.mean(list(s.values()))),
                        any_leaf_rule=a["abaxial"] if RULE[dis] == "abaxial" else any(a.values()))
    return out


def flag_summary(lv, lids):
    """Specificity summary of healthy-leaf values under both face rules and all flag criteria."""
    out = {}
    for fr_name, sk, ak in (("either_face", "sev_either", "any_either"), ("leaf_rule", "sev_leaf_rule", "any_leaf_rule")):
        sev = np.array([lv[l][sk] for l in lids]); anyp = np.array([lv[l][ak] for l in lids])
        flags = {"any_pixel": anyp, "ge_0.5pct": sev >= 0.5, "ge_5pct": sev >= 5}
        e = dict(face_rule=FACE_RULES[fr_name], pred_sev_median=float(np.median(sev)), pred_sev_mean=float(sev.mean()),
                 pred_sev_max=float(sev.max()), pred_sev_min=float(sev.min()))
        for fl_name, fl in flags.items():
            kf = int(fl.sum())
            e[fl_name] = dict(flagged=kf, n=len(lids), fp_rate=kf / len(lids), fp_rate_wilson95=wilson(kf, len(lids)),
                              specificity=1 - kf / len(lids))
        e["predicted_grades"] = {f"G{g}": int(sum(grade(s) == g for s in sev)) for g in range(4)}
        out[fr_name] = e
    return out


# ----------------------------------------------------------------------------------------------------------------
def main():
    t0 = time.time()
    res = dict(meta=dict(
        script=os.path.abspath(__file__), created=time.strftime("%Y-%m-%d %H:%M"),
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        input=PER_IMAGE, evaluation_space="1024 x 1024 aspect-preserving letterbox, black padding; RGB bilinear, labels nearest (run_inference.py load())",
        leaf_rules="rust: abaxial face only; American leaf spot: mean of the two faces (IoU = mean of face IoUs; severity = mean of face severities)",
        reference_grades="rust: from reference severity with cut-points 0.5/5/20 %; leaf spot: annotator's visual grade (G1-G3)",
        predicted_grades="predicted leaf severity with cut-points 0.5/5/20 % (G0-G3)",
        bootstrap=f"{B} resamples of leaves, percentile 95% CI, numpy default_rng(seed={SEED}) re-seeded per disease; one index matrix per disease shared by all methods (paired differences); undefined resamples dropped and counted",
        ccc="Lin's concordance correlation coefficient with 1/n moments", qwk="quadratic-weighted kappa, K = 4 (G0-G3)",
        loa="Bland-Altman limits: bias +/- 1.96 SD (SD with n-1) of predicted minus reference severity (percentage points)",
        proportion_ci="Wilson score, two-sided 95%",
        healthy_coverage="21 healthy rust-pack TEST leaves are present locally (EARTH site); the paper used 54. The other 33 are not in the local pack. Healthy VALIDATION leaves: 21 (EARTH).",
        healthy_face_rules=FACE_RULES, healthy_flag_rules=FLAG_RULES,
        baseline_leaf_mask="reference leaf mask (letterbox label >= 1, i.e. alpha > 8), so the baseline's severity denominator is the exact reference leaf area; the networks segment the leaf themselves (their denominator is their predicted leaf)",
        baseline_postprocessing="none (no smoothing, morphology or speck removal), matching the raw-argmax evaluation of the networks",
        tie_handling="ri and exr global thresholds compared in exact integer arithmetic on the 8-bit letterboxed RGB (ri >= k/1000 <=> 1000(r-g) >= k(r+g+b); exr >= m/1000 <=> 200(7r-5g) >= m(r+g+b)); pixels exactly on the threshold count as lesion. a* compared in float64. Otsu thresholds are histogram bin centres compared in float64 (lesion = index > t).",
        otsu_rule="per LEAF: one threshold from the pooled leaf pixels of the faces used by the evaluation (diseased leaves: rust abaxial, leaf spot both faces; healthy leaves, either-face rule: both faces). Per-face Otsu reported as a secondary variant.",
        indices=INDEX_NAME, grids={k: [float(v[0]), float(v[-1]), float(v[1] - v[0])] for k, v in GRID.items()},
        tuning_set="validation DISEASED leaves only (21 rust, 55 leaf spot); leaf IoU is undefined on healthy leaves, which are used only to report specificity",
        selection="per disease, the configuration with the highest validation mean leaf IoU; evaluated once on test"))

    # ---------------- 0. reproduce published network values ----------------
    repro, ok_all = {}, True
    net_test = {}
    for dis in DIS:
        lids = diseased_ids(dis, "test")
        for arch in ARCHS:
            leaves = leaf_rows(dis, faces(dis, arch, dis, "test"), RULE[dis])
            m = to_arrays(dis, leaves, lids); pm = point_metrics(m)
            net_test[(dis, arch)] = m
            comp = dict(iou=pm["iou"], mae=pm["mae"], bias=pm["bias"], qwk=pm["qwk"], loa=pm["loa"])
            pub = PUBLISHED[(dis, arch)]
            match = {}
            for k, v in comp.items():
                if k == "loa":
                    match[k] = all(abs(round(v[i], 2) - pub[k][i]) < 1e-9 for i in range(2))
                else:
                    match[k] = abs(round(v, PUB_DP[k]) - pub[k]) < 1e-9
            ok_all &= all(match.values())
            repro[f"{dis}/{arch}"] = dict(n=len(lids), computed=comp, published=dict(pub, loa=list(pub["loa"])), match=match)
    res["reproduction"] = dict(all_match=bool(ok_all), models=repro)
    print("reproduction of published values:", "ALL MATCH" if ok_all else "MISMATCH", flush=True)
    if not ok_all:
        json.dump(res, open(f"{OUT}/results.json", "w"), indent=2)
        raise SystemExit("published values not reproduced; stopping")
    assert len(diseased_ids("roya", "test")) == 21 and len(diseased_ids("ojo", "test")) == 55
    assert len(healthy_ids("test")) == 21 and len(healthy_ids("val")) == 21
    # revision_brief.md item 10 (rust networks, healthy test leaves, maximum over both faces)
    b10 = {}
    for a in ARCHS:
        lv = net_healthy("roya", a, "test", healthy_ids("test"))
        n_any = sum(lv[l]["any_either"] for l in lv); mx = max(lv[l]["sev_either"] for l in lv)
        b10[a] = dict(any_pixel_either_face=int(n_any), max_sev_either_face=mx, brief=list(BRIEF_ITEM10[a]),
                      match=bool(n_any == BRIEF_ITEM10[a][0] and abs(round(mx, 2) - BRIEF_ITEM10[a][1]) < 1e-9))
    res["reproduction"]["brief_item10_healthy_counts"] = b10
    assert all(v["match"] for v in b10.values()), b10
    print("revision brief item 10 healthy-leaf counts reproduced", flush=True)

    # ---------------- 1. rebuild images, 2. indices ----------------
    split_rows = {p: [r for r in csv.DictReader(open(PACKS[p][1], newline="")) if r["split"] in ("val", "test")] for p in DIS}
    ref_counts = {(r["image_pack"], r["file"]): (int(r["ref_leaf_px"]), int(r["ref_lesion_px"]))
                  for r in PER if r["arch"] == "segformer" and r["model_disease"] == r["image_pack"]}
    IMG = {}
    n_checked = 0
    for pack in DIS:
        for r in split_rows[pack]:
            rgb, lab = load(pack, r["file"])
            leaf = lab >= 1; les = lab == 2
            key = (pack, r["file"])
            assert (int(leaf.sum()), int(les.sum())) == ref_counts[key], key
            n_checked += 1
            face = "abaxial" if "abaxial" in r["file"] else "adaxial"
            rec = dict(pack=pack, split=r["split"], leaf_id=r["leaf_id"], face=face, grade_label=r["grade"],
                       n_leaf=int(leaf.sum()), n_les=int(les.sum()), les=les[leaf],
                       edge=(leaf & ~ndimage.binary_erosion(leaf, iterations=EDGE_PX))[leaf],
                       edge12=(leaf & ~ndimage.binary_erosion(leaf, iterations=12))[leaf])
            for name in INDICES:
                rec[name] = index_image(rgb, name)[leaf].astype(np.float64)
            for name in EXACT:
                num, den = index_num_den(rgb, name)
                num, den = num[leaf], den[leaf]
                rec[name + "_key"] = (num // den).astype(np.int16)          # |key| <= 1400
                rec[name + "_divisible"] = (num % den) == 0                 # index is exactly key/1000
            IMG[key] = rec
    print(f"rebuilt {n_checked} images; reference counts match per_image.csv ({time.time() - t0:.0f}s)", flush=True)
    res["meta"]["reference_rebuild_check"] = f"{n_checked} images: rebuilt reference leaf and lesion pixel counts equal per_image.csv exactly"

    def kv(rec, index):
        return rec[index + "_key"] if index in EXACT else rec[index]

    by_leaf = defaultdict(dict)
    for (pack, file), rec in IMG.items():
        by_leaf[(pack, rec["split"], rec["leaf_id"])][rec["face"]] = (pack, file)

    def face_keys(pack, split, lid, which):
        f = by_leaf[(pack, split, lid)]
        return [f["abaxial"]] if which == "abaxial" else [f["abaxial"], f["adaxial"]]

    def predict(keys, index, rule, thr=None):
        """Per-face lesion predictions. otsu = one per-leaf threshold on the pooled pixels of these faces;
        otsu_face = one threshold per face; global = exact comparison with thr."""
        if rule == "otsu":
            t = otsu(np.concatenate([IMG[k][index] for k in keys]))
            return [IMG[k][index] > t for k in keys], [t] * len(keys)
        if rule == "otsu_face":
            ts = [otsu(IMG[k][index]) for k in keys]
            return [IMG[k][index] > t for k, t in zip(keys, ts)], ts
        tk = thr_key(index, thr)
        return [kv(IMG[k], index) >= tk for k in keys], [float(thr)] * len(keys)

    def face_stats(key, pred):
        rec = IMG[key]
        inter = int((pred & rec["les"]).sum()); union = int((pred | rec["les"]).sum())
        return dict(pred_px=int(pred.sum()), inter=inter, union=union, iou=inter / union if union else float("nan"),
                    sev=100.0 * pred.sum() / rec["n_leaf"], ref_sev=100.0 * rec["n_les"] / rec["n_leaf"])

    def leaf_eval(dis, pack, split, lids, index, rule, thr=None, excl=None):
        out = {}
        for lid in lids:
            keys = face_keys(pack, split, lid, RULE[dis])
            preds, ts = predict(keys, index, rule, thr)
            if excl:
                preds = [p & ~IMG[k][excl] for k, p in zip(keys, preds)]
            fs = [face_stats(k, p) for k, p in zip(keys, preds)]
            out[lid] = dict(iou=float(np.mean([f["iou"] for f in fs])) if all(np.isfinite(f["iou"]) for f in fs) else float("nan"),
                            ref_sev=float(np.mean([f["ref_sev"] for f in fs])), pred_sev=float(np.mean([f["sev"] for f in fs])),
                            any_px=any(f["pred_px"] > 0 for f in fs), grade_label=IMG[keys[0]]["grade_label"],
                            thr=ts, face_iou=[f["iou"] for f in fs], face_sev=[f["sev"] for f in fs])
        return out

    def healthy_eval(dis, split, lids, index, rule, thr=None, excl=None):
        """Healthy-leaf values of a baseline configuration under both face rules. Both faces are predicted; for per-leaf
        Otsu the either-face rule pools both faces and the leaf rule pools the faces of the disease's leaf rule."""
        out = {}
        for lid in lids:
            both = face_keys("roya", split, lid, "both")
            preds, _ = predict(both, index, rule, thr)
            if excl:
                preds = [p & ~IMG[k][excl] for k, p in zip(both, preds)]
            sev = {IMG[k]["face"]: 100.0 * p.sum() / IMG[k]["n_leaf"] for k, p in zip(both, preds)}
            anyp = {IMG[k]["face"]: bool(p.any()) for k, p in zip(both, preds)}
            if RULE[dis] == "abaxial":
                if rule == "otsu":      # per-leaf Otsu of the leaf rule = abaxial pixels only
                    k0 = both[0]; p0 = predict([k0], index, rule, thr)[0][0]
                    s_lr, a_lr = 100.0 * p0.sum() / IMG[k0]["n_leaf"], bool(p0.any())
                else:
                    s_lr, a_lr = sev["abaxial"], anyp["abaxial"]
            else:
                s_lr, a_lr = float(np.mean(list(sev.values()))), any(anyp.values())
            out[lid] = dict(sev_abaxial=sev["abaxial"], sev_adaxial=sev["adaxial"], sev_either=max(sev.values()),
                            any_abaxial=anyp["abaxial"], any_adaxial=anyp["adaxial"], any_either=any(anyp.values()),
                            sev_leaf_rule=s_lr, any_leaf_rule=a_lr)
        return out

    # ---------------- 3. validation tuning / selection ----------------
    val_curves = {}
    configs = {}
    hval = healthy_ids("val")
    for dis in DIS:
        lids = diseased_ids(dis, "val")
        # global threshold: exact counts at every grid threshold via sorted keys
        for index in INDICES:
            grid = GRID[index]; gk = grid_keys(index)
            leaf_iou = np.zeros((len(lids), len(grid)))
            for i, lid in enumerate(lids):
                fi = []
                for key in face_keys(dis, "val", lid, RULE[dis]):
                    rec = IMG[key]; v = kv(rec, index)
                    sl = np.sort(v[rec["les"]]); sn = np.sort(v[~rec["les"]])
                    tp = len(sl) - np.searchsorted(sl, gk, "left")
                    fp = len(sn) - np.searchsorted(sn, gk, "left")
                    fi.append(tp / (len(sl) + fp))            # len(sl) > 0 on every diseased face
                leaf_iou[i] = np.mean(fi, axis=0)
            curve = leaf_iou.mean(0)
            j = int(np.argmax(curve)); thr = float(grid[j])
            assert 0 < j < len(grid) - 1, (dis, index, "optimum at grid edge")
            near = grid[curve >= curve[j] - 0.005]
            # healthy VALIDATION leaves (rust pack, EARTH): face severities at every grid threshold (secondary analysis)
            hs_either = np.zeros((len(hval), len(grid))); hs_lr = np.zeros((len(hval), len(grid)))
            for i, lid in enumerate(hval):
                fs = {}
                for key in face_keys("roya", "val", lid, "both"):
                    rec = IMG[key]; sv = np.sort(kv(rec, index))
                    fs[rec["face"]] = 100.0 * (len(sv) - np.searchsorted(sv, gk, "left")) / rec["n_leaf"]
                hs_either[i] = np.maximum(fs["abaxial"], fs["adaxial"])
                hs_lr[i] = fs["abaxial"] if RULE[dis] == "abaxial" else (fs["abaxial"] + fs["adaxial"]) / 2
            hflag = (hs_either >= 0.5).sum(0); hflag_lr = (hs_lr >= 0.5).sum(0)
            val_curves[(dis, index)] = (curve, hflag, hflag_lr)
            # direct re-evaluation at the chosen threshold must equal the grid value
            chk = leaf_eval(dis, dis, "val", lids, index, "global", thr)
            assert abs(np.mean([chk[l]["iou"] for l in lids]) - curve[j]) < 1e-12
            configs[(dis, index, "global")] = dict(threshold=thr, val_iou=float(curve[j]),
                                                   plateau_within_0_005_of_max=[float(near.min()), float(near.max())],
                                                   healthy_val_flagged_ge_0_5pct=int(hflag[j]),
                                                   healthy_val_flagged_ge_0_5pct_leaf_rule=int(hflag_lr[j]))
            # secondary: best threshold subject to no healthy validation leaf reaching 0.5 % (primary either-face rule)
            for name, hf in (("global_constrained", hflag), ("global_constrained_leaf_rule", hflag_lr)):
                okc = np.where(hf == 0, curve, -np.inf)
                jc = int(np.argmax(okc))
                configs[(dis, index, name)] = dict(
                    threshold=float(grid[jc]), val_iou=float(curve[jc]), healthy_val_flagged_ge_0_5pct=int(hf[jc]),
                    lowest_threshold_with_0_healthy_flagged=float(grid[np.argmax(hf == 0)]))
        for index in INDICES:
            for orule in ("otsu", "otsu_face"):
                ev = leaf_eval(dis, dis, "val", lids, index, orule)
                configs[(dis, index, orule)] = dict(threshold=None, val_iou=float(np.mean([ev[l]["iou"] for l in lids])),
                                                    otsu_thr_median=float(np.median([t for l in lids for t in ev[l]["thr"]])))
    with open(f"{OUT}/validation_curves.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["disease", "index", "threshold", "val_mean_leaf_iou_diseased", "healthy_val_leaves_ge_0.5pct_either_face",
                    "healthy_val_leaves_ge_0.5pct_leaf_rule"])
        for (dis, index), (curve, hflag, hflag_lr) in val_curves.items():
            for t, c, h, hl_ in zip(GRID[index], curve, hflag, hflag_lr):
                w.writerow([dis, index, f"{t:g}", f"{c:.6f}", int(h), int(hl_)])

    selected = {}
    selected_constrained = {}
    selected_constrained_lr = {}
    res["validation_configs"] = {}
    for dis in DIS:
        rows = {f"{index}_{rule}": dict(index=INDEX_NAME[index], rule=rule, **configs[(dis, index, rule)])
                for index in INDICES for rule in RULES}
        best = max(rows, key=lambda k: rows[k]["val_iou"])
        idx_, rule_ = best.split("_", 1)
        selected[dis] = (idx_, rule_, configs[(dis, idx_, rule_)]["threshold"])
        blk = dict(n_val_diseased=len(diseased_ids(dis, "val")), n_val_healthy=len(hval), configs=rows, selected=best,
                   otsu_per_face_secondary={f"{index}_otsu_face": configs[(dis, index, "otsu_face")] for index in INDICES})
        for name, store, desc in (("global_constrained", selected_constrained,
                                   "global threshold maximizing validation mean leaf IoU subject to 0 of 21 healthy validation leaves at >= 0.5 % predicted severity on EITHER face (primary face rule)"),
                                  ("global_constrained_leaf_rule", selected_constrained_lr,
                                   "same, with the paper's leaf rule for healthy leaves (rust: abaxial face; leaf spot: mean of faces)")):
            cons = {index: configs[(dis, index, name)] for index in INDICES}
            bc = max(cons, key=lambda k: cons[k]["val_iou"])
            store[dis] = (bc, "global", cons[bc]["threshold"])
            blk["secondary_specificity_constrained" + ("" if name == "global_constrained" else "_leaf_rule")] = dict(
                rule=desc, candidates=cons, selected=f"{bc}_global")
        res["validation_configs"][dis] = blk
        print(dis, "validation:", {k: round(v["val_iou"], 4) for k, v in rows.items()}, "->", best, selected[dis][2],
              "| constrained:", selected_constrained[dis], "| constrained (leaf rule):", selected_constrained_lr[dis], flush=True)

    # networks' validation IoU (context only)
    res["network_validation_iou"] = {}
    for dis in DIS:
        lids = diseased_ids(dis, "val")
        for arch in ARCHS:
            lv = leaf_rows(dis, faces(dis, arch, dis, "val"), RULE[dis])
            res["network_validation_iou"][f"{dis}/{arch}"] = float(np.mean([lv[l]["iou"] for l in lids]))

    # ---------------- tie diagnostics at the selected thresholds (ri/exr) ----------------
    ties = {}
    for dis in DIS:
        index, rule, thr = selected[dis]
        if index not in EXACT or rule != "global":
            continue
        m = thr_key(index, thr); per_img = []; float_below = 0; per_face = defaultdict(list)
        for (pack, file), rec in IMG.items():
            if pack != dis:
                continue
            tie = (rec[index + "_key"] == m) & rec[index + "_divisible"]
            per_img.append(int(tie.sum())); per_face[rec["face"]].append(int(tie.sum()))
            float_below += int((tie & (rec[index] < thr)).sum())
            # outside ties, the float comparison and the exact comparison must agree
            dis_agree = (rec[index] >= thr) != (rec[index + "_key"] >= m)
            assert not (dis_agree & ~tie).any(), (dis, file)
        ties[dis] = dict(index=index, threshold=thr, images=len(per_img), tie_pixels_total=int(sum(per_img)),
                         tie_pixels_mean_per_image=float(np.mean(per_img)),
                         tie_pixels_mean_per_image_by_face={k: float(np.mean(v)) for k, v in per_face.items()},
                         tie_pixels_classified_below_by_float64=float_below,
                         note="pixels whose index equals the threshold exactly (leaf pixels of all validation and test images of the disease's pack); counted as lesion by the exact comparison")
    res["meta"]["tie_diagnostics_selected_thresholds"] = ties

    # ---------------- 4. test evaluation (selected configuration, once) ----------------
    res["test"] = {}
    per_leaf_rows = []
    for dis in DIS:
        lids = diseased_ids(dis, "test")
        index, rule, thr = selected[dis]
        ev = leaf_eval(dis, dis, "test", lids, index, rule, thr)
        mb = to_arrays(dis, ev, lids)
        ci_, cr_, ct_ = selected_constrained[dis]
        mbc = to_arrays(dis, leaf_eval(dis, dis, "test", lids, ci_, cr_, ct_), lids)
        li_, lr_, lt_ = selected_constrained_lr[dis]
        mbl = to_arrays(dis, leaf_eval(dis, dis, "test", lids, li_, lr_, lt_), lids)
        methods = {"baseline": mb, "baseline_constrained": mbc, "baseline_constrained_leaf_rule": mbl,
                   **{a: net_test[(dis, a)] for a in ARCHS}}
        rng = np.random.default_rng(SEED)
        IDX = rng.integers(0, len(lids), size=(B, len(lids)))
        boots = {k: boot_metrics(m, IDX) for k, m in methods.items()}
        block = dict(n=len(lids), selected_config=dict(index=index, index_name=INDEX_NAME[index], rule=rule, threshold=thr),
                     secondary_constrained_config=dict(index=ci_, index_name=INDEX_NAME[ci_], rule="global (specificity-constrained on validation, either-face rule)", threshold=ct_,
                                                       note="secondary analysis; not the primary baseline"),
                     secondary_constrained_config_leaf_rule=dict(index=li_, index_name=INDEX_NAME[li_], rule="global (specificity-constrained on validation, paper's leaf rule)", threshold=lt_,
                                                                 note="secondary analysis; reported for comparison with the previous version"),
                     methods={})
        for k, m in methods.items():
            pm = point_metrics(m)
            ci = {}; drop = {}
            for s in ("iou", "mae", "rmse", "bias", "pearson_r", "ccc", "grade_acc", "qwk"):
                ci[s], drop[s] = pct_ci(boots[k][s])
            pm["ci95"] = ci; pm["boot_dropped"] = {s: v for s, v in drop.items() if v}
            pm["grade_acc_wilson95"] = wilson(pm["grade_correct"], pm["n"])
            block["methods"][k] = pm
        # paired differences network - baseline
        for ref_key, out_key, stats in (("baseline", "paired_network_minus_baseline", ("iou", "mae", "qwk", "ccc")),
                                        ("baseline_constrained", "paired_network_minus_baseline_constrained", ("iou", "mae", "qwk"))):
            block[out_key] = {}
            for a in ARCHS:
                dd = {}
                for s in stats:
                    c, nd = pct_ci(boots[a][s] - boots[ref_key][s])
                    dd[s] = dict(diff=float(block["methods"][a][s] - block["methods"][ref_key][s]), ci95=c, dropped=nd,
                                 ci_excludes_zero=excludes_zero(c))
                if ref_key == "baseline":
                    c = pct_ci(np.abs(boots[a]["bias"]) - np.abs(boots["baseline"]["bias"]))[0]
                    dd["abs_bias"] = dict(diff=abs(block["methods"][a]["bias"]) - abs(block["methods"]["baseline"]["bias"]), ci95=c,
                                          ci_excludes_zero=excludes_zero(c))
                    dd["leaves_network_iou_higher"] = int((methods[a]["iou"] > mb["iou"]).sum())
                    dd["leaves_baseline_iou_higher"] = int((methods[a]["iou"] < mb["iou"]).sum())
                block[out_key][a] = dd
        # IoU by reference grade (descriptive)
        strat = {}
        rg = mb["ref_grade"]
        for g in sorted(set(rg.tolist())):
            sel = rg == g
            strat[f"G{g}"] = dict(n=int(sel.sum()), **{k: float(m["iou"][sel].mean()) for k, m in methods.items()},
                                  **{f"{k}_mae": float(np.abs(m["pred_sev"][sel] - m["ref_sev"][sel]).mean()) for k, m in methods.items()})
        block["by_reference_grade"] = strat
        cm = np.zeros((K, K), int)
        for x, y in zip(mb["ref_grade"], mb["pred_grade"]): cm[x, y] += 1
        block["baseline_confusion_rows_ref_cols_pred_G0_G3"] = cm.tolist()
        res["test"][dis] = block
        for i, lid in enumerate(lids):
            row = dict(disease=dis, leaf_id=lid, ref_sev=mb["ref_sev"][i], ref_grade=int(mb["ref_grade"][i]),
                       baseline_iou=mb["iou"][i], baseline_sev=mb["pred_sev"][i], baseline_grade=int(mb["pred_grade"][i]),
                       baseline_face_iou=";".join(f"{x:.6f}" for x in ev[lid]["face_iou"]),
                       baseline_face_sev=";".join(f"{x:.6f}" for x in ev[lid]["face_sev"]),
                       baseline_thr=";".join(f"{x:.6g}" for x in ev[lid]["thr"]),
                       constrained_iou=mbc["iou"][i], constrained_sev=mbc["pred_sev"][i], constrained_grade=int(mbc["pred_grade"][i]))
            for a in ARCHS:
                row[f"{a}_iou"] = net_test[(dis, a)]["iou"][i]; row[f"{a}_sev"] = net_test[(dis, a)]["pred_sev"][i]
                row[f"{a}_grade"] = int(net_test[(dis, a)]["pred_grade"][i])
            per_leaf_rows.append(row)
        print(dis, "test baseline:", {k: round(v, 4) if isinstance(v, float) else v for k, v in point_metrics(mb).items() if k != "loa"},
              "loa", [round(x, 3) for x in point_metrics(mb)["loa"]], flush=True)

    with open(f"{OUT}/per_leaf_test.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_leaf_rows[0].keys())); w.writeheader()
        for r in per_leaf_rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})

    # baseline on validation (in-sample for the tuned threshold; context)
    res["baseline_validation_metrics"] = {}
    for dis in DIS:
        lids = diseased_ids(dis, "val"); index, rule, thr = selected[dis]
        ev = leaf_eval(dis, dis, "val", lids, index, rule, thr)
        res["baseline_validation_metrics"][dis] = point_metrics(to_arrays(dis, ev, lids))

    # ---------------- 5. specificity on healthy leaves ----------------
    res["specificity"] = dict(face_rules=FACE_RULES, flag_rules=FLAG_RULES)
    healthy_rows = []
    for split in ("test", "val"):
        hl = healthy_ids(split)
        blk = dict(n=len(hl), split=split, methods={})
        for dis in DIS:
            entries = {"baseline": healthy_eval(dis, split, hl, *selected[dis]),
                       "baseline_constrained": healthy_eval(dis, split, hl, *selected_constrained[dis]),
                       "baseline_constrained_leaf_rule": healthy_eval(dis, split, hl, *selected_constrained_lr[dis])}
            for a in ARCHS:
                lv = net_healthy(dis, a, split, hl)
                if lv is not None:
                    entries[a] = lv
            for k, lv in entries.items():
                blk["methods"][f"{dis}/{k}"] = flag_summary(lv, hl)
                if k == "baseline_constrained_leaf_rule":
                    continue
                for l in hl:
                    e = lv[l]
                    healthy_rows.append(dict(split=split, leaf_id=l, model=f"{dis}/{k}",
                                             pred_sev_abaxial=f"{e['sev_abaxial']:.6f}", pred_sev_adaxial=f"{e['sev_adaxial']:.6f}",
                                             pred_sev_either_face_max=f"{e['sev_either']:.6f}", pred_sev_leaf_rule=f"{e['sev_leaf_rule']:.6f}",
                                             any_px_abaxial=int(e["any_abaxial"]), any_px_adaxial=int(e["any_adaxial"])))
            for a in ARCHS:
                if f"{dis}/{a}" not in blk["methods"]:
                    blk["methods"][f"{dis}/{a}"] = "not available: no network predictions for these leaves in per_image.csv (inference not re-run)"
        # where do the baseline's lesion pixels fall on healthy leaves? (share inside the 3-px leaf-margin band)
        loc = {}
        for dis in DIS:
            index, rule, thr = selected[dis]
            loc[dis] = {}
            for which, name in (("both", "both_faces"), (RULE[dis], "leaf_rule_faces")):
                tot = edge_hits = edge_area = leaf_area = 0
                for lid in hl:
                    keys = face_keys("roya", split, lid, which)
                    preds, _ = predict(keys, index, rule, thr)
                    for key, pred in zip(keys, preds):
                        rec = IMG[key]
                        tot += int(pred.sum()); edge_hits += int((pred & rec["edge"]).sum())
                        edge_area += int(rec["edge"].sum()); leaf_area += rec["n_leaf"]
                loc[dis][name] = dict(faces="abaxial and adaxial" if which != "abaxial" else "abaxial",
                                      flagged_px=tot, flagged_px_in_margin_band=edge_hits,
                                      share_in_margin_band=edge_hits / tot if tot else float("nan"),
                                      margin_band_share_of_leaf_area=edge_area / leaf_area, margin_band_px=EDGE_PX)
        blk["baseline_flagged_pixel_location"] = loc
        res["specificity"][split] = blk
    with open(f"{OUT}/per_leaf_healthy.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(healthy_rows[0].keys())); w.writeheader(); w.writerows(healthy_rows)

    # ---------------- 6. sensitivity: the non-selected configurations on test (NOT used for selection) ----------
    res["sensitivity_all_configs_on_test"] = dict(
        note="Reported for transparency only; selection used validation alone and the primary comparison is the selected configuration. Healthy columns: either-face rule (primary) and paper's leaf rule.",
        results={})
    hl = healthy_ids("test")
    for dis in DIS:
        lids = diseased_ids(dis, "test")
        for index in INDICES:
            for rule in RULES + ("otsu_face",):
                thr = configs[(dis, index, rule)]["threshold"]
                pm = point_metrics(to_arrays(dis, leaf_eval(dis, dis, "test", lids, index, rule, thr), lids))
                fs = flag_summary(healthy_eval(dis, "test", hl, index, rule, thr), hl)
                res["sensitivity_all_configs_on_test"]["results"][f"{dis}/{index}_{rule}"] = dict(
                    val_iou=configs[(dis, index, rule)]["val_iou"], threshold=thr, test_iou=pm["iou"], test_mae=pm["mae"],
                    test_bias=pm["bias"], test_ccc=pm["ccc"], test_qwk=pm["qwk"], test_grade_acc=pm["grade_acc"],
                    healthy_test_flagged_ge_0_5pct=fs["either_face"]["ge_0.5pct"]["flagged"],
                    healthy_test_flagged_any_pixel=fs["either_face"]["any_pixel"]["flagged"],
                    healthy_test_pred_sev_median=fs["either_face"]["pred_sev_median"],
                    healthy_test_flagged_ge_0_5pct_leaf_rule=fs["leaf_rule"]["ge_0.5pct"]["flagged"],
                    healthy_test_pred_sev_median_leaf_rule=fs["leaf_rule"]["pred_sev_median"])

    # ---------------- 7. POST-HOC sensitivity: exclude a leaf-margin band from the lesion prediction ----------
    # Added after inspecting where the primary baseline's false-positive pixels on healthy leaves fell (mostly the
    # leaf margin band). Band = leaf minus its binary erosion by 3 px (EDGE_PX) or 12 px (the erosion used by the
    # deployed application at a 1024-px canvas). Global threshold re-tuned on the validation diseased leaves for each
    # index; best index by validation IoU; evaluated on test.
    ph = dict(note="POST HOC (added after seeing where the primary baseline's healthy-leaf false positives fell); secondary only",
              band_definition="reference leaf minus its binary erosion (3x3 cross, scipy default) by 3 or 12 letterbox px; lesion pixels predicted inside the band are removed; severity denominator unchanged",
              results={})
    for dis in DIS:
        lids_v = diseased_ids(dis, "val"); lids_t = diseased_ids(dis, "test")
        for excl in ("edge", "edge12"):
            cand = {}
            for index in INDICES:
                grid = GRID[index]; gk = grid_keys(index)
                li = np.zeros((len(lids_v), len(grid)))
                for i, lid in enumerate(lids_v):
                    fi = []
                    for key in face_keys(dis, "val", lid, RULE[dis]):
                        rec = IMG[key]; v = kv(rec, index); keep = ~rec[excl]
                        sl = np.sort(v[rec["les"] & keep]); sn = np.sort(v[~rec["les"] & keep])
                        tp = len(sl) - np.searchsorted(sl, gk, "left"); fp = len(sn) - np.searchsorted(sn, gk, "left")
                        fi.append(tp / (rec["n_les"] + fp))
                    li[i] = np.mean(fi, axis=0)
                curve = li.mean(0); j = int(np.argmax(curve))
                assert 0 < j < len(grid) - 1
                cand[index] = (float(grid[j]), float(curve[j]))
            bi = max(cand, key=lambda k: cand[k][1]); thr = cand[bi][0]
            mt = to_arrays(dis, leaf_eval(dis, dis, "test", lids_t, bi, "global", thr, excl=excl), lids_t)
            pm = point_metrics(mt)
            IDX = np.random.default_rng(SEED).integers(0, len(lids_t), size=(B, len(lids_t)))
            bm = boot_metrics(mt, IDX)
            entry = dict(band_px=EDGE_PX if excl == "edge" else 12, index=bi, index_name=INDEX_NAME[bi], threshold=thr,
                         val_iou=cand[bi][1], val_candidates={k: dict(threshold=v[0], val_iou=v[1]) for k, v in cand.items()},
                         test=dict(iou=pm["iou"], iou_ci95=pct_ci(bm["iou"])[0], mae=pm["mae"], mae_ci95=pct_ci(bm["mae"])[0],
                                   bias=pm["bias"], ccc=pm["ccc"], grade_acc=pm["grade_acc"], qwk=pm["qwk"]))
            for split in ("test", "val"):
                hl_ = healthy_ids(split)
                fs = flag_summary(healthy_eval(dis, split, hl_, bi, "global", thr, excl=excl), hl_)
                entry[f"healthy_{split}"] = dict(n=len(hl_),
                                                 flagged_ge_0_5pct=fs["either_face"]["ge_0.5pct"]["flagged"],
                                                 flagged_any_pixel=fs["either_face"]["any_pixel"]["flagged"],
                                                 pred_sev_median=fs["either_face"]["pred_sev_median"],
                                                 pred_sev_max=fs["either_face"]["pred_sev_max"],
                                                 flagged_ge_0_5pct_leaf_rule=fs["leaf_rule"]["ge_0.5pct"]["flagged"],
                                                 flagged_any_pixel_leaf_rule=fs["leaf_rule"]["any_pixel"]["flagged"])
            ph["results"][f"{dis}/margin{entry['band_px']}px"] = entry
            print("post-hoc", dis, excl, bi, thr, round(pm["iou"], 4), entry["healthy_test"], flush=True)
    res["posthoc_margin_excluded"] = ph

    res["meta"]["runtime_s"] = round(time.time() - t0, 1)
    json.dump(res, open(f"{OUT}/results.json", "w"), indent=2)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
