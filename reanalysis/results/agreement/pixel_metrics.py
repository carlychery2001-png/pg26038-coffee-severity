"""Pixel-level diagnostics for the severity-agreement reanalysis (PG-26038 major revision).

For every test image that enters the leaf-level severity analysis (rust: diseased test leaves, abaxial face;
American leaf spot: diseased test leaves, both faces) and each of the three models of that disease:
  1. Rebuild the reference label map in 1024 x 1024 letterbox space EXACTLY as run_inference.py:load()
     (leaf = alpha > 8, lesion = mask > 127 & leaf, aspect-preserving letterbox, black padding, NEAREST for labels)
     and check the pixel counts against out/per_image.csv (ref leaf/lesion px, pred leaf/lesion px, intersection,
     union must match exactly; the script stops otherwise).
  2. Decompose lesion-area error into false-positive (FP) and false-negative (FN) pixels, and split the FP pixels into
       - spurious: in a predicted lesion component (8-connected) that does not overlap any reference lesion,
       - boundary band: in a component that overlaps a reference lesion and <= 2 px (Euclidean) from it,
       - extension: in a component that overlaps a reference lesion and > 2 px from it,
     and the FN pixels into missed (reference component with no predicted lesion pixel) and partial.
     FP pixels that fall outside the reference leaf mask are also counted.
  3. Boundary F1 of the lesion class at tolerances of 1, 2, 3 and 5 px (Euclidean, letterbox space). Boundary =
     mask minus its 3 x 3 (8-connected) erosion; precision = share of predicted boundary pixels within the tolerance of
     a reference boundary pixel, recall = the converse; both empty -> F1 = 1, one empty -> 0.
No model inference is run: predictions are the saved class maps in out/preds (0 background, 1 leaf, 2 lesion).

Output: work/pixel_metrics.csv (one row per model x image).
"""
import os as _os  # (release) all paths below are relative to ROOT (repository root by default; see README)
ROOT = (_os.environ.get("PG26038_ROOT") or _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))).replace(chr(92), "/")
import csv
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

BASE = ROOT
OUT = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(OUT, "work")
os.makedirs(WORK, exist_ok=True)
PACKS = {"roya": f"{BASE}/datasets/pg26038-roya-real-pack", "ojo": f"{BASE}/datasets/pg26038-ojo-real-pack"}
PREDS = f"{BASE}/reanalysis/out/preds"
SIZE = 1024
TOLS = (1, 2, 3, 5)
BAND = 2  # px, FP "boundary band" width (same as the boundary-F1 tolerance)
EIGHT = np.ones((3, 3), bool)


def resize_pad(arr, nearest=False):  # verbatim from run_inference.py
    im = Image.fromarray(arr); w, h = im.size; s = SIZE / max(w, h)
    im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.NEAREST if nearest else Image.BILINEAR)
    canvas = Image.new(im.mode, (SIZE, SIZE), 0); canvas.paste(im, ((SIZE - im.width) // 2, (SIZE - im.height) // 2))
    return np.asarray(canvas)


def load_label(pack, file):  # run_inference.py:load(), label part only
    root = PACKS[pack]
    rgba = np.asarray(Image.open(f"{root}/images/{file}").convert("RGBA"))
    leaf = rgba[..., 3] > 8
    les = (np.asarray(Image.open(f"{root}/masks/{file}").convert("L")) > 127) & leaf
    lab = np.zeros(leaf.shape, np.uint8); lab[leaf] = 1; lab[les] = 2
    return resize_pad(lab, nearest=True), rgba.shape[1], rgba.shape[0]


def boundary(m):
    return m & ~ndi.binary_erosion(m, structure=EIGHT, border_value=0)


def boundary_counts(pred_les, ref_les):
    bp, br = boundary(pred_les), boundary(ref_les)
    npb, nrb = int(bp.sum()), int(br.sum())
    out = {"pred_boundary_px": npb, "ref_boundary_px": nrb}
    d_to_ref = ndi.distance_transform_edt(~br) if nrb else None
    d_to_pred = ndi.distance_transform_edt(~bp) if npb else None
    for t in TOLS:
        out[f"bmatch_pred_t{t}"] = int((d_to_ref[bp] <= t).sum()) if (npb and nrb) else 0
        out[f"bmatch_ref_t{t}"] = int((d_to_pred[br] <= t).sum()) if (npb and nrb) else 0
    return out


def fp_fn_split(pred_les, ref_les, ref_leaf):
    fp = pred_les & ~ref_les
    fn = ref_les & ~pred_les
    lab_p, n_p = ndi.label(pred_les, structure=EIGHT)
    lab_r, n_r = ndi.label(ref_les, structure=EIGHT)
    # predicted components touching a reference lesion
    touch_p = np.zeros(n_p + 1, bool)
    touch_p[np.unique(lab_p[pred_les & ref_les])] = True
    touch_p[0] = False
    touch_r = np.zeros(n_r + 1, bool)
    touch_r[np.unique(lab_r[pred_les & ref_les])] = True
    touch_r[0] = False
    spurious = fp & ~touch_p[lab_p]
    attached = fp & touch_p[lab_p]
    d_ref = ndi.distance_transform_edt(~ref_les) if ref_les.any() else np.full(ref_les.shape, np.inf)
    band = attached & (d_ref <= BAND)
    ext = attached & (d_ref > BAND)
    missed = fn & ~touch_r[lab_r]
    return {
        "fp_px": int(fp.sum()), "fn_px": int(fn.sum()),
        "fp_spurious_px": int(spurious.sum()), "fp_band_px": int(band.sum()), "fp_extension_px": int(ext.sum()),
        "fp_outside_ref_leaf_px": int((fp & ~ref_leaf).sum()),
        "fn_missed_px": int(missed.sum()), "fn_partial_px": int((fn & touch_r[lab_r]).sum()),
        "n_pred_components": int(n_p), "n_pred_components_spurious": int((~touch_p[1:]).sum()),
        "n_ref_components": int(n_r), "n_ref_components_missed": int((~touch_r[1:]).sum()),
    }


def main():
    rows = list(csv.DictReader(open(f"{BASE}/reanalysis/out/per_image.csv")))
    sel = [r for r in rows if r["model_disease"] == r["image_pack"] and r["split"] == "test"
           and r["grade_label"] != "healthy" and not (r["model_disease"] == "roya" and r["face"] != "abaxial")]
    sel.sort(key=lambda r: (r["image_pack"], r["file"], r["arch"]))
    print(len(sel), "model x image rows")
    cache = {}
    out = []
    for i, r in enumerate(sel):
        key = (r["image_pack"], r["file"])
        if key not in cache:
            cache.clear()
            cache[key] = load_label(*key)
        lab, w0, h0 = cache[key]
        pred = np.asarray(Image.open(f"{PREDS}/{r['model_disease']}_{r['arch']}/{r['image_pack']}_{r['split']}/{r['file']}"))
        assert pred.shape == (SIZE, SIZE), pred.shape
        ref_leaf, ref_les = lab >= 1, lab == 2
        pred_leaf, pred_les = pred >= 1, pred == 2
        chk = {"ref_leaf_px": int(ref_leaf.sum()), "ref_lesion_px": int(ref_les.sum()),
               "pred_leaf_px": int(pred_leaf.sum()), "pred_lesion_px": int(pred_les.sum()),
               "inter_lesion": int((pred_les & ref_les).sum()), "union_lesion": int((pred_les | ref_les).sum())}
        bad = {k: (v, r[k]) for k, v in chk.items() if v != int(r[k])}
        if bad:
            sys.exit(f"MISMATCH with per_image.csv for {r['model_disease']}_{r['arch']} {r['file']}: {bad}")
        rec = {k: r[k] for k in ("model_disease", "arch", "image_pack", "split", "file", "leaf_id", "face", "grade_label")}
        rec.update(chk)
        rec.update({"orig_w": w0, "orig_h": h0, "letterbox_scale": SIZE / max(w0, h0)})
        rec.update(fp_fn_split(pred_les, ref_les, ref_leaf))
        rec.update(boundary_counts(pred_les, ref_les))
        out.append(rec)
        if (i + 1) % 50 == 0:
            print(i + 1, flush=True)
    with open(os.path.join(WORK, "pixel_metrics.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        wr.writeheader(); wr.writerows(out)
    print("all pixel counts match per_image.csv;", len(out), "rows written")


if __name__ == "__main__":
    main()
