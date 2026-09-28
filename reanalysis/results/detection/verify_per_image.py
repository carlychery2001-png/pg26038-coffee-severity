"""Independent check of reanalysis/out/per_image.csv for every TEST row used by the detection analysis.

For each (model, image pack, test file) the stored predicted class map (0 background, 1 leaf, 2 lesion; 1024 x 1024
letterbox space) is re-read, the reference label is rebuilt exactly as in reanalysis/run_inference.py (load():
leaf = alpha > 8, lesion = mask > 127 & leaf, aspect-preserving letterbox with black padding, NEAREST for labels),
and the pixel counts written in per_image.csv are recomputed. No model inference is run.

Output: verify_per_image.json (this folder).
"""
import os as _os  # (release) all paths below are relative to ROOT (repository root by default; see README)
ROOT = (_os.environ.get("PG26038_ROOT") or _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))).replace(chr(92), "/")
import csv
import json
import os

import numpy as np
from PIL import Image

BASE = ROOT
OUT = os.path.dirname(os.path.abspath(__file__))
PACKS = {"roya": f"{BASE}/datasets/pg26038-roya-real-pack", "ojo": f"{BASE}/datasets/pg26038-ojo-real-pack"}
PREDS = f"{BASE}/reanalysis/out/preds"
SIZE = 1024
COLS = ["ref_leaf_px", "ref_lesion_px", "pred_leaf_px", "pred_lesion_px", "inter_lesion", "union_lesion"]


def resize_pad(arr, nearest=False):          # identical to run_inference.resize_pad
    im = Image.fromarray(arr); w, h = im.size; s = SIZE / max(w, h)
    im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.NEAREST if nearest else Image.BILINEAR)
    canvas = Image.new(im.mode, (SIZE, SIZE), 0); canvas.paste(im, ((SIZE - im.width) // 2, (SIZE - im.height) // 2))
    return np.asarray(canvas)


_ref = {}
def ref_label(pack, file):                   # identical to run_inference.load (label part only)
    k = (pack, file)
    if k not in _ref:
        root = PACKS[pack]
        rgba = np.asarray(Image.open(f"{root}/images/{file}").convert("RGBA"))
        leaf = rgba[..., 3] > 8
        les = (np.asarray(Image.open(f"{root}/masks/{file}").convert("L")) > 127) & leaf
        lab = np.zeros(leaf.shape, np.uint8); lab[leaf] = 1; lab[les] = 2
        _ref[k] = resize_pad(lab, nearest=True)
    return _ref[k]


rows = [r for r in csv.DictReader(open(f"{BASE}/reanalysis/out/per_image.csv")) if r["split"] == "test"]
rows.sort(key=lambda r: (r["image_pack"], r["file"]))
n_ok, mismatches, max_abs_sev = 0, [], 0.0
for r in rows:
    lab = ref_label(r["image_pack"], r["file"])
    pred = np.asarray(Image.open(f"{PREDS}/{r['model_disease']}_{r['arch']}/{r['image_pack']}_test/{r['file']}"))
    got = {"ref_leaf_px": int((lab >= 1).sum()), "ref_lesion_px": int((lab == 2).sum()),
           "pred_leaf_px": int((pred >= 1).sum()), "pred_lesion_px": int((pred == 2).sum()),
           "inter_lesion": int(np.logical_and(pred == 2, lab == 2).sum()),
           "union_lesion": int(np.logical_or(pred == 2, lab == 2).sum())}
    bad = {c: (int(r[c]), got[c]) for c in COLS if int(r[c]) != got[c]}
    ps = 100 * got["pred_lesion_px"] / got["pred_leaf_px"] if got["pred_leaf_px"] else 0.0
    max_abs_sev = max(max_abs_sev, abs(ps - float(r["pred_sev"])))
    if bad:
        mismatches.append({"model": f"{r['model_disease']}_{r['arch']}", "pack": r["image_pack"], "file": r["file"], "diff": bad})
    else:
        n_ok += 1
res = {"rows_checked": len(rows), "rows_matching_all_pixel_counts": n_ok, "n_mismatches": len(mismatches),
       "max_abs_diff_pred_sev_pct_vs_csv_rounding": max_abs_sev, "mismatches": mismatches[:50]}
json.dump(res, open(f"{OUT}/verify_per_image.json", "w"), indent=2)
print(json.dumps({k: v for k, v in res.items() if k != "mismatches"}, indent=2))
