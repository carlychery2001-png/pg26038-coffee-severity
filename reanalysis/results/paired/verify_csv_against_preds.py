"""Integrity check: recompute the per-image pixel counts in per_image.csv from the stored predicted class maps
(reanalysis/out/preds) and the reference labels rebuilt exactly as in reanalysis/run_inference.py::load()
(leaf = alpha > 8, lesion = mask > 127 & leaf, aspect-preserving letterbox to 1024 x 1024, black padding,
NEAREST for labels). No model inference is run.

Scope: every TEST image used by paired_analysis.py (within-disease test images of all six models, and the
rust-pack healthy test images scored by the leaf-spot models in the cross-disease runs).
Writes verify_csv.json in this folder.
"""
from __future__ import annotations
import os as _os  # (release) all paths below are relative to ROOT (repository root by default; see README)
ROOT = (_os.environ.get("PG26038_ROOT") or _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))).replace(chr(92), "/")

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

BASE = ROOT
PACK_ROOT = {"roya": f"{BASE}/datasets/pg26038-roya-real-pack", "ojo": f"{BASE}/datasets/pg26038-ojo-real-pack"}
PREDS = Path(f"{BASE}/reanalysis/out/preds")
CSV = Path(f"{BASE}/reanalysis/out/per_image.csv")
OUT = Path(__file__).resolve().parent
SIZE = 1024


def resize_pad(arr, nearest=False):  # verbatim logic of run_inference.resize_pad
    im = Image.fromarray(arr); w, h = im.size; s = SIZE / max(w, h)
    im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.NEAREST if nearest else Image.BILINEAR)
    canvas = Image.new(im.mode, (SIZE, SIZE), 0); canvas.paste(im, ((SIZE - im.width) // 2, (SIZE - im.height) // 2))
    return np.asarray(canvas)


def ref_label(pack, file):  # label part of run_inference.load
    root = PACK_ROOT[pack]
    rgba = np.asarray(Image.open(f"{root}/images/{file}").convert("RGBA"))
    leaf = rgba[..., 3] > 8
    les = (np.asarray(Image.open(f"{root}/masks/{file}").convert("L")) > 127) & leaf
    lab = np.zeros(leaf.shape, np.uint8); lab[leaf] = 1; lab[les] = 2
    return resize_pad(lab, nearest=True)


def main():
    rows = list(csv.DictReader(CSV.open(newline="")))
    use = [r for r in rows if r["split"] == "test" and (
        r["image_pack"] == r["model_disease"] or (r["image_pack"] == "roya" and r["grade_label"] == "healthy"))]
    cache = {}
    fields = ["ref_leaf_px", "ref_lesion_px", "pred_leaf_px", "pred_lesion_px", "inter_lesion", "union_lesion"]
    mismatches = []
    for r in use:
        k = (r["image_pack"], r["file"])
        if k not in cache:
            cache[k] = ref_label(*k)
        lab = cache[k]
        pred = np.asarray(Image.open(PREDS / f"{r['model_disease']}_{r['arch']}" / f"{r['image_pack']}_{r['split']}" / r["file"]))
        assert pred.shape == (SIZE, SIZE) and pred.max() <= 2
        got = dict(ref_leaf_px=int((lab >= 1).sum()), ref_lesion_px=int((lab == 2).sum()),
                   pred_leaf_px=int((pred >= 1).sum()), pred_lesion_px=int((pred == 2).sum()),
                   inter_lesion=int(((pred == 2) & (lab == 2)).sum()), union_lesion=int(((pred == 2) | (lab == 2)).sum()))
        bad = {f: (int(r[f]), got[f]) for f in fields if int(r[f]) != got[f]}
        if bad:
            mismatches.append(dict(model=f"{r['model_disease']}_{r['arch']}", pack=r["image_pack"], file=r["file"], diff=bad))
    res = dict(images_checked=len(use), unique_reference_images=len(cache), mismatches=len(mismatches),
               mismatch_examples=mismatches[:20], fields=fields)
    (OUT / "verify_csv.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"checked {len(use)} rows ({len(cache)} reference images); mismatches: {len(mismatches)}")


if __name__ == "__main__":
    main()
