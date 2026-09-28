"""Compute reference severity (lesion px / leaf px) for every face in both packs.
leaf = alpha > 8 ; lesion = mask > 127 AND leaf.  Native resolution (no letterbox)."""
import os as _os  # (release) all paths below are relative to ROOT (repository root by default; see README)
ROOT = (_os.environ.get("PG26038_ROOT") or _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))).replace(chr(92), "/")
import csv, json, os
import numpy as np
from PIL import Image

OUT = os.path.dirname(os.path.abspath(__file__))
PACKS = {
    "rust": ((ROOT + "/datasets/pg26038-roya-real-pack"), "split.csv"),
    "leafspot": ((ROOT + "/datasets/pg26038-ojo-real-pack"), "meta/split.csv"),
}
rows = []
for disease, (root, split) in PACKS.items():
    with open(os.path.join(root, split), newline="") as f:
        for r in csv.DictReader(f):
            img = np.asarray(Image.open(os.path.join(root, "images", r["file"])))
            m = np.asarray(Image.open(os.path.join(root, "masks", r["file"])).convert("L"))
            leaf = img[..., 3] > 8
            les = (m > 127) & leaf
            outside = int(((m > 127) & ~leaf).sum())
            face = "abaxial" if "__abaxial" in r["file"] else "adaxial"
            rows.append(dict(disease=disease, file=r["file"], leaf_id=r["leaf_id"], face=face,
                             label=r["grade"], split=r["split"], leaf_px=int(leaf.sum()),
                             lesion_px=int(les.sum()), mask_outside_leaf_px=outside,
                             severity_pct=100.0 * les.sum() / max(1, leaf.sum()),
                             w=img.shape[1], h=img.shape[0]))
with open(os.path.join(OUT, "work", "all_face_severity.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(len(rows))
