"""Run the six thesis models (SegFormer-B5, DeepLabV3+, UNet++ x roya/ojo) on the 60 expert-form leaves
and compare their grades with the four experts.

Each form page shows one leaf: abaxial (left) and adaxial (right) on white, which is converted to the
training format (leaf on black, 1583x2048 portrait frame, leaf at the typical dataset scale) before
inference with coffee-app/backend/infer.py (same letterbox / normalization / argmax / cut points as the thesis).
Leaf severity = mean of the two surfaces, as in the paper."""
import os as _os  # (release) all paths below are relative to ROOT (repository root by default; see README)
ROOT = (_os.environ.get("PG26038_ROOT") or _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))).replace(chr(92), "/")
import csv, glob, os, sys
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
sys.path.insert(0, _os.environ.get("PG26038_APP_DIR", "coffee-app") + "/backend")  # (release) app backend with infer.py; not in this repository
import infer

DS = (ROOT + r"/datasets")
FW, FH = 1583, 2048
os.makedirs("overlays", exist_ok=True)

def leaf_mask(a, white):
    fg = a.min(2) < 225 if white else a.max(2) > 25
    fg = ndi.binary_opening(fg, np.ones((3, 3)))
    lab, n = ndi.label(fg)
    if not n: return fg
    big = lab == 1 + int(np.argmax(ndi.sum(fg, lab, range(1, n + 1))))
    return ndi.binary_fill_holes(big)

# typical leaf height / frame height in the real training scans
ratios = []
for f in sorted(glob.glob(f"{DS}/pg26038-ojo-lesions/images/*.png"))[::40] + sorted(glob.glob(f"{DS}/pg26038-sam2-roya/images/*.png"))[::20]:
    m = leaf_mask(np.asarray(Image.open(f).convert("RGB")), False); ys = np.nonzero(m)[0]
    if len(ys): ratios.append((ys.max() - ys.min()) / m.shape[0])
R = float(np.median(ratios)); print(f"dataset leaf height ratio: {R:.3f} (n={len(ratios)})")

def to_training_format(panel):
    m = leaf_mask(panel, True)
    ys, xs = np.nonzero(m); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    leaf = panel[y0:y1, x0:x1] * m[y0:y1, x0:x1, None]
    s = R * FH / (y1 - y0)
    im = Image.fromarray(leaf.astype(np.uint8)).resize((max(1, round((x1 - x0) * s)), round((y1 - y0) * s)), Image.BILINEAR)
    canvas = Image.new("RGB", (FW, FH)); canvas.paste(im, ((FW - im.width) // 2, (FH - im.height) // 2))
    return canvas

def grade(s): return 0 if s < 0.5 else 1 if s < 5 else 2 if s < 20 else 3

rows = []
for fp in sorted(glob.glob("imagenes/*.jpg")):
    fid = os.path.basename(fp)[:-4]; disease = "ojo" if fid.startswith("ojo") else "roya"
    a = np.asarray(Image.open(fp).convert("RGB")); w = a.shape[1]; top = a[70:940]
    sides = {"abaxial": to_training_format(top[:, : w // 2]), "adaxial": to_training_format(top[:, w // 2 :])}
    row = {"leaf_id": fid, "disease": disease}
    for arch in ("segformer", "deeplab", "unetpp"):
        sev = {}
        for side, img in sides.items():
            r = infer.analyze(img, disease, arch); sev[side] = r["severity"]
            if arch == "segformer": r["overlay"].save(f"overlays/{fid}_{side}.jpg", quality=85)
        mean = (sev["abaxial"] + sev["adaxial"]) / 2
        row.update({f"{arch}_sev_ab": sev["abaxial"], f"{arch}_sev_ad": sev["adaxial"], f"{arch}_sev": round(mean, 3), f"{arch}_grade": grade(mean)})
    rows.append(row); print(fid, {k: v for k, v in row.items() if k.endswith(("_sev", "_grade"))}, flush=True)

with open("model_predictions.csv", "w", newline="") as fo:
    wr = csv.DictWriter(fo, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
