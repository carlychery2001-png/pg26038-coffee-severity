"""Round 4 / A2: run the published and the matched-protocol (lrB, seeds 0-2) ONNX models on the 60 expert-form
images with the EXACT protocol of validation/run_models.py.

Protocol (copied from validation/run_models.py and coffee-app/backend/infer.py, not changed):
  * form page = abaxial face (left half) + adaxial face (right half) of rows 70:940, leaf on white;
  * each face is cut out (min channel < 225, 3x3 opening, largest component, holes filled), pasted on black
    in a 1583 x 2048 portrait frame and scaled so that leaf height / frame height = R, the median ratio of the
    real training scans (same glob and subsampling as run_models.py);
  * letterbox to 1024 x 1024 (bilinear, black pad), ImageNet normalisation, logits upsampled bilinearly to 1024 if
    needed (SegFormer), raw argmax, crop to the letterbox content;
  * severity = lesion px / (predicted leaf + lesion px) x 100, rounded to 2 dp (as infer.analyze).
Each model is run only on its own disease's 30 form images, both faces (as run_models.py).

Published models: modal-recovery/serving/*.onnx (verified by hash to be identical to coffee-app/models/*.onnx,
which run_models.py used). Their re-run here must reproduce validation/model_predictions.csv (reproduction check).
Matched models: reanalysis/kaggle_runs/pg26038-mb*-lrB*/output/*.onnx (12 files).

Writes only into the folder of this script: form_predictions.csv, run_forms.json.
"""
import os as _os  # (release) all paths below are relative to ROOT (repository root by default; see README)
ROOT = (_os.environ.get("PG26038_ROOT") or _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))).replace(chr(92), "/")
import csv, glob, hashlib, json, os, time
import numpy as np
import onnxruntime as ort
from PIL import Image
from scipy import ndimage as ndi

OUT = os.path.dirname(os.path.abspath(__file__))
BASE = ROOT
VAL = f"{BASE}/validation"
DS = f"{BASE}/datasets"
PUB_DIR = f"{BASE}/modal-recovery/serving"
APP_DIR = _os.environ.get("PG26038_APP_DIR", "coffee-app") + "/models"  # (release) app model copies (hash check only); not in this repository
PUB = {("roya", "segformer"): "roya_segformer_realpack_1024.onnx", ("roya", "deeplab"): "roya_deeplab_realpack_1024.onnx",
       ("roya", "unetpp"): "unetpp_roya_realpack_1024.onnx", ("ojo", "segformer"): "ojo_segformer_realpack_1024.onnx",
       ("ojo", "deeplab"): "ojo_deeplab_realpack_1024.onnx", ("ojo", "unetpp"): "unetpp_ojo_realpack_1024.onnx"}
FW, FH = 1583, 2048
SIZE = 1024
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


# ---------------- run_models.py preprocessing (verbatim logic) ----------------
def leaf_mask(a, white):
    fg = a.min(2) < 225 if white else a.max(2) > 25
    fg = ndi.binary_opening(fg, np.ones((3, 3)))
    lab, n = ndi.label(fg)
    if not n: return fg
    big = lab == 1 + int(np.argmax(ndi.sum(fg, lab, range(1, n + 1))))
    return ndi.binary_fill_holes(big)


ratios = []
for f in sorted(glob.glob(f"{DS}/pg26038-ojo-lesions/images/*.png"))[::40] + sorted(glob.glob(f"{DS}/pg26038-sam2-roya/images/*.png"))[::20]:
    m = leaf_mask(np.asarray(Image.open(f).convert("RGB")), False); ys = np.nonzero(m)[0]
    if len(ys): ratios.append((ys.max() - ys.min()) / m.shape[0])
R = float(np.median(ratios)); print(f"dataset leaf height ratio: {R:.3f} (n={len(ratios)})", flush=True)


def to_training_format(panel):
    m = leaf_mask(panel, True)
    ys, xs = np.nonzero(m); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    leaf = panel[y0:y1, x0:x1] * m[y0:y1, x0:x1, None]
    s = R * FH / (y1 - y0)
    im = Image.fromarray(leaf.astype(np.uint8)).resize((max(1, round((x1 - x0) * s)), round((y1 - y0) * s)), Image.BILINEAR)
    canvas = Image.new("RGB", (FW, FH)); canvas.paste(im, ((FW - im.width) // 2, (FH - im.height) // 2))
    return canvas


# ---------------- infer.analyze (verbatim logic, any ONNX path) ----------------
def letterbox(img):
    img = img.convert("RGB"); w, h = img.size; scale = SIZE / max(w, h)
    nw, nh = round(w * scale), round(h * scale)
    resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (SIZE, SIZE), (0, 0, 0)); x0, y0 = (SIZE - nw) // 2, (SIZE - nh) // 2
    canvas.paste(resized, (x0, y0))
    arr = (np.asarray(canvas, np.float32) / 255.0 - MEAN) / STD
    return arr.transpose(2, 0, 1)[None].astype(np.float32), (x0, y0, nw, nh)


def severity(sess, img):
    tensor, (x0, y0, nw, nh) = letterbox(img)
    logits = sess.run(None, {sess.get_inputs()[0].name: tensor})[0][0]
    if logits.shape[1] != SIZE:
        logits = np.stack([np.asarray(Image.fromarray(logits[c]).resize((SIZE, SIZE), Image.BILINEAR)) for c in range(logits.shape[0])])
    pred = logits.argmax(0)[y0:y0 + nh, x0:x0 + nw]
    leaf_px = int((pred >= 1).sum()); lesion_px = int((pred == 2).sum())
    return round((lesion_px / leaf_px * 100) if leaf_px else 0.0, 2), leaf_px, lesion_px


# ---------------- form faces ----------------
faces = {}
for fp in sorted(glob.glob(f"{VAL}/imagenes/*.jpg")):
    fid = os.path.basename(fp)[:-4]
    a = np.asarray(Image.open(fp).convert("RGB")); w = a.shape[1]; top = a[70:940]
    faces[fid] = {"abaxial": to_training_format(top[:, : w // 2]), "adaxial": to_training_format(top[:, w // 2:])}
assert len(faces) == 60

# ---------------- model list ----------------
models = []
for (dis, arch), fn in PUB.items():
    models.append(dict(run=f"published_{dis}_{arch}", kind="published", disease=dis, arch=arch, seed="", path=f"{PUB_DIR}/{fn}"))
for p in sorted(glob.glob(f"{BASE}/reanalysis/kaggle_runs/pg26038-mb*-lrB*/output/*.onnx")):
    run = os.path.basename(os.path.dirname(os.path.dirname(p))); parts = run.split("-")
    seed = int(run.split("seed")[1]) if "seed" in run else 0
    models.append(dict(run=run, kind="matched", disease=parts[2], arch=parts[3], seed=seed, path=p))
assert sum(m["kind"] == "matched" for m in models) == 12

meta = dict(R=R, n_ratio_images=len(ratios), hashes={}, app_models_identical={}, timing_s={})
for (dis, arch), fn in PUB.items():
    hs, ha = sha256(f"{PUB_DIR}/{fn}"), sha256(f"{APP_DIR}/{fn}")
    meta["hashes"][fn] = hs; meta["app_models_identical"][fn] = hs == ha
for m in models:
    if m["kind"] == "matched": meta["hashes"][m["run"]] = sha256(m["path"])

opts = ort.SessionOptions(); opts.intra_op_num_threads = 8
rows = []
order = sorted(models, key=lambda m: (m["arch"] == "segformer", m["run"]))  # fast models first
for m in order:
    sess = ort.InferenceSession(m["path"], opts, providers=["CPUExecutionProvider"]); t0 = time.time()
    for fid in sorted(faces):
        if not fid.startswith(m["disease"]): continue
        r = dict(run=m["run"], kind=m["kind"], disease=m["disease"], arch=m["arch"], seed=m["seed"], form_id=fid)
        for side in ("abaxial", "adaxial"):
            s, lp, les = severity(sess, faces[fid][side])
            r[f"sev_{side[:2]}"] = s; r[f"leafpx_{side[:2]}"] = lp; r[f"lesionpx_{side[:2]}"] = les
        rows.append(r)
    meta["timing_s"][m["run"]] = round(time.time() - t0, 1)
    print(m["run"], f"{time.time() - t0:.0f}s", flush=True)
    del sess

with open(f"{OUT}/form_predictions.csv", "w", newline="") as fo:
    wr = csv.DictWriter(fo, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)

# ---------------- reproduction check against validation/model_predictions.csv ----------------
grade = lambda s: 0 if s < 0.5 else 1 if s < 5 else 2 if s < 20 else 3
P = {r["leaf_id"]: r for r in csv.DictReader(open(f"{VAL}/model_predictions.csv"))}
chk = {}
for r in rows:
    if r["kind"] != "published": continue
    p = P[r["form_id"]]; a = r["arch"]
    d_ab = abs(r["sev_ab"] - float(p[f"{a}_sev_ab"])); d_ad = abs(r["sev_ad"] - float(p[f"{a}_sev_ad"]))
    c = chk.setdefault(r["run"], dict(n=0, max_abs_diff=0.0, n_exact_both_faces=0))
    c["n"] += 1; c["max_abs_diff"] = max(c["max_abs_diff"], d_ab, d_ad); c["n_exact_both_faces"] += int(d_ab < 1e-9 and d_ad < 1e-9)
meta["reproduction_check_vs_model_predictions_csv"] = chk
json.dump(meta, open(f"{OUT}/run_forms.json", "w"), indent=1)
print(json.dumps(chk, indent=1))
