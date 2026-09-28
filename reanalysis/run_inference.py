"""Re-run the six thesis models on the local test and validation packs with the EXACT evaluation protocol of the
training kernels (modal-recovery/kernels/*_realpack_kernel.py): leaf = alpha > 8, lesion = mask > 127 & leaf,
aspect-preserving letterbox to 1024 x 1024 (black pad), ImageNet normalisation, raw argmax, metrics in letterbox space.

Outputs (reanalysis/out/):
  per_image.csv   one row per (model disease, model arch, image): reference and predicted pixel counts, IoU, severities
  preds/<model>/<split>/<stem>.png   predicted class map (0/1/2) in letterbox space, for figures and later analyses
Cross-disease runs: each disease's models are also applied to the other disease's TEST images (false-positive study).
"""
import os as _os  # (release) all paths below are relative to ROOT (repository root by default; see README)
ROOT = (_os.environ.get("PG26038_ROOT") or _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))).replace(chr(92), "/")
import csv, os, sys, time
import numpy as np
import onnxruntime as ort
from PIL import Image

BASE = ROOT
MODELS = f"{BASE}/modal-recovery/serving"
PACKS = {"roya": (f"{BASE}/datasets/pg26038-roya-real-pack", f"{BASE}/datasets/pg26038-roya-real-pack/split.csv"),
         "ojo": (f"{BASE}/datasets/pg26038-ojo-real-pack", f"{BASE}/datasets/pg26038-ojo-real-pack/meta/split.csv")}
FILES = {("roya", "segformer"): "roya_segformer_realpack_1024.onnx", ("roya", "deeplab"): "roya_deeplab_realpack_1024.onnx",
         ("roya", "unetpp"): "unetpp_roya_realpack_1024.onnx", ("ojo", "segformer"): "ojo_segformer_realpack_1024.onnx",
         ("ojo", "deeplab"): "ojo_deeplab_realpack_1024.onnx", ("ojo", "unetpp"): "unetpp_ojo_realpack_1024.onnx"}
SIZE = 1024
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
OUT = f"{BASE}/reanalysis/out"; os.makedirs(OUT, exist_ok=True)


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


def rows_of(pack, splits):
    return [r for r in csv.DictReader(open(PACKS[pack][1])) if r["split"] in splits]


sessions = {}
opts = ort.SessionOptions(); opts.intra_op_num_threads = 8
def session(dis, arch):
    k = (dis, arch)
    if k not in sessions: sessions[k] = ort.InferenceSession(f"{MODELS}/{FILES[k]}", opts, providers=["CPUExecutionProvider"])
    return sessions[k]


def predict(sess, rgb):
    x = ((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    logits = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
    if logits.shape[1] != SIZE:      # SegFormer outputs 1/4 resolution; upsample logits bilinearly like the kernel
        logits = np.stack([np.asarray(Image.fromarray(c).resize((SIZE, SIZE), Image.BILINEAR)) for c in logits])
    return logits.argmax(0).astype(np.uint8)


jobs = []   # (model disease, arch, image pack, split set label, rows)
for dis in ("roya", "ojo"):
    for arch in ("deeplab", "unetpp", "segformer"):          # fast models first
        jobs.append((dis, arch, dis, rows_of(dis, {"test", "val"})))
        other = "ojo" if dis == "roya" else "roya"
        jobs.append((dis, arch, other, rows_of(other, {"test"})))

done = set()
csv_path = f"{OUT}/per_image.csv"
if os.path.exists(csv_path):
    for r in csv.DictReader(open(csv_path)): done.add((r["model_disease"], r["arch"], r["image_pack"], r["file"]))
fh = open(csv_path, "a", newline=""); wr = csv.writer(fh)
if not done:
    wr.writerow(["model_disease", "arch", "image_pack", "split", "file", "leaf_id", "face", "grade_label",
                 "ref_leaf_px", "ref_lesion_px", "pred_leaf_px", "pred_lesion_px", "inter_lesion", "union_lesion",
                 "ref_sev", "pred_sev", "lesion_iou"])
t0 = time.time(); n = 0
for dis, arch, pack, rows in jobs:
    sess = session(dis, arch)
    for r in rows:
        key = (dis, arch, pack, r["file"])
        if key in done: continue
        rgb, lab = load(pack, r["file"])
        pred = predict(sess, rgb)
        d = f"{OUT}/preds/{dis}_{arch}/{pack}_{r['split']}"; os.makedirs(d, exist_ok=True)
        Image.fromarray(pred).save(f"{d}/{r['file']}")
        ref_leaf, ref_les = int((lab >= 1).sum()), int((lab == 2).sum())
        p_leaf, p_les = int((pred >= 1).sum()), int((pred == 2).sum())
        inter = int(np.logical_and(pred == 2, lab == 2).sum()); union = int(np.logical_or(pred == 2, lab == 2).sum())
        face = "abaxial" if "abaxial" in r["file"] else "adaxial"
        wr.writerow([dis, arch, pack, r["split"], r["file"], r["leaf_id"], face, r["grade"], ref_leaf, ref_les, p_leaf, p_les,
                     inter, union, round(100 * ref_les / ref_leaf, 5) if ref_leaf else 0,
                     round(100 * p_les / p_leaf, 5) if p_leaf else 0, round(inter / union, 6) if union else ""])
        n += 1
        if n % 20 == 0:
            fh.flush(); print(f"{n} images | {dis}_{arch} on {pack} | {time.time() - t0:.0f}s", flush=True)
    fh.flush(); print(f"finished {dis}_{arch} on {pack}", flush=True)
fh.close(); print("ALL DONE", time.time() - t0)
