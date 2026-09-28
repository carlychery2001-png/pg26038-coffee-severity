"""A5 rust operating point, step 1: CPU inference that KEEPS the softmax lesion probability.

Protocol copied verbatim from reanalysis/run_inference.py (and eval_retrained.py):
  leaf = alpha > 8; lesion = mask > 127 AND leaf; aspect-preserving letterbox to 1024 x 1024 (black pad, NEAREST for
  labels, BILINEAR for RGB); ImageNet normalisation; SegFormer logits (1/4 resolution) upsampled bilinearly per channel
  with PIL exactly as run_inference.py; metrics in letterbox space.
New here: softmax over the 3 classes (0 background, 1 leaf, 2 lesion) of the (upsampled) logits is kept.

Images: rust pack, abaxial faces only: diseased validation (21), diseased test (21) and healthy test (21) leaves.
Models: published rust SegFormer-B5, UNet++, DeepLabV3+ (modal-recovery/serving) and matched-protocol rust CNNs
(lrB, seeds 0-2; reanalysis/kaggle_runs/*roya*lrB*/output/*.onnx).

Per model x image, written to work/prob/<model>/<split>_<grade>/<stem>.npz (compressed):
  argmax  uint8 (0/1/2)   raw 3-class argmax (the published operating point)
  c01     uint8 (0/1)     argmax over background vs leaf only (class a pixel takes when it is NOT called lesion)
  p2      uint16          softmax lesion probability x 65535 (rounded)
The argmax pixel counts are checked against out/per_image.csv (published) or out/per_image_retrained.csv (matched)
and the script stops on any mismatch. Resumable (skips existing npz).
"""
import os as _os  # (release) all paths below are relative to ROOT (repository root by default; see README)
ROOT = (_os.environ.get("PG26038_ROOT") or _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))).replace(chr(92), "/")
import csv, glob, os, sys, time
import numpy as np
import onnxruntime as ort
from PIL import Image

BASE = ROOT
HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.join(HERE, "work", "prob")
PACK = f"{BASE}/datasets/pg26038-roya-real-pack"
SIZE = 1024
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)

MODELS = {"pub_segformer": f"{BASE}/modal-recovery/serving/roya_segformer_realpack_1024.onnx",
          "pub_unetpp": f"{BASE}/modal-recovery/serving/unetpp_roya_realpack_1024.onnx",
          "pub_deeplab": f"{BASE}/modal-recovery/serving/roya_deeplab_realpack_1024.onnx"}
for f in sorted(glob.glob(f"{BASE}/reanalysis/kaggle_runs/*roya*lrB*/output/*.onnx")):
    run = os.path.basename(os.path.dirname(os.path.dirname(f)))
    arch = run.split("-")[3]; seed = run.split("seed")[1] if "seed" in run else "0"
    MODELS[f"m_{arch}_s{seed}"] = f
RUN_OF = {k: os.path.basename(os.path.dirname(os.path.dirname(v))) for k, v in MODELS.items() if k.startswith("m_")}
PUB_ARCH = {"pub_segformer": "segformer", "pub_unetpp": "unetpp", "pub_deeplab": "deeplab"}


def resize_pad(arr, nearest=False):  # verbatim from run_inference.py
    im = Image.fromarray(arr); w, h = im.size; s = SIZE / max(w, h)
    im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.NEAREST if nearest else Image.BILINEAR)
    canvas = Image.new(im.mode, (SIZE, SIZE), 0); canvas.paste(im, ((SIZE - im.width) // 2, (SIZE - im.height) // 2))
    return np.asarray(canvas)


def load(file):  # verbatim from run_inference.py:load()
    rgba = np.asarray(Image.open(f"{PACK}/images/{file}").convert("RGBA"))
    leaf = rgba[..., 3] > 8
    les = (np.asarray(Image.open(f"{PACK}/masks/{file}").convert("L")) > 127) & leaf
    lab = np.zeros(leaf.shape, np.uint8); lab[leaf] = 1; lab[les] = 2
    return resize_pad(rgba[..., :3]), resize_pad(lab, nearest=True)


def logits_of(sess, rgb):  # run_inference.py:predict() without the final argmax
    x = ((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    logits = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
    if logits.shape[1] != SIZE:
        logits = np.stack([np.asarray(Image.fromarray(c).resize((SIZE, SIZE), Image.BILINEAR)) for c in logits])
    return logits


def main():
    rows = [r for r in csv.DictReader(open(f"{PACK}/split.csv"))
            if "abaxial" in r["file"] and ((r["split"] in ("val", "test") and r["grade"] == "roya")
                                           or (r["split"] == "test" and r["grade"] == "healthy"))]
    print(len(rows), "images;", len(MODELS), "models", flush=True)
    # reference counts from earlier verified runs (for the argmax reproduction check)
    pub = {(r["arch"], r["file"]): r for r in csv.DictReader(open(f"{BASE}/reanalysis/out/per_image.csv"))
           if r["model_disease"] == "roya" and r["image_pack"] == "roya"}
    ret = {(r["run"], r["file"], r["split"]): r for r in csv.DictReader(open(f"{BASE}/reanalysis/out/per_image_retrained.csv"))}
    opts = ort.SessionOptions(); opts.intra_op_num_threads = 8
    order = [k for k in MODELS if k != "pub_segformer"] + ["pub_segformer"]  # slow model last
    checked = mism = 0; t0 = time.time()
    for m in order:
        sess = None
        for r in rows:
            d = os.path.join(PROB, m, f"{r['split']}_{r['grade']}"); os.makedirs(d, exist_ok=True)
            out = os.path.join(d, r["file"].replace(".png", ".npz"))
            if os.path.exists(out):
                continue
            if sess is None:
                sess = ort.InferenceSession(MODELS[m], opts, providers=["CPUExecutionProvider"])
            rgb, lab = load(r["file"])
            lg = logits_of(sess, rgb).astype(np.float64)
            am = lg.argmax(0).astype(np.uint8)
            e = np.exp(lg - lg.max(0, keepdims=True)); p = e / e.sum(0, keepdims=True)
            c01 = (lg[1] > lg[0]).astype(np.uint8)  # argmax over {0,1}; np.argmax picks index 0 on ties
            p2 = np.round(p[2] * 65535).astype(np.uint16)
            # reproduction check of the argmax counts
            ref = pub.get((PUB_ARCH[m], r["file"])) if m.startswith("pub_") else ret.get((RUN_OF[m], r["file"], r["split"]))
            chk = {"ref_leaf_px": int((lab >= 1).sum()), "ref_lesion_px": int((lab == 2).sum()),
                   "pred_leaf_px": int((am >= 1).sum()), "pred_lesion_px": int((am == 2).sum()),
                   "inter_lesion": int(((am == 2) & (lab == 2)).sum()), "union_lesion": int(((am == 2) | (lab == 2)).sum())}
            if ref is None:
                sys.exit(f"no earlier row for {m} {r['file']} {r['split']}")
            bad = {k: (v, int(ref[k])) for k, v in chk.items() if v != int(ref[k])}
            checked += 1
            if bad:
                mism += 1
                print("MISMATCH", m, r["file"], bad, flush=True)
                sys.exit(1)
            np.savez_compressed(out, argmax=am, c01=c01, p2=p2)
        print(f"{m} done | {time.time() - t0:.0f}s", flush=True)
    print(f"ALL DONE; argmax counts checked on {checked} new model x image pairs, mismatches {mism}", flush=True)


if __name__ == "__main__":
    main()
