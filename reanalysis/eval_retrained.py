"""Evaluate retrained Kaggle models (kaggle_runs/*/output/*.onnx) with the EXACT protocol of run_inference.py
on the validation and test splits of their own disease. Appends to out/per_image_retrained.csv (resumable).
Then summarises leaf-level metrics (paper rules) into out/retrained_summary.json."""
import os as _os  # (release) all paths below are relative to ROOT (repository root by default; see README)
ROOT = (_os.environ.get("PG26038_ROOT") or _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))).replace(chr(92), "/")
import csv, glob, json, os, sys, time
import numpy as np, onnxruntime as ort
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
import importlib.util
spec = importlib.util.spec_from_file_location("ri", os.path.join(os.path.dirname(__file__), "run_inference.py"))
BASE = ROOT; RUNS = f"{BASE}/reanalysis/kaggle_runs"; OUT = f"{BASE}/reanalysis/out"

# --- reuse protocol helpers without executing run_inference's main loop ---
SIZE = 1024
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
PACKS = {"roya": (f"{BASE}/datasets/pg26038-roya-real-pack", f"{BASE}/datasets/pg26038-roya-real-pack/split.csv"),
         "ojo": (f"{BASE}/datasets/pg26038-ojo-real-pack", f"{BASE}/datasets/pg26038-ojo-real-pack/meta/split.csv")}
def resize_pad(arr, nearest=False):
    im = Image.fromarray(arr); w, h = im.size; s = SIZE / max(w, h)
    im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.NEAREST if nearest else Image.BILINEAR)
    c = Image.new(im.mode, (SIZE, SIZE), 0); c.paste(im, ((SIZE - im.width) // 2, (SIZE - im.height) // 2)); return np.asarray(c)
def load(pack, file):
    root = PACKS[pack][0]; rgba = np.asarray(Image.open(f"{root}/images/{file}").convert("RGBA")); leaf = rgba[..., 3] > 8
    les = (np.asarray(Image.open(f"{root}/masks/{file}").convert("L")) > 127) & leaf
    lab = np.zeros(leaf.shape, np.uint8); lab[leaf] = 1; lab[les] = 2
    return resize_pad(rgba[..., :3]), resize_pad(lab, nearest=True)
def predict(sess, rgb):
    x = ((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    lg = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
    if lg.shape[1] != SIZE: lg = np.stack([np.asarray(Image.fromarray(c).resize((SIZE, SIZE), Image.BILINEAR)) for c in lg])
    return lg.argmax(0).astype(np.uint8)

csv_path = f"{OUT}/per_image_retrained.csv"
done = set()
if os.path.exists(csv_path):
    done = {(r["run"], r["file"], r["split"]) for r in csv.DictReader(open(csv_path))}
fh = open(csv_path, "a", newline=""); wr = csv.writer(fh)
if not done:
    wr.writerow(["run", "disease", "arch", "split", "file", "leaf_id", "face", "grade_label", "ref_leaf_px", "ref_lesion_px",
                 "pred_leaf_px", "pred_lesion_px", "inter_lesion", "union_lesion", "ref_sev", "pred_sev", "lesion_iou"])
opts = ort.SessionOptions(); opts.intra_op_num_threads = 8
for onnx in sorted(glob.glob(f"{RUNS}/*/output/*.onnx")):
    run = os.path.basename(os.path.dirname(os.path.dirname(onnx)))          # e.g. pg26038-mb-roya-deeplab-lrB
    parts = run.split("-"); dis, arch = parts[2], parts[3]
    rows = [r for r in csv.DictReader(open(PACKS[dis][1])) if r["split"] in ("val", "test")]
    todo = [r for r in rows if (run, r["file"], r["split"]) not in done]
    if not todo: continue
    sess = ort.InferenceSession(onnx, opts, providers=["CPUExecutionProvider"]); t0 = time.time()
    for r in todo:
        rgb, lab = load(dis, r["file"]); pred = predict(sess, rgb)
        rl, rles = int((lab >= 1).sum()), int((lab == 2).sum()); pl, ples = int((pred >= 1).sum()), int((pred == 2).sum())
        inter = int(np.logical_and(pred == 2, lab == 2).sum()); union = int(np.logical_or(pred == 2, lab == 2).sum())
        wr.writerow([run, dis, arch, r["split"], r["file"], r["leaf_id"], "abaxial" if "abaxial" in r["file"] else "adaxial", r["grade"],
                     rl, rles, pl, ples, inter, union, round(100 * rles / rl, 5) if rl else 0, round(100 * ples / pl, 5) if pl else 0,
                     round(inter / union, 6) if union else ""])
    fh.flush(); print(f"{run}: {len(todo)} images in {time.time() - t0:.0f}s", flush=True)
fh.close()

# --- leaf-level summary (paper rules) ---
def grade(s): return 0 if s < 0.5 else 1 if s < 5 else 2 if s < 20 else 3
def qwk(a, b, K=4):
    O = np.zeros((K, K))
    for x, y in zip(a, b): O[x, y] += 1
    E = np.outer(O.sum(1), O.sum(0)) / O.sum(); W = (np.subtract.outer(np.arange(K), np.arange(K)) ** 2) / (K - 1) ** 2
    return float(1 - (W * O).sum() / (W * E).sum())
rows = list(csv.DictReader(open(csv_path))); summ = {}
for run in sorted({r["run"] for r in rows}):
    summ[run] = {}
    for split in ("val", "test"):
        rr = [r for r in rows if r["run"] == run and r["split"] == split and r["grade_label"] != "healthy"]
        dis = rr[0]["disease"] if rr else None
        per = {}
        for r in rr:
            if dis == "roya" and r["face"] != "abaxial": continue
            per.setdefault(r["leaf_id"], []).append(r)
        if not per: continue
        iou = [np.mean([float(x["lesion_iou"]) for x in v]) for v in per.values()]
        ps = [np.mean([float(x["pred_sev"]) for x in v]) for v in per.values()]; rs = [np.mean([float(x["ref_sev"]) for x in v]) for v in per.values()]
        rg = [grade(s) for s in rs] if dis == "roya" else [int(v[0]["grade_label"][1]) for v in per.values()]
        pg = [grade(s) for s in ps]
        heal = [r for r in rows if r["run"] == run and r["split"] == split and r["grade_label"] == "healthy"]
        summ[run][split] = {"n_leaves": len(per), "leaf_iou": float(np.mean(iou)), "mae": float(np.mean(np.abs(np.array(ps) - rs))),
                            "bias": float(np.mean(np.array(ps) - rs)), "grade_acc": float(np.mean(np.array(pg) == rg)), "qwk": qwk(rg, pg),
                            "healthy_images_flagged_gt0": int(sum(float(h["pred_sev"]) > 0 for h in heal)), "healthy_images": len(heal)}
json.dump(summ, open(f"{OUT}/retrained_summary.json", "w"), indent=1)
for run, v in summ.items():
    print(run, {s: {k: round(x, 3) if isinstance(x, float) else x for k, x in m.items()} for s, m in v.items()})
