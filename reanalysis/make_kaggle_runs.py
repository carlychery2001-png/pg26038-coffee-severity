"""Generate Kaggle kernels for the reviewer-requested retraining, from the ORIGINAL realpack CNN kernels.
Only the training budget / learning rate / seed change; data, loss, augmentation, model selection and export are kept.

  mb_*  : matched budget (60 epochs, patience 15, as SegFormer-B5) x two learning-rate pairs, chosen on validation later
  rs_*  : leaf-spot re-seeding at the ORIGINAL protocol (30 epochs, patience 12, LR 2e-5/2e-4), seeds 1 and 2
Each kernel also writes history.json (per-epoch train loss, val lesion IoU, val FP, score) for learning curves.
"""
import os as _os  # (release) all paths below are relative to ROOT (repository root by default; see README)
ROOT = (_os.environ.get("PG26038_ROOT") or _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))).replace(chr(92), "/")
import json, os, re

SRC = (ROOT + "/reanalysis/kaggle_src")
OUT = (ROOT + "/reanalysis/kaggle_runs")
TEMPLATE = {"deeplab": f"{SRC}/pg26038-roya-deeplab-realpack/pg26038-roya-deeplab-realpack.py",
            "unetpp": f"{SRC}/pg26038-roya-unetpp-realpack/pg26038-roya-unetpp-realpack.py"}
DATA = {"roya": "carlychery2001/pg26038-roya-real-pack", "ojo": "carlychery2001/pg26038-ojo-real-pack"}
LRS = {"lrA": ("2e-5", "2e-4"), "lrB": ("6e-5", "6e-4")}

def sub1(s, old, new):
    assert s.count(old) == 1, f"template anchor not unique/absent: {old[:60]!r} ({s.count(old)})"
    return s.replace(old, new)

def make(slug, arch, dis, epochs, patience, lr, seed, photometric=True):
    s = open(TEMPLATE[arch], encoding="utf-8").read()
    # the original CNN kernels apply photometric jitter (p = 0.8); SegFormer-B5 was trained with flips/rotations only.
    # Matched-budget runs switch it off so the CNN protocol equals the transformer's.
    s = sub1(s, "            if np.random.rand() < 0.8:\n                rgb = _photometric(rgb)",
             f"            if PHOTOMETRIC and np.random.rand() < 0.8:\n                rgb = _photometric(rgb)")
    s = sub1(s, "Image.MAX_IMAGE_PIXELS = None\n", f"Image.MAX_IMAGE_PIXELS = None\nPHOTOMETRIC = {photometric}\n")
    enc, dec = lr
    s = sub1(s, 'opt = torch.optim.AdamW([{"params": enc, "lr": 2e-5}, {"params": dec, "lr": 2e-4}], weight_decay=0.01)',
             f'opt = torch.optim.AdamW([{{"params": enc, "lr": {enc}}}, {{"params": dec, "lr": {dec}}}], weight_decay=0.01)')
    s = sub1(s, "EPOCHS = 30\n", f"EPOCHS = {epochs}\n")
    s = sub1(s, "patience, bad = 12, 0", f"patience, bad = {patience}, 0")
    # reproducibility: fixed seed right after imports
    s = sub1(s, "Image.MAX_IMAGE_PIXELS = None\n",
             f"Image.MAX_IMAGE_PIXELS = None\nimport random as _rnd\nSEED = {seed}\n_rnd.seed(SEED); np.random.seed(SEED)\n")
    s = s.replace("net = Net(build()).to(DEV)", f"torch.manual_seed({seed}); torch.cuda.manual_seed_all({seed})\nnet = Net(build()).to(DEV)", 1)
    # leaf-spot pack has no healthy rows -> fp_rate() returns NaN; treat as 0 so model selection = val lesion IoU
    s = sub1(s, 'score = va["lesion_iou"] - fp_rt', 'score = va["lesion_iou"] - (0.0 if fp_rt != fp_rt else fp_rt)')
    # per-epoch history for learning curves
    s = sub1(s, "best = -2.0; best_state = None;", "HIST = []\nbest = -2.0; best_state = None;")
    s = sub1(s, "    if score > best:\n",
             "    HIST.append({'epoch': ep, 'train_loss': tl / max(1, len(tr_dl)), 'val_lesion_iou': va['lesion_iou'], "
             "'val_fp': (None if fp_rt != fp_rt else fp_rt), 'score': score})\n"
             "    json.dump(HIST, open(os.path.join(OUT, 'history.json'), 'w'), indent=1)\n    if score > best:\n")
    # distinct ONNX / summary names
    s = re.sub(r'"(roya_deeplab_realpack_1024|unetpp_roya_realpack_1024)\.onnx"', f'"{slug}.onnx"', s)
    s = s.replace('"arch": "deeplab_mobilenet_1024_roya_realpack"', f'"arch": "{slug}"').replace('"arch": "unetpp_mobilenet_1024_roya_realpack"', f'"arch": "{slug}"')
    s = s.replace('json.dump({"best_val_lesion_iou": best,', 'json.dump({"best_epoch": max(HIST, key=lambda h: h["score"])["epoch"] if HIST else None, "epochs_run": len(HIST), "seed": SEED, "best_val_lesion_iou": best,', 1)
    d = f"{OUT}/{slug}"; os.makedirs(d, exist_ok=True)
    open(f"{d}/{slug}.py", "w", encoding="utf-8").write(s)
    meta = {"id": f"carlychery2001/{slug}", "title": slug.replace("-", " ").upper().replace("PG26038", "PG26038"),
            "code_file": f"{slug}.py", "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "keywords": ["gpu"],
            "dataset_sources": [DATA[dis]], "kernel_sources": [], "competition_sources": [], "model_sources": [],
            "machine_shape": "NvidiaTeslaT4"}
    json.dump(meta, open(f"{d}/kernel-metadata.json", "w"), indent=2)
    return slug

runs = []
for dis in ("roya", "ojo"):
    for arch in ("deeplab", "unetpp"):
        for lrk, lr in LRS.items():
            runs.append(make(f"pg26038-mb-{dis}-{arch}-{lrk}", arch, dis, 60, 15, lr, 0, photometric=False))
for arch in ("deeplab", "unetpp"):
    for seed in (1, 2):
        runs.append(make(f"pg26038-rs-ojo-{arch}-seed{seed}", arch, "ojo", 30, 12, LRS["lrA"], seed))
json.dump(runs, open(f"{OUT}/queue.json", "w"), indent=1)
print(len(runs), "kernels:", *runs, sep="\n  ")
