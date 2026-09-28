"""PG 26038 - Kaggle kernel: SegFormer-B5 at 1024px on the SAME balanced dataset + split
as the other seg models (carlychery2001/pg26038-ojo-balanced).

Same recipe/loss as lesion_segformer_b5_kernel.py -- only the input resolution grows to 1024
(finer native grid -> sharper lesion boundaries, like B3 v3 did vs B3 v1). MiT-B5 (~82M) at
1024 is very heavy: batch 1 x accum 32 (effective 32) + gradient checkpointing to fit a T4.
3 classes {0 bg, 1 leaf, 2 lesion}, ImageNet norm, FocalTversky+wCE, AdamW enc6e-5/dec6e-4
cosine, FP16 AMP. split.csv respected (train = real+synthetic, val/test = REAL only).
Writes best.pt + summary.json + ONNX (roya_segformer_realpack_1024.onnx, 1x3x1024x1024, dynamo=False).
"""
import csv, glob, json, os, time, zipfile
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
INPUT = "/content/input"
print("kaggle input:", os.listdir(INPUT) if os.path.isdir(INPUT) else "MISSING")
for zp in glob.glob(os.path.join(INPUT, "**", "*.zip"), recursive=True):
    try:
        zipfile.ZipFile(zp).extractall("/content/working/data")
    except Exception as exc:
        print("zip extract failed:", zp, exc)


def _find(name, want_dir=False):
    for root in ("/content/working/data", INPUT):
        for h in glob.glob(os.path.join(root, "**", name), recursive=True):
            if (os.path.isdir(h) if want_dir else os.path.isfile(h)):
                return h
    return None


images = _find("images", True)
masks = _find("masks", True)
split_csv = _find("split.csv")
print("images:", images, "| masks:", masks, "| split:", split_csv)
assert images and masks and split_csv, "dataset not found under /content/input"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation

DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__} | device {DEV} | {torch.cuda.get_device_name(0) if DEV=='cuda' else 'CPU'}")
import transformers as _tf
print("transformers", _tf.__version__)

SIZE = 1024
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
NUM_CLASSES = 3  # 0 bg, 1 leaf, 2 lesion


def read_split():
    by = {"train": [], "val": [], "test": []}
    with open(split_csv, newline="") as fh:
        for r in csv.DictReader(fh):
            ip = os.path.join(images, r["file"]); mp = os.path.join(masks, r["file"])
            if os.path.isfile(ip) and os.path.isfile(mp):
                by.get(r["split"], by["train"]).append((r["file"], r["leaf_id"], r["grade"], ip, mp))
    return by


def _resize_pad(arr, size, nearest=False):
    im = Image.fromarray(arr)
    w, h = im.size
    s = size / max(w, h)
    im = im.resize((max(1, round(w * s)), max(1, round(h * s))),
                   Image.NEAREST if nearest else Image.BILINEAR)
    canvas = Image.new(im.mode, (size, size), 0)
    canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return np.asarray(canvas)


class LeafSet(Dataset):
    def __init__(self, items, train):
        self.items = items; self.train = train

    def __len__(self):
        return len(self.items)

    def _label(self, ip, mp):
        rgba = np.asarray(Image.open(ip).convert("RGBA"))
        leaf = rgba[..., 3] > 8
        les = (np.asarray(Image.open(mp).convert("L")) > 127) & leaf
        lab = np.zeros(leaf.shape, np.uint8)
        lab[leaf] = 1; lab[les] = 2
        return rgba[..., :3], lab

    def __getitem__(self, i):
        _, _, _, ip, mp = self.items[i]
        rgb, lab = self._label(ip, mp)
        rgb = _resize_pad(rgb, SIZE); lab = _resize_pad(lab, SIZE, nearest=True)
        if self.train:
            if np.random.rand() < 0.5: rgb, lab = rgb[:, ::-1], lab[:, ::-1]
            if np.random.rand() < 0.5: rgb, lab = rgb[::-1], lab[::-1]
            k = np.random.randint(4)
            if k: rgb, lab = np.rot90(rgb, k), np.rot90(lab, k)
        x = ((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1).copy()
        return torch.from_numpy(x), torch.from_numpy(lab.astype(np.int64).copy())


by = read_split()
n_syn = sum(1 for it in by["train"] if it[0].startswith("synth_"))
print(f"split -> train {len(by['train'])} (synth {n_syn}) | val {len(by['val'])} | test {len(by['test'])}")

BATCH, ACCUM = 1, 32          # effective 32 (B5 @1024 is very heavy)
tr_dl = DataLoader(LeafSet(by["train"], True), batch_size=BATCH, shuffle=True,
                   num_workers=2, drop_last=True, pin_memory=True)


def build():
    for src in ("nvidia/mit-b5", "nvidia/segformer-b5-finetuned-ade-640-640"):
        try:
            m = SegformerForSemanticSegmentation.from_pretrained(
                src, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True)
            print("loaded pretrained:", src)
            return m
        except Exception as exc:
            print("could not load", src, "->", str(exc)[:120])
    from transformers import SegformerConfig
    print("WARNING: random-init MiT-B5 (no pretrained).")
    return SegformerForSemanticSegmentation(SegformerConfig(
        num_labels=NUM_CLASSES, depths=[3, 6, 40, 3], hidden_sizes=[64, 128, 320, 512],
        decoder_hidden_size=768))


model = build().to(DEV)
try:
    model.gradient_checkpointing_enable()
    print("gradient checkpointing ON")
except Exception as exc:
    print("grad-checkpointing unavailable:", str(exc)[:80])
enc = [p for n, p in model.named_parameters() if "decode_head" not in n]
dec = [p for n, p in model.named_parameters() if "decode_head" in n]
opt = torch.optim.AdamW([{"params": enc, "lr": 6e-5}, {"params": dec, "lr": 6e-4}], weight_decay=0.01)
EPOCHS = 60
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
scaler = torch.cuda.amp.GradScaler(enabled=(DEV == "cuda"))
ce_w = torch.tensor([0.25, 0.5, 3.0], device=DEV)


def up(logits):
    return F.interpolate(logits, size=(SIZE, SIZE), mode="bilinear", align_corners=False)


def focal_tversky(logits, y, a=0.3, b=0.7, gexp=0.75, eps=1e-6):
    p = torch.softmax(logits, 1)[:, 2]
    t = (y == 2).float()
    tp = (p * t).sum(); fp = (p * (1 - t)).sum(); fn = ((1 - p) * t).sum()
    ti = (tp + eps) / (tp + a * fp + b * fn + eps)
    return (1 - ti) ** gexp


def loss_fn(logits, y):
    return 0.7 * focal_tversky(logits, y) + 0.3 * F.cross_entropy(logits, y, weight=ce_w)


@torch.no_grad()
def evaluate(items):
    model.eval()
    inter = np.zeros(NUM_CLASSES); union = np.zeros(NUM_CLASSES); dsum = 0.0; dn = 0
    per_leaf = {}
    dl = DataLoader(LeafSet(items, False), batch_size=BATCH, shuffle=False, num_workers=2)
    idx = 0
    for x, y in dl:
        with torch.cuda.amp.autocast(enabled=(DEV == "cuda")):
            logit = up(model(pixel_values=x.to(DEV)).logits)
        pred = logit.argmax(1).cpu().numpy(); y = y.numpy()
        for c in range(NUM_CLASSES):
            pc = pred == c; yc = y == c
            inter[c] += np.logical_and(pc, yc).sum(); union[c] += np.logical_or(pc, yc).sum()
        pl = (pred == 2); yl = (y == 2)
        for j in range(pred.shape[0]):
            i2, u2 = np.logical_and(pl[j], yl[j]).sum(), (pl[j].sum() + yl[j].sum())
            dsum += (2 * i2 / u2) if u2 else 1.0; dn += 1
            leaf = (pred[j] >= 1).sum(); les = (pred[j] == 2).sum()
            _, lid, grade, _, _ = items[idx]
            sev = 100.0 * les / leaf if leaf else 0.0
            per_leaf.setdefault(lid, {"g": grade, "sev": []})["sev"].append(sev)
            idx += 1
    iou = inter / np.maximum(union, 1)
    return dict(lesion_iou=float(iou[2]), leaf_iou=float(iou[1]),
                miou=float(iou[1:].mean()), dice=float(dsum / max(dn, 1)), per_leaf=per_leaf)


OUT = "/content/working"
best = -1.0; best_state = None; patience, bad = 15, 0
t0 = time.time()
for ep in range(1, EPOCHS + 1):
    model.train(); tl = 0.0; opt.zero_grad()
    for i, (x, y) in enumerate(tr_dl):
        with torch.cuda.amp.autocast(enabled=(DEV == "cuda")):
            logit = up(model(pixel_values=x.to(DEV)).logits)
            loss = loss_fn(logit, y.to(DEV)) / ACCUM
        scaler.scale(loss).backward(); tl += float(loss) * ACCUM
        if (i + 1) % ACCUM == 0:
            scaler.step(opt); scaler.update(); opt.zero_grad()
    sched.step()
    va = evaluate(by["val"])
    print(f"  epoch {ep:3d}/{EPOCHS}  loss={tl/max(1,len(tr_dl)):.4f}  "
          f"val lesion-IoU={va['lesion_iou']:.3f}  mIoU={va['miou']:.3f}  Dice={va['dice']:.3f}  "
          f"[{(time.time()-t0)/60:.0f}m]")
    if va["lesion_iou"] > best:
        best = va["lesion_iou"]; bad = 0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save({"state_dict": best_state, "val_lesion_iou": best, "arch": "SegFormer-MiT-B5-1024-roya-realpack"},
                   os.path.join(OUT, "best.pt"))
    else:
        bad += 1
        if bad >= patience:
            print(f"early stop at epoch {ep} (no val gain in {patience})"); break
    if (time.time() - t0) / 3600 > 8.0:
        print("wall-clock guard: stopping"); break

if best_state:
    model.load_state_dict(best_state)
te = evaluate(by["test"])
order = {"G1": 0, "G2": 1, "G3": 2}


def grade(s):
    return 0 if s < 5 else (1 if s < 20 else 2)


yt, yp = [], []
for lid, d in te["per_leaf"].items():
    yt.append(order[d["g"]]); yp.append(grade(max(d["sev"])))
yt, yp = np.array(yt), np.array(yp)
acc = float((yt == yp).mean()) if len(yt) else 0.0
try:
    from sklearn.metrics import cohen_kappa_score, confusion_matrix
    qwk = float(cohen_kappa_score(yt, yp, weights="quadratic")); cm = confusion_matrix(yt, yp, labels=[0, 1, 2]).tolist()
except Exception:
    qwk = float("nan"); cm = [[0]*3 for _ in range(3)]
print(f"\nTEST  lesion-IoU {te['lesion_iou']:.3f} | mIoU {te['miou']:.3f} | Dice {te['dice']:.3f}")
print(f"TEST grade  acc {acc:.3f} | QWK {qwk:.3f} | confusion {cm}")

json.dump({"best_val_lesion_iou": best, "test_lesion_iou": te["lesion_iou"],
           "test_miou": te["miou"], "test_dice": te["dice"], "test_leaf_iou": te["leaf_iou"],
           "test_grade_acc": acc, "test_qwk": qwk, "confusion": cm,
           "n_train": len(by["train"]), "n_val": len(by["val"]), "n_test": len(by["test"]),
           "n_synth_train": n_syn, "arch": "SegFormer-MiT-B5-1024-roya-realpack", "input": SIZE},
          open(os.path.join(OUT, "summary.json"), "w"), indent=2)

try:
    model.eval()
    dummy = torch.zeros(1, 3, SIZE, SIZE, device=DEV)
    kw = dict(input_names=["pixel_values"], output_names=["logits"],
              dynamic_axes={"pixel_values": {0: "n"}, "logits": {0: "n"}}, opset_version=17)
    try:
        torch.onnx.export(model, (dummy,), os.path.join(OUT, "roya_segformer_realpack_1024.onnx"), dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(model, (dummy,), os.path.join(OUT, "roya_segformer_realpack_1024.onnx"), **kw)
    print("ONNX exported: roya_segformer_realpack_1024.onnx")
except Exception as exc:
    print("ONNX export skipped:", str(exc)[:200])

print("DONE. /content/working:", os.listdir(OUT))
