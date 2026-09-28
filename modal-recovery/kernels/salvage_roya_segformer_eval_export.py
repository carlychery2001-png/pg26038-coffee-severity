"""Salvage: the Kaggle-trained roya SegFormer-B5 (100% real pack, val IoU 0.619)
died before test-eval/export. Load its best.pt, evaluate on the real held-out
test split, export ONNX. Runs on Colab A100 (paths under /content)."""
import csv, glob, json, os
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
INPUT = "/content/input"
OUT = "/content/working"
os.makedirs(OUT, exist_ok=True)

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SIZE = 1024
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
NUM_CLASSES = 3
BATCH = 1


def _find(name, want_dir=False):
    for h in glob.glob(os.path.join(INPUT, "**", name), recursive=True):
        if os.path.isdir(h) if want_dir else os.path.isfile(h):
            return h
    raise FileNotFoundError(name)


images, masks, split_csv = _find("images", True), _find("masks", True), _find("split.csv")
ckpt_path = _find("best.pt")
print("data:", images, "| ckpt:", ckpt_path)


def resize_pad(arr, size, nearest=False):
    im = Image.fromarray(arr); w, h = im.size; s = size / max(w, h)
    im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.NEAREST if nearest else Image.BILINEAR)
    canvas = Image.new(im.mode, (size, size), 0)
    canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return np.asarray(canvas)


class EvalSet(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        fn, grade, ip, mp = self.items[i]
        rgba = np.asarray(Image.open(ip).convert("RGBA"))
        leaf = rgba[..., 3] > 8
        les = ((np.asarray(Image.open(mp).convert("L")) > 127) & leaf) if (mp and os.path.isfile(mp)) else np.zeros(leaf.shape, bool)
        lab = np.zeros(leaf.shape, np.uint8); lab[leaf] = 1; lab[les] = 2
        rgb = resize_pad(rgba[..., :3], SIZE); lab = resize_pad(lab, SIZE, nearest=True)
        x = ((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1).copy()
        return torch.from_numpy(x), torch.from_numpy(lab.astype(np.int64).copy())


rows = {"train": [], "val": [], "test": []}
for r in csv.DictReader(open(split_csv, newline="")):
    ip = os.path.join(images, r["file"]); mp = os.path.join(masks, r["file"])
    if os.path.isfile(ip):
        rows.setdefault(r["split"], rows["train"]).append(
            (r["file"], (r.get("grade", "") or "").lower(), ip, mp if os.path.isfile(mp) else None))
test_les = [r for r in rows["test"] if "healthy" not in r[1]]
test_heal = [r for r in rows["test"] if "healthy" in r[1]]
print(f"test roya {len(test_les)} | test healthy {len(test_heal)}")

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b5", num_labels=NUM_CLASSES, ignore_mismatched_sizes=True).to(DEV)
ck = torch.load(ckpt_path, map_location=DEV, weights_only=False)
model.load_state_dict(ck["state_dict"])
val_iou = ck.get("val_lesion_iou")
print("checkpoint loaded | val_lesion_iou:", val_iou)


def up(logits):
    return F.interpolate(logits, size=(SIZE, SIZE), mode="bilinear", align_corners=False)


@torch.no_grad()
def evaluate(items):
    model.eval()
    inter = np.zeros(NUM_CLASSES); union = np.zeros(NUM_CLASSES); dsum = 0.0; dn = 0
    for x, y in DataLoader(EvalSet(items), batch_size=BATCH, num_workers=2):
        with torch.cuda.amp.autocast(enabled=(DEV == "cuda")):
            logit = up(model(pixel_values=x.to(DEV)).logits)
        pred = logit.argmax(1).cpu().numpy(); y = y.numpy()
        for c in range(NUM_CLASSES):
            inter[c] += np.logical_and(pred == c, y == c).sum()
            union[c] += np.logical_or(pred == c, y == c).sum()
        pl = (pred == 2); yl = (y == 2)
        for j in range(pred.shape[0]):
            i2, u2 = np.logical_and(pl[j], yl[j]).sum(), (pl[j].sum() + yl[j].sum())
            dsum += (2 * i2 / u2) if u2 else 1.0; dn += 1
    iou = inter / np.maximum(union, 1)
    return dict(lesion_iou=float(iou[2]), leaf_iou=float(iou[1]),
                miou=float(iou[1:].mean()), dice=float(dsum / max(dn, 1)))


@torch.no_grad()
def fp_rate(items):
    if not items:
        return float("nan"), 0.0
    model.eval(); sevs = []
    for x, _ in DataLoader(EvalSet(items), batch_size=BATCH, num_workers=2):
        with torch.cuda.amp.autocast(enabled=(DEV == "cuda")):
            pred = up(model(pixel_values=x.to(DEV)).logits).argmax(1).cpu().numpy()
        for j in range(pred.shape[0]):
            leaf = (pred[j] >= 1).sum(); les = (pred[j] == 2).sum()
            sevs.append(100.0 * les / leaf if leaf else 0.0)
    sevs = np.array(sevs)
    return float(np.median(sevs)), float((sevs >= 5).mean())


te = evaluate(test_les)
fp_med, fp_rt = fp_rate(test_heal)
print(f"TEST(real roya) lesion-IoU {te['lesion_iou']:.3f} | mIoU {te['miou']:.3f} | Dice {te['dice']:.3f}")
print(f"TEST(real healthy) FP {fp_rt:.3f} | sev median {fp_med:.2f}%")

json.dump({"best_val_lesion_iou": val_iou, "test_lesion_iou": te["lesion_iou"],
           "test_miou": te["miou"], "test_dice": te["dice"], "test_leaf_iou": te["leaf_iou"],
           "test_healthy_fp_rate": fp_rt, "test_healthy_sev_median": fp_med,
           "n_test_roya": len(test_les), "n_test_healthy": len(test_heal),
           "arch": "SegFormer-MiT-B5-1024-roya-realpack", "input": SIZE,
           "note": "salvaged from Kaggle ckpt; trained 100% real roya pack"},
          open(os.path.join(OUT, "summary.json"), "w"), indent=2)

model.eval()
dummy = torch.zeros(1, 3, SIZE, SIZE, device=DEV)
kw = dict(input_names=["pixel_values"], output_names=["logits"],
          dynamic_axes={"pixel_values": {0: "n"}, "logits": {0: "n"}}, opset_version=17)
try:
    torch.onnx.export(model, (dummy,), os.path.join(OUT, "roya_segformer_realpack_1024.onnx"), dynamo=False, **kw)
except TypeError:
    torch.onnx.export(model, (dummy,), os.path.join(OUT, "roya_segformer_realpack_1024.onnx"), **kw)
print("ONNX exported: roya_segformer_realpack_1024.onnx")
print("DONE:", os.listdir(OUT))
