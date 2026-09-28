"""PG 26038 - Kaggle kernel: FINE-TUNE the deployed roya lesion model to be robust to
LIGHT REFLECTION (specular glare), photometric variation, and out-of-domain leaves.

Starts from the existing checkpoint (roya_deeplab_mobilenet1024/best.pt) and continues
training with:
  1. synthetic GLARE + photometric augmentation on the existing labelled data
     (masks stay valid -> teaches 'reflection/highlight/colour-shift != lesion', no new labels)
  2. EXTRA data merged into TRAIN only:
       - RoCoLe healthy leaves  -> cutout + EMPTY lesion mask (hard negatives: glossy Robusta veins)
       - BRACOL rust leaves     -> cutout + lesion mask (extra real rust positives)
Everything else matches the base recipe so it stays comparable and the ONNX drops into infer.py:
  3 classes {0 bg, 1 leaf, 2 lesion}, 1024 letterbox, ImageNet norm,
  loss = 0.7*FocalTversky(a=.3 b=.7 g=.75) + 0.3*weighted-CE.
Fine-tune specifics: lower LR (enc 2e-5 / dec 2e-4), EPOCHS 30, cosine.
VAL/TEST stay your REAL split so you can confirm NO regression on your own domain; a separate
RoCoLe-healthy false-positive rate is also reported.

Kaggle inputs expected (add as datasets):
  - base dataset  : images/  masks/  split.csv         (carlychery2001/pg26038-ojo-balanced)
  - init ckpt     : best.pt   (the current roya_deeplab_mobilenet1024 checkpoint)
  - extra pack    : extra/images/  extra/masks/  extra_split.csv   (built locally, see prep script)
Writes best.pt + summary.json + unetpp_ojo_1024.onnx to /kaggle/working.
"""
import csv, glob, json, os, time, zipfile
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
INPUT = "/kaggle/input"
print("kaggle input:", os.listdir(INPUT) if os.path.isdir(INPUT) else "MISSING")
for zp in glob.glob(os.path.join(INPUT, "**", "*.zip"), recursive=True):
    try:
        zipfile.ZipFile(zp).extractall("/kaggle/working/data")
    except Exception as exc:
        print("zip extract failed:", zp, exc)


def _find(name, want_dir=False, must=True):
    for root in ("/kaggle/working/data", INPUT):
        for h in glob.glob(os.path.join(root, "**", name), recursive=True):
            if (os.path.isdir(h) if want_dir else os.path.isfile(h)):
                return h
    if must:
        raise FileNotFoundError(name)
    return None


images = _find("images", True)
masks = _find("masks", True)
split_csv = _find("split.csv")
def _find_ckpt():                                              # checkpoint may be named *best*.pt
    for pat in ("*best*.pt", "*.pt"):
        for root in ("/kaggle/working/data", INPUT):
            hits = [h for h in glob.glob(os.path.join(root, "**", pat), recursive=True) if os.path.isfile(h)]
            if hits:
                return sorted(hits, key=len)[0]
    raise FileNotFoundError("checkpoint .pt")


try:
    init_ckpt = _find_ckpt()                                   # optional warm-start ckpt
except Exception:
    init_ckpt = None
extra_split = _find("extra_split.csv", must=False)             # optional extra train pack
print("base images:", images, "| masks:", masks, "| split:", split_csv)
print("init ckpt:", init_ckpt, "| extra_split:", extra_split)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__} | device {DEV} | {torch.cuda.get_device_name(0) if DEV=='cuda' else 'CPU'}")

SIZE = 1024
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
NUM_CLASSES = 3


def _resize_pad(arr, size, nearest=False):
    im = Image.fromarray(arr); w, h = im.size; s = size / max(w, h)
    im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.NEAREST if nearest else Image.BILINEAR)
    canvas = Image.new(im.mode, (size, size), 0)
    canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return np.asarray(canvas)


# ---------------- augmentation: glare + photometric (train only) ----------------
def _photometric(rgb):
    """brightness / contrast / gamma / per-channel colour jitter â€” mild, domain-robustness."""
    x = rgb.astype(np.float32)
    x *= np.random.uniform(0.82, 1.18)                          # brightness
    m = x.mean()
    x = (x - m) * np.random.uniform(0.82, 1.18) + m            # contrast
    x = np.clip(x, 0, 255)
    x = 255.0 * (x / 255.0) ** np.random.uniform(0.8, 1.25)   # gamma
    x *= np.random.uniform(0.94, 1.06, size=3)                 # colour cast
    return np.clip(x, 0, 255).astype(np.uint8)


def _add_glare(rgb, leaf, n=(1, 4)):
    """Paint soft specular highlights on the LEAF (bright, low-saturation Gaussian blobs).
    Teaches the model that a glossy reflection is not a lesion. Labels are unchanged."""
    ys, xs = np.where(leaf)
    if ys.size < 50:
        return rgb
    H, W = rgb.shape[:2]
    out = rgb.astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    for _ in range(np.random.randint(n[0], n[1] + 1)):
        k = np.random.randint(ys.size); cy, cx = ys[k], xs[k]
        ry = np.random.uniform(0.02, 0.10) * H; rx = np.random.uniform(0.02, 0.10) * W
        g = np.exp(-(((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2))
        a = (np.random.uniform(0.35, 0.9) * g)[..., None]      # blend strength toward white
        tint = np.random.uniform(235, 255)                     # slightly warm/cool white
        out = out * (1 - a) + tint * a
    return np.clip(out, 0, 255).astype(np.uint8)


class LeafSet(Dataset):
    def __init__(self, items, train, glare=True):
        self.items = items; self.train = train; self.glare = glare

    def __len__(self):
        return len(self.items)

    def _label(self, ip, mp):
        rgba = np.asarray(Image.open(ip).convert("RGBA"))
        leaf = rgba[..., 3] > 8
        if mp and os.path.isfile(mp):
            les = (np.asarray(Image.open(mp).convert("L")) > 127) & leaf
        else:                                                   # empty mask -> pure negative (healthy)
            les = np.zeros(leaf.shape, bool)
        lab = np.zeros(leaf.shape, np.uint8); lab[leaf] = 1; lab[les] = 2
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
            rgb = np.ascontiguousarray(rgb); lab = np.ascontiguousarray(lab)
            if self.glare and np.random.rand() < 0.6:
                rgb = _add_glare(rgb, lab >= 1)
            if np.random.rand() < 0.8:
                rgb = _photometric(rgb)
        x = ((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1).copy()
        return torch.from_numpy(x), torch.from_numpy(lab.astype(np.int64).copy())


# ---------------- data (base split + extra train packs) ----------------
def read_rows():
    by = {"train": [], "val": [], "test": []}
    with open(split_csv, newline="") as fh:
        for r in csv.DictReader(fh):
            ip = os.path.join(images, r["file"]); mp = os.path.join(masks, r["file"])
            if os.path.isfile(ip) and os.path.isfile(mp):
                by.get(r["split"], by["train"]).append((r["file"], r["leaf_id"], r["grade"], ip, mp))
    return by


# Train ONLY on Carly's own real leaves (pg26038-roya-real-pack): real ROYA (lesion masks) +
# real HEALTHY (empty masks). No synthetic, no BRACOL, no RoCoLe. Val/test come from this pack's
# own split; roya rows -> lesion IoU, healthy rows -> false-positive check.
by = read_rows()
n_extra = 0
GR = lambda r: (r[2] or "").lower()
val_les = [r for r in by["val"] if "healthy" not in GR(r)]      # real ROYA -> lesion IoU
rocole_val = [r for r in by["val"] if "healthy" in GR(r)]       # real HEALTHY -> FP check
print(f"split -> train {len(by['train'])} | val_roya {len(val_les)} | val_healthy {len(rocole_val)} | "
      f"test {len(by['test'])} | roya_train {sum('healthy' not in GR(r) for r in by['train'])} "
      f"healthy_train {sum('healthy' in GR(r) for r in by['train'])}")

BATCH, ACCUM = 2, 16
tr_dl = DataLoader(LeafSet(by["train"], True, glare=False), batch_size=BATCH, shuffle=True,
                   num_workers=2, drop_last=True, pin_memory=True)


# ---------------- model + load init checkpoint ----------------
def build():
    os.system("pip -q install segmentation-models-pytorch >/dev/null 2>&1")
    import segmentation_models_pytorch as smp
    return smp.UnetPlusPlus(encoder_name="mobilenet_v2", encoder_weights="imagenet",
                            in_channels=3, classes=NUM_CLASSES)


class Net(nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, x):
        o = self.m(x); o = o["out"] if isinstance(o, dict) else o
        if o.shape[-1] != SIZE or o.shape[-2] != SIZE:
            o = F.interpolate(o, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
        return o


net = Net(build()).to(DEV)
if init_ckpt:                                    # warm-start the mobilenet_v2 ENCODER from the roya
    ck = torch.load(init_ckpt, map_location=DEV) # deeplab ckpt (UNet++ decoder differs -> re-learned)
    sd = ck.get("state_dict", ck); msd = net.state_dict()
    keep = {k: v for k, v in sd.items() if k in msd and v.shape == msd[k].shape}
    net.load_state_dict(keep, strict=False)
    print(f"warm-start encoder from {os.path.basename(init_ckpt)} | loaded {len(keep)}/{len(sd)} tensors")

enc = [p for n, p in net.named_parameters() if (".encoder." in n or ".backbone." in n)]
dec = [p for n, p in net.named_parameters() if not (".encoder." in n or ".backbone." in n)]
opt = torch.optim.AdamW([{"params": enc, "lr": 2e-5}, {"params": dec, "lr": 2e-4}], weight_decay=0.01)
EPOCHS = 30
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
scaler = torch.cuda.amp.GradScaler(enabled=(DEV == "cuda"))
ce_w = torch.tensor([0.25, 0.5, 3.0], device=DEV)


def focal_tversky(logits, y, a=0.3, b=0.7, gexp=0.75, eps=1e-6):
    p = torch.softmax(logits, 1)[:, 2]; t = (y == 2).float()
    tp = (p * t).sum(); fp = (p * (1 - t)).sum(); fn = ((1 - p) * t).sum()
    return (1 - (tp + eps) / (tp + a * fp + b * fn + eps)) ** gexp


def loss_fn(logits, y):
    return 0.7 * focal_tversky(logits, y) + 0.3 * F.cross_entropy(logits, y, weight=ce_w)


@torch.no_grad()
def evaluate(items):
    net.eval(); inter = np.zeros(NUM_CLASSES); union = np.zeros(NUM_CLASSES); dsum = 0.0; dn = 0; per_leaf = {}
    dl = DataLoader(LeafSet(items, False), batch_size=BATCH, shuffle=False, num_workers=2); idx = 0
    for x, y in dl:
        with torch.cuda.amp.autocast(enabled=(DEV == "cuda")):
            logit = net(x.to(DEV))
        pred = logit.argmax(1).cpu().numpy(); y = y.numpy()
        for c in range(NUM_CLASSES):
            inter[c] += np.logical_and(pred == c, y == c).sum(); union[c] += np.logical_or(pred == c, y == c).sum()
        pl = (pred == 2); yl = (y == 2)
        for j in range(pred.shape[0]):
            i2, u2 = np.logical_and(pl[j], yl[j]).sum(), (pl[j].sum() + yl[j].sum())
            dsum += (2 * i2 / u2) if u2 else 1.0; dn += 1
            leaf = (pred[j] >= 1).sum(); les = (pred[j] == 2).sum()
            _, lid, grade, _, _ = items[idx]
            per_leaf.setdefault(lid, {"g": grade, "sev": []})["sev"].append(100.0 * les / leaf if leaf else 0.0)
            idx += 1
    iou = inter / np.maximum(union, 1)
    return dict(lesion_iou=float(iou[2]), leaf_iou=float(iou[1]), miou=float(iou[1:].mean()),
                dice=float(dsum / max(dn, 1)), per_leaf=per_leaf)


@torch.no_grad()
def fp_rate(items):
    """False-positive severity on RoCoLe healthy negatives (predicted diseased = sev>=5%)."""
    if not items:
        return float("nan"), float("nan")
    net.eval(); dl = DataLoader(LeafSet(items, False), batch_size=BATCH, shuffle=False, num_workers=2); sevs = []
    for x, _ in dl:
        with torch.cuda.amp.autocast(enabled=(DEV == "cuda")):
            pred = net(x.to(DEV)).argmax(1).cpu().numpy()
        for j in range(pred.shape[0]):
            leaf = (pred[j] >= 1).sum(); les = (pred[j] == 2).sum()
            sevs.append(100.0 * les / leaf if leaf else 0.0)
    sevs = np.array(sevs)
    return float(np.median(sevs)), float((sevs >= 5).mean())


# ---------------- fine-tune ----------------
OUT = "/kaggle/working"
best = -2.0; best_state = None; patience, bad = 12, 0; t0 = time.time()   # score in [-1,1]; -2 => always save once
for ep in range(1, EPOCHS + 1):
    net.train(); tl = 0.0; opt.zero_grad()
    for i, (x, y) in enumerate(tr_dl):
        with torch.cuda.amp.autocast(enabled=(DEV == "cuda")):
            loss = loss_fn(net(x.to(DEV)), y.to(DEV)) / ACCUM
        scaler.scale(loss).backward(); tl += float(loss) * ACCUM
        if (i + 1) % ACCUM == 0:
            scaler.step(opt); scaler.update(); opt.zero_grad()
    sched.step()
    va = evaluate(val_les); fp_med, fp_rt = fp_rate(rocole_val)
    score = va["lesion_iou"] - fp_rt                  # want high lesion-IoU AND low healthy false-positive
    print(f"  ep {ep:3d}/{EPOCHS} loss={tl/max(1,len(tr_dl)):.4f} | val lesion-IoU={va['lesion_iou']:.3f} "
          f"Dice={va['dice']:.3f} | RoCoLe-healthy FP={fp_rt:.3f} (sevmed={fp_med:.1f}%) | score={score:.3f} "
          f"[{(time.time()-t0)/60:.0f}m]", flush=True)
    if score > best:
        best = score; bad = 0
        best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        torch.save({"state_dict": best_state, "val_lesion_iou": va["lesion_iou"], "val_fp": fp_rt,
                    "score": best, "arch": "UnetPlusPlus-MobileNetV2-1024"}, os.path.join(OUT, "best.pt"))
    else:
        bad += 1
        if bad >= patience:
            print(f"early stop at epoch {ep}"); break

# ---- test on best (YOUR real roya/healthy test split) ----
if best_state:
    net.load_state_dict(best_state)
test_roya = [r for r in by["test"] if "healthy" not in GR(r)]
test_heal = [r for r in by["test"] if "healthy" in GR(r)]
te = evaluate(test_roya if test_roya else val_les)              # lesion IoU on real roya test leaves
fp_med, fp_rt = fp_rate(test_heal if test_heal else rocole_val)  # FP on real healthy test leaves
print(f"\nTEST(real roya) lesion-IoU {te['lesion_iou']:.3f} | mIoU {te['miou']:.3f} | Dice {te['dice']:.3f}")
print(f"TEST(real healthy) FP rate {fp_rt:.3f} | median predicted severity {fp_med:.1f}%")

json.dump({"best_val_lesion_iou": best, "test_lesion_iou": te["lesion_iou"], "test_miou": te["miou"],
           "test_dice": te["dice"], "test_healthy_fp_rate": fp_rt, "test_healthy_sev_median": fp_med,
           "n_train": len(by["train"]), "n_test_roya": len(test_roya), "n_test_healthy": len(test_heal),
           "arch": "unetpp_mobilenet_1024_ojo"},
          open(os.path.join(OUT, "summary.json"), "w"), indent=2)

# ---- ONNX (drops into infer.py) ----
try:
    net.eval(); dummy = torch.zeros(1, 3, SIZE, SIZE, device=DEV)
    torch.onnx.export(net, (dummy,), os.path.join(OUT, "unetpp_ojo_1024.onnx"),
                      input_names=["pixel_values"], output_names=["logits"],
                      dynamic_axes={"pixel_values": {0: "n"}, "logits": {0: "n"}}, opset_version=16, dynamo=False)
    print("ONNX exported: unetpp_ojo_1024.onnx")
except Exception as exc:
    print("ONNX export skipped:", str(exc)[:200])
print("DONE:", os.listdir(OUT))
