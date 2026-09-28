"""Identify which dataset leaf each expert-form image (ojo01..roya30) shows.
Form pages: abaxial (left) + adaxial (right) on white, stems trimmed. Dataset scans: black background.
Everything is downscaled to 256 px tall first; matching uses the leaf blade (stem removed),
testing flips / 180-degree rotation and swapped sides."""
import os as _os  # (release) all paths below are relative to ROOT (repository root by default; see README)
ROOT = (_os.environ.get("PG26038_ROOT") or _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))).replace(chr(92), "/")
import glob, os, re, csv, time
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
DS = (ROOT + r"/datasets")
H = 256
S = (48, 96)

def small(im):
    im = im.convert("RGB"); w, h = im.size
    return np.asarray(im.resize((max(1, round(w * H / h)), H), Image.BILINEAR))

def blade(fg):
    op = ndi.binary_opening(fg, structure=np.ones((9, 9)))
    lab, n = ndi.label(op)
    if n == 0: return None
    m = lab == 1 + int(np.argmax(ndi.sum(op, lab, range(1, n + 1))))
    return ndi.binary_fill_holes(m)

def desc(a, fg):
    m = blade(fg)
    if m is None or m.sum() < 300: return None
    ys, xs = np.nonzero(m); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    g = a[y0:y1, x0:x1].astype(np.float32).mean(2); mm = m[y0:y1, x0:x1]
    g = (g - g[mm].mean()) / (g[mm].std() + 1e-6) * mm
    gi = np.asarray(Image.fromarray(g).resize(S, Image.BILINEAR))
    mi = np.asarray(Image.fromarray((mm * 255).astype(np.uint8)).resize(S, Image.BILINEAR), np.float32) / 255 > .5
    return gi, mi, (x1 - x0) / (y1 - y0)

def variants(d):
    g, m, ar = d
    return [(f(g), f(m), ar) for f in (lambda x: x, np.fliplr, np.flipud, lambda x: np.rot90(x, 2))]

def sim(a, b):
    (g1, m1, _), (g2, m2, _) = a, b
    both = m1 & m2; iou = both.sum() / ((m1 | m2).sum() + 1e-6)
    x, y = g1[both] - g1[both].mean(), g2[both] - g2[both].mean()
    return 0.5 * (x @ y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-6) + 0.5 * iou

t0 = time.time()
pool = {}
for sub in ["pg26038-ojo-lesions", "pg26038-sam2-roya", "pg26038-roya-real-pack", "pg26038-ojo-real-pack", "pg26038-sam2-ojo", "pg26038-ojo-balanced", "pg26038-roya-grade-pack-v2"]:
    for f in glob.glob(f"{DS}/{sub}/images/*.png"):
        m = re.match(r"(\d{8}_.+?_\d{3})__(abaxial|adaxial)\.png$", os.path.basename(f))
        if m: pool.setdefault(m.group(1), {}).setdefault(m.group(2), f)
D = {}
for k, v in sorted(pool.items()):
    if len(v) < 2: continue
    ds = [desc(a, a.max(2) > 25) for a in (small(Image.open(v["abaxial"])), small(Image.open(v["adaxial"])))]
    if None not in ds: D[k] = [variants(d) for d in ds]
print(f"candidate leaves: {len(D)}  ({time.time()-t0:.0f}s)", flush=True)

rows = []
for fp in sorted(glob.glob("imagenes/*.jpg")):
    fid = os.path.basename(fp)[:-4]
    a = np.asarray(Image.open(fp).convert("RGB")); w = a.shape[1]; top = a[70:940]
    fab, fad = [desc(s, s.min(2) < 225) for s in (small(Image.fromarray(top[:, : w // 2])), small(Image.fromarray(top[:, w // 2 :])))]
    sc = []
    for k, (vab, vad) in D.items():
        s = max(max((sim(fab, vab[i]) + sim(fad, vad[i])) / 2 for i in range(4)),
                max((sim(fab, vad[i]) + sim(fad, vab[i])) / 2 for i in range(4)))
        sc.append((s, k))
    sc.sort(reverse=True)
    (s1, k1), (s2, k2) = sc[0], sc[1]
    rows.append([fid, k1, round(s1, 4), k2, round(s2, 4), round(s1 - s2, 4)])
    print(f"{fid}: {k1}  {s1:.3f} | 2nd {k2} {s2:.3f} | margin {s1-s2:.3f}", flush=True)
with open("form_image_key.csv", "w", newline="") as fo:
    wr = csv.writer(fo); wr.writerow(["form_id", "leaf_id", "score", "second_best", "second_score", "margin"]); wr.writerows(rows)
print(f"done in {time.time()-t0:.0f}s")
