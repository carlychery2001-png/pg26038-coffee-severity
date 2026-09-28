"""Shared data access, leaf-level aggregation, bootstrap and figure style for the revision figures
(paper Figures 4, 12, 13 and 14). Nothing here re-runs model inference.

Inputs (read only)
  <ROOT>/reanalysis/out/per_image.csv          per-image pixel counts from run_inference.py
  <ROOT>/reanalysis/out/preds/<model>/<pack>_<split>/<file>   predicted class maps (0/1/2), 1024 x 1024 letterbox
  <ROOT>/datasets/pg26038-{roya,ojo}-real-pack  images, masks, split files
  <ROOT>/reanalysis/run_inference.py            the reference-label rebuild (functions resize_pad and load) is
                                                        taken verbatim from this file via ast, so it cannot drift.

Leaf-level rules (as in the paper)
  rust (roya):       abaxial face only; diseased test leaves only (grade_label == 'roya'); reference grade from the
                     reference severity with cut-points 0.5 / 5 / 20 %.
  leaf spot (ojo):   IoU = mean of the two face IoUs, severity = mean of the two face severities; reference grade =
                     annotator's visual grade (grade_label G1/G2/G3).
  Severities are computed from the integer pixel counts (100 * lesion_px / leaf_px) rather than the 5-decimal
  rounded CSV columns; the band/grade assignment is asserted to be identical either way.
"""
from __future__ import annotations
import os as _os  # (release) all paths below are relative to ROOT (repository root by default; see README)
ROOT = (_os.environ.get("PG26038_ROOT") or _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))).replace(chr(92), "/")

import ast
import csv
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = Path(ROOT)
REANALYSIS = BASE / "reanalysis"
PER_IMAGE = REANALYSIS / "out" / "per_image.csv"
PREDS = REANALYSIS / "out" / "preds"
RUN_INFERENCE = REANALYSIS / "run_inference.py"
OUT = Path(__file__).resolve().parent

SEED = 20260927
B = 10_000

DISEASES = ["roya", "ojo"]
ARCHS = ["segformer", "unetpp", "deeplab"]
MODEL_NAME = {"segformer": "SegFormer-B5", "unetpp": "UNet++", "deeplab": "DeepLabV3+"}
PANEL = {
    "roya": ("(a) Coffee leaf rust", "Hemileia vastatrix"),
    "ojo": ("(b) American leaf spot", "Mycena citricolor"),
}
DISEASE_NAME = {"roya": "coffee leaf rust", "ojo": "American leaf spot"}

# Reproduced published values (task sanity checks), 3 / 2 decimals as printed in the paper.
PUBLISHED = {
    "leaf_iou": {"roya": {"segformer": 0.767, "unetpp": 0.647, "deeplab": 0.632},
                 "ojo": {"segformer": 0.884, "unetpp": 0.853, "deeplab": 0.815}},
    "mae": {"roya": {"segformer": 0.99, "unetpp": 1.77, "deeplab": 1.47},
            "ojo": {"segformer": 0.22, "unetpp": 0.15, "deeplab": 0.25}},
    "bias": {"roya": {"segformer": 0.99, "unetpp": 1.75, "deeplab": 1.41},
             "ojo": {"segformer": 0.16, "unetpp": -0.03, "deeplab": 0.08}},
    "qwk": {"roya": {"segformer": 0.88, "unetpp": 0.79, "deeplab": 0.88},
            "ojo": {"segformer": 0.82, "unetpp": 0.85, "deeplab": 0.84}},
    "loa": {"roya": {"segformer": (-1.68, 3.66), "unetpp": (-2.68, 6.17), "deeplab": (-3.07, 5.89)},
            "ojo": {"segformer": (-0.47, 0.79), "unetpp": (-0.50, 0.44), "deeplab": (-0.66, 0.82)}},
}

N_TEST_DISEASED = {"roya": 21, "ojo": 55}
N_HEALTHY_RUST_TEST_LOCAL = 21
N_HEALTHY_RUST_TEST_PAPER = 54


def grade_from_severity(s: float) -> int:
    return 0 if s < 0.5 else 1 if s < 5 else 2 if s < 20 else 3


def read_rows() -> list[dict]:
    with open(PER_IMAGE, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def leaf_table(rows: list[dict], disease: str, arch: str, split: str = "test") -> list[dict]:
    """Leaf-level records for the model's own disease pack, diseased leaves only, sorted by leaf_id."""
    sel = [r for r in rows if r["model_disease"] == disease and r["arch"] == arch and r["image_pack"] == disease
           and r["split"] == split and r["grade_label"] != "healthy"]
    faces: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in sel:
        assert r["face"] not in faces[r["leaf_id"]], ("duplicate face", r["file"])
        faces[r["leaf_id"]][r["face"]] = r
    out = []
    for lid in sorted(faces):
        f = faces[lid]
        assert set(f) == {"abaxial", "adaxial"}, (lid, set(f))

        def face_vals(r):
            inter, union = int(r["inter_lesion"]), int(r["union_lesion"])
            iou = inter / union
            assert abs(iou - float(r["lesion_iou"])) < 1e-6, r["file"]
            rs = 100 * int(r["ref_lesion_px"]) / int(r["ref_leaf_px"])
            ps = 100 * int(r["pred_lesion_px"]) / int(r["pred_leaf_px"])
            assert abs(rs - float(r["ref_sev"])) < 1e-4 and abs(ps - float(r["pred_sev"])) < 1e-4, r["file"]
            return dict(iou=iou, iou_exact=Fraction(inter, union), ref_sev=rs, pred_sev=ps,
                        ref_sev_csv=float(r["ref_sev"]), pred_sev_csv=float(r["pred_sev"]),
                        inter=inter, union=union, file=r["file"])

        ab, ad = face_vals(f["abaxial"]), face_vals(f["adaxial"])
        if disease == "roya":
            rec = dict(leaf_id=lid, iou=ab["iou"], iou_exact=ab["iou_exact"], ref_sev=ab["ref_sev"],
                       pred_sev=ab["pred_sev"], ref_sev_csv=ab["ref_sev_csv"],
                       ref_grade=grade_from_severity(ab["ref_sev"]),
                       ref_grade_csv=grade_from_severity(ab["ref_sev_csv"]))
        else:
            gl = f["abaxial"]["grade_label"]
            assert gl == f["adaxial"]["grade_label"] and gl in ("G1", "G2", "G3"), lid
            rec = dict(leaf_id=lid, iou=(ab["iou"] + ad["iou"]) / 2, iou_exact=(ab["iou_exact"] + ad["iou_exact"]) / 2,
                       ref_sev=(ab["ref_sev"] + ad["ref_sev"]) / 2, pred_sev=(ab["pred_sev"] + ad["pred_sev"]) / 2,
                       ref_sev_csv=(ab["ref_sev_csv"] + ad["ref_sev_csv"]) / 2,
                       ref_grade=int(gl[1]), ref_grade_csv=int(gl[1]))
        rec["pred_grade"] = grade_from_severity(rec["pred_sev"])
        rec["faces"] = {"abaxial": ab, "adaxial": ad}
        out.append(rec)
    assert len(out) == N_TEST_DISEASED[disease], (disease, arch, len(out))
    return out


def qwk(a, b, k: int = 4) -> float:
    a, b = np.asarray(a), np.asarray(b)
    o = np.bincount(k * a + b, minlength=k * k).reshape(k, k).astype(float)
    e = np.outer(o.sum(1), o.sum(0)) / o.sum()
    w = (np.subtract.outer(np.arange(k), np.arange(k)) ** 2) / (k - 1) ** 2
    return float(1 - (w * o).sum() / (w * e).sum())


def boot_indices(n: int, seed: int = SEED, b: int = B) -> np.ndarray:
    """Leaf resampling indices (b x n). A fresh generator with the fixed seed is used for every stratum, and the
    same indices are applied to all three models of that stratum (paired resampling)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(b, n))


def percentile_ci(samples: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


# --------------------------------------------------------------------------- reference-label rebuild (verbatim)
def _load_rebuild_namespace() -> dict:
    """Execute only the constants and the resize_pad/load functions of run_inference.py (no model, no jobs)."""
    src = RUN_INFERENCE.read_text(encoding="utf-8")
    tree = ast.parse(src)
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = {a.name for a in node.names}
            if names & {"onnxruntime"}:
                continue
            keep.append(node)
        elif isinstance(node, ast.Assign):
            targets = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if targets & {"BASE", "PACKS", "SIZE"}:
                keep.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in {"resize_pad", "load"}:
            keep.append(node)
    mod = ast.Module(body=keep, type_ignores=[])
    ns: dict = {}
    exec(compile(mod, str(RUN_INFERENCE), "exec"), ns)
    assert ns["SIZE"] == 1024 and callable(ns["load"]) and callable(ns["resize_pad"])
    return ns


_NS = None


def load_letterbox(pack: str, file: str):
    """(rgb uint8 1024x1024x3, label uint8 1024x1024 with 0 bg / 1 leaf / 2 lesion), exactly as run_inference.load()."""
    global _NS
    if _NS is None:
        _NS = _load_rebuild_namespace()
    return _NS["load"](pack, file)


def load_pred(disease: str, arch: str, pack: str, split: str, file: str) -> np.ndarray:
    from PIL import Image
    p = np.asarray(Image.open(PREDS / f"{disease}_{arch}" / f"{pack}_{split}" / file))
    assert p.shape == (1024, 1024) and p.dtype == np.uint8 and set(np.unique(p)) <= {0, 1, 2}, file
    return p


# --------------------------------------------------------------------------- style (as figure5 / rebuilt figure11)
INK = "#1a1a1a"
GREY_TXT = "#555555"
GREY_TICK = "#8a8a8a"
GREY_AXIS = "#c8c8c8"
STYLE = {
    "segformer": dict(color="#0072B2", marker="^", ms=7.0),
    "unetpp": dict(color="#009E73", marker="s", ms=6.2),
    "deeplab": dict(color="#E69F00", marker="o", ms=6.6),
}
CM = 1 / 2.54


def setup_style() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.edgecolor": GREY_AXIS,
        "axes.linewidth": 0.9,
        "xtick.color": GREY_AXIS,
        "ytick.color": GREY_AXIS,
        "xtick.labelcolor": GREY_TICK,
        "ytick.labelcolor": GREY_TICK,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "image.composite_image": False,
    })


def panel_title(ax, disease: str, dy_title: float = 17, dy_species: float = 5) -> None:
    title, species = PANEL[disease]
    ax.annotate(title, xy=(0, 1), xycoords="axes fraction", xytext=(0, dy_title), textcoords="offset points",
                ha="left", va="bottom", fontsize=10.2, fontweight="bold", color=INK)
    ax.annotate(species, xy=(0, 1), xycoords="axes fraction", xytext=(0, dy_species), textcoords="offset points",
                ha="left", va="bottom", fontsize=8.4, fontstyle="italic", color=GREY_TXT)


def save(fig, stem: str) -> dict:
    """Vector PDF (TrueType fonts) + 600 dpi PNG flattened to opaque RGB."""
    from PIL import Image
    pdf, png = OUT / f"{stem}.pdf", OUT / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=600)
    with Image.open(png) as im:
        rgb = Image.new("RGB", im.size, "white")
        rgb.paste(im, mask=im.convert("RGBA").getchannel("A"))
    rgb.save(png, dpi=(600, 600))
    return {"pdf": str(pdf).replace("\\", "/"), "png": str(png).replace("\\", "/")}
