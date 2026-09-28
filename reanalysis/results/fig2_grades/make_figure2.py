"""Figure 2 (grade exemplars) for the PG-26038 MDPI Agriculture revision.

Two rows (coffee leaf rust, American leaf spot) x four ordinal grades (G0-G3).
Each cell shows one leaf face on white (alpha compositing), the reference lesion
boundary as a thin magenta vector outline, and the face's reference severity.

Reference severity (per face) = lesion px / leaf px, with leaf = alpha > 8 and
lesion = mask > 127 AND leaf, computed on the images and masks of the local
training packs (pg26038-roya-real-pack, pg26038-ojo-real-pack) at their native
resolution (no letterbox).

Display source: the rust-pack images are 1024 px tall. For print sharpness the
leaf picture is drawn from the 2048 px copy of the SAME scan (rust leaves:
pg26038-sam2-roya/images; healthy leaves: pg26038-ojo-lesions/images). The
script checks that a Lanczos 2x downscale of each 2048 px image reproduces the
pack image (leaf-pixel mean |diff| and alpha agreement are written to
results.json). The outline is always the pack mask, drawn as a vector contour in
the pack's pixel frame, so the boundary and the printed severity are exactly the
reference.

Outputs (this folder): figure2.pdf (vector text/outlines, TrueType fonts,
native-resolution leaf images), figure2.png (600 dpi, opaque white), results.json.
"""
from __future__ import annotations
import os as _os  # (release) all paths below are relative to ROOT (repository root by default; see README)
ROOT = (_os.environ.get("PG26038_ROOT") or _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))).replace(chr(92), "/")

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

OUT = Path(__file__).resolve().parent
DS = Path((ROOT + "/datasets"))
PACK = {
    "rust": DS / "pg26038-roya-real-pack",
    "leafspot": DS / "pg26038-ojo-real-pack",
}
SPLIT = {
    "rust": PACK["rust"] / "split.csv",
    "leafspot": PACK["leafspot"] / "meta" / "split.csv",
}
HIRES = {  # 2048 px copies of the same scans (display only)
    "Hvastatrix": DS / "pg26038-sam2-roya" / "images",
    "Healthy": DS / "pg26038-ojo-lesions" / "images",
    "Mcitricolor": None,  # leaf-spot pack is already 2048 px
}

# ---------------------------------------------------------------- selection
GRADES = [
    ("G0", "healthy or trace, <0.5%", 0.0, 0.5),
    ("G1", "mild, 0.5 to <5%", 0.5, 5.0),
    ("G2", "moderate, 5 to <20%", 5.0, 20.0),
    ("G3", "severe, \u226520%", 20.0, 100.0),
]
ROWS = [
    ("rust", "(a) Coffee leaf rust", "Hemileia vastatrix"),
    ("leafspot", "(b) American leaf spot", "Mycena citricolor"),
]
# pack -> file per grade. G0 cells use healthy control leaves: the healthy set is
# stored in the rust pack but was collected at EARTH (Guacimo), the leaf-spot
# site, and serves as the healthy (G0) set for both diseases. The leaf-spot pack
# has no healthy leaves. Two different healthy leaves are used for the two rows.
SELECTED = {
    ("rust", "G0"): ("rust", "20260612_Healthy_051__abaxial.png"),
    ("rust", "G1"): ("rust", "20260819_Hvastatrix_GX_130__abaxial.png"),
    ("rust", "G2"): ("rust", "20260819_Hvastatrix_GX_114__abaxial.png"),
    ("rust", "G3"): ("rust", "20260819_Hvastatrix_GX_119__abaxial.png"),
    ("leafspot", "G0"): ("rust", "20260615_Healthy_117__abaxial.png"),
    ("leafspot", "G1"): ("leafspot", "20260621_Mcitricolor_GX_008__abaxial.png"),
    ("leafspot", "G2"): ("leafspot", "20260628_Mcitricolor_GX_014__abaxial.png"),
    ("leafspot", "G3"): ("leafspot", "20260628_Mcitricolor_GX_130__abaxial.png"),
}
# Selection rule (G1-G3): train/val faces only (the test split is withheld pending
# peer review, manuscript 06_backmatter.tex). Faces in the band (leaf spot: annotator
# grade equal to the band) ranked by |abaxial severity - band target|; the nearest
# face without lesion mask on the petiole end or the apex (screened with the end
# zones = top/bottom 8% of the leaf's vertical extent) and with typical lesions on
# visual inspection was chosen. Every face nearer the target than the chosen one is
# listed in PASSED_OVER with the reason; main() asserts the list is complete.
EXCLUDED_SPLITS = {"test"}
END_ZONE_FRAC = 0.08
MARGIN_BAND_PX = 3  # lesion component "touches the margin" if within 3 px of the leaf edge
RATIONALE = {
    ("rust", "G0"): "Healthy control leaf (EARTH healthy set, stored in the rust pack); symmetric, no marks.",
    ("rust", "G1"): "Third-nearest rust abaxial face to the arithmetic band middle (2.75%); 0.045 percentage points (pp) away. The two nearer faces are listed in passed_over: GX_030 (2.77%) has lesion mask in both end zones; GX_129 (2.73%, 0.024 pp away, no end-zone mask) was passed over on appearance only, because its pustules are concentrated along the right margin. GX_130 has discrete sporulating pustules spread over the lamina, 18% of lesion area in pustules touching the margin, no mask pixels in either end zone. Both GX_129 and GX_130 print as 2.7%. Adaxial face also G1 (2.81%).",
    ("rust", "G2"): "Nearest rust abaxial face (any split) to the arithmetic band middle (12.5%); typical sporulating pustules, 15% of lesion area in pustules touching the margin. Petiole-end zone unmarked; 22 lesion px (0.5% of lesion area) fall in the apex zone, the lower edge of a pustule at the base of the acumen (the acumen itself is unmarked). The left margin has a natural fold. Visually cleaner alternative further from the middle: GX_136 (9.56%). Rust grades use the abaxial face only (adaxial 4.76%).",
    ("rust", "G3"): "Only two rust leaves reach >=20% on the abaxial face (GX_072 21.73%, GX_119 23.41%), equidistant from their median; GX_119 has typical coalescing sporulating lesions and no mask pixels in either end zone, but 62% of its lesion area is in lesions touching the margin. GX_072 has a damaged, deeply lobed margin and mask in both end zones.",
    ("leafspot", "G0"): "Healthy control leaf from the rust pack (EARTH healthy set, the leaf-spot collection site); the leaf-spot pack has no healthy leaves. Leaf-spot-pack faces with <0.5% exist (22 abaxial faces) but every one carries the annotator's visual grade G1, so none was used as a G0 exemplar.",
    ("leafspot", "G1"): "Train split; 0.136 pp from the band middle (2.75%). All four nearer faces are listed in passed_over (three have lesion mask at the petiole end, one has a single lesion). Two lesions on the lamina, away from the margin, tip and petiole end: an upper subcircular eyespot with a reddish-brown halo beside the midrib (with two small mask specks on the midrib next to it) and a lower lesion formed by two coalesced spots. No lesion area touches the margin; no mask pixels in either end zone. Replaces 20260628 GX_017 (3.12%, test split, withheld). Annotator grade G1; adaxial 3.06%.",
    ("leafspot", "G2"): "Train split; 0.21 pp from the band middle (12.5%), the nearest face after the four listed in passed_over (three with petiole-end mask, one with irregular marginal lesions). 18 mask components, mostly subcircular eyespots scattered over the lamina, some with very dark centres, plus a few thin masked streaks along the midrib and one lateral vein. 3.8% of lesion area in lesions touching the margin; no mask pixels in either end zone. Replaces 20260628 GX_094 (13.20%, 0.70 pp from the middle, 15% of lesion area touching the margin, paler leaf). Annotator grade G2; adaxial 10.58%.",
    ("leafspot", "G3"): "Train split; nearest train/val face to the G3 target (26.3%, median of the pack's >=20% abaxial faces) with no lesion mask in either end zone, 2.9 pp below it. The six nearer faces are listed in passed_over: four train faces with tip, petiole-end or margin necrosis (95-100% of lesion area touching the margin for three of them), and two test faces that are withheld. Four large irregular lesions on the mid-lamina, formed by coalescing spots, plus two small ones. These are blotches rather than discrete subcircular eyespots; no train/val G3 face near the target shows discrete eyespots without tip or margin necrosis. 47% of lesion area is in lesions touching the leaf margin (the left lesion, the upper-right lesion and a small right-margin spot); the petiole end and the apex are unmarked. Farther alternatives inspected: GX_053 (22.98%, train; chlorotic, senescent-looking, petiole-end mask) and GX_076 (22.49%, val; the most eyespot-like, but a large lesion covers the apex and 76% of lesion area touches the margin). Replaces 20260628 GX_031 (27.33%, test split, withheld; necrotic masked acumen, petiole-end dot, 70% of lesion area touching the margin). Annotator grade G3; adaxial 22.54%.",
}
# every face nearer its band target than the chosen exemplar (all splits), with the reason
PASSED_OVER = {
    ("rust", "G1"): {
        "20260819_Hvastatrix_GX_030__abaxial.png": "Lesion mask in both end zones (35 px petiole end, 159 px apex; 10% of lesion area) and 70% of lesion area in pustules touching the margin.",
        "20260819_Hvastatrix_GX_129__abaxial.png": "No end-zone mask and 0.02 pp nearer than GX_130 (both print as 2.7%). Passed over on appearance: pustules are concentrated along the right margin, two of them coalescing against the edge (34% of lesion area touching the margin vs 18% for GX_130).",
    },
    ("rust", "G2"): {},
    ("rust", "G3"): {
        "20260819_Hvastatrix_GX_072__abaxial.png": "Equidistant from the target (median of the two >=20% faces). Damaged, deeply lobed margin; lesion mask in both end zones (49 + 581 px, 10% of lesion area); 64% of lesion area touching the margin.",
    },
    ("leafspot", "G1"): {
        "20260621_Mcitricolor_GX_221__abaxial.png": "Only one lesion (two coalesced spots forming a single mask component).",
        "20260621_Mcitricolor_GX_158__abaxial.png": "Masked cut petiole end (458 px in the petiole-end zone, 9% of lesion area); 32% of lesion area touching the margin.",
        "20260621_Mcitricolor_GX_160__abaxial.png": "Masked petiole end (256 px) and a long masked streak along the upper midrib.",
        "20260621_Mcitricolor_GX_240__abaxial.png": "Lesion mask at the petiole end (297 px) and the apex (95 px).",
    },
    ("leafspot", "G2"): {
        "20260628_Mcitricolor_GX_120__abaxial.png": "No end-zone mask, but irregular lesions, 64% of lesion area touching the margin, and dark streaking along the veins over the lamina.",
        "20260628_Mcitricolor_GX_126__abaxial.png": "Val split. Masked petiole end (293 px in the petiole-end zone).",
        "20260628_Mcitricolor_GX_119__abaxial.png": "Masked petiole end (725 px); irregular lesions, one reaching the base of the acumen; vein streaking.",
        "20260621_Mcitricolor_GX_009__abaxial.png": "Val split. Masked petiole end (751 px) and a large necrotic lesion covering the apex (4967 px in the apex zone; 18% of lesion area in the end zones).",
    },
    ("leafspot", "G3"): {
        "20260628_Mcitricolor_GX_095__abaxial.png": "Necrosis running along the midrib to the petiole end (800 px in the petiole-end zone); 78% of lesion area touching the margin.",
        "20260628_Mcitricolor_GX_033__abaxial.png": "Hole through the lamina; lesions along both margins (99.6% of lesion area touching the margin); 1241 px in the apex zone.",
        "20260628_Mcitricolor_GX_079__abaxial.png": "Marginal necrosis along the whole left margin down to the apex (99% of lesion area touching the margin; 2640 px apex, 410 px petiole end).",
        "20260628_Mcitricolor_GX_088__abaxial.png": "Marginal necrosis along both margins down to the apex (95% of lesion area touching the margin; 1944 px apex, 341 px petiole end).",
        "20260628_Mcitricolor_GX_031__abaxial.png": "Test split (withheld); previously used here. Necrotic masked acumen (3017 px apex zone) and a masked dot at the petiole end (207 px); irregular blotches, 70% of lesion area touching the margin.",
        "20260628_Mcitricolor_GX_064__abaxial.png": "Test split (withheld). A large lesion covers the acumen (1174 px apex zone); 100% of lesion area touching the margin.",
    },
}

# ---------------------------------------------------------------- style
INK = "#1a1a1a"
GREY_TXT = "#555555"
GREY_NOTE = "#7a7a7a"
GREY_RULE = "#c8c8c8"
MAGENTA = "#FF00FF"
OUTLINE_LW = 0.45  # pt
HEADER_FS = 7.9  # pt, column headers
CM = 1 / 2.54


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "image.interpolation": "none",
    })


# ---------------------------------------------------------------- data
def split_rows() -> dict:
    rows = {}
    for pack, p in SPLIT.items():
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                rows[(pack, r["file"])] = r
    return rows


def load_face(pack: str, fname: str) -> dict:
    img = np.asarray(Image.open(PACK[pack] / "images" / fname).convert("RGBA"))
    msk = np.asarray(Image.open(PACK[pack] / "masks" / fname).convert("L"))
    leaf = img[..., 3] > 8
    les = (msk > 127) & leaf
    return dict(img=img, leaf=leaf, les=les,
                sev=100.0 * les.sum() / leaf.sum(),
                leaf_px=int(leaf.sum()), lesion_px=int(les.sum()))


def lesion_geometry(leaf: np.ndarray, les: np.ndarray) -> dict:
    """End-zone lesion px (petiole end = top, apex = bottom in both packs) and the
    share of lesion area in components (8-connected) that touch the leaf margin."""
    from scipy import ndimage as ndi
    ys = np.where(leaf.any(1))[0]
    y0, y1 = ys.min(), ys.max()
    hz = int(END_ZONE_FRAC * (y1 - y0 + 1))
    top = int(les[y0: y0 + hz].sum())
    bot = int(les[y1 - hz + 1: y1 + 1].sum())
    border = leaf & ~ndi.binary_erosion(leaf, iterations=MARGIN_BAND_PX)
    lab, n = ndi.label(les, structure=np.ones((3, 3)))
    touching = np.unique(lab[border & les])
    touching = touching[touching > 0]
    marg = int(np.isin(lab, touching).sum()) if touching.size else 0
    tot = int(les.sum())
    return {
        "petiole_end_zone_lesion_px": top, "apex_zone_lesion_px": bot,
        "end_zone_share_of_lesion_area": round((top + bot) / tot, 4) if tot else 0.0,
        "margin_touching_share_of_lesion_area": round(marg / tot, 4) if tot else 0.0,
        "n_lesion_components": int(n),
    }


def display_source(pack: str, fname: str, ref_img: np.ndarray) -> tuple[np.ndarray, tuple, dict]:
    """Return (RGBA display image, (sx, sy) = pack px per display px, check dict)."""
    kind = next(k for k in HIRES if k in fname)
    src = HIRES[kind]
    if src is None:
        return ref_img, (1.0, 1.0), {"display_file": str(PACK[pack] / "images" / fname).replace("\\", "/"),
                                     "display_px_per_pack_px": 1.0}
    hi_im = Image.open(src / fname).convert("RGBA")
    h, w = ref_img.shape[:2]
    # the pack image is a Lanczos resize of the 2048 px scan (1583 x 2048 -> 791 x 1024)
    assert abs(hi_im.size[1] / h - 2) < 0.01 and abs(hi_im.size[0] / w - 2) < 0.01, (fname, hi_im.size, (w, h))
    down = np.asarray(hi_im.resize((w, h), Image.LANCZOS)).astype(float)
    leaf = ref_img[..., 3] > 8
    chk = {
        "display_file": str(src / fname).replace("\\", "/"),
        "display_size_px": list(hi_im.size), "pack_size_px": [w, h],
        "check_lanczos_down_leaf_rgb_mean_abs_diff": round(float(np.abs(down[..., :3] - ref_img[..., :3])[leaf].mean()), 3),
        "check_alpha_gt8_agreement": round(float(((down[..., 3] > 8) == leaf).mean()), 6),
    }
    assert chk["check_lanczos_down_leaf_rgb_mean_abs_diff"] < 1.0 and chk["check_alpha_gt8_agreement"] > 0.9999, chk
    return np.asarray(hi_im), (w / hi_im.size[0], h / hi_im.size[1]), chk


def composite_white(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    rgb = rgba[..., :3].astype(np.float32) * a + 255.0 * (1.0 - a)
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- figure
LAYOUT = dict(
    W=16.0,          # cm
    margin_l=0.10, margin_r=0.10, margin_t=0.12, margin_b=0.18,
    header_h=0.62,   # grade headers
    title_h=1.22,    # panel title + species + clearance above the petiole
    leaf_h=4.85,     # leaf image box height
    label_h=0.78,    # severity text under the leaf
    row_gap=0.22,
    leaf_w_frac=0.80,  # max leaf-box width as a fraction of the column
)


def fig_height(L=LAYOUT) -> float:
    return (L["margin_t"] + L["header_h"] + 2 * (L["title_h"] + L["leaf_h"] + L["label_h"])
            + L["row_gap"] + L["margin_b"])


def draw(cells: dict, interpolation: str) -> plt.Figure:
    setup_style()
    L = LAYOUT
    W, H = L["W"], fig_height()
    fig = plt.figure(figsize=(W * CM, H * CM))
    col_w = (W - L["margin_l"] - L["margin_r"]) / 4

    def fx(x_cm):
        return x_cm / W

    def fy(y_cm_from_top):
        return 1 - y_cm_from_top / H

    # ---- grade headers (once, above both rows): bold grade code + description
    y_head = L["margin_t"] + L["header_h"] * 0.55
    renderer = fig.canvas.get_renderer()
    for j, (g, desc, _, _) in enumerate(GRADES):
        xc = L["margin_l"] + (j + 0.5) * col_w
        t1 = fig.text(0, 0, g, fontsize=HEADER_FS, fontweight="bold", color=INK, ha="left", va="baseline")
        t2 = fig.text(0, 0, " \u2014 " + desc, fontsize=HEADER_FS, color=INK, ha="left", va="baseline")
        w1 = t1.get_window_extent(renderer).width / fig.dpi * 2.54
        w2 = t2.get_window_extent(renderer).width / fig.dpi * 2.54
        assert w1 + w2 < col_w - 0.25, (g, w1 + w2, col_w)  # header must stay inside its column
        x0 = xc - (w1 + w2) / 2
        t1.set_position((fx(x0), fy(y_head)))
        t2.set_position((fx(x0 + w1), fy(y_head)))
    # thin rule under the headers
    y_rule = L["margin_t"] + L["header_h"] - 0.02
    fig.add_artist(plt.Line2D([fx(L["margin_l"] + 0.05), fx(W - L["margin_r"] - 0.05)],
                              [fy(y_rule)] * 2, color=GREY_RULE, lw=0.6))

    y = L["margin_t"] + L["header_h"]
    for i, (disease, title, species) in enumerate(ROWS):
        if i == 1:
            y += L["row_gap"]
        # ---- panel title, left-aligned, bold; species grey italic beneath
        fig.text(fx(L["margin_l"] + 0.05), fy(y + 0.50), title, fontsize=10.2, fontweight="bold",
                 color=INK, ha="left", va="baseline")
        fig.text(fx(L["margin_l"] + 0.05), fy(y + 0.86), species, fontsize=8.4, fontstyle="italic",
                 color=GREY_TXT, ha="left", va="baseline")
        y_leaf_top = y + L["title_h"]
        for j, (g, _, _, _) in enumerate(GRADES):
            c = cells[(disease, g)]
            # crop to the leaf bbox (pack frame) with a 2 px margin
            ys, xs = np.where(c["leaf"])
            y0, y1 = max(0, ys.min() - 2), min(c["leaf"].shape[0] - 1, ys.max() + 2)
            x0, x1 = max(0, xs.min() - 2), min(c["leaf"].shape[1] - 1, xs.max() + 2)
            sx, sy = c["scale"]
            U0, U1 = int(np.floor(x0 / sx)), int(np.ceil((x1 + 1) / sx))
            V0, V1 = int(np.floor(y0 / sy)), int(np.ceil((y1 + 1) / sy))
            disp = composite_white(c["disp"][V0:V1, U0:U1])
            les = c["les"][y0: y1 + 1, x0: x1 + 1]
            hpx, wpx = y1 - y0 + 1, x1 - x0 + 1
            # fit into the leaf box, same height for all leaves unless width-limited
            box_w = col_w * L["leaf_w_frac"]
            k = min(L["leaf_h"] / hpx, box_w / wpx)
            w_cm, h_cm = wpx * k, hpx * k
            # effective resolution of the embedded leaf image at print size
            c["print_size_cm"] = [round(w_cm, 3), round(h_cm, 3)]
            c["embedded_crop_px"] = [int(U1 - U0), int(V1 - V0)]
            c["print_ppi"] = round((V1 - V0) / (h_cm / 2.54), 1)
            xc = L["margin_l"] + (j + 0.5) * col_w
            top = y_leaf_top + (L["leaf_h"] - h_cm) / 2  # vertically centred in the leaf box
            ax = fig.add_axes([fx(xc - w_cm / 2), fy(top + h_cm), w_cm / W, h_cm / H])
            # display pixel u covers pack coordinates [u*sx - 0.5, (u+1)*sx - 0.5]
            ax.imshow(disp, extent=(U0 * sx - 0.5, U1 * sx - 0.5, V1 * sy - 0.5, V0 * sy - 0.5),
                      interpolation=interpolation, interpolation_stage="rgba", resample=True)
            if les.any():
                pad = np.pad(les.astype(float), 1)
                xx = np.arange(x0 - 1, x1 + 2)
                yy = np.arange(y0 - 1, y1 + 2)
                ax.contour(xx, yy, pad, levels=[0.5], colors=[MAGENTA], linewidths=OUTLINE_LW,
                           antialiased=True)
            ax.set_xlim(x0 - 0.5, x1 + 0.5)
            ax.set_ylim(y1 + 0.5, y0 - 0.5)
            ax.set_axis_off()
            # ---- severity under the leaf box
            y_lab = y_leaf_top + L["leaf_h"] + 0.36
            fig.text(fx(xc), fy(y_lab), f"{c['sev']:.1f}%", fontsize=8.8, color=INK,
                     ha="center", va="baseline")
            if c["is_healthy_control"]:
                fig.text(fx(xc), fy(y_lab + 0.33), "healthy control leaf", fontsize=6.8,
                         color=GREY_NOTE, ha="center", va="baseline")
        y = y_leaf_top + L["leaf_h"] + L["label_h"]
    return fig


def band_target(all_rows: list, disease: str, g: str) -> float:
    lo, hi = {"G1": (0.5, 5.0), "G2": (5.0, 20.0), "G3": (20.0, 100.0)}[g]
    if g != "G3":
        return (lo + hi) / 2
    # open band: median of the pack's >=20% abaxial faces (all splits)
    vals = sorted(float(r["severity_pct"]) for r in all_rows
                  if r["disease"] == disease and r["face"] == "abaxial"
                  and float(r["severity_pct"]) >= 20 and r["label"] != "healthy")
    return float(np.median(vals))


def ranked_candidates(all_rows: list, disease: str, g: str, target: float) -> list:
    """Abaxial faces in the band (leaf spot: annotator grade equal to the band), all splits,
    ranked by distance to the band target."""
    lo, hi = {"G1": (0.5, 5.0), "G2": (5.0, 20.0), "G3": (20.0, 100.0)}[g]
    out = []
    for r in all_rows:
        if r["disease"] != disease or r["face"] != "abaxial" or r["label"] == "healthy":
            continue
        s = float(r["severity_pct"])
        if not (lo <= s < hi):
            continue
        if disease == "leafspot" and r["label"] != g:
            continue
        out.append((abs(s - target), r))
    out.sort(key=lambda t: t[0])
    return out


def main() -> None:
    splits = split_rows()
    cells, rec = {}, []
    for (disease, g), (pack, fname) in SELECTED.items():
        c = load_face(pack, fname)
        disp, scale, chk = display_source(pack, fname, c["img"])
        c.update(disp=disp, scale=scale, is_healthy_control="Healthy" in fname)
        cells[(disease, g)] = c
        band = next(b for b in GRADES if b[0] == g)
        assert band[2] <= c["sev"] < band[3], (fname, c["sev"], g)
        other = fname.replace("__abaxial", "__adaxial") if "__abaxial" in fname else fname.replace("__adaxial", "__abaxial")
        oc = load_face(pack, other)
        sr = splits[(pack, fname)]
        assert sr["split"] not in EXCLUDED_SPLITS, (fname, sr["split"])  # test split is withheld
        rec.append({
            "row": disease, "grade": g, "column_header": f"{g} \u2014 {band[1]}",
            "file": fname, "leaf_id": sr["leaf_id"], "face": "abaxial" if "__abaxial" in fname else "adaxial",
            "source_pack": str(PACK[pack]).replace("\\", "/"),
            "image": str(PACK[pack] / "images" / fname).replace("\\", "/"),
            "mask": str(PACK[pack] / "masks" / fname).replace("\\", "/"),
            "pack_label": sr["grade"], "split": sr["split"],
            "healthy_control": c["is_healthy_control"],
            "leaf_px": c["leaf_px"], "lesion_px": c["lesion_px"],
            "severity_pct": round(c["sev"], 4), "severity_printed": f"{c['sev']:.1f}%",
            "other_face_file": other, "other_face_severity_pct": round(oc["sev"], 4),
            "leaf_mean_severity_pct": round((c["sev"] + oc["sev"]) / 2, 4),
            **lesion_geometry(c["leaf"], c["les"]),
            "display": chk,
            "rationale": RATIONALE[(disease, g)],
        })

    fig = draw(cells, interpolation="none")
    fig.savefig(OUT / "figure2.pdf")
    plt.close(fig)
    for r in rec:
        c = cells[(r["row"], r["grade"])]
        r["display"].update(embedded_crop_px=c["embedded_crop_px"], print_size_cm=c["print_size_cm"],
                            print_ppi=c["print_ppi"])
    fig = draw(cells, interpolation="lanczos")
    fig.savefig(OUT / "figure2.png", dpi=600)
    plt.close(fig)
    with Image.open(OUT / "figure2.png") as im:
        rgb = Image.new("RGB", im.size, "white")
        rgb.paste(im, mask=im.convert("RGBA").getchannel("A"))
    rgb.save(OUT / "figure2.png", dpi=(600, 600))

    # candidate context: every face nearer the band target than the chosen exemplar
    # (must match PASSED_OVER exactly), the exemplar, and the next three farther faces
    table = OUT / "work" / "all_face_severity.csv"  # written by compute_severity.py
    with open(table, newline="") as f:
        all_rows = list(csv.DictReader(f))
    context = {}
    for disease in ("rust", "leafspot"):
        for g in ("G1", "G2", "G3"):
            tgt = band_target(all_rows, disease, g)
            ranked = ranked_candidates(all_rows, disease, g, tgt)
            chosen = SELECTED[(disease, g)][1]
            d_ch = next(d for d, r in ranked if r["file"] == chosen)
            nearer = [r["file"] for d, r in ranked if d <= d_ch + 1e-9 and r["file"] != chosen]
            assert set(nearer) == set(PASSED_OVER[(disease, g)]), (disease, g, sorted(set(nearer) ^ set(PASSED_OVER[(disease, g)])))
            farther = [r for d, r in ranked if d > d_ch + 1e-9][:3]
            listed = [r for d, r in ranked if d <= d_ch + 1e-9] + farther
            items = []
            for r in listed:
                fc = load_face(disease, r["file"])
                if r["file"] == chosen:
                    status = "chosen"
                elif r["file"] in PASSED_OVER[(disease, g)]:
                    status = "passed over: " + PASSED_OVER[(disease, g)][r["file"]]
                else:
                    status = "farther from the target than the chosen face"
                items.append({"file": r["file"], "split": r["split"], "label": r["label"],
                              "severity_pct": round(float(r["severity_pct"]), 3),
                              "distance_to_target_pp": round(abs(float(r["severity_pct"]) - tgt), 3),
                              **lesion_geometry(fc["leaf"], fc["les"]), "status": status})
            context[f"{disease}_{g}"] = {
                "target_pct": round(tgt, 2),
                "n_faces_in_band_all_splits": len(ranked),
                "n_faces_in_band_train_val": sum(1 for _, r in ranked if r["split"] not in EXCLUDED_SPLITS),
                "chosen_rank_all_splits": 1 + len(nearer),
                "candidates": items,
            }

    H = fig_height()
    res = {
        "figure": "Figure 2, grade exemplars (rebuilt)",
        "outputs": {"pdf": str(OUT / "figure2.pdf").replace("\\", "/"),
                    "png": str(OUT / "figure2.png").replace("\\", "/"),
                    "script": str(Path(__file__).resolve()).replace("\\", "/")},
        "size_cm": [16.0, round(H, 2)], "png_dpi": 600,
        "definitions": {
            "leaf": "alpha > 8", "lesion": "mask > 127 AND leaf",
            "severity": "100 * lesion px / leaf px on the displayed face, pack images/masks at native resolution (no letterbox)",
            "grade_bands": "G0 <0.5%, G1 0.5 to <5%, G2 5 to <20%, G3 >=20%",
            "band_middle_rule": "arithmetic middle for closed bands (G1 2.75%, G2 12.5%); for the open G3 band, the median of the pack's >=20% abaxial faces",
            "outline": "reference lesion boundary (mask > 127 AND leaf), vector contour at the 0.5 level of the binary mask in the pack pixel frame, magenta #FF00FF, 0.45 pt",
            "display": "leaves composited on white via alpha; each leaf scaled individually to a common height (no common physical scale; scan resolution is not recorded in the packs)",
            "print_ppi": "embedded leaf-crop height in px / printed leaf height in inches (PDF embeds the crop at native resolution; the 600 dpi PNG resamples it)",
            "split_policy": "exemplars restricted to the train and val splits; the test split is withheld pending peer review (manuscript sections/06_backmatter.tex, data availability statement)",
            "end_zones": "top and bottom 8% of the leaf's vertical extent (petiole end at the top, apex at the bottom in both packs); used to screen out faces whose mask marks petiole-end or apex necrosis",
            "margin_touching_share": "share of lesion area in 8-connected lesion components that come within 3 px of the leaf edge",
            "selection_rule": "per cell, faces in the band (leaf spot: annotator grade equal to the band) ranked by distance of the abaxial severity to the band target; the nearest train/val face without end-zone mask and with typical lesions on visual inspection was chosen; every face nearer the target (any split) is listed with the reason it was passed over (selection_context, status field)",
        },
        "faces": "abaxial for all eight cells (leaf spot: the adaxial face was not clearly more representative for any chosen leaf)",
        "g0_source_note": "Both G0 cells are healthy control leaves from the rust pack (EARTH, Guacimo healthy set, the same site as American leaf spot); labelled 'healthy control leaf' in the figure. The leaf-spot pack contains no healthy leaves, and its <0.5% faces all carry annotator grade G1.",
        "cells": rec,
        "selection_context": context,
    }
    (OUT / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    for r in rec:
        print(f"{r['row']:8s} {r['grade']} {r['file']:45s} {r['severity_printed']:>6s} "
              f"(other face {r['other_face_severity_pct']:.2f}%, label {r['pack_label']}, {r['split']})")


if __name__ == "__main__":
    main()
