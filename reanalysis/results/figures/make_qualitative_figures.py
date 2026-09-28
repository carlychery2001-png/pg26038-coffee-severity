"""Rebuild the qualitative SegFormer-B5 figures (paper Figure 13, coffee leaf rust; paper Figure 14, American leaf spot)
with registered overlays. No re-inference: predictions are the saved class maps of run_inference.py.

Selection
  Test leaves with disease (rust 21, leaf spot 55) ranked by leaf-level lesion IoU of SegFormer-B5; worst = rank 1,
  median = rank (n + 1) / 2 (n is odd for both diseases, so the median is one real leaf), best = rank n.
  Rust: the abaxial face (the face that defines the rust leaf IoU).
  Leaf spot: the requested rule "the face whose IoU is closest to the leaf mean" cannot discriminate, because the leaf
  IoU is the mean of exactly two face IoUs, so both faces are equidistant from it by construction
  (|IoU_ab - m| = |IoU_ad - m| = |IoU_ab - IoU_ad| / 2; asserted below with exact fractions). The tie is broken by a
  rule fixed before looking at the images and independent of IoU: the abaxial face (as for rust and paper Figure 2).
  Both face IoUs and the leaf IoU are printed in the figure.

Rendering (letterbox space)
  Image and reference labels rebuilt exactly as run_inference.load() (the function is executed verbatim from that
  file); the prediction is the saved 1024 x 1024 class map. All three share one pixel grid, so no inverse transform is
  applied. Each row is cropped to the bounding box of the reference leaf and of every reference or predicted lesion
  pixel (+ margin), so no lesion pixel is cut off and the IoU of the plotted crop equals the full-frame IoU.
  Columns: input | reference lesion outline (green) | predicted lesion outline (magenta) | error map
  (TP white, FP magenta, FN green, other reference-leaf pixels grey, background black). Rows share one pixel scale.

Verification (all asserted)
  1. full-frame pixel counts of the rebuilt reference and the saved prediction equal per_image.csv;
  2. IoU from the plotted crop arrays = IoU from the error-map raster = CSV value (exact integer counts);
  3. the outlines are registered: pixel centres inside the drawn contour paths reproduce each mask exactly;
  4. after saving, the error-map raster embedded in the PDF is extracted with PyMuPDF and is pixel-identical to the
     array computed here; the IoU recounted from the PDF raster equals the printed IoU;
  5. leaf spot: leaf IoU = mean of the shown face IoU (from plotted masks) and the other face IoU (recomputed from its
     masks, not plotted) = CSV leaf mean.
  6. severities: the printed LEAF severity (rust: abaxial face; leaf spot: mean of the two face severities) equals the
     value that sets the leaf's Figure 12 band (quant_results.json); leaf spot additionally prints the shown-face
     (abaxial) severity under a label that says so. Face severities are recomputed from the full-frame masks
     (the predicted leaf mask, the denominator of the predicted severity, is not plotted).

Outputs (this folder): figure13.pdf/.png, figure14.pdf/.png, qualitative_results.json
"""
from __future__ import annotations

import json
from fractions import Fraction

import numpy as np

import common as C

C.setup_style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

ARCH = "segformer"
# leaf table of make_quant_figures.py (run first): used only to assert that the printed leaf severity is the value
# that places the leaf in its Figure 12 band
try:
    Q_LEAF = {d: {r["leaf_id"]: r for r in t} for d, t in
              json.loads((C.OUT / "quant_results.json").read_text(encoding="utf-8"))["leaf_tables"].items()}
except FileNotFoundError:
    Q_LEAF = None
GREEN = "#22F25A"      # reference: outline and FN
MAGENTA = "#FF22FF"    # prediction: outline and FP
TP_RGB = (255, 255, 255)
FP_RGB = (255, 34, 255)
FN_RGB = (34, 242, 90)
LEAF_RGB = (84, 84, 84)
BG_RGB = (0, 0, 0)
MARGIN = 10            # px around the crop box
OUTLINE_LW = 0.55
HALO_LW = 1.35


def hex_of(rgb):
    return "#%02X%02X%02X" % rgb


assert hex_of(FP_RGB) == MAGENTA and hex_of(FN_RGB) == GREEN


# ------------------------------------------------------------------------------------------------ selection
def select(rows, disease):
    tab = C.leaf_table(rows, disease, ARCH)
    order = sorted(tab, key=lambda r: (r["iou_exact"], r["leaf_id"]))
    n = len(order)
    assert n % 2 == 1
    picks = {"Worst": 0, "Median": (n - 1) // 2, "Best": n - 1}
    out = []
    for role, k in picks.items():
        r = order[k]
        # no ties at the selected ranks
        if k > 0:
            assert order[k - 1]["iou_exact"] != r["iou_exact"]
        if k < n - 1:
            assert order[k + 1]["iou_exact"] != r["iou_exact"]
        ab, ad = r["faces"]["abaxial"], r["faces"]["adaxial"]
        if disease == "roya":
            face, rule = "abaxial", "rust leaf IoU is defined on the abaxial face"
            tie = None
        else:
            m = r["iou_exact"]
            d_ab, d_ad = abs(ab["iou_exact"] - m), abs(ad["iou_exact"] - m)
            assert d_ab == d_ad, "faces are equidistant from their mean by construction"
            face = "abaxial"
            rule = ("tie: both faces equidistant from the leaf mean by construction; pre-specified IoU-independent "
                    "tie-break = abaxial face")
            tie = dict(abs_diff_abaxial_to_leaf_mean=float(d_ab), abs_diff_adaxial_to_leaf_mean=float(d_ad),
                       exactly_equal=True)
        out.append(dict(role=role, rank=k + 1, n=n, leaf_id=r["leaf_id"], leaf_iou=r["iou"],
                        leaf_iou_exact=r["iou_exact"], ref_sev=r["ref_sev"], pred_sev=r["pred_sev"],
                        ref_grade=r["ref_grade"], face=face, face_rule=rule, tie=tie,
                        face_iou={"abaxial": ab["iou"], "adaxial": ad["iou"]},
                        face_iou_exact={"abaxial": ab["iou_exact"], "adaxial": ad["iou_exact"]},
                        face_file={"abaxial": ab["file"], "adaxial": ad["file"]},
                        face_sev={"abaxial": (ab["ref_sev"], ab["pred_sev"]), "adaxial": (ad["ref_sev"], ad["pred_sev"])}))
    ranking = [dict(rank=i + 1, leaf_id=r["leaf_id"], leaf_iou=r["iou"],
                    iou_abaxial=r["faces"]["abaxial"]["iou"], iou_adaxial=r["faces"]["adaxial"]["iou"])
               for i, r in enumerate(order)]
    return out, ranking


# ------------------------------------------------------------------------------------------------ masks
def csv_row(rows, disease, file):
    rr = [r for r in rows if r["model_disease"] == disease and r["arch"] == ARCH and r["image_pack"] == disease
          and r["split"] == "test" and r["file"] == file]
    assert len(rr) == 1
    return rr[0]


def face_masks(rows, disease, file):
    rgb, lab = C.load_letterbox(disease, file)
    pred = C.load_pred(disease, ARCH, disease, "test", file)
    ref_leaf, ref_les = lab >= 1, lab == 2
    pred_leaf, pred_les = pred >= 1, pred == 2
    r = csv_row(rows, disease, file)
    full = dict(ref_leaf_px=int(ref_leaf.sum()), ref_lesion_px=int(ref_les.sum()), pred_leaf_px=int(pred_leaf.sum()),
                pred_lesion_px=int(pred_les.sum()), inter_lesion=int((ref_les & pred_les).sum()),
                union_lesion=int((ref_les | pred_les).sum()))
    for k, v in full.items():
        assert v == int(r[k]), (file, k, v, r[k])
    return rgb, ref_leaf, ref_les, pred_les, full


def crop_box(ref_leaf, ref_les, pred_les):
    anym = ref_leaf | ref_les | pred_les
    ys, xs = np.nonzero(anym)
    y0, y1 = max(0, ys.min() - MARGIN), min(1024, ys.max() + 1 + MARGIN)
    x0, x1 = max(0, xs.min() - MARGIN), min(1024, xs.max() + 1 + MARGIN)
    return int(y0), int(y1), int(x0), int(x1)


def error_map(ref_leaf, ref_les, pred_les):
    em = np.zeros(ref_leaf.shape + (3,), np.uint8)
    em[:] = BG_RGB
    em[ref_leaf] = LEAF_RGB
    em[ref_les & pred_les] = TP_RGB
    em[~ref_les & pred_les] = FP_RGB
    em[ref_les & ~pred_les] = FN_RGB
    return em


def counts_from_raster(em):
    def cnt(rgb):
        return int(np.all(em == np.array(rgb, np.uint8), axis=-1).sum())
    return dict(tp=cnt(TP_RGB), fp=cnt(FP_RGB), fn=cnt(FN_RGB))


def iou_of(c):
    return Fraction(c["tp"], c["tp"] + c["fp"] + c["fn"])


def contour_registration(cs, mask):
    """Pixel centres inside the drawn 0.5-level contour paths must reproduce the mask exactly."""
    paths = cs.get_paths()
    verts, codes = [], []
    for p in paths:
        if len(p.vertices):
            verts.append(p.vertices)
            codes.append(p.codes if p.codes is not None else
                         np.r_[MPath.MOVETO, np.full(len(p.vertices) - 1, MPath.LINETO)])
    h, w = mask.shape
    if not verts:
        return bool(not mask.any()), 0
    path = MPath(np.concatenate(verts), np.concatenate(codes))
    inside = np.zeros((h, w), bool)
    # even-odd over sub-paths (holes are separate closed loops); each loop tested only inside its bounding box
    for sub in path.to_polygons(closed_only=False):
        x0, y0 = np.floor(sub.min(0)).astype(int)
        x1, y1 = np.ceil(sub.max(0)).astype(int)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w - 1), min(y1, h - 1)
        yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        hit = MPath(sub).contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(yy.shape)
        inside[y0:y1 + 1, x0:x1 + 1] ^= hit
    mism = int((inside != mask).sum())
    return mism == 0, mism


# ------------------------------------------------------------------------------------------------ figure
def build(disease, picks, rows):
    data = []
    for p in picks:
        rgb, ref_leaf, ref_les, pred_les, full = face_masks(rows, disease, p["face_file"][p["face"]])
        y0, y1, x0, x1 = crop_box(ref_leaf, ref_les, pred_les)
        sl = (slice(y0, y1), slice(x0, x1))
        rgb_c, leaf_c, ref_c, pred_c = rgb[sl], ref_leaf[sl], ref_les[sl], pred_les[sl]
        assert ref_c.sum() == ref_les.sum() and pred_c.sum() == pred_les.sum(), "lesion pixels outside crop"
        em = error_map(leaf_c, ref_c, pred_c)
        c_arr = dict(tp=int((ref_c & pred_c).sum()), fp=int((~ref_c & pred_c).sum()), fn=int((ref_c & ~pred_c).sum()))
        c_ras = counts_from_raster(em)
        assert c_arr == c_ras
        iou_plot = iou_of(c_ras)
        assert c_ras["tp"] == full["inter_lesion"] and sum(c_ras.values()) == full["union_lesion"]
        assert iou_plot == p["face_iou_exact"][p["face"]]
        # printed face severities: recomputed here from the full-frame masks (the predicted leaf mask is not plotted)
        fs = (100 * full["ref_lesion_px"] / full["ref_leaf_px"], 100 * full["pred_lesion_px"] / full["pred_leaf_px"])
        assert fs == tuple(p["face_sev"][p["face"]]), (fs, p["face_sev"][p["face"]])
        # printed leaf severity must be the value that sets the leaf's Figure 12 band (quant_results.json)
        assert Q_LEAF is not None, "run make_quant_figures.py first"
        q = Q_LEAF[disease][p["leaf_id"]]
        assert q["ref_sev_pct"] == p["ref_sev"] and q["pred_sev_segformer"] == p["pred_sev"], p["leaf_id"]
        other = None
        if disease == "ojo":
            of = "adaxial" if p["face"] == "abaxial" else "abaxial"
            _, _, o_ref, o_pred, o_full = face_masks(rows, disease, p["face_file"][of])
            o_iou = Fraction(int((o_ref & o_pred).sum()), int((o_ref | o_pred).sum()))
            assert o_iou == p["face_iou_exact"][of]
            leaf_iou_re = (iou_plot + o_iou) / 2
            assert leaf_iou_re == p["leaf_iou_exact"]
            o_fs = (100 * o_full["ref_lesion_px"] / o_full["ref_leaf_px"],
                    100 * o_full["pred_lesion_px"] / o_full["pred_leaf_px"])
            assert o_fs == tuple(p["face_sev"][of])
            # leaf severity = mean of the two face severities (recomputed from both faces' masks)
            assert abs((fs[0] + o_fs[0]) / 2 - p["ref_sev"]) < 1e-12 and abs((fs[1] + o_fs[1]) / 2 - p["pred_sev"]) < 1e-12
            other = dict(face=of, iou=float(o_iou), counts=dict(tp=int((o_ref & o_pred).sum()),
                         fp=int((~o_ref & o_pred).sum()), fn=int((o_ref & ~o_pred).sum())))
        else:
            assert iou_plot == p["leaf_iou_exact"]
        data.append(dict(pick=p, rgb=rgb_c, ref=ref_c, pred=pred_c, em=em, box=(y0, y1, x0, x1), counts=c_ras,
                         iou_plot=iou_plot, other=other, full=full))
    return data


def draw(disease, data):
    fig_w = 16.0
    left_txt = 3.85          # cm, text column
    gap = 0.14               # cm between image slots
    right_pad = 0.05
    head = 0.62              # cm, column headers
    row_gap = 0.30
    legend_h = 1.05
    max_w = max(d["rgb"].shape[1] for d in data)
    sum_h = sum(d["rgb"].shape[0] for d in data)
    s_w = (fig_w - left_txt - 3 * gap - right_pad) / (4 * max_w)
    s_h = (20.2 - head - legend_h - 2 * row_gap - 0.15) / sum_h
    s = min(s_w, s_h)        # cm per letterbox pixel, common to all panels of the figure
    slot = max_w * s
    fig_h = head + sum_h * s + 2 * row_gap + legend_h + 0.15
    fig = plt.figure(figsize=(fig_w * C.CM, fig_h * C.CM))

    def fx(cm):
        return cm / fig_w

    def fy(cm_from_top):
        return 1 - cm_from_top / fig_h

    headers = ["Input", "Reference", "SegFormer-B5", "Error map"]
    for j, h in enumerate(headers):
        xc = left_txt + j * (slot + gap) + slot / 2
        fig.text(fx(xc), fy(head - 0.18), h, ha="center", va="bottom", fontsize=8.4, color=C.INK)

    y_top = head
    drawn = []
    text_artists = []
    for i, d in enumerate(data):
        h_px, w_px = d["rgb"].shape[:2]
        hh, ww = h_px * s, w_px * s
        axes = []
        for j in range(4):
            x_left = left_txt + j * (slot + gap) + (slot - ww) / 2
            ax = fig.add_axes([fx(x_left), fy(y_top + hh), fx(ww), hh / fig_h])
            ax.set_axis_off()
            axes.append(ax)
        a_in, a_ref, a_pred, a_err = axes
        for ax in (a_in, a_ref, a_pred):
            ax.imshow(d["rgb"], interpolation="none")
        halo = [patheffects.Stroke(linewidth=HALO_LW, foreground="black", alpha=0.55), patheffects.Normal()]
        cs_ref = a_ref.contour(d["ref"].astype(float), levels=[0.5], colors=[GREEN], linewidths=OUTLINE_LW)
        cs_pred = a_pred.contour(d["pred"].astype(float), levels=[0.5], colors=[MAGENTA], linewidths=OUTLINE_LW)
        for cs in (cs_ref, cs_pred):
            cs.set_path_effects(halo)
        a_err.imshow(d["em"], interpolation="none")
        for ax in axes:
            ax.set_xlim(-0.5, w_px - 0.5)
            ax.set_ylim(h_px - 0.5, -0.5)
        ok_ref, mis_ref = contour_registration(cs_ref, d["ref"])
        ok_pred, mis_pred = contour_registration(cs_pred, d["pred"])
        assert ok_ref and ok_pred, (mis_ref, mis_pred)
        d["outline_registration"] = dict(reference_mismatched_px=mis_ref, prediction_mismatched_px=mis_pred)

        # ---- row text (left column; asserted below not to reach the first image)
        p, c = d["pick"], d["counts"]
        short = p["leaf_id"].split("_", 2)[-1]
        iou_txt = f"{float(d['iou_plot']):.3f}"
        rs, ps = p["face_sev"][p["face"]]            # shown face (recomputed from full-frame masks in build())
        lrs, lps = p["ref_sev"], p["pred_sev"]       # leaf level (rust: = abaxial face; leaf spot: mean of faces)
        leaf_sev_txt = f"{lrs:.2f}% / {lps:.2f}%"
        face_sev_txt = f"{rs:.2f}% / {ps:.2f}%"
        x_t = 0.12
        yt = y_top + 0.05
        lines = [(p["role"], 9.0, C.INK, "bold", 0.50)]
        sub = f"rank {p['rank']} of {p['n']} · {short}"
        if disease == "ojo":
            sub += f" · G{p['ref_grade']}"
        lines.append((sub, 6.9, C.GREY_TXT, "normal", 0.42))
        if disease == "roya":
            # rust leaf-level values ARE the abaxial-face values (the face shown), so one block suffices
            assert (lrs, lps) == (rs, ps) and p["face"] == "abaxial"
            lines.append(("Leaf IoU = " + iou_txt, 8.0, C.INK, "normal", 0.44))
            lines.append(("abaxial face (shown) = leaf", 6.7, C.GREY_TXT, "normal", 0.36))
            lines.append((f"TP {c['tp']:,} · FP {c['fp']:,} · FN {c['fn']:,} px", 6.4, C.GREY_TXT, "normal", 0.33))
            lines.append(("leaf severity, ref. / pred.:", 6.4, C.GREY_TXT, "normal", 0.31))
            lines.append((leaf_sev_txt, 6.4, C.GREY_TXT, "normal", 0.33))
        else:
            o = d["other"]
            # leaf-level block (both faces)
            lines.append(("Leaf IoU = " + f"{float(p['leaf_iou_exact']):.3f}", 8.0, C.INK, "normal", 0.44))
            lines.append((f"= mean of {p['face']} {iou_txt}", 6.7, C.GREY_TXT, "normal", 0.33))
            lines.append((f"(shown) and {o['face']} {o['iou']:.3f}", 6.7, C.GREY_TXT, "normal", 0.36))
            lines.append(("leaf severity (mean of faces),", 6.4, C.GREY_TXT, "normal", 0.31))
            lines.append(("ref. / pred.: " + leaf_sev_txt, 6.4, C.GREY_TXT, "normal", 0.45))
            # shown-face block (from the plotted masks)
            lines.append((f"{p['face'].capitalize()} face (shown):", 6.7, C.INK, "normal", 0.35))
            lines.append((f"TP {c['tp']:,} · FP {c['fp']:,} · FN {c['fn']:,} px", 6.4, C.GREY_TXT, "normal", 0.31))
            lines.append((f"{p['face']}-face severity,", 6.4, C.GREY_TXT, "normal", 0.31))
            lines.append(("ref. / pred.: " + face_sev_txt, 6.4, C.GREY_TXT, "normal", 0.33))
        first_img_left = left_txt + (slot - ww) / 2
        for txt, fs, col, fw, step in lines:
            t = fig.text(fx(x_t), fy(yt), txt, ha="left", va="top", fontsize=fs, color=col, fontweight=fw)
            text_artists.append((t, first_img_left))
            yt += step
        drawn.append(dict(role=p["role"], printed_iou=iou_txt,
                          printed_leaf_iou=(iou_txt if disease == "roya" else f"{float(p['leaf_iou_exact']):.3f}"),
                          printed_leaf_severity_ref_pred=leaf_sev_txt,
                          printed_shown_face_severity_ref_pred=(None if disease == "roya" else face_sev_txt),
                          text_lines=[ln[0] for ln in lines],
                          crop_px=[int(h_px), int(w_px)], panel_cm=[round(hh, 3), round(ww, 3)]))
        y_top += hh + row_gap

    # ---- legend
    handles = [
        Line2D([], [], color=GREEN, lw=1.2, path_effects=[patheffects.Stroke(linewidth=2.2, foreground="black", alpha=0.55), patheffects.Normal()]),
        Line2D([], [], color=MAGENTA, lw=1.2, path_effects=[patheffects.Stroke(linewidth=2.2, foreground="black", alpha=0.55), patheffects.Normal()]),
        Patch(facecolor=hex_of(TP_RGB), edgecolor="#9a9a9a", lw=0.6),
        Patch(facecolor=hex_of(FP_RGB), edgecolor="none"),
        Patch(facecolor=hex_of(FN_RGB), edgecolor="none"),
        Patch(facecolor=hex_of(LEAF_RGB), edgecolor="none"),
    ]
    labels = ["Reference lesion outline", "Predicted lesion outline", "TP: lesion in both",
              "FP: prediction only", "FN: reference only", "Leaf, no lesion in either"]
    lax = fig.add_axes([fx(0.1), 0.0, fx(fig_w - 0.2), (legend_h - 0.05) / fig_h])
    lax.set_axis_off()
    lax.legend(handles, labels, ncol=3, loc="center", frameon=False, fontsize=7.2, handlelength=1.5,
               handleheight=0.9, columnspacing=1.6, labelcolor=C.INK)
    # no row text may reach the image column (checked in rendered coordinates)
    renderer = fig.canvas.get_renderer()
    fig_w_px = fig.get_size_inches()[0] * fig.dpi
    for t, limit_cm in text_artists:
        right_cm = t.get_window_extent(renderer).x1 / fig_w_px * fig_w
        assert right_cm < limit_cm - 0.08, (t.get_text(), right_cm, limit_cm)
    meta = dict(scale_cm_per_px=s, fig_size_cm=[fig_w, round(fig_h, 3)])
    return fig, drawn, meta


def pdf_verify(pdf_path, data):
    """Check the saved PDF, not only the arrays.
    (a) Storage: matplotlib stores each raster bottom-up and places it with a y-flipped matrix. The stored error-map
        raster, flipped back, must be pixel-identical to the computed array (the input crop likewise, 3 placements).
    (b) Display: the page is rendered by MuPDF without anti-aliasing at 16x zoom inside each error-map / input
        placement, and sampled at every image-pixel centre; the samples must equal the computed arrays exactly,
        which also proves the upright orientation. TP/FP/FN are recounted from the rendered error map.
    """
    import pymupdf
    doc = pymupdf.open(pdf_path)
    assert len(doc) == 1
    page = doc[0]
    stored = {}
    for img in page.get_images(full=True):
        pix = pymupdf.Pixmap(doc, img[0])
        if pix.alpha:
            pix = pymupdf.Pixmap(pix, 0)
        stored[img[0]] = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3].copy()
    placements = page.get_image_info(xrefs=True)
    aa = int(pymupdf.TOOLS.show_aa_level()["graphics"])
    pymupdf.TOOLS.set_aa_level(0)
    zoom = 16

    def render_sample(bbox, shape):
        x0, y0, x1, y1 = bbox
        h, w = shape[:2]
        pm = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=pymupdf.Rect(x0, y0, x1, y1), alpha=False)
        img = np.frombuffer(pm.samples, np.uint8).reshape(pm.height, pm.width, pm.n)[..., :3]
        jj = np.floor((x0 + (np.arange(w) + 0.5) / w * (x1 - x0)) * zoom - pm.irect[0]).astype(int)
        ii = np.floor((y0 + (np.arange(h) + 0.5) / h * (y1 - y0)) * zoom - pm.irect[1]).astype(int)
        return img[ii][:, jj]

    out = []
    try:
        for d in data:
            em_x = [x for x, r in stored.items() if r.shape == d["em"].shape and np.array_equal(r[::-1], d["em"])]
            in_x = [x for x, r in stored.items() if r.shape == d["rgb"].shape and np.array_equal(r[::-1], d["rgb"])]
            # the input crop is embedded once per column (3 XObjects with identical pixels, or 1 if deduplicated)
            assert len(em_x) == 1 and len(in_x) in (1, 3), (em_x, in_x)
            em_pl = [pl for pl in placements if pl["xref"] == em_x[0]]
            in_pl = sorted([pl for pl in placements if pl["xref"] in in_x], key=lambda pl: pl["bbox"][0])
            assert len(em_pl) == 1 and len(in_pl) == 3, (len(em_pl), len(in_pl))
            assert em_pl[0]["transform"][3] < 0 and all(pl["transform"][3] < 0 for pl in in_pl)  # flipped placement
            em_r = render_sample(em_pl[0]["bbox"], d["em"].shape)
            in_r = render_sample(in_pl[0]["bbox"], d["rgb"].shape)   # the uncovered input column
            assert np.array_equal(em_r, d["em"]), int((em_r != d["em"]).any(-1).sum())
            assert np.array_equal(in_r, d["rgb"]), int((in_r != d["rgb"]).any(-1).sum())
            c = counts_from_raster(em_r)
            assert c == d["counts"] and iou_of(c) == d["iou_plot"]
            out.append(dict(role=d["pick"]["role"], stored_error_map_identical=True, stored_input_identical=True,
                            input_placements=3, rendered_error_map_identical=True, rendered_input_identical=True,
                            rendered_counts=c, rendered_iou=float(iou_of(c)), render="MuPDF, AA off, zoom 16"))
    finally:
        pymupdf.TOOLS.set_aa_level(aa)
        doc.close()
    return out


def main():
    rows = C.read_rows()
    res = {"model": C.MODEL_NAME[ARCH], "colours": dict(reference_outline_and_FN=GREEN, prediction_outline_and_FP=MAGENTA,
           TP=hex_of(TP_RGB), leaf_no_lesion=hex_of(LEAF_RGB), background=hex_of(BG_RGB)),
           "outline": dict(level=0.5, linewidth_pt=OUTLINE_LW, halo="black, alpha 0.55, %.2f pt" % HALO_LW),
           "crop_margin_px": MARGIN, "figures": {}}
    for disease, stem, paper_fig, old in (("roya", "figure13", 13, "figures/figure9.png"),
                                           ("ojo", "figure14", 14, "figures/figure10.png")):
        picks, ranking = select(rows, disease)
        data = build(disease, picks, rows)
        fig, drawn, meta = draw(disease, data)
        files = C.save(fig, stem)
        plt.close(fig)
        pdfv = pdf_verify(files["pdf"], data)
        entry = dict(paper_figure=paper_fig, replaces_manuscript_file=old, files=files, layout=meta,
                     n_test_leaves=len(ranking), rows=[])
        for d, dr, pv in zip(data, drawn, pdfv):
            p = d["pick"]
            entry["rows"].append(dict(
                role=p["role"], rank=p["rank"], leaf_id=p["leaf_id"], face_shown=p["face"], face_rule=p["face_rule"],
                tie_check=p["tie"], file=p["face_file"][p["face"]],
                leaf_iou=float(p["leaf_iou_exact"]), leaf_iou_3dp=f"{float(p['leaf_iou_exact']):.3f}",
                face_iou_abaxial=float(p["face_iou_exact"]["abaxial"]), face_iou_adaxial=float(p["face_iou_exact"]["adaxial"]),
                iou_recomputed_from_plotted_masks=float(d["iou_plot"]), iou_recomputed_exact=str(d["iou_plot"]),
                counts_plotted=d["counts"], full_frame_counts=d["full"], other_face=d["other"],
                ref_severity_pct_shown_face=p["face_sev"][p["face"]][0], pred_severity_pct_shown_face=p["face_sev"][p["face"]][1],
                leaf_ref_severity_pct=p["ref_sev"], leaf_pred_severity_pct=p["pred_sev"], ref_grade=p["ref_grade"],
                crop_box_yx=dict(zip(("y0", "y1", "x0", "x1"), d["box"])), outline_registration=d["outline_registration"],
                printed=dr, pdf_check=pv,
                verified=dict(csv_counts=True, plotted_arrays_eq_raster_eq_csv=True, outlines_registered=True,
                              pdf_raster_identical=True, leaf_mean_consistent=True,
                              printed_leaf_severity_eq_figure12_value=True)))
        entry["ranking"] = ranking
        res["figures"][stem] = entry
        for r in entry["rows"]:
            extra = "" if disease == "roya" else f" (ab {r['face_iou_abaxial']:.3f}, ad {r['face_iou_adaxial']:.3f})"
            print(f"{stem} {r['role']:6s} rank {r['rank']:2d} {r['leaf_id']} {r['face_shown']}: "
                  f"leaf IoU {r['leaf_iou_3dp']}{extra}; plotted-face IoU {r['iou_recomputed_from_plotted_masks']:.6f} "
                  f"TP/FP/FN {r['counts_plotted']}  crop {r['printed']['crop_px']}")
    (C.OUT / "qualitative_results.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
