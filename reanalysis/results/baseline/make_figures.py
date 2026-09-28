"""Figures for the color-threshold baseline (PG-26038 major revision).

Reads results.json, per_leaf_test.csv and per_leaf_healthy.csv written by baseline.py, and (for the overlays)
rebuilds the evaluation-space images with baseline.load() and reads the published SegFormer-B5 class maps from
reanalysis/out/preds. No inference is run.

Outputs (this folder):
  figure_baseline_comparison.pdf/.png  leaf IoU, severity MAE and healthy-leaf flag rate, baseline vs networks
  figure_baseline_overlays.pdf/.png    error maps (baseline and SegFormer-B5) for 3 test leaves per disease:
                                       median baseline IoU, largest |SegFormer-B5 - baseline| IoU, and the healthy
                                       test leaf with the median baseline predicted severity (paper's leaf rule);
                                       the healthy-leaf panels carry a zoom inset of the 32 x 32-px window that holds
                                       the most color-threshold lesion pixels
Style: figure5 of the manuscript (Arial, bold panel titles with grey italic species, light grey axes), 16 cm wide,
vector PDF with TrueType fonts (pdf.fonttype 42) and 600 dpi PNG.
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
import matplotlib
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, OUT)
import baseline as bl  # noqa: E402

PREDS = f"{bl.BASE}/reanalysis/out/preds"
CM = 1 / 2.54
INK, GREY_TXT, GREY_TICK, GREY_AXIS, GREY_NOTE = "#1a1a1a", "#555555", "#8a8a8a", "#c8c8c8", "#7a7a7a"
DIS_COLOR = {"roya": "#C96A3D", "ojo": "#3884AD"}          # rust orange / leaf-spot blue of figure5
PANELS = [("roya", "(a) Coffee leaf rust", "Hemileia vastatrix"),
          ("ojo", "(b) American leaf spot", "Mycena citricolor")]
NETS = [("segformer", "SegFormer-B5", "#0072B2", "^", 6.6),
        ("unetpp", "UNet++", "#009E73", "s", 5.6),
        ("deeplab", "DeepLabV3+", "#E69F00", "o", 6.0)]
BASE_LABEL = "Color threshold"
BASE_COLOR = "#4D4D4D"                                       # neutral dark grey: distinct from all three model colors
ZOOM_WIN = 32                                                # zoom-inset window (letterbox px) for the healthy leaves
INSET_FRAC = 0.46                                            # inset width as a fraction of the panel width
TP_C, FP_C, FN_C, REF_C = "#F0E442", "#CC79A7", "#56B4E9", "#FF00FF"


def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial"], "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "axes.edgecolor": GREY_AXIS, "axes.linewidth": 0.9,
        "xtick.color": GREY_AXIS, "ytick.color": GREY_AXIS, "xtick.labelcolor": GREY_TICK, "ytick.labelcolor": INK,
        "xtick.major.width": 0.9, "ytick.major.width": 0.9, "xtick.major.size": 3.2, "ytick.major.size": 3.2,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})


def save(fig, stem):
    fig.savefig(f"{OUT}/{stem}.pdf")
    fig.savefig(f"{OUT}/{stem}.png", dpi=600)
    plt.close(fig)
    with Image.open(f"{OUT}/{stem}.png") as im:           # opaque white RGB
        rgb = Image.new("RGB", im.size, "white"); rgb.paste(im, mask=im.convert("RGBA").getchannel("A"))
    rgb.save(f"{OUT}/{stem}.png", dpi=(600, 600))


def panel_title(ax, title, species, dy=15):
    ax.annotate(title, xy=(0, 1), xycoords="axes fraction", xytext=(0, dy + 11), textcoords="offset points",
                ha="left", va="bottom", fontsize=10.2, fontweight="bold", color=INK)
    ax.annotate(species, xy=(0, 1), xycoords="axes fraction", xytext=(0, dy), textcoords="offset points",
                ha="left", va="bottom", fontsize=8.4, fontstyle="italic", color=GREY_TXT)


# ------------------------------------------------------------------------------------------------ figure 1
def figure_comparison(res):
    setup_style()
    fig, axes = plt.subplots(3, 2, figsize=(16 * CM, 13.2 * CM))
    fig.subplots_adjust(left=0.175, right=0.965, bottom=0.075, top=0.885, wspace=0.14, hspace=0.62)
    ypos = {"segformer": 3.0, "unetpp": 2.0, "deeplab": 1.0, "baseline": 0.0}
    rows = [("iou", "Leaf lesion IoU (95% bootstrap CI)"),
            ("mae", "Severity MAE, pp (95% bootstrap CI)"),
            ("fp", "Healthy leaves ≥G1 on either face, % (95% CI)")]
    xlim = {("iou", "roya"): (0.40, 1.0), ("iou", "ojo"): (0.40, 1.0),
            ("mae", "roya"): (0.0, 4.2), ("mae", "ojo"): (0.0, 0.40),
            ("fp", "roya"): (-4, 116), ("fp", "ojo"): (-4, 116)}
    xticks = {("iou", "roya"): [0.4, 0.6, 0.8, 1.0], ("iou", "ojo"): [0.4, 0.6, 0.8, 1.0],
              ("mae", "roya"): [0, 1, 2, 3, 4], ("mae", "ojo"): [0, 0.1, 0.2, 0.3, 0.4],
              ("fp", "roya"): [0, 25, 50, 75, 100], ("fp", "ojo"): [0, 25, 50, 75, 100]}
    spine_end = {("iou", "roya"): 1.0, ("iou", "ojo"): 1.0, ("mae", "roya"): 4.0, ("mae", "ojo"): 0.4,
                 ("fp", "roya"): 100, ("fp", "ojo"): 100}

    def value(dis, key, metric):
        t = res["test"][dis]["methods"][key]
        if metric == "fp":
            e = res["specificity"]["test"]["methods"][f"{dis}/{key}"]["either_face"]["ge_0.5pct"]
            return 100 * e["fp_rate"], [100 * v for v in e["fp_rate_wilson95"]], f"{e['flagged']}/{e['n']}"
        fmt = "{:.3f}" if metric == "iou" else "{:.2f}"
        return t[metric], t["ci95"][metric], fmt.format(t[metric])

    for c, (dis, title, species) in enumerate(PANELS):
        for r, (metric, xlabel) in enumerate(rows):
            ax = axes[r, c]
            x0, x1 = xlim[(metric, dis)]
            span = x1 - x0
            ax.axhline(0.5, color="#e6e6e6", lw=0.7, zorder=0)          # separates the baseline from the networks
            items = [(a, name, col, mk, ms, ypos[a], "full") for a, name, col, mk, ms in NETS]
            items.append(("baseline", BASE_LABEL, BASE_COLOR, "D", 5.4, ypos["baseline"], "full"))
            cons_differs = res["test"][dis]["secondary_constrained_config"]["threshold"] != res["test"][dis]["selected_config"]["threshold"] or \
                res["test"][dis]["secondary_constrained_config"]["index"] != res["test"][dis]["selected_config"]["index"]
            if cons_differs:
                items[-1] = ("baseline", BASE_LABEL, BASE_COLOR, "D", 5.4, 0.17, "full")
                items.append(("baseline_constrained", "constrained", BASE_COLOR, "D", 5.0, -0.25, "none"))
            for key, name, col, mk, ms, y, fill in items:
                v, (lo, hi), lab = value(dis, key, metric)
                ax.plot([lo, hi], [y, y], color=col, lw=2.0 if fill == "full" else 1.3, solid_capstyle="butt", zorder=2,
                        alpha=1.0 if fill == "full" else 0.9)
                if fill == "full":
                    ax.plot(v, y, marker=mk, ms=ms, color=col, mec=col, mew=0.6, ls="none", zorder=3, clip_on=False)
                else:
                    ax.plot(v, y, marker=mk, ms=ms, mfc="white", mec=col, mew=1.3, ls="none", zorder=3, clip_on=False)
                txt = lab if key != "baseline_constrained" else f"{lab}  constrained"
                # keep the label clear of the marker when the point sits at (or near) the upper CI limit
                ax_w_pt = ax.get_position().width * fig.get_figwidth() * 72
                gap_pt = (hi - v) / span * ax_w_pt
                dx = 4 + max(0.0, ms / 2 + 1 - gap_pt)
                ax.annotate(txt, xy=(hi, y), xytext=(dx, 0), textcoords="offset points", ha="left", va="center",
                            fontsize=6.6, color=GREY_TXT if fill == "full" else GREY_NOTE, annotation_clip=False)
            ax.set_ylim(-0.62, 3.55)
            ax.set_xlim(x0, x1)
            ax.set_xticks(xticks[(metric, dis)])
            if metric == "iou":
                ax.set_xticklabels([f"{t:.1f}" for t in xticks[(metric, dis)]])
            ax.tick_params(axis="x", labelsize=7.6, pad=2)
            ax.set_yticks([3, 2, 1, 0])
            if c == 0:
                ax.set_yticklabels(["SegFormer-B5", "UNet++", "DeepLabV3+", BASE_LABEL], fontsize=8.2)
            else:
                ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0, pad=4)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.spines["left"].set_bounds(-0.62, 3.55)
            ax.spines["bottom"].set_bounds(x0, spine_end[(metric, dis)])
            ax.set_xlabel(xlabel, fontsize=7.8, color=GREY_TXT, labelpad=2.5)
            if r == 0:
                panel_title(ax, title, species)
    return fig


# ------------------------------------------------------------------------------------------------ figure 2
def load_face(pack, split, file):
    rgb, lab = bl.load(pack, file)
    return rgb, lab


def baseline_mask(rgb, lab, cfg):
    """Selected configuration on one face (global thresholds compared exactly, as in baseline.py)."""
    index, rule, thr = cfg
    leaf = lab >= 1
    if rule == "global":
        return bl.global_pred_image(rgb, index, thr) & leaf
    v = bl.index_image(rgb, index)
    return (v > bl.otsu(v[leaf])) & leaf


def zoom_window(pred, win=ZOOM_WIN):
    """Top-left corner of the win x win window holding the most predicted pixels."""
    dens = ndimage.uniform_filter(pred.astype(np.float64), size=win, mode="constant")
    cy, cx = np.unravel_index(int(np.argmax(dens)), dens.shape)
    return int(np.clip(cx - win // 2, 0, bl.SIZE - win)), int(np.clip(cy - win // 2, 0, bl.SIZE - win))


def inset_corner(lab, box, win, frac, aspect):
    """Panel corner (axes-fraction origin) for the inset: least leaf under it, not covering the zoom window."""
    x0, x1, y0, y1 = box
    wd = (x1 - x0) * frac                       # inset side in data px (square in print)
    leaf = lab >= 1
    best = None
    for bx in (0.0, 1.0 - frac):
        for by in (0.0, 1.0 - frac / aspect):
            cx0 = x0 + bx * (x1 - x0); cy0 = y1 - (by + frac / aspect) * (y1 - y0)   # image rows grow downwards
            X0, Y0 = int(max(cx0, 0)), int(max(cy0, 0))
            X1, Y1 = int(min(cx0 + wd, bl.SIZE)), int(min(cy0 + wd, bl.SIZE))
            cover = int(leaf[Y0:Y1, X0:X1].sum()) if X1 > X0 and Y1 > Y0 else 0
            zx, zy = win
            hits = not (zx + ZOOM_WIN < cx0 or zx > cx0 + wd or zy + ZOOM_WIN < cy0 or zy > cy0 + wd)
            score = (hits, cover)
            if best is None or score < best[0]:
                best = (score, bx, by)
    return best[1], best[2]


def seg_mask(model_dis, image_pack, split, file):
    p = np.asarray(Image.open(f"{PREDS}/{model_dis}_segformer/{image_pack}_{split}/{file}"))
    return p == 2


def error_rgb(rgb, lab, pred):
    """Leaf in grey, TP/FP/FN colored, white background outside the reference leaf."""
    leaf = lab >= 1; ref = lab == 2
    lum = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]) / 255.0
    g = 0.34 + 0.34 * np.clip(lum / max(np.percentile(lum[leaf], 99), 1e-6), 0, 1)
    out = np.ones(rgb.shape, np.float64)
    out[leaf] = g[leaf, None]
    to = lambda h: np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    out[pred & ref] = to(TP_C); out[pred & ~ref] = to(FP_C); out[~pred & ref] = to(FN_C)
    return out


def scan_rgb(rgb, lab):
    leaf = lab >= 1
    out = np.ones(rgb.shape, np.float64); out[leaf] = rgb[leaf] / 255.0
    return out


def crop_box(lab, aspect, pad=10):
    ys, xs = np.nonzero(lab >= 1)
    y0, y1, x0, x1 = ys.min() - pad, ys.max() + pad, xs.min() - pad, xs.max() + pad
    h, w = y1 - y0, x1 - x0
    if h / w > aspect:                       # too tall -> widen
        nw = h / aspect; x0 -= (nw - w) / 2; x1 += (nw - w) / 2
    else:
        nh = w * aspect; y0 -= (nh - h) / 2; y1 += (nh - h) / 2
    return x0, x1, y0, y1


def pick_examples(res):
    per = list(csv.DictReader(open(f"{OUT}/per_leaf_test.csv", newline="")))
    hea = list(csv.DictReader(open(f"{OUT}/per_leaf_healthy.csv", newline="")))
    ex = {}
    for dis in bl.DIS:
        rr = [r for r in per if r["disease"] == dis]
        b = np.array([float(r["baseline_iou"]) for r in rr]); s = np.array([float(r["segformer_iou"]) for r in rr])
        order = np.argsort(b, kind="stable"); med = rr[order[len(order) // 2]]          # n odd: exact median
        big = rr[int(np.argmax(np.abs(s - b)))]
        hh = sorted([h for h in hea if h["split"] == "test" and h["model"] == f"{dis}/baseline"],
                    key=lambda h: (float(h["pred_sev_leaf_rule"]), h["leaf_id"]))
        hmed = hh[len(hh) // 2]
        ex[dis] = [("Median threshold\nIoU", med["leaf_id"], dis, False),
                   ("Largest IoU\ndifference", big["leaf_id"], dis, False),
                   ("Healthy leaf,\nmedian severity", hmed["leaf_id"], "roya", True)]
    return ex


def figure_overlays(res):
    setup_style()
    ex = pick_examples(res)
    cfg = {dis: (res["test"][dis]["selected_config"]["index"], res["test"][dis]["selected_config"]["rule"],
                 res["test"][dis]["selected_config"]["threshold"]) for dis in bl.DIS}
    # layout in centimetres
    ncol, aspect = 6, 1.50
    W, L, R, GX, GG = 16.0, 2.05, 0.08, 0.10, 0.42          # width, left (row labels), right, column gap, group gap
    cw = (W - L - R - GG - GX * (ncol - 2)) / ncol
    ch = cw * aspect
    T_TITLE, T_COL, CAP, ROWGAP, LEG, BOT = 1.28, 0.62, 0.40, 0.22, 0.55, 0.12
    H = T_TITLE + T_COL + 3 * ch + 3 * CAP + 2 * ROWGAP + LEG + BOT
    fig = plt.figure(figsize=(W * CM, H * CM))
    fx = lambda v: v / W
    fy = lambda v: v / H
    meta = {"examples": {}}
    row_names = ["Scan and\nreference", BASE_LABEL.replace(" ", "\n"), "SegFormer-B5"]
    for g, (dis, title, species) in enumerate(PANELS):
        for k, (label, lid, pack, healthy) in enumerate(ex[dis]):
            col = g * 3 + k
            x = fx(L + col * cw + (col - g) * GX + g * GG)
            file = f"{lid}__abaxial.png"
            split = "test"
            rgb, lab = load_face(pack, split, file)
            pb = baseline_mask(rgb, lab, cfg[dis])
            ps = seg_mask(dis, pack, split, file)
            leaf = lab >= 1; ref = lab == 2
            ref_sev = 100.0 * ref.sum() / leaf.sum()
            sev_b = 100.0 * pb.sum() / leaf.sum()
            seg_leaf = np.asarray(Image.open(f"{PREDS}/{dis}_segformer/{pack}_{split}/{file}")) >= 1
            sev_s = 100.0 * ps.sum() / seg_leaf.sum()
            iou = lambda p: (int((p & ref).sum()) / int((p | ref).sum())) if (p | ref).any() else float("nan")
            x0, x1, y0, y1 = crop_box(lab, aspect)
            imgs = [scan_rgb(rgb, lab), error_rgb(rgb, lab, pb), error_rgb(rgb, lab, ps)]
            if healthy:
                zx, zy = zoom_window(pb)
                ibx, iby = inset_corner(lab, (x0, x1, y0, y1), (zx, zy), INSET_FRAC, aspect)
                zoom_mag = (x1 - x0) * INSET_FRAC / ZOOM_WIN
                in_win = lambda p: int(p[zy:zy + ZOOM_WIN, zx:zx + ZOOM_WIN].sum())
            if healthy:
                caps = [f"healthy, ref {ref_sev:.1f}%", f"{sev_b:.2f}%", f"{sev_s:.2f}%"]
            else:
                caps = [f"ref {ref_sev:.1f}% (G{bl.grade(ref_sev)})", f"IoU {iou(pb):.2f} · {sev_b:.1f}%",
                        f"IoU {iou(ps):.2f} · {sev_s:.1f}%"]
            meta["examples"][f"{dis}/{k}"] = dict(role=label.replace("\n", " "), leaf_id=lid, face="abaxial", pack=pack, healthy=healthy,
                                                   ref_sev=ref_sev, baseline_face_iou=iou(pb), baseline_face_sev=sev_b,
                                                   segformer_face_iou=iou(ps), segformer_face_sev=sev_s)
            if healthy:
                meta["examples"][f"{dis}/{k}"]["zoom_inset"] = dict(
                    window_xy_letterbox=[zx, zy], window_px=ZOOM_WIN, magnification_vs_panel=round(zoom_mag, 2),
                    baseline_px_in_window=in_win(pb), baseline_px_total=int(pb.sum()),
                    segformer_px_in_window=in_win(ps), segformer_px_total=int(ps.sum()),
                    rule="window with the most color-threshold lesion pixels on this face")
            for r in range(3):
                y_top = H - T_TITLE - T_COL - r * (ch + CAP + ROWGAP)
                ax = fig.add_axes([x, fy(y_top - ch), fx(cw), fy(ch)])
                ax.imshow(imgs[r], interpolation="none" if r else "antialiased")
                if r == 0 and ref.any():
                    ax.contour(ref.astype(float), levels=[0.5], colors=[REF_C], linewidths=0.35)
                ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                if healthy:
                    ax.add_patch(Rectangle((zx - 0.5, zy - 0.5), ZOOM_WIN, ZOOM_WIN, fill=False, ec=INK, lw=0.45, zorder=5))
                    axi = ax.inset_axes([ibx, iby, INSET_FRAC, INSET_FRAC / aspect])
                    axi.imshow(imgs[r], interpolation="none" if r else "antialiased")
                    axi.set_xlim(zx - 0.5, zx + ZOOM_WIN - 0.5); axi.set_ylim(zy + ZOOM_WIN - 0.5, zy - 0.5)
                    axi.set_xticks([]); axi.set_yticks([])
                    for sp in axi.spines.values():
                        sp.set_edgecolor(INK); sp.set_linewidth(0.45)
                    axi.text(0.05, 0.05, f"×{zoom_mag:.0f}", transform=axi.transAxes, ha="left", va="bottom",
                             fontsize=5.2, color=INK, bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.85))
                ax.text(0.5, -0.02, caps[r], transform=ax.transAxes, ha="center", va="top", fontsize=6.5, color=GREY_TXT)
                if r == 0:
                    ax.text(0.5, 1.025, label, transform=ax.transAxes, ha="center", va="bottom", fontsize=6.6,
                            color=INK, linespacing=1.1)
                if col == 0:
                    ax.text(-0.06, 0.5, row_names[r], transform=ax.transAxes, ha="right", va="center", fontsize=7.6,
                            color=INK, linespacing=1.15)
                if r == 0 and k == 0:
                    panel_title(ax, title, species, dy=24)
    handles = [Line2D([], [], color=REF_C, lw=1.0, label="Reference outline"),
               Patch(facecolor=TP_C, edgecolor="none", label="Overlap (true lesion)"),
               Patch(facecolor=FP_C, edgecolor="none", label="Predicted only (false lesion)"),
               Patch(facecolor=FN_C, edgecolor="none", label="Reference only (missed)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7.0, handlelength=1.3,
               handleheight=0.9, columnspacing=1.4, bbox_to_anchor=(fx(L + (W - L - R) / 2), fy(BOT)))
    return fig, meta


def main():
    res = json.load(open(f"{OUT}/results.json"))
    fig = figure_comparison(res)
    save(fig, "figure_baseline_comparison")
    fig, meta = figure_overlays(res)
    save(fig, "figure_baseline_overlays")
    json.dump(meta, open(f"{OUT}/overlay_examples.json", "w"), indent=2)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
