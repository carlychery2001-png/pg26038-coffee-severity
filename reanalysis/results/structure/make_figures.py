"""Figures for the validation-set / data-structure reanalysis (Supplementary Materials).

Reads results.json (written by structure_analysis.py) and writes, in this folder:
  figure_val_test.pdf/.png          leaf IoU and severity MAE, validation vs test, six models, 95% leaf-bootstrap CIs
  figure_grade_distribution.pdf/.png reference grade mix per split (rust: mask-derived; leaf spot: visual and mask-derived)
Style follows the paper's two-panel figures (figure5): Arial, bold panel titles with the grey italic species
name beneath, light grey axes, Okabe-Ito model colours (SegFormer-B5 blue triangle, UNet++ green square,
DeepLabV3+ orange circle), 16 cm wide, vector PDF with TrueType fonts (pdf.fonttype 42) and a 600 dpi PNG.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image

OUT = Path(__file__).resolve().parent
RES = json.loads((OUT / "results.json").read_text(encoding="utf-8"))

INK = "#1a1a1a"
GREY_TXT = "#555555"
GREY_TICK = "#8a8a8a"
GREY_AXIS = "#c8c8c8"
PANELS = [("roya", "(a) Coffee leaf rust", "Hemileia vastatrix"),
          ("ojo", "(b) American leaf spot", "Mycena citricolor")]
MODELS = ["segformer", "unetpp", "deeplab"]
MNAME = {"segformer": "SegFormer-B5", "unetpp": "UNet++", "deeplab": "DeepLabV3+"}
STYLE = {"segformer": dict(color="#0072B2", marker="^", ms=6.6),
         "unetpp": dict(color="#009E73", marker="s", ms=5.6),
         "deeplab": dict(color="#E69F00", marker="o", ms=6.0)}
# Disease hues of figure5 (rust orange, leaf-spot blue); light-to-dark ramps for grades G0-G3.
RAMP = {"roya": ["#F6E0D3", "#E4A27F", "#C4602F", "#7A3413"],
        "ojo": ["#DCEAF3", "#8FBAD6", "#2A78A5", "#113F5E"]}
GRADE_LBL = ["G0 (<0.5%)", "G1 (0.5–5%)", "G2 (5–20%)", "G3 (≥20%)"]
CM = 1 / 2.54


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial"],
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.edgecolor": GREY_AXIS, "axes.linewidth": 0.9,
        "xtick.color": GREY_AXIS, "ytick.color": GREY_AXIS,
        "xtick.labelcolor": GREY_TICK, "ytick.labelcolor": INK,
        "xtick.major.width": 0.9, "ytick.major.width": 0.9,
        "xtick.major.size": 3.5, "ytick.major.size": 3.5,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    })


def panel_title(ax, title, species):
    ax.annotate(title, xy=(0, 1), xycoords="axes fraction", xytext=(0, 17), textcoords="offset points",
                ha="left", va="bottom", fontsize=10.2, fontweight="bold", color=INK)
    ax.annotate(species, xy=(0, 1), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points",
                ha="left", va="bottom", fontsize=8.4, fontstyle="italic", color=GREY_TXT)


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=600)
    plt.close(fig)
    with Image.open(OUT / f"{stem}.png") as im:     # flatten to opaque RGB with a true white background
        rgb = Image.new("RGB", im.size, "white")
        rgb.paste(im, mask=im.convert("RGBA").getchannel("A"))
    rgb.save(OUT / f"{stem}.png", dpi=(600, 600))


# ---------------------------------------------------------------------------------------------------------------
def figure_val_test():
    vt = RES["val_vs_test"]["results"]
    fig, axes = plt.subplots(2, 2, figsize=(16 * CM, 10.6 * CM))
    fig.subplots_adjust(left=0.165, right=0.975, bottom=0.155, top=0.885, wspace=0.45, hspace=0.55)
    rows = [("iou", "Leaf lesion IoU", {"roya": (0.56, 0.84, [0.6, 0.7, 0.8]), "ojo": (0.74, 0.92, [0.75, 0.8, 0.85, 0.9])}),
            ("mae", "Severity MAE (percentage points)", {"roya": (0, 3.0, [0, 1, 2, 3]), "ojo": (0, 0.85, [0, 0.2, 0.4, 0.6, 0.8])})]
    ypos = {m: 2 - i for i, m in enumerate(MODELS)}
    off = {"val": 0.17, "test": -0.17}
    for r, (key, xlabel, lims) in enumerate(rows):
        for c, (dis, title, species) in enumerate(PANELS):
            ax = axes[r, c]
            lo_x, hi_x, ticks = lims[dis]
            for m in MODELS:
                st = STYLE[m]
                for s in ("val", "test"):
                    e = vt[dis][m][s]
                    y = ypos[m] + off[s]
                    lo, hi = e[f"{key}_ci95"]
                    filled = s == "test"
                    ax.plot([lo, hi], [y, y], color=st["color"], lw=2.1 if filled else 1.3,
                            solid_capstyle="butt", zorder=2, alpha=1.0 if filled else 0.85)
                    ax.plot(e[key], y, marker=st["marker"], ms=st["ms"] + (0.7 if filled else 0), ls="none", zorder=3,
                            mfc=st["color"] if filled else "white", mec=st["color"] if not filled else "white",
                            mew=1.3 if not filled else 1.0, clip_on=False)
            ax.set_yticks([ypos[m] for m in MODELS], [MNAME[m] for m in MODELS], fontsize=8.4)
            ax.set_ylim(-0.55, 2.55)
            pad = 0.025 * (hi_x - lo_x)
            ax.set_xlim(lo_x - pad, hi_x + pad)
            ax.set_xticks(ticks)
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
                lambda v, _p, k=key: f"{v:.2f}".rstrip("0").rstrip(".") if k == "iou" else f"{v:g}"))
            ax.tick_params(axis="x", labelsize=7.8, pad=2.5)
            ax.tick_params(axis="y", pad=3)
            ax.set_xlabel(xlabel, fontsize=8.2, color=GREY_TXT, labelpad=3)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_bounds(lo_x - pad, hi_x + pad)
            if r == 0:
                panel_title(ax, title, species)
    grey = "#6f6f6f"
    handles = [Line2D([0], [0], color=grey, lw=1.3, marker="o", ms=5.5, mfc="white", mec=grey, mew=1.3,
                      label="Validation"),
               Line2D([0], [0], color=grey, lw=2.1, marker="o", ms=5.5, mfc=grey, mec="white", mew=1.0,
                      label="Test")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=7.8,
               bbox_to_anchor=(0.57, 0.0), handlelength=2.2, columnspacing=2.4, labelcolor=GREY_TXT)
    save(fig, "figure_val_test")


# ---------------------------------------------------------------------------------------------------------------
def figure_grade_distribution():
    gd = RES["grade_distributions"]["results"]
    fig, axes = plt.subplots(1, 2, figsize=(16 * CM, 7.9 * CM))
    fig.subplots_adjust(left=0.13, right=0.965, bottom=0.255, top=0.84, wspace=0.42)
    split_name = {"train": "Train", "val": "Validation", "test": "Test"}
    for ax, (dis, title, species) in zip(axes, PANELS):
        H = 0.40 if dis == "roya" else 0.58      # similar printed bar thickness in both panels
        bars, heads = [], []          # bars: (y, tick label, counts); heads: (y, text)
        y = 0.0
        for s in ("test", "val", "train"):          # drawn bottom-up; train at the top
            e = gd[dis][s]
            if dis == "roya":
                bars.append((y, "Mask-derived", [e["mask_grade_counts"][f"G{g}"] for g in range(4)]))
                top = y
            else:
                bars.append((y, "Mask-derived", [e["mask_grade_counts"][f"G{g}"] for g in range(4)]))
                bars.append((y + 0.72, "Visual", [e["visual_grade_counts"][f"G{g}"] for g in range(4)]))
                top = y + 0.72
            heads.append((top + H / 2 + 0.1, f"{split_name[s]}, n = {e['n_diseased_leaves']} leaves"))
            y = top + 1.25
        for yb, _lbl, cnt in bars:
            tot = sum(cnt); left = 0.0
            for g, k in enumerate(cnt):
                if k == 0:
                    continue
                w = 100 * k / tot
                ax.barh(yb, w, left=left, height=H, color=RAMP[dis][g], edgecolor="white", linewidth=0.8, zorder=2)
                if w >= 4.5:
                    ax.text(left + w / 2, yb, str(k), ha="center", va="center", fontsize=6.6 if w >= 7 else 6.0,
                            color="white" if g >= 2 else INK, zorder=3)
                elif g == 3:     # thin last segment: count just outside the bar end
                    ax.text(101.2, yb, str(k), ha="left", va="center", fontsize=6.0, color=GREY_TXT, clip_on=False)
                left += w
        for yh, txt in heads:
            ax.text(0, yh, txt, ha="left", va="bottom", fontsize=7.4, color=GREY_TXT)
        ax.set_yticks([b[0] for b in bars], [b[1] for b in bars], fontsize=7.6)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.tick_params(axis="x", labelsize=7.8, pad=2.5)
        ax.tick_params(axis="y", length=0, pad=4)
        ax.set_xlabel("Share of diseased leaves (%)", fontsize=8.2, color=GREY_TXT, labelpad=3)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.set_ylim(-H / 2 - 0.12, max(h[0] for h in heads) + 0.42)
        panel_title(ax, title, species)
        handles = [Patch(facecolor=RAMP[dis][g], edgecolor="none", label=GRADE_LBL[g]) for g in range(4)]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.01, -0.175), ncol=2, frameon=False,
                  fontsize=7.0, handlelength=1.1, handleheight=0.9, handletextpad=0.45, columnspacing=1.6,
                  labelspacing=0.35, labelcolor=GREY_TXT)
    save(fig, "figure_grade_distribution")


if __name__ == "__main__":
    setup_style()
    figure_val_test()
    figure_grade_distribution()
    print("figures written")
