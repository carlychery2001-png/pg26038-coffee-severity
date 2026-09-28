"""Figures for the severity-agreement reanalysis (PG-26038 major revision).

figure5.pdf / .png          NEW Figure 5: SegFormer-B5, (a) coffee leaf rust, (b) American leaf spot.
                            Points = leaves (difference predicted - reference vs mean of the two); dashed line = OLS
                            regression of differences on means; dotted lines = regression-based 95% limits of agreement
                            (Bland & Altman 1999); solid light grey line = zero (no difference).
figureS_agreement_all.pdf / .png
                            Supplementary: the same for all six models (rows = disease, columns = model), y axis shared
                            within a disease.
Input: results.json and work/leaf_level.csv (agreement.py). Style follows the paper's figure5.png: Arial, bold
"(a) Disease" titles with the species in grey italic, light grey axes, 16 cm wide, vector PDF with TrueType fonts
(pdf.fonttype 42) and a 600 dpi PNG. No in-figure headline titles.
"""
import csv
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

OUT = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(OUT, "results.json")))
LEAVES = list(csv.DictReader(open(os.path.join(OUT, "work", "leaf_level.csv"))))
K_SD = math.sqrt(math.pi / 2)
CM = 1 / 2.54

INK = "#1b1b1b"
GREY_TXT = "#565656"
GREY_TICK = "#8f8f8f"
GREY_AXIS = "#c8c8c8"
GREY_LOA = "#7a7a7a"
# Disease colours of the published figure5.png: drawn at alpha 0.85 over white these render as #cc7951 / #448bb0,
# the fill colours measured in figure5.png.
DIS_COL = {"rust": "#C36132", "leaf_spot": "#2376A2"}
PANELS = [("rust", "Coffee leaf rust", "Hemileia vastatrix"), ("leaf_spot", "American leaf spot", "Mycena citricolor")]
MODELS = [("segformer", "SegFormer-B5"), ("unetpp", "UNet++"), ("deeplab", "DeepLabV3+")]
STYLE = {"SegFormer-B5": dict(color="#0072B2", marker="^", s=30),
         "UNet++": dict(color="#009E73", marker="s", s=22),
         "DeepLabV3+": dict(color="#E69F00", marker="o", s=26)}


def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial"],
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.edgecolor": GREY_AXIS, "axes.linewidth": 0.9,
        "xtick.color": GREY_AXIS, "ytick.color": GREY_AXIS,
        "xtick.labelcolor": GREY_TICK, "ytick.labelcolor": GREY_TICK,
        "xtick.major.width": 0.9, "ytick.major.width": 0.9, "xtick.major.size": 3.2, "ytick.major.size": 3.2,
        "axes.labelcolor": GREY_TXT,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.unicode_minus": True,
    })


def data(dis, model_name):
    rows = [r for r in LEAVES if r["disease"] == dis and r["model"] == model_name]
    return [float(r["mean"]) for r in rows], [float(r["diff"]) for r in rows]


def lines(dis, arch):
    m = R["models"][f"{dis}/{arch}"]
    g, rl = m["regression_of_differences_on_means"], m["regression_based_loa"]
    b0, b1 = g["intercept"], g["slope"]
    c0, c1 = rl["abs_residual_intercept_c0"], rl["abs_residual_slope_c1"]
    lo, hi = m["mean_of_A"]["min"], m["mean_of_A"]["max"]
    n = 200
    xs = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    fit = [b0 + b1 * x for x in xs]
    sd = [K_SD * (c0 + c1 * x) for x in xs]
    assert min(sd) > 0
    return xs, fit, [f - 1.96 * s for f, s in zip(fit, sd)], [f + 1.96 * s for f, s in zip(fit, sd)], m


def fmt_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"


def slope_text(m):
    g = m["regression_of_differences_on_means"]
    lo, hi = g["slope_ci95_t"]
    return (f"slope {g['slope']:.3f} (95% CI {lo:.3f} to {hi:.3f})\n"
            f"{fmt_p(g['slope_p'])}, n = {m['n_leaves']} leaves").replace("-", "−")


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="both", pad=2.5)


def draw_panel(ax, dis, arch, model_name, point_style, label_lines, fs_tick, fs_note):
    xs, fit, lo, hi, m = lines(dis, arch)
    ax.axhline(0, color=GREY_AXIS, lw=1.0, zorder=1)
    ax.plot(xs, lo, color=GREY_LOA, lw=1.1, ls=(0, (1, 1.8)), zorder=2, solid_capstyle="butt")
    ax.plot(xs, hi, color=GREY_LOA, lw=1.1, ls=(0, (1, 1.8)), zorder=2, solid_capstyle="butt")
    ax.plot(xs, fit, color=INK, lw=1.3, ls=(0, (5, 2.6)), zorder=3)
    x, y = data(dis, model_name)
    ax.scatter(x, y, zorder=4, edgecolors="white", linewidths=0.6, alpha=0.85, clip_on=False, **point_style)
    ax.tick_params(axis="both", labelsize=fs_tick)
    style_axes(ax)
    if label_lines:
        for yv, txt, col in ((hi[-1], "upper 95% LoA", GREY_TICK), (fit[-1], "regression", INK),
                             (lo[-1], "lower 95% LoA", GREY_TICK)):
            ax.annotate(txt, xy=(xs[-1], yv), xytext=(5, 0), textcoords="offset points", ha="left", va="center",
                        fontsize=fs_note, color=col, annotation_clip=False)
    return m


def save(fig, stem):
    fig.savefig(os.path.join(OUT, f"{stem}.pdf"))
    png = os.path.join(OUT, f"{stem}.png")
    fig.savefig(png, dpi=600)
    plt.close(fig)
    with Image.open(png) as im:  # flatten RGBA to opaque RGB on white
        rgb = Image.new("RGB", im.size, "white")
        rgb.paste(im, mask=im.convert("RGBA").getchannel("A"))
    rgb.save(png, dpi=(600, 600))


def figure5():
    fig, axes = plt.subplots(1, 2, figsize=(16 * CM, 6.9 * CM))
    fig.subplots_adjust(left=0.075, right=0.885, bottom=0.165, top=0.80, wspace=0.42)
    for i, (ax, (dis, title, species)) in enumerate(zip(axes, PANELS)):
        m = draw_panel(ax, dis, "segformer", "SegFormer-B5",
                       dict(color=DIS_COL[dis], marker="o", s=24), True, 8.0, 6.8)
        ax.set_xlabel("Mean of predicted and reference severity (%)", fontsize=8.4, labelpad=3)
        if i == 0:
            ax.set_ylabel("Predicted − reference (pp)", fontsize=8.4, labelpad=3)
        ax.annotate(f"({'ab'[i]}) {title}", xy=(0, 1), xycoords="axes fraction", xytext=(0, 17),
                    textcoords="offset points", ha="left", va="bottom", fontsize=10.2, fontweight="bold", color=INK)
        ax.annotate(species, xy=(0, 1), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points",
                    ha="left", va="bottom", fontsize=8.4, fontstyle="italic", color=GREY_TXT)
        ax.annotate(slope_text(m), xy=(0.03, 0.97), xycoords="axes fraction", ha="left", va="top",
                    fontsize=6.8, color=GREY_TXT, linespacing=1.15)
    ax0, ax1 = axes
    ax0.set_xlim(-0.8, 22.6); ax0.set_xticks([0, 5, 10, 15, 20])
    ax0.set_ylim(-1.1, 6.8); ax0.set_yticks(range(-1, 7))
    ax1.set_xlim(-1.0, 28.6); ax1.set_xticks([0, 5, 10, 15, 20, 25])
    ax1.set_ylim(-0.62, 2.0); ax1.set_yticks([-0.5, 0, 0.5, 1.0, 1.5, 2.0])
    for ax in axes:
        ax.spines["left"].set_bounds(*ax.get_ylim())
    save(fig, "figure5")


def figure_supp():
    fig, axes = plt.subplots(2, 3, figsize=(16 * CM, 12.8 * CM), sharex="row", sharey="row")
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.135, top=0.885, wspace=0.12, hspace=0.80)
    renderer = fig.canvas.get_renderer()
    for r, (dis, title, species) in enumerate(PANELS):
        for c, (arch, name) in enumerate(MODELS):
            ax = axes[r, c]
            m = draw_panel(ax, dis, arch, name, STYLE[name], False, 7.2, 6.2)
            ax.set_title(name, fontsize=8.2, color=INK, loc="left", pad=4)
            g = m["regression_of_differences_on_means"]
            lo, hi = g["slope_ci95_t"]
            txt = f"slope {g['slope']:.3f}\n(95% CI {lo:.3f} to {hi:.3f})".replace("-", "−")
            ax.annotate(txt, xy=(0.03, 0.97), xycoords="axes fraction", ha="left", va="top",
                        fontsize=6.2, color=GREY_TXT, linespacing=1.1)
            if c == 0:
                ax.set_ylabel("Predicted − reference (pp)", fontsize=7.6, labelpad=2)
            ax.set_xlabel("Mean severity (%)", fontsize=7.6, labelpad=2)
            ax.tick_params(axis="y", labelleft=(c == 0))
        # row heading: bold disease name with the species in grey italic after it, above the model titles
        pos = axes[r, 0].get_position()
        x0, y0 = 0.012, pos.y1 + 0.052
        t = fig.text(x0, y0, f"({'ab'[r]}) {title}", ha="left", va="baseline", fontsize=9.6, fontweight="bold",
                     color=INK)
        bb = t.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        fig.text(bb.x1 + 0.018, y0, species, ha="left", va="baseline", fontsize=8.0, fontstyle="italic",
                 color=GREY_TXT)
    axes[0, 0].set_xlim(-0.9, 23.2); axes[0, 0].set_xticks([0, 5, 10, 15, 20])
    axes[0, 0].set_ylim(-3.4, 13.2); axes[0, 0].set_yticks([-3, 0, 3, 6, 9, 12])
    axes[1, 0].set_xlim(-1.1, 28.6); axes[1, 0].set_xticks([0, 10, 20])
    axes[1, 0].set_ylim(-1.3, 2.25); axes[1, 0].set_yticks([-1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0])
    for ax in axes.flat:
        ax.spines["left"].set_bounds(*ax.get_ylim())
    handles = [Line2D([], [], color=INK, lw=1.3, ls=(0, (5, 2.6))),
               Line2D([], [], color=GREY_LOA, lw=1.1, ls=(0, (1, 1.8))),
               Line2D([], [], color=GREY_AXIS, lw=1.0)]
    labels = ["regression of differences on means", "regression-based 95% limits of agreement", "zero difference"]
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=7.0, handlelength=2.6,
               columnspacing=1.6, bbox_to_anchor=(0.53, 0.0), labelcolor=GREY_TXT)
    save(fig, "figureS_agreement_all")


if __name__ == "__main__":
    setup_style()
    figure5()
    figure_supp()
    print("figures written")
