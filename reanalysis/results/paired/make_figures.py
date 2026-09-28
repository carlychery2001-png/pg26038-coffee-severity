"""Figures for the paired comparison of the published models (reads results.json from paired_analysis.py).

figure_paired_differences.pdf/.png  Paired differences (first minus second model) in MAE, |bias|, QWK and grade
                                     accuracy, per disease, with 95% intervals and exact McNemar counts.
figure_grade_recall.pdf/.png         Per-grade recall with 95% Wilson intervals for every model (rust; leaf spot vs
                                     visual grade; leaf spot vs mask-derived grade) and leaf-spot accuracy/QWK
                                     against the two reference standards.

Style follows the manuscript's figure5.png: Arial, bold "(a) Coffee leaf rust" / "(b) American leaf spot" titles
with grey italic species names, light grey axes, no in-figure headline titles, 16 cm wide, vector PDF with
TrueType fonts (pdf.fonttype 42) and a 600 dpi PNG.

CI rendering: filled markers are small with a thin white edge, and CI lines plus end caps are drawn ABOVE them, so
short whiskers (e.g., recall 0.96 -> upper bound 0.99) stay readable. Open markers keep the line underneath, with caps
on top.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

OUT = Path(__file__).resolve().parent
R = json.loads((OUT / "results.json").read_text(encoding="utf-8"))

CM = 1 / 2.54
INK = "#1a1a1a"
GREY_TXT = "#555555"
GREY_TICK = "#8a8a8a"
GREY_AXIS = "#c8c8c8"
GREY_REF = "#9a9a9a"
DIS_COLOR = {"roya": "#CC7951", "ojo": "#448BB0"}  # marker colours sampled from figure5.png (dominant fills)
TITLE = {"roya": "Coffee leaf rust", "ojo": "American leaf spot"}
SPECIES = {"roya": "Hemileia vastatrix", "ojo": "Mycena citricolor"}
ARCHS = ["segformer", "unetpp", "deeplab"]
NAME = {"segformer": "SegFormer-B5", "unetpp": "UNet++", "deeplab": "DeepLabV3+"}
STYLE = {  # marker sizes kept small so that short CI whiskers stay visible next to the marker
    "segformer": dict(color="#0072B2", marker="^", ms=5.0),
    "unetpp": dict(color="#009E73", marker="s", ms=4.2),
    "deeplab": dict(color="#E69F00", marker="o", ms=4.6),
}
MEW = 0.5  # thin white marker edge
CAP_MS, CAP_MEW = 3.6, 1.0  # CI end caps (points), drawn above the markers


def ci_caps(ax, x, y, ci, est, color, horizontal=False, z=4):
    """Draw an end cap at each CI bound that differs from the estimate, above the marker."""
    for b in ci:
        if abs(b - est) > 1e-9:
            if horizontal:
                ax.plot(b, y, marker="|", ms=CAP_MS + 0.6, mew=CAP_MEW, color=color, ls="none", zorder=z)
            else:
                ax.plot(x, b, marker="_", ms=CAP_MS, mew=CAP_MEW, color=color, ls="none", zorder=z,
                        clip_on=False)
PAIRS = ["segformer_vs_deeplab", "segformer_vs_unetpp", "unetpp_vs_deeplab"]


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial"],
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.edgecolor": GREY_AXIS, "axes.linewidth": 0.9,
        "xtick.color": GREY_AXIS, "ytick.color": GREY_AXIS,
        "xtick.labelcolor": GREY_TICK, "ytick.labelcolor": GREY_TICK,
        "xtick.major.width": 0.9, "ytick.major.width": 0.9,
        "xtick.major.size": 3.2, "ytick.major.size": 3.2,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    })


def panel_title(ax, letter, dis, extra=None, dy=(17, 5)):
    ax.annotate(f"({letter}) {TITLE[dis]}", xy=(0, 1), xycoords="axes fraction", xytext=(0, dy[0]),
                textcoords="offset points", ha="left", va="bottom", fontsize=10.2, fontweight="bold", color=INK)
    ax.annotate(SPECIES[dis], xy=(0, 1), xycoords="axes fraction", xytext=(0, dy[1]), textcoords="offset points",
                ha="left", va="bottom", fontsize=8.4, fontstyle="italic", color=GREY_TXT)
    if extra:
        ax.annotate(extra, xy=(1, 1), xycoords="axes fraction", xytext=(0, dy[1]), textcoords="offset points",
                    ha="right", va="bottom", fontsize=7.4, color=GREY_TXT)


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=600)
    plt.close(fig)
    with Image.open(OUT / f"{stem}.png") as im:  # flatten RGBA to opaque white RGB
        rgb = Image.new("RGB", im.size, "white")
        rgb.paste(im, mask=im.convert("RGBA").getchannel("A"))
    rgb.save(OUT / f"{stem}.png", dpi=(600, 600))


def nice_lim(lo, hi, step):
    import math
    m = max(abs(lo), abs(hi))
    m = math.ceil(m / step * 1.08) * step
    return -m, m


def fmt_p(p):
    return "1.00" if p >= 0.995 else f"{p:.2f}"


# ------------------------------------------------------------------------------------------------ figure A
def figure_paired():
    setup_style()
    fig = plt.figure(figsize=(16 * CM, 11.8 * CM))
    cols = [("d_mae_pp", "ΔMAE (pp)", "low"), ("d_abs_bias_pp", "Δ|bias| (pp)", "low"),
            ("d_qwk", "ΔQWK", "high"), ("d_accuracy", "ΔGrade accuracy", "high")]
    left, right, gap = 0.175, 0.845, 0.022
    w = (right - left - gap * 3) / 4
    rows_y = [(0.595, 0.3), (0.095, 0.3)]  # (bottom, height) for rust, leaf spot
    steps = {"roya": {"d_mae_pp": 0.5, "d_abs_bias_pp": 0.5, "d_qwk": 0.1, "d_accuracy": 0.1},
             "ojo": {"d_mae_pp": 0.05, "d_abs_bias_pp": 0.05, "d_qwk": 0.05, "d_accuracy": 0.05}}
    ticks = {"roya": {"d_mae_pp": [-1, 0, 1], "d_abs_bias_pp": [-1, 0, 1], "d_qwk": [-0.2, 0, 0.2],
                      "d_accuracy": [-0.2, 0, 0.2]},
             "ojo": {"d_mae_pp": [-0.1, 0, 0.1], "d_abs_bias_pp": [-0.2, 0, 0.2], "d_qwk": [-0.05, 0, 0.05],
                     "d_accuracy": [-0.1, 0, 0.1]}}
    for r, (dis, letter) in enumerate([("roya", "a"), ("ojo", "b")]):
        P = R["paired"][dis]
        bottom, height = rows_y[r]
        for c, (key, label, better) in enumerate(cols):
            ax = fig.add_axes([left + c * (w + gap), bottom, w, height])
            vals = []
            for i, pk in enumerate(PAIRS):
                d = P[pk][key]
                ci = d["tango_score_ci95"] if key == "d_accuracy" else d["ci95"]
                vals += ci
            lo, hi = nice_lim(min(vals), max(vals), steps[dis][key])
            ax.axvline(0, color=GREY_REF, lw=0.9, zorder=1)
            for i, pk in enumerate(PAIRS):
                y = len(PAIRS) - 1 - i
                d = P[pk][key]
                ci = d["tango_score_ci95"] if key == "d_accuracy" else d["ci95"]
                col = DIS_COLOR[dis]
                # marker first, CI line and caps drawn above it so that short whiskers stay readable
                ax.plot(d["estimate"], y, marker="o", ms=5.0, color=col, mec="white", mew=0.6, ls="none", zorder=2)
                if ci[1] - ci[0] > 1e-12:
                    ax.plot(ci, [y, y], color=col, lw=1.8, solid_capstyle="butt", zorder=3)
                    ci_caps(ax, None, y, ci, d["estimate"], col, horizontal=True, z=3)
                if ci[1] - ci[0] <= 1e-12:
                    ax.annotate("identical\ngrades", xy=(d["estimate"], y), xytext=(6, 0),
                                textcoords="offset points", ha="left", va="center", fontsize=6.3,
                                color=GREY_TXT, linespacing=1.0)
            ax.set_xlim(lo, hi)
            ax.set_ylim(-0.6, len(PAIRS) - 0.4)
            ax.set_xticks(ticks[dis][key])
            ax.set_xticklabels([("0" if t == 0 else f"{t:+g}".replace("-", "−")) for t in ticks[dis][key]])
            ax.tick_params(axis="x", labelsize=7.4, pad=2)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if c == 0:
                ax.set_yticks(range(len(PAIRS)))
                ax.set_yticklabels([f"{NAME[p.split('_vs_')[0]]} vs\n{NAME[p.split('_vs_')[1]]}" for p in PAIRS[::-1]],
                                   fontsize=7.6, color=INK, linespacing=1.05)
                ax.tick_params(axis="y", pad=3, labelcolor=INK)
                panel_title(ax, letter, dis, dy=(19, 7))
            else:
                ax.set_yticks([])
                ax.spines["left"].set_visible(False)
            ax.set_xlabel(label, fontsize=8.2, color=GREY_TXT, labelpad=2.5)
            # direction hint (which side favours the first-named model)
            txt, xa, ha = (("← first better", 0.02, "left") if better == "low"
                           else ("first better →", 0.98, "right"))
            ax.text(xa, 1.0, txt, transform=ax.transAxes, ha=ha, va="bottom", fontsize=6.2, color=GREY_TICK)
        # McNemar column
        axm = fig.add_axes([right + 0.012, bottom, 1 - right - 0.014, height])
        axm.set_axis_off()
        axm.set_ylim(-0.6, len(PAIRS) - 0.4)
        axm.set_xlim(0, 1)
        axm.text(0.0, len(PAIRS) - 0.4, "McNemar\nb / c; p", ha="left", va="bottom", fontsize=6.6,
                 color=GREY_TXT, linespacing=1.05)
        for i, pk in enumerate(PAIRS):
            y = len(PAIRS) - 1 - i
            mc = P[pk]["mcnemar"]
            axm.text(0.0, y, f"{mc['first_correct_second_wrong']} / {mc['first_wrong_second_correct']}; "
                     f"{fmt_p(mc['p_exact'])}", ha="left", va="center", fontsize=7.2, color=GREY_TXT)
    save(fig, "figure_paired_differences")


# ------------------------------------------------------------------------------------------------ figure B
def recall_panel(ax, dis, ref, letter, extra, healthy=False):
    grades = ["G0", "G1", "G2", "G3"]
    cats = grades + (["Healthy"] if healthy else [])
    off = {"segformer": -0.24, "unetpp": 0.0, "deeplab": 0.24}
    ax.axhline(1.0, color="#e6e6e6", lw=0.8, zorder=0)
    for j, g in enumerate(cats):
        for a in ARCHS:
            st = STYLE[a]
            if g == "Healthy":
                h = R["healthy_g0"][dis][a]
                n, k, ci = h["n_healthy_local"], h["predicted_G0"], h["g0_recall_wilson95"]
            else:
                rr = R["per_model"][dis][a]["grades"][ref]["per_grade_recall"][g]
                n, k, ci = rr["n"], rr["correct"], rr["wilson95"]
            if n == 0:
                continue
            x = j + off[a]
            # filled marker below; CI line and end caps above it (short upper whiskers were hidden before)
            ax.plot(x, k / n, marker=st["marker"], ms=st["ms"], color=st["color"], mec="white", mew=MEW,
                    ls="none", zorder=3, clip_on=False)
            ax.plot([x, x], ci, color=st["color"], lw=1.3, solid_capstyle="butt", zorder=4, clip_on=False)
            ci_caps(ax, x, None, ci, k / n, st["color"])
        if g == "Healthy":
            n = R["healthy_g0"][dis]["segformer"]["n_healthy_local"]
        else:
            n = R["per_model"][dis]["segformer"]["grades"][ref]["per_grade_recall"][g]["n"]
        if n == 0:
            ax.text(j, 0.5, "no test\nleaves", ha="center", va="center", fontsize=6.6, color=GREY_TICK,
                    linespacing=1.05)
    labels = []
    for g in cats:
        if g == "Healthy":
            n = R["healthy_g0"][dis]["segformer"]["n_healthy_local"]
            labels.append(f"Healthy†\nn = {n}")
        else:
            n = R["per_model"][dis]["segformer"]["grades"][ref]["per_grade_recall"][g]["n"]
            labels.append(f"{g}\nn = {n}")
    if healthy:
        ax.axvline(len(grades) - 0.5, color=GREY_AXIS, lw=0.8, ls=(0, (2, 2)), zorder=0)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(labels, fontsize=7.4, linespacing=1.1)
    ax.set_xlim(-0.55, len(cats) - 0.45)
    ax.set_ylim(0, 1.04)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.00"])
    ax.tick_params(axis="both", labelsize=7.6, pad=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_bounds(0, 1.0)
    panel_title(ax, letter, dis, extra)


def reference_panel(ax, letter):
    dis = "ojo"
    cats = [("accuracy", "Grade accuracy"), ("qwk", "QWK")]
    off = {"segformer": -0.26, "unetpp": 0.0, "deeplab": 0.26}
    inner = {"reference": -0.055, "mask_derived": 0.055}
    agree = R["leafspot_reference_agreement"]
    ref_vals = {"accuracy": agree["visual_eq_mask_derived"] / agree["n"], "qwk": agree["qwk_visual_vs_mask"]}
    for j, (m, _) in enumerate(cats):
        ax.plot([j - 0.42, j + 0.42], [ref_vals[m]] * 2, color=GREY_REF, lw=0.9, ls=(0, (1, 1.6)), zorder=1)
        for a in ARCHS:
            st = STYLE[a]
            for rname, dx in inner.items():
                g = R["per_model"][dis][a]["grades"][rname]
                est, ci = (g["accuracy"], g["accuracy_wilson95"]) if m == "accuracy" else (g["qwk"], g["qwk_ci95"])
                x = j + off[a] + dx
                filled = rname == "reference"
                # filled markers: CI line above the marker; open markers: line below (white fill stays clean).
                # End caps always on top.
                ax.plot([x, x], ci, color=st["color"], lw=1.3, solid_capstyle="butt", zorder=4 if filled else 2)
                ax.plot(x, est, marker=st["marker"], ms=st["ms"] + (0 if filled else 0.4), color=st["color"],
                        mfc=st["color"] if filled else "white", mec=st["color"] if not filled else "white",
                        mew=0.9 if not filled else MEW, ls="none", zorder=3)
                ci_caps(ax, x, None, ci, est, st["color"], z=5)
    ax.annotate("visual vs\nmask-derived\nreference", xy=(1 + 0.42, ref_vals["qwk"]), xytext=(3, 0),
                textcoords="offset points", ha="left", va="center", fontsize=6.3, color=GREY_TICK, linespacing=1.0)
    hv = Line2D([], [], marker="o", ms=4.6, color=GREY_TXT, mec="white", mew=MEW, ls="none",
                label="vs visual grade (filled)")
    hm = Line2D([], [], marker="o", ms=5.0, color=GREY_TXT, mfc="white", mec=GREY_TXT, mew=0.9, ls="none",
                label="vs mask-derived grade (open)")
    ax.legend(handles=[hv, hm], loc="lower right", frameon=False, fontsize=6.8, handletextpad=0.2,
              borderaxespad=0.2, labelcolor=GREY_TXT)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([c[1] for c in cats], fontsize=7.6)
    ax.set_xlim(-0.55, len(cats) - 0.45 + 0.62)
    ax.set_ylim(0.6, 1.01)
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(["0.60", "0.70", "0.80", "0.90", "1.00"])
    ax.tick_params(axis="both", labelsize=7.6, pad=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_bounds(0.6, 1.0)
    ax.spines["bottom"].set_bounds(-0.55, len(cats) - 0.45)
    panel_title(ax, letter, dis, "accuracy and QWK by reference")


def figure_recall():
    setup_style()
    fig = plt.figure(figsize=(16 * CM, 13.6 * CM))
    L, Rr, wsp = 0.085, 0.975, 0.105
    w = (Rr - L - wsp) / 2
    top_b, bot_b, h = 0.612, 0.148, 0.285
    axa = fig.add_axes([L, top_b, w, h])
    axb = fig.add_axes([L + w + wsp, top_b, w, h])
    axc = fig.add_axes([L, bot_b, w, h])
    axd = fig.add_axes([L + w + wsp, bot_b, w, h])
    recall_panel(axa, "roya", "reference", "a", "reference: mask-derived grade", healthy=True)
    recall_panel(axb, "ojo", "reference", "b", "reference: visual grade", healthy=True)
    recall_panel(axc, "ojo", "mask_derived", "c", "reference: mask-derived grade")
    reference_panel(axd, "d")
    for ax in (axa, axc):
        ax.set_ylabel("Recall (95% Wilson CI)", fontsize=8.2, color=GREY_TXT, labelpad=3)
    axd.set_ylabel("Agreement (95% CI)", fontsize=8.2, color=GREY_TXT, labelpad=3)
    handles = [Line2D([], [], marker=STYLE[a]["marker"], color=STYLE[a]["color"], ms=STYLE[a]["ms"] + 1.0,
                      mec="white", mew=MEW, ls="none", label=NAME[a]) for a in ARCHS]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8.0,
               bbox_to_anchor=(0.5, 0.0), handletextpad=0.3, columnspacing=1.8, labelcolor=INK)
    save(fig, "figure_grade_recall")


if __name__ == "__main__":
    figure_paired()
    figure_recall()
    print("figures written")
