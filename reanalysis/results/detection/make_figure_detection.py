"""Supplementary figure: leaf-level predicted severity of each disease's three models on the target-disease test
leaves, the other disease's test leaves (cross-disease) and the 21 local healthy test leaves (EARTH controls).

Reads leaf_level.csv (written by detection_analysis.py). Primary leaf-level rule = the rule of the disease the model
was trained for (rust models: abaxial face; leaf-spot models: mean of both faces).
Outputs: figure_detection.pdf (vector, TrueType fonts) and figure_detection.png (600 dpi).
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator

OUT = os.path.dirname(os.path.abspath(__file__))
CM = 1 / 2.54
INK, INK2, TICK, AXIS = "#1b1b1b", "#565656", "#8f8f8f", "#d2d2d2"
RUST_C, LS_C = "#C76C3F", "#3080A8"          # disease colours sampled from Figure 5 (rust orange, leaf-spot blue)
MODELS = [("segformer", "SegFormer-B5", "#0072B2", "^", 4.3),
          ("unetpp", "UNet++", "#009E73", "s", 3.5),
          ("deeplab", "DeepLabV3+", "#E69F00", "o", 3.9)]
PANELS = [("roya", "(a) Coffee leaf rust", "Hemileia vastatrix"),
          ("ojo", "(b) American leaf spot", "Mycena citricolor")]
GROUP_NAME = {"rust": "Rust", "leaf_spot": "Leaf spot", "healthy": "Healthy"}
GROUP_ROLE = {"target": "target disease", "cross": "other disease", "healthy": "EARTH controls"}
TINT = {"rust": RUST_C, "leaf_spot": LS_C, "healthy": None}
THR = 0.5
LINTHRESH, LINSCALE = 0.1, 0.45

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8.5, "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5, "axes.edgecolor": AXIS, "axes.linewidth": 0.9, "axes.labelcolor": INK2,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": TICK,
    "xtick.major.width": 0.9, "ytick.major.width": 0.9, "xtick.major.size": 0, "ytick.major.size": 3.2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.facecolor": "white", "figure.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 600, "legend.frameon": False,
})

leaf = list(csv.DictReader(open(f"{OUT}/leaf_level.csv")))


def values(m, a, pack, cls):
    v = [float(x["pred_sev"]) for x in leaf if x["model_disease"] == m and x["arch"] == a and x["image_pack"] == pack
         and x["split"] == "test" and x["leaf_class"] == cls]
    return np.array(v)


def jitter(n, seed, width=0.075):
    # deterministic, evenly spread horizontal offsets (ranked) so points do not depend on a random draw
    rng = np.random.default_rng(seed)
    return (rng.permutation(n) / max(n - 1, 1) - 0.5) * 2 * width if n > 1 else np.zeros(1)


fig = plt.figure(figsize=(16 * CM, 8.0 * CM))
left, right, bottom, top, gap = 0.075, 0.99, 0.265, 0.815, 0.085
w = (right - left - gap) / 2
axes = [fig.add_axes([left + i * (w + gap), bottom, w, top - bottom]) for i in range(2)]

for ax, (m, title, species) in zip(axes, PANELS):
    own = "rust" if m == "roya" else "leaf_spot"
    other = "leaf_spot" if m == "roya" else "rust"
    groups = [(own, "target", m, own), (other, "cross", "ojo" if m == "roya" else "roya", other),
              ("healthy", "healthy", "roya", "healthy")]
    ax.set_yscale("symlog", linthresh=LINTHRESH, linscale=LINSCALE)
    ymax = 190.0
    ax.set_ylim(-0.012, ymax)
    ax.set_xlim(-0.55, 2.78)
    # grade cut-points: 0.5% (G0/G1; also the >=0.5% detection threshold), 5% and 20%
    for y in (5, 20):
        ax.axhline(y, color="#bdbdbd", lw=0.7, ls=(0, (1, 2.2)), zorder=1)
    ax.axhline(THR, color=INK, lw=0.9, ls=(0, (4.5, 2.5)), zorder=1)
    for gi, (key, role, pack, cls) in enumerate(groups):
        if TINT[key]:
            ax.axvspan(gi - 0.47, gi + 0.47, ymin=0, ymax=1, color=TINT[key], alpha=0.07, lw=0, zorder=0)
        else:
            ax.axvspan(gi - 0.47, gi + 0.47, ymin=0, ymax=1, color="#9a9a9a", alpha=0.06, lw=0, zorder=0)
        for mi, (a, name, col, mk, ms) in enumerate(MODELS):
            v = values(m, a, pack, cls)
            x0 = gi + (mi - 1) * 0.29
            xs = x0 + jitter(len(v), seed=1000 * gi + 10 * mi + (0 if m == "roya" else 500))
            ax.plot(xs, v, ls="none", marker=mk, ms=ms, mfc=col, mec="white", mew=0.35, alpha=0.88, zorder=3,
                    clip_on=False)
            med = float(np.median(v))
            ax.plot([x0 - 0.11, x0 + 0.11], [med, med], color=INK, lw=1.1, solid_capstyle="butt", zorder=4)
            k = int((v >= THR).sum())
            ax.text(x0, 100, f"{k}", ha="center", va="center", fontsize=7.6, color=INK)
    # row label for the counts: number of leaves with predicted leaf severity >= 0.5%
    ax.annotate("≥0.5%", xy=(0, 100), xycoords=("axes fraction", "data"), xytext=(-4, 0),
                textcoords="offset points", ha="right", va="center", fontsize=7.2, color=INK2)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([GROUP_NAME[g[0]] for g in groups], color=INK, fontsize=8.5)
    for gi, (key, role, pack, cls) in enumerate(groups):
        n = len(values(m, "segformer", pack, cls))
        ax.annotate(f"n = {n}", xy=(gi, 0), xycoords=("data", "axes fraction"), xytext=(0, -16.5),
                    textcoords="offset points", ha="center", va="top", fontsize=7.6, color=INK2)
        ax.annotate(GROUP_ROLE[role], xy=(gi, 0), xycoords=("data", "axes fraction"), xytext=(0, -26.5),
                    textcoords="offset points", ha="center", va="top", fontsize=7.6, color=INK2, fontstyle="italic")
    ax.tick_params(axis="x", pad=4)
    ax.yaxis.set_major_locator(FixedLocator([0, 0.1, 0.5, 1, 5, 20, 50]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels(["0", "0.1", "0.5", "1", "5", "20", "50"])
    ax.spines["left"].set_bounds(0, 60)
    ax.spines["bottom"].set_visible(False)
    # grade bands, labelled at the right edge of each panel
    for lab, y in (("G0", 0.2), ("G1", 1.6), ("G2", 10), ("G3", 38)):
        ax.text(2.77, y, lab, ha="right", va="center", fontsize=7.2, color=TICK)
    ax.annotate(title, xy=(0, 1), xycoords="axes fraction", xytext=(0, 20), textcoords="offset points",
                ha="left", va="bottom", fontsize=10.5, fontweight="bold", color=INK)
    ax.annotate(species, xy=(0, 1), xycoords="axes fraction", xytext=(0, 7), textcoords="offset points",
                ha="left", va="bottom", fontsize=8, fontstyle="italic", color=INK2)

axes[0].set_ylabel("Predicted severity (%)")
axes[0].yaxis.set_label_coords(-0.105, 0.43)   # centred on the data range, clear of the count-row label
handles = [Line2D([], [], ls="none", marker=mk, ms=ms + 0.9, mfc=col, mec="white", mew=0.35, label=name)
           for _, name, col, mk, ms in MODELS]
handles += [Line2D([], [], color=INK, lw=1.1, label="median"),
            Line2D([], [], color=INK, lw=0.9, ls=(0, (4.5, 2.5)), label="0.5% threshold (G0/G1)")]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.0), ncol=5, handlelength=1.6,
           handletextpad=0.5, columnspacing=1.5, fontsize=8.0, labelcolor=INK2)

fig.savefig(f"{OUT}/figure_detection.pdf")
fig.savefig(f"{OUT}/figure_detection.png", dpi=600)
print("saved")
