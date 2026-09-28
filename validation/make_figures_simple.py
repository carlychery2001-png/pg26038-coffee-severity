"""Plain-language figures for the expert-panel validation (thesis + MDPI).
Each chart answers one question and states its takeaway in the title.
Outputs figures/Fig_simple_*.pdf (vector) and .png (600 dpi), 16 cm wide.
Primary set = 53 leaves never used to train the models; rust graded on the abaxial face."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

exec(open("final_analysis.py", encoding="utf-8").read().split("res = {}")[0])   # E, KEY, P, EXP, model_grade, qwk
T = json.load(open("tiefree_results.json")); C = json.load(open("closeness.json"))
TRAIN = {f for f, r in KEY.items() if r["model_split"] == "train"}
os.makedirs("figures", exist_ok=True)

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
EXPERT = "#9a9994"
RATERS = [("expert", "Typical expert", EXPERT), ("segformer", "SegFormer-B5", "#2a78d6"),
          ("deeplab", "DeepLabV3+", "#eb6834"), ("unetpp", "UNet++", "#1baf7a")]
DIS = [("ojo", "American leaf spot"), ("roya", "Coffee leaf rust")]
CM = 1 / 2.54
plt.rcParams.update({
    "font.family": "Arial", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8.5,
    "legend.fontsize": 8, "axes.edgecolor": INK2, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK,
    "axes.labelcolor": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "savefig.dpi": 600,
})

def headline(fig, text, sub=None):
    fig.text(0.01, 0.995, text, ha="left", va="top", fontsize=10.5, fontweight="bold", color=INK)
    if sub: fig.text(0.01, 0.935, sub, ha="left", va="top", fontsize=8.3, color=INK2)

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/{name}.{ext}", bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig); print("saved", name)

def rows_for(d): return [r for r in E if r["leaf_id"].startswith(d) and r["leaf_id"] not in TRAIN]
def sev(lid):
    p = P[lid]
    return float(p["segformer_sev_ab"]) if lid.startswith("roya") else (float(p["segformer_sev_ab"]) + float(p["segformer_sev_ad"])) / 2

# =====================================================================================
# A - Who agrees best with the expert panel? (simple bars)
# =====================================================================================
fig, axes = plt.subplots(1, 2, figsize=(16 * CM, 6.3 * CM), sharey=True)
for ax, (d, name) in zip(axes, DIS):
    vals = [T[f"{d}_segformer"]["fair_expert"]] + [T[f"{d}_{a}"]["fair_model"] for a, _, _ in RATERS[1:]]
    y = np.arange(len(RATERS))[::-1]
    for yi, v, (_, lab, c) in zip(y, vals, RATERS):
        ax.barh(yi, v, height=0.62, color=c, zorder=2)
        ax.text(v + 0.012, yi, f"{v:.2f}", va="center", ha="left", fontsize=8.5, fontweight="bold" if lab == "SegFormer-B5" else "normal", color=INK)
    ax.axvline(vals[0], color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=3)
    ax.set_yticks(y, [lab for _, lab, _ in RATERS]); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1.0); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.50", "0.75", "1\n(perfect)"])
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0); ax.spines["left"].set_visible(False)
    ax.set_title(f"{name}  ({len(rows_for(d))} leaves)", loc="left", fontweight="bold", pad=6)
    ax.set_xlabel("Agreement with the expert panel")
headline(fig, "SegFormer-B5 agrees with the expert panel more closely than a typical expert does (clearly for leaf spot)",
         "Agreement score = weighted kappa with the panel’s consensus grade (the other experts’ median). Dashed line = typical expert.")
fig.tight_layout(rect=(0, 0, 1, 0.86))
save(fig, "Fig_simple_1_agreement")

# =====================================================================================
# B - How far off are the grades? (100% stacked bars)
# =====================================================================================
SEG = [("Same grade as the panel", "#1c5cab"), ("1 grade off", "#9ec5f4"), ("2 or more grades off", "#e34948")]
fig, axes = plt.subplots(1, 2, figsize=(16 * CM, 6.3 * CM), sharey=True)
for ax, (d, name) in zip(axes, DIS):
    y = np.arange(len(RATERS))[::-1]
    for yi, (key, lab, _) in zip(y, RATERS):
        left = 0
        for k, (slab, c) in enumerate(SEG):
            w = C[d][key][k] * 100
            if w <= 0: continue
            ax.barh(yi, w, left=left, height=0.62, color=c, edgecolor="white", linewidth=1.2, zorder=2)
            if w >= 6:
                ax.text(left + w / 2, yi, f"{w:.0f}%", ha="center", va="center", fontsize=8,
                        color="white" if k == 0 else INK, fontweight="bold" if k == 0 else "normal")
            left += w
    ax.set_yticks(y, [lab for _, lab, _ in RATERS]); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.spines["left"].set_visible(False)
    ax.set_title(f"{name}  ({len(rows_for(d))} leaves)", loc="left", fontweight="bold", pad=6)
    ax.set_xlabel("Share of leaves")
handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=l) for l, c in SEG]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
e, s = C["all"]["expert"], C["all"]["segformer"]
headline(fig, f"SegFormer-B5 gives the panel’s grade on {s[0]*100:.0f}% of leaves; a typical expert does on {e[0]*100:.0f}%",
         "Grade compared with the consensus of the other experts. Misses by the best models are never more than one grade.")
fig.tight_layout(rect=(0, 0.07, 1, 0.86))
save(fig, "Fig_simple_2_closeness")

# =====================================================================================
# C - Leaf by leaf: model grade vs the spread of expert grades
# =====================================================================================
fig, axes = plt.subplots(2, 1, figsize=(16 * CM, 10.2 * CM))
for ax, (d, name) in zip(axes, DIS):
    rows = rows_for(d)
    M = np.array([[int(r[e]) for r in rows] for e in EXP])
    g = np.array([model_grade(r["leaf_id"], "segformer") for r in rows])
    sv = np.array([sev(r["leaf_id"]) for r in rows])
    order = np.lexsort((sv, M.mean(0))); M, g = M[:, order], g[order]
    x = np.arange(len(rows)); n_in = 0
    for i in x:
        lo, hi = M[:, i].min(), M[:, i].max()
        ax.add_patch(plt.Rectangle((i - 0.34, lo - 0.3), 0.68, hi - lo + 0.6, facecolor="#e4e3df", edgecolor="none", zorder=1))
        inside = lo <= g[i] <= hi; n_in += inside
        ax.plot(i, g[i], "o", ms=6.5, mfc="#2a78d6" if inside else "white", mec="#2a78d6" if inside else "#e34948", mew=1.6, zorder=3)
    ax.set_yticks(range(4), ["G0 healthy", "G1 mild", "G2 moderate", "G3 severe"]); ax.set_ylim(-0.6, 3.6)
    ax.set_xlim(-0.8, len(rows) - 0.2); ax.set_xticks([]); ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0); ax.spines["left"].set_visible(False)
    ax.set_title(f"{name}: model inside the experts’ range on {n_in} of {len(rows)} leaves", loc="left", fontweight="bold")
axes[1].set_xlabel("Each column is one leaf (sorted from healthiest to most diseased, by the experts’ average)")
handles = [plt.Rectangle((0, 0), 1, 1, facecolor="#e4e3df", label="Grades given by the 4 experts (from lowest to highest)"),
           Line2D([], [], marker="o", ls="", mfc="#2a78d6", mec="#2a78d6", ms=6, label="SegFormer-B5 grade"),
           Line2D([], [], marker="o", ls="", mfc="white", mec="#e34948", mew=1.6, ms=6, label="SegFormer-B5 outside the experts’ range")]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=7.6)
headline(fig, "Leaf by leaf, the model’s grade falls where the experts’ grades fall")
fig.tight_layout(rect=(0, 0.05, 1, 0.94))
save(fig, "Fig_simple_3_perleaf")

# =====================================================================================
# D - Model severity % against the experts' average grade
# =====================================================================================
fig, axes = plt.subplots(1, 2, figsize=(16 * CM, 7.0 * CM), sharey=True)
rng_ = np.random.default_rng(3)
from scipy.stats import spearmanr
for ax, (d, name) in zip(axes, DIS):
    rows = rows_for(d)
    mg = np.array([np.mean([int(r[e]) for e in EXP]) for r in rows]); sv = np.array([sev(r["leaf_id"]) for r in rows])
    FLOOR = 0.01; s = np.maximum(sv, FLOOR)
    bands = [(FLOOR * 0.6, 0.5, "G0 healthy"), (0.5, 5, "G1 mild"), (5, 20, "G2 moderate"), (20, 100, "G3 severe")]
    for k, (lo, hi, lab) in enumerate(bands):
        if k % 2 == 0: ax.axhspan(lo, hi, color="#f3f2ef", zorder=0, lw=0)
        if d == "roya":   # grade names once, outside the right-hand panel, clear of the data
            ax.text(1.02, np.sqrt(max(lo, FLOOR) * hi), lab, transform=ax.get_yaxis_transform(), va="center", ha="left",
                    fontsize=7.4, color=INK2, clip_on=False)
    ax.plot(mg + rng_.uniform(-0.06, 0.06, len(rows)), s, "o", ms=5.5, mfc="#2a78d6", mec="white", mew=0.8, zorder=3)
    ax.text(0.02, 0.97, f"rank correlation {spearmanr(mg, sv).correlation:.2f}", transform=ax.transAxes, va="top", ha="left", fontsize=7.6, color=INK2)
    ax.set_yscale("log"); ax.set_ylim(FLOOR * 0.6, 100)
    ax.set_yticks([0.01, 0.1, 0.5, 1, 5, 20, 100], ["≈0", "0.1", "0.5", "1", "5", "20", "100"])
    ax.minorticks_off()
    ax.set_xlim(-0.25, 3.35); ax.set_xticks([0, 1, 2, 3], ["G0", "G1", "G2", "G3"])
    ax.set_title(f"{name}  ({len(rows)} leaves)", loc="left", fontweight="bold", pad=6)
    ax.set_xlabel("Experts’ average grade")
axes[0].set_ylabel("Diseased leaf area measured\nby SegFormer-B5 (%)")
headline(fig, "The more disease the experts see, the more the model measures",
         "Each dot is one leaf. Shaded bands show which grade the measured area corresponds to.")
fig.tight_layout(rect=(0, 0, 1, 0.86))
save(fig, "Fig_simple_4_severity")
