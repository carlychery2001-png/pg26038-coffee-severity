"""Publication figures for the expert-panel validation (thesis + MDPI paper).
Outputs figures/*.pdf (vector) and figures/*.png (600 dpi), 16 cm wide.
Rust = abaxial face, American leaf spot = mean of both faces (thesis rule).
Primary set = 53 leaves never used to train the deployed models."""
import csv, json, os, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

exec(open("final_analysis.py", encoding="utf-8").read().split("res = {}")[0])   # data, qwk, model_grade
T = json.load(open("tiefree_results.json"))
os.makedirs("figures", exist_ok=True)
TRAIN = {f for f, r in KEY.items() if r["model_split"] == "train"}

# ---- style: validated categorical slots 1-3 for the models, neutral ink for humans ----
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
MODEL = {"segformer": ("SegFormer-B5", "#2a78d6", "o"), "deeplab": ("DeepLabV3+", "#eb6834", "s"), "unetpp": ("UNet++", "#1baf7a", "^")}
EXPERT_C = "#6f6e69"
DIS = [("ojo", "American leaf spot"), ("roya", "Coffee leaf rust")]
CM = 1 / 2.54
plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.edgecolor": INK2, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "axes.labelcolor": INK, "text.color": INK, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 600,
})

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/{name}.{ext}", bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig); print("saved", name)

def rows_for(d):
    return [r for r in E if (d == "all" or r["leaf_id"].startswith(d)) and r["leaf_id"] not in TRAIN]

def loo(rows):
    M = np.array([[int(r[e]) for r in rows] for e in EXP])
    refs = [np.array([int(np.median(np.delete(M, j, 0)[:, i])) for i in range(len(rows))]) for j in range(4)]
    return M, refs

# =====================================================================================
# Figure 1 - forest plot: model minus evaluators (leave-one-out QWK), 95% bootstrap CI
# =====================================================================================
fig, ax = plt.subplots(figsize=(16 * CM, 7.2 * CM))
groups = [("ojo", "American leaf spot", 30), ("roya", "Coffee leaf rust", 23), ("all", "Pooled", 53)]
y, yt, ylab, sep = 0, [], [], []
for gi, (d, name, n) in enumerate(groups):
    for a in ("segformer", "deeplab", "unetpp"):
        t = T[f"{d}_{a}"]; lab, c, mk = MODEL[a]
        lo, hi = t["diff_ci"]
        ax.plot([lo, hi], [y, y], color=c, lw=2, solid_capstyle="round", zorder=2)
        ax.plot(t["diff"], y, mk, ms=6.5, mfc=c, mec="white", mew=1.2, zorder=3)
        fmt = lambda v: ("+" if round(v, 2) >= 0 else "−") + f"{abs(round(v, 2)):.2f}"
        ax.text(0.305, y, f"{fmt(t['diff'])}  [{fmt(lo)}, {fmt(hi)}]", va="center", ha="left", fontsize=7.2, color=INK2)
        yt.append(y); ylab.append(lab); y -= 1
    ax.text(-0.215, y + 3.55, f"{name} (n = {n})", ha="left", va="center", fontsize=8, fontweight="bold")
    if gi < 2: sep.append(y + 0.1)
    y -= 1.1
for s in sep: ax.axhline(s - 0.4, color=GRID, lw=0.6, zorder=0)
ax.axvline(0, color=INK2, lw=0.8, zorder=1)
ax.set_yticks(yt, ylab); ax.tick_params(axis="y", length=0)
ax.set_xlim(-0.2, 0.3); ax.set_ylim(y + 0.1, 0.9)
ax.set_xticks(np.arange(-0.2, 0.31, 0.1))
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.1f}".replace("-", "−") if abs(v) > 1e-9 else "0"))
ax.set_xlabel("Difference in quadratic-weighted kappa, model minus expert evaluators")
ax.text(0.305, 1.25, "Difference [95% CI]", ha="left", va="bottom", fontsize=7.2, color=INK2, fontweight="bold")
ax.annotate("", xy=(0.28, y + 0.45), xytext=(0.03, y + 0.45), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.7))
ax.text(0.155, y + 0.62, "model agrees with the panel more closely", ha="center", va="bottom", fontsize=6.8, color=MUTED)
ax.annotate("", xy=(-0.18, y + 0.45), xytext=(-0.03, y + 0.45), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.7))
ax.text(-0.105, y + 0.62, "experts agree more closely", ha="center", va="bottom", fontsize=6.8, color=MUTED)
ax.spines["left"].set_visible(False)
ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
save(fig, "Fig_expert_1_forest")

# =====================================================================================
# Figure 2 - leave-one-out QWK: each evaluator vs each model, per disease
# =====================================================================================
fig, axes = plt.subplots(1, 2, figsize=(16 * CM, 6.4 * CM), sharey=True)
for ax, (d, name) in zip(axes, DIS):
    rows = rows_for(d); M, refs = loo(rows)
    ek = [qwk(M[j], refs[j]) for j in range(4)]
    xs = np.arange(4)
    ax.axhspan(min(ek), max(ek), color="#efeeea", zorder=0, lw=0)
    ax.axhline(np.mean(ek), color=EXPERT_C, lw=1.2, ls=(0, (4, 2)), zorder=1)
    ax.text(0.2, np.mean(ek) - 0.006, f"mean {np.mean(ek):.2f}", va="top", ha="left", fontsize=6.8, color=INK2)
    ly = {}                                            # de-overlap the E1-E4 labels (min gap 0.02)
    for j in sorted(range(4), key=lambda j: ek[j]):
        y_ = ek[j] if not ly else max(ek[j], max(ly.values()) + 0.02)
        ly[j] = y_
    for j, k in enumerate(ek):
        ax.plot(0, k, "D", ms=5, mfc=EXPERT_C, mec="white", mew=1, zorder=3)
        ax.text(-0.14, ly[j], f"E{j + 1}", ha="right", va="center", fontsize=6.6, color=INK2)
    for i, a in enumerate(("segformer", "deeplab", "unetpp"), start=1):
        lab, c, mk = MODEL[a]
        k = np.mean([qwk(np.array([model_grade(r["leaf_id"], a) for r in rows]), refs[j]) for j in range(4)])
        ks = [qwk(np.array([model_grade(r["leaf_id"], a) for r in rows]), refs[j]) for j in range(4)]
        ax.plot([i, i], [min(ks), max(ks)], color=c, lw=2, solid_capstyle="round", alpha=0.45, zorder=2)
        ax.plot(i, k, mk, ms=7.5, mfc=c, mec="white", mew=1.2, zorder=3)
        ax.text(i + 0.14, k, f"{k:.2f}", va="center", ha="left", fontsize=7, color=INK)
    ax.set_xticks(xs, ["Expert\nevaluators", "SegFormer-\nB5", "DeepLab-\nV3+", "UNet++"])
    ax.set_xlim(-0.55, 3.6); ax.set_title(f"{name} (n = {len(rows)})", loc="left", fontweight="bold")
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0); ax.tick_params(axis="x", length=0)
axes[0].set_ylabel("Quadratic-weighted kappa vs\nleave-one-out panel consensus")
axes[0].set_ylim(0.5, 1.0)
handles = [Line2D([], [], marker="D", ls="", mfc=EXPERT_C, mec="white", ms=5, label="Individual expert (E1–E4)"),
           Line2D([], [], color=EXPERT_C, ls=(0, (4, 2)), lw=1.2, label="Expert mean"),
           Line2D([], [], color=MUTED, lw=2, alpha=0.45, label="Model range across the four references")]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.07))
fig.tight_layout(rect=(0, 0.05, 1, 1))
save(fig, "Fig_expert_2_raters")

# =====================================================================================
# Figure 3 - per-leaf: spread of the four expert grades vs the SegFormer-B5 grade
# =====================================================================================
fig, axes = plt.subplots(2, 1, figsize=(16 * CM, 9.6 * CM))
for ax, (d, name) in zip(axes, DIS):
    rows = rows_for(d)
    M = np.array([[int(r[e]) for r in rows] for e in EXP])
    g = np.array([model_grade(r["leaf_id"], "segformer") for r in rows])
    sev = np.array([float(P[r["leaf_id"]]["segformer_sev_ab"]) if d == "roya" else (float(P[r["leaf_id"]]["segformer_sev_ab"]) + float(P[r["leaf_id"]]["segformer_sev_ad"])) / 2 for r in rows])
    order = np.lexsort((sev, M.mean(0)))
    M, g = M[:, order], g[order]
    x = np.arange(len(rows))
    for i in x:
        lo, hi = M[:, i].min(), M[:, i].max()
        ax.add_patch(plt.Rectangle((i - 0.32, lo - 0.3), 0.64, hi - lo + 0.6, facecolor="#e9e8e4", edgecolor="none", zorder=1))
        for gval, cnt in zip(*np.unique(M[:, i], return_counts=True)):
            offs = (np.arange(cnt) - (cnt - 1) / 2) * 0.14
            ax.plot(i + offs, [gval] * cnt, "o", ms=2.6, color=EXPERT_C, zorder=2)
        inside = lo <= g[i] <= hi
        ax.plot(i, g[i], "D", ms=6, mfc="#2a78d6" if inside else "white", mec="#2a78d6", mew=1.4, zorder=3)
    ax.set_yticks(range(4), ["G0", "G1", "G2", "G3"]); ax.set_ylim(-0.6, 3.6)
    ax.set_xlim(-0.8, len(rows) - 0.2); ax.set_xticks([])
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    n_in = int(sum(M[:, i].min() <= g[i] <= M[:, i].max() for i in x))
    ax.set_title(f"{name}: SegFormer-B5 grade inside the experts’ range for {n_in} of {len(rows)} leaves", loc="left", fontweight="bold")
    ax.set_ylabel("Severity grade")
axes[1].set_xlabel("Leaves, ordered by mean expert grade")
handles = [plt.Rectangle((0, 0), 1, 1, facecolor="#e9e8e4", label="Range of the four expert grades"),
           Line2D([], [], marker="o", ls="", color=EXPERT_C, ms=3, label="Individual expert grade"),
           Line2D([], [], marker="D", ls="", mfc="#2a78d6", mec="#2a78d6", ms=5.5, label="SegFormer-B5, inside range"),
           Line2D([], [], marker="D", ls="", mfc="white", mec="#2a78d6", mew=1.4, ms=5.5, label="SegFormer-B5, outside range")]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.045))
fig.tight_layout(rect=(0, 0.04, 1, 1))
save(fig, "Fig_expert_3_perleaf")

# =====================================================================================
# Figure 4 - continuous SegFormer-B5 severity vs mean expert grade (log scale, cut-points)
# =====================================================================================
fig, axes = plt.subplots(1, 2, figsize=(16 * CM, 6.6 * CM), sharey=True)
rng_ = np.random.default_rng(3)
for ax, (d, name) in zip(axes, DIS):
    rows = rows_for(d)
    mg = np.array([np.mean([int(r[e]) for e in EXP]) for r in rows])
    sev = np.array([float(P[r["leaf_id"]]["segformer_sev_ab"]) if d == "roya" else (float(P[r["leaf_id"]]["segformer_sev_ab"]) + float(P[r["leaf_id"]]["segformer_sev_ad"])) / 2 for r in rows])
    FLOOR = 0.01
    s = np.maximum(sev, FLOOR)
    for cut, lab in ((0.5, "0.5%"), (5, "5%"), (20, "20%")):
        ax.axhline(cut, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    for lo, hi, gl in ((FLOOR * 0.6, 0.5, "G0"), (0.5, 5, "G1"), (5, 20, "G2"), (20, 100, "G3")):
        ax.text(3.08, np.sqrt(max(lo, FLOOR * 0.8) * hi), gl, va="center", ha="left", fontsize=7, color=INK2, fontweight="bold")
    jit = rng_.uniform(-0.06, 0.06, len(rows))
    ax.plot(mg + jit, s, "o", ms=5, mfc="#2a78d6", mec="white", mew=0.8, alpha=0.9, zorder=3)
    rho = __import__("scipy.stats", fromlist=["spearmanr"]).spearmanr(mg, sev).correlation
    ax.text(0.02, 0.97, f"Spearman ρ = {rho:.2f}", transform=ax.transAxes, va="top", ha="left", fontsize=7.2, color=INK2)
    ax.set_yscale("log"); ax.set_ylim(FLOOR * 0.6, 100)
    ax.set_yticks([0.01, 0.1, 0.5, 1, 5, 20, 100], ["≤0.01", "0.1", "0.5", "1", "5", "20", "100"])
    ax.set_xlim(-0.2, 3.3); ax.set_xticks([0, 1, 2, 3])
    ax.set_title(f"{name} (n = {len(rows)})", loc="left", fontweight="bold")
    ax.set_xlabel("Mean grade of the four experts (G0 = 0 … G3 = 3)")
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
axes[0].set_ylabel("SegFormer-B5 predicted severity (%)")
fig.tight_layout()
save(fig, "Fig_expert_4_severity")
