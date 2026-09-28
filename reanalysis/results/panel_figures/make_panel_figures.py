"""Journal-style expert-panel figures for the PG-26038 major revision (no in-figure headline titles).

Outputs (this folder only):
  figure_panel_agreement.pdf/.png  2 x 2: (a, b) QWK vs leave-one-out panel consensus with 95% CI;
                                   (c, d) 100% stacked bars over the 4n comparisons with the LOO consensus:
                                          same grade / 1 grade off / >=2 grades off
  figure_panel_perleaf.pdf/.png    per-leaf range of the four expert grades vs SegFormer-B5 grade
  figure_panel_severity.pdf/.png   SegFormer-B5 severity vs mean expert grade (Supplement); overlapping markers are
                                   dodged by their real outline in mm (asserted: no two triangles overlap)
  panel_figure_numbers.json        every plotted number, plus consistency checks against the published JSON

Data: <ROOT>/validation (read only). Primary set = the 53 leaves not used to train the models
(the 7 healthy rust training leaves are excluded). Model grade: rust = abaxial face; American leaf spot =
mean of both faces. Same definitions as validation/final_analysis.py and validation/tiefree.py.
Style follows the paper's matplotlib figures (figure4/5/7.png): Arial, bold "(a) Disease" panel titles with
the species in grey italic beneath, light grey axes, Okabe-Ito model colours, 16 cm wide, fonttype 42.
"""
import os as _os  # (release) all paths below are relative to ROOT (repository root by default; see README)
ROOT = (_os.environ.get("PG26038_ROOT") or _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))).replace(chr(92), "/")
import csv, json, os
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerBase

SRC = (ROOT + "/validation")
OUT = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------------------------------------ data
E = list(csv.DictReader(open(f"{SRC}/expert_consensus.csv", encoding="utf-8")))
P = {r["leaf_id"]: r for r in csv.DictReader(open(f"{SRC}/model_predictions.csv", encoding="utf-8"))}
KEY = {r["form_id"]: r for r in csv.DictReader(open(f"{SRC}/form_image_key.csv", encoding="utf-8"))}
TIEFREE = json.load(open(f"{SRC}/tiefree_results.json"))
CLOSE = json.load(open(f"{SRC}/closeness.json"))
FINAL = json.load(open(f"{SRC}/final_results.json"))
EXP = [k for k in E[0] if k not in ("leaf_id", "consensus_grade", "range")]
TRAIN = {f for f, r in KEY.items() if r["model_split"] == "train"}
K = 4
grade = lambda s: 0 if s < 0.5 else 1 if s < 5 else 2 if s < 20 else 3


def severity(lid, arch="segformer"):
    p = P[lid]
    if lid.startswith("roya"):
        return float(p[f"{arch}_sev_ab"])                      # rust: abaxial face
    return (float(p[f"{arch}_sev_ab"]) + float(p[f"{arch}_sev_ad"])) / 2   # leaf spot: mean of both faces


def qwk(a, b):
    O = np.bincount(K * np.asarray(a) + np.asarray(b), minlength=K * K).reshape(K, K).astype(float)
    Ex = np.outer(O.sum(1), O.sum(0)) / O.sum()
    W = (np.subtract.outer(np.arange(K), np.arange(K)) ** 2) / (K - 1) ** 2
    den = (W * Ex).sum()
    return np.nan if den == 0 else 1 - (W * O).sum() / den


def loo_refs(M):          # median of the other three experts (always an integer, cannot tie)
    return [np.median(np.delete(M, j, 0), axis=0).astype(int) for j in range(M.shape[0])]


def fair(M, G):
    """LOO agreement: typical expert = mean_j QWK(expert j, ref_j); model = mean_j QWK(model, ref_j)."""
    refs = loo_refs(M)
    out = {"expert": float(np.mean([qwk(M[j], refs[j]) for j in range(len(refs))]))}
    for a, g in G.items():
        out[a] = float(np.mean([qwk(g, r) for r in refs]))
    return out


DIS = [("roya", "Coffee leaf rust", "Hemileia vastatrix"), ("ojo", "American leaf spot", "Mycena citricolor")]
ARCH = ["segformer", "unetpp", "deeplab"]
B, SEED = (200 if os.environ.get("PANEL_FAST") else 10000), 7   # same resample stream as validation/tiefree.py,
                                                                   # so the paired differences reproduce

N = {"checks": {}}
for d, name, _ in DIS:
    rows = [r for r in E if r["leaf_id"].startswith(d) and r["leaf_id"] not in TRAIN]
    lids = [r["leaf_id"] for r in rows]
    M = np.array([[int(r[e]) for r in rows] for e in EXP])
    G = {a: np.array([grade(severity(l, a)) for l in lids]) for a in ARCH}
    n = len(rows)
    point = fair(M, G)
    per_eval = [float(qwk(M[j], r)) for j, r in enumerate(loo_refs(M))]

    # paired leaf bootstrap (percentile), identical index stream to tiefree.py
    rng = np.random.default_rng(SEED); boot = {k: [] for k in point}; dropped = 0
    for _ in range(B):
        i = rng.integers(0, n, n)
        v = fair(M[:, i], {a: g[i] for a, g in G.items()})
        if not all(np.isfinite(list(v.values()))):
            dropped += 1; continue
        for k in v: boot[k].append(v[k])
    boot = {k: np.array(v) for k, v in boot.items()}
    ci = {k: [float(x) for x in np.percentile(v, [2.5, 97.5])] for k, v in boot.items()}
    diff_ci = {a: [float(x) for x in np.percentile(boot[a] - boot["expert"], [2.5, 97.5])] for a in ARCH}

    # closeness: |grade - LOO ref| in {0, 1, >=2}, averaged over the four references
    refs = loo_refs(M)
    def dist(dev):
        dev = np.abs(np.asarray(dev)).ravel()
        return [float(np.mean(dev == 0)), float(np.mean(dev == 1)), float(np.mean(dev >= 2))]
    close = {"expert": dist([M[j] - refs[j] for j in range(len(refs))])}
    for a in ARCH:
        close[a] = dist([G[a] - r for r in refs])

    # per leaf + severity
    lo, hi = M.min(0), M.max(0)
    sev = np.array([severity(l) for l in lids])
    mean_g = M.mean(0)
    inside = (G["segformer"] >= lo) & (G["segformer"] <= hi)
    outside = [dict(form_id=l, expert_grades=M[:, k].tolist(), segformer_grade=int(G["segformer"][k]),
                    segformer_severity=float(sev[k])) for k, l in enumerate(lids) if not inside[k]]

    N[d] = dict(n=n, loo_qwk=point, loo_qwk_ci95=ci, individual_evaluators_loo_qwk=dict(zip([f"E{j + 1}" for j in range(len(EXP))], per_eval)),
                diff_vs_expert=dict((a, point[a] - point["expert"]) for a in ARCH), diff_vs_expert_ci95=diff_ci,
                bootstrap=dict(B=B, seed=SEED, type="paired leaf resampling, percentile", dropped_undefined=dropped),
                closeness=close, segformer_in_expert_range=[int(inside.sum()), n], segformer_outside_range=outside,
                spearman_segformer_severity_vs_mean_grade=float(spearmanr(mean_g, sev).correlation),
                unanimous_leaves=int(np.sum(lo == hi)))
    N[d]["_arrays"] = dict(M=M, G=G, lo=lo, hi=hi, sev=sev, mean_g=mean_g, inside=inside)
    N[d]["_lids"] = lids

    # consistency with the published analysis files
    for a in ARCH:
        t = TIEFREE[f"{d}_{a}"]
        N["checks"][f"{d}_{a}_fair_model"] = abs(point[a] - t["fair_model"]) < 1e-9
        N["checks"][f"{d}_{a}_diff_ci"] = [round(x, 3) for x in diff_ci[a]] == t["diff_ci"]
        N["checks"][f"{d}_{a}_closeness"] = np.allclose(close[a], CLOSE[d][a])
        N["checks"][f"{d}_{a}_in_range"] = abs(np.mean((G[a] >= lo) & (G[a] <= hi)) - FINAL[f"primary_{d}"][a]["in_range"]) < 1e-3
    N["checks"][f"{d}_expert_fair"] = abs(point["expert"] - TIEFREE[f"{d}_segformer"]["fair_expert"]) < 1e-9
    N["checks"][f"{d}_expert_closeness"] = np.allclose(close["expert"], CLOSE[d]["expert"])

# ------------------------------------------------------------------------------------------------ style
CM = 1 / 2.54
INK, INK2, TICK, AXIS = "#1b1b1b", "#565656", "#8f8f8f", "#d2d2d2"
EXPERT_C = "#8c8c8c"
RATERS = [  # key, label, colour, marker, markersize
    ("expert", "Typical expert", EXPERT_C, "D", 5.8),
    ("segformer", "SegFormer-B5", "#0072B2", "^", 7.0),
    ("unetpp", "UNet++", "#009E73", "s", 6.0),
    ("deeplab", "DeepLabV3+", "#E69F00", "o", 6.6),
]
plt.rcParams.update({
    "font.family": "Arial", "font.size": 8.5, "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5, "axes.edgecolor": AXIS, "axes.linewidth": 0.9, "axes.labelcolor": INK2,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": TICK, "ytick.labelcolor": TICK,
    "xtick.major.size": 2.4, "ytick.major.size": 2.4, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.pad": 2.5, "ytick.major.pad": 2.5, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.facecolor": "white", "figure.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 600, "legend.frameon": False,
})


def panel_title(ax, letter, name, species, species_gap=3.0):
    ax.annotate(f"({letter}) {name}", xy=(0, 1), xycoords="axes fraction", xytext=(0, species_gap + 12.5),
                textcoords="offset points", ha="left", va="bottom", fontsize=10.5, fontweight="bold", color=INK)
    ax.annotate(species, xy=(0, 1), xycoords="axes fraction", xytext=(0, species_gap),
                textcoords="offset points", ha="left", va="bottom", fontsize=8, fontstyle="italic", color=INK2)


def mix(c, w):          # blend colour c with white; w = weight of white
    rgb = np.array(matplotlib.colors.to_rgb(c))
    return tuple(rgb * (1 - w) + w)


def text_on(c):         # dark or white text, whichever has the higher WCAG contrast on colour c
    def lum(rgb):
        rgb = np.array(matplotlib.colors.to_rgb(rgb))
        rgb = np.where(rgb <= 0.03928, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    L = lum(c)
    return "white" if (1.05) / (L + 0.05) > (L + 0.05) / (lum(INK) + 0.05) else INK


def axes_cm(fig, W, H, left, bottom, width, height):   # place axes in absolute centimetres
    return fig.add_axes([left / W, bottom / H, width / W, height / H])


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}", facecolor="white")   # fixed 16 cm canvas, no bbox cropping
    plt.close(fig)
    print("saved", name)


# ================================================================================================ Figure A
TINT = 0.62
SEG_LAB = ["Same grade", "1 grade off", "\u22652 grades off"]
BEYOND = "#1b1b1b"


def seg_colours(c):
    return [c, mix(c, TINT), BEYOND]


class Swatch:           # legend proxy: one small stripe per rater colour
    def __init__(self, colours): self.colours = colours


class SwatchHandler(HandlerBase):
    # Matplotlib centres the handle box on the text's font box (descender included), which leaves the swatch
    # ~0.4 mm above the optical centre of the label (midway between baseline and cap height). Lower it by
    # DROP x handle height; checked on the 600 dpi PNG (swatch centre vs baseline/cap-height midpoint).
    DROP = 0.16

    def create_artists(self, legend, orig, x0, y0, width, height, fontsize, trans):
        k = len(orig.colours); w = width / k; y0 -= self.DROP * height
        return [Rectangle((x0 + i * w, y0), w, height, facecolor=c, edgecolor="none", transform=trans)
                for i, c in enumerate(orig.colours)]


# layout in cm: title block above each axes ~0.9 cm, x-label block below ~0.95 cm
WA, LA, GAPA, RA, HAX = 16.0, 2.45, 1.15, 0.35, 3.5
WAX = (WA - LA - GAPA - RA) / 2
B_BOT = 0.08 + 0.45 + 0.22 + 0.95                   # legend + gap + x-label block
B_TOP = B_BOT + HAX + 0.95 + 0.40 + 0.92            # x-label block + gap + title block
HA = B_TOP + HAX + 1.02
fig = plt.figure(figsize=(WA * CM, HA * CM))
ax_at = lambda row, col: axes_cm(fig, WA, HA, LA + col * (WAX + GAPA), B_TOP if row == 0 else B_BOT, WAX, HAX)
ypos = np.arange(len(RATERS))[::-1].astype(float)
letters = iter("abcd")
for col, (d, name, species) in enumerate(DIS):
    D = N[d]
    ax = ax_at(0, col)
    ax.axvline(D["loo_qwk"]["expert"], color=EXPERT_C, lw=0.7, ls=(0, (2, 2)), zorder=1)
    for y, (k, lab, c, mk, ms) in zip(ypos, RATERS):
        v, (l, h) = D["loo_qwk"][k], D["loo_qwk_ci95"][k]
        ax.plot([l, h], [y, y], color=c, lw=2.0, solid_capstyle="round", zorder=2)
        ax.plot(v, y, mk, ms=ms, mfc=c, mec="white", mew=1.1, zorder=3)
        ax.text(h + 0.012, y, f"{v:.2f}", va="center", ha="left", fontsize=8.5, color=INK2)
    ax.set_ylim(-0.6, len(RATERS) - 0.4)
    ax.set_xlim(0.40, 1.0); ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xticklabels(["0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"])
    ax.set_yticks(ypos)
    if col == 0:
        ax.set_yticklabels([r[1] for r in RATERS], color=INK, fontsize=8.5)
    else:
        ax.set_yticklabels([])
    ax.set_xlabel("Quadratic-weighted kappa (95% CI)")
    panel_title(ax, next(letters), name, species)
ax_bottom = []
for col, (d, name, species) in enumerate(DIS):
    D = N[d]
    ax = ax_at(1, col); ax_bottom.append(ax)
    for y, (k, lab, c, mk, ms) in zip(ypos, RATERS):
        fr = D["closeness"][k]; left = 0.0
        for s, (w, sc) in enumerate(zip(fr, seg_colours(c))):
            w *= 100
            if w <= 0:
                continue
            ax.barh(y, w, left=left, height=0.62, color=sc, edgecolor="white", linewidth=0.8, zorder=2)
            if w >= 8:
                ax.text(left + w / 2, y, f"{w:.0f}%", ha="center", va="center", fontsize=8, color=text_on(sc))
            else:     # small segment (e.g. >=2 grades off): label just beyond the bar end
                ax.text(left + w + 1.5, y, f"{w:.0f}%", ha="left", va="center", fontsize=8, color=INK, clip_on=False)
            left += w
    ax.set_ylim(-0.6, len(RATERS) - 0.4)
    ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks(ypos)
    if col == 0:
        ax.set_yticklabels([r[1] for r in RATERS], color=INK, fontsize=8.5)
    else:
        ax.set_yticklabels([])
    ax.set_xlabel("Comparisons with consensus (%)")   # unit = rater grade vs one LOO consensus grade (4 x n per rater)
    panel_title(ax, next(letters), name, species)
handles = [Swatch([seg_colours(r[2])[s] for r in RATERS]) for s in range(2)] + [Swatch([BEYOND])]
fig.legend(handles, SEG_LAB, handler_map={Swatch: SwatchHandler()}, loc="lower center", ncol=3,
           bbox_to_anchor=((LA + (WA - RA)) / 2 / WA, 0.08 / HA), handlelength=2.6, handleheight=0.9,
           columnspacing=2.2, borderaxespad=0, borderpad=0)
save(fig, "figure_panel_agreement")

# ================================================================================================ Figure B
SEG_C = RATERS[1][2]
RANGE_C = "#dcdcdc"
nr, no = N["roya"]["n"], N["ojo"]["n"]
WB, LB, GAPB, RB, HBX = 16.0, 1.55, 0.35, 0.15, 4.2
BB = 0.08 + 0.45 + 0.25 + 0.55                      # legend + gap + x-label block (no tick labels)
HB = BB + HBX + 1.02
span = {"roya": nr + 0.6, "ojo": no + 0.6}           # x-limits span n + 0.6 columns: same column width in both panels
unit = (WB - LB - GAPB - RB) / (span["roya"] + span["ojo"])
fig = plt.figure(figsize=(WB * CM, HB * CM))
for col, (d, name, species) in enumerate(DIS):
    A = N[d]["_arrays"]; n = N[d]["n"]
    order = np.lexsort((A["sev"], A["mean_g"]))
    lo, hi, g, ins = A["lo"][order], A["hi"][order], A["G"]["segformer"][order], A["inside"][order]
    ax = axes_cm(fig, WB, HB, LB if col == 0 else LB + span["roya"] * unit + GAPB, BB, span[d] * unit, HBX)
    for yy in range(4):
        ax.axhline(yy, color="#efefef", lw=0.6, zorder=0)
    for i in range(n):
        ax.add_patch(Rectangle((i - 0.36, lo[i] - 0.28), 0.72, hi[i] - lo[i] + 0.56, facecolor=RANGE_C,
                               edgecolor="none", zorder=1))
        ax.plot(i, g[i], "^", ms=5.6, mfc=SEG_C if ins[i] else "white", mec=SEG_C, mew=1.0 if ins[i] else 1.2, zorder=3)
    ax.set_xlim(-0.8, n - 0.2); ax.set_xticks([])
    ax.spines["bottom"].set_visible(True)
    ax.set_ylim(-0.55, 3.55); ax.set_yticks(range(4))
    ax.set_yticklabels(["G0", "G1", "G2", "G3"] if col == 0 else [])
    if col == 0:
        ax.set_ylabel("Severity grade")
    ax.set_xlabel("Leaves, ordered by mean expert grade")
    panel_title(ax, "ab"[col], name, species)
handles = [Rectangle((0, 0), 1, 1, facecolor=RANGE_C, edgecolor="none"),
           Line2D([], [], marker="^", ls="", ms=5.6, mfc=SEG_C, mec=SEG_C, mew=1.0),
           Line2D([], [], marker="^", ls="", ms=5.6, mfc="white", mec=SEG_C, mew=1.2)]
fig.legend(handles, ["Expert grade range", "SegFormer-B5 within range", "SegFormer-B5 outside range"],
           loc="lower center", ncol=3, bbox_to_anchor=((LB + WB - RB) / 2 / WB, 0.08 / HB), handlelength=1.4,
           columnspacing=2.2, borderaxespad=0, borderpad=0)
save(fig, "figure_panel_perleaf")

# ================================================================================================ Figure C
FLOOR = 0.01
BANDS = [(FLOOR * 0.55, 0.5, "G0"), (0.5, 5, "G1"), (5, 20, "G2"), (20, 100, "G3")]
WC, LC, GAPC, RC, HCX = 16.0, 1.5, 0.3, 0.65, 4.8
WCX = (WC - LC - GAPC - RC) / 2
BC = 0.1 + 0.95
HC = BC + HCX + 1.02
XLIM, YLIM = (-0.15, 3.15), (FLOOR * 0.55, 100)
MS_C, MEW_C, CLEAR_MM = 4.6, 0.6, 0.15          # marker size (pt), white edge (pt), min fill-to-fill clearance (mm)
fig = plt.figure(figsize=(WC * CM, HC * CM))

# --- dodge: markers are spaced by their real outline in millimetres on the printed 16 cm figure ------------
PT_MM = 25.4 / 72


def tri_mm(cx, cy, w):   # outline of matplotlib's '^' marker: bbox centred on the data point, width = height = ms
    h = w / 2
    return np.array([[cx, cy + h], [cx - h, cy - h], [cx + h, cy - h]])


_AXES = [np.array([0.0, 1.0]), np.array([1.0, 0.0]), np.array([2.0, 1.0]) / np.sqrt(5), np.array([-2.0, 1.0]) / np.sqrt(5)]


def tri_gap(p, q):       # separating-axis distance bound between two triangles (exact for overlap / no overlap)
    return max(max((q @ a).min() - (p @ a).max(), (p @ a).min() - (q @ a).max()) for a in _AXES)


def need_dx(dy, w, g):
    """Smallest |dx| (mm) at which two '^' markers of width w, dy mm apart vertically, are >= g mm apart.
    Derived from the three edge normals of the triangle; a marker may tuck beside a neighbour it partly clears."""
    dy = abs(dy)
    if dy - w >= g:
        return 0.0
    return max(0.0, min(w + g, (np.sqrt(5) * g + 2 * w - dy) / 2))


def dodge_mm(col_mm, Y, w, g):
    """Horizontal offsets (mm). Leaves in one mean-grade column whose markers would overlap form a cluster.
    For every ordering of a cluster the markers are packed left to right as tightly as their triangular outlines
    allow; the narrowest ordering wins (ties: closest to severity increasing left to right), and the packed
    cluster is centred on its true mean grade."""
    import itertools
    off = np.zeros(len(Y))
    for c0 in np.unique(col_mm):
        idx = sorted(np.flatnonzero(col_mm == c0), key=lambda i: Y[i])
        clusters, cur = [], [idx[0]]
        for i in idx[1:]:
            if any(need_dx(Y[i] - Y[j], w, g) > 0 for j in cur):
                cur.append(i)
            else:
                clusters.append(cur); cur = [i]
        clusters.append(cur)
        for c in (c for c in clusters if len(c) > 1):
            best = None
            for perm in itertools.permutations(c):
                x = {perm[0]: 0.0}
                for k in perm[1:]:
                    cands = sorted({0.0} | {x[j] + s * need_dx(Y[k] - Y[j], w, g) for j in x for s in (1, -1)})
                    x[k] = next(cx for cx in cands if cx >= 0 and
                                all(abs(cx - x[j]) >= need_dx(Y[k] - Y[j], w, g) - 1e-9 for j in x))
                key = (round(max(x.values()) - min(x.values()), 6), sum(abs(perm.index(i) - r) for r, i in enumerate(c)))
                if best is None or key < best[0]:
                    best = (key, x)
            x = best[1]; mid = (max(x.values()) + min(x.values())) / 2
            for i in c:
                off[i] = x[i] - mid
    return off


UX = WCX * 10 / (XLIM[1] - XLIM[0])                                # mm per grade unit
UY = HCX * 10 / (np.log10(YLIM[1]) - np.log10(YLIM[0]))            # mm per decade
W_MM = MS_C * PT_MM
swarm = {}
for col, (d, name, species) in enumerate(DIS):
    A = N[d]["_arrays"]
    ax = axes_cm(fig, WC, HC, LC + col * (WCX + GAPC), BC, WCX, HCX)
    for k, (lo, hi, lab) in enumerate(BANDS):
        if k % 2 == 0:
            ax.axhspan(lo, hi, color="#f4f4f4", lw=0, zorder=0)
        if col == 1:
            ax.text(1.025, np.sqrt(max(lo, FLOOR) * hi), lab, transform=ax.get_yaxis_transform(), ha="left",
                    va="center", fontsize=8.5, color=INK2, clip_on=False)
    y = np.maximum(A["sev"], FLOOR)
    Ymm, X0mm = np.log10(y) * UY, A["mean_g"] * UX
    off = dodge_mm(X0mm, Ymm, W_MM, CLEAR_MM)
    Xmm = X0mm + off
    pairs = [(tri_gap(tri_mm(Xmm[i], Ymm[i], W_MM), tri_mm(Xmm[j], Ymm[j], W_MM)), i, j)
             for i in range(len(y)) for j in range(i + 1, len(y))]
    clear, wi, wj = min(pairs)
    assert clear >= CLEAR_MM - 1e-6, (d, clear)                      # no two markers overlap anywhere in the panel
    x = Xmm / UX
    lids = N[d]["_lids"]
    swarm[d] = dict(max_abs_shift_grade_units=float(np.abs(off).max() / UX),
                    min_clearance_mm=float(clear), closest_pair=[lids[wi], lids[wj]],
                    shifted={lids[i]: round(float(off[i] / UX), 4) for i in np.flatnonzero(np.abs(off) > 1e-9)})
    ax.plot(x, y, "^", ms=MS_C, mfc=SEG_C, mec="white", mew=MEW_C, zorder=3)
    ax.text(0.03, 0.97, f"Spearman \u03c1 = {N[d]['spearman_segformer_severity_vs_mean_grade']:.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color=INK2)
    ax.set_yscale("log"); ax.set_ylim(*YLIM); ax.minorticks_off()
    ax.set_yticks([0.01, 0.1, 0.5, 1, 5, 20, 100])
    ax.set_yticklabels(["\u22640.01", "0.1", "0.5", "1", "5", "20", "100"] if col == 0 else [])
    ax.set_xlim(*XLIM); ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("Mean expert grade")
    if col == 0:
        ax.set_ylabel("SegFormer-B5 severity (%)")
    panel_title(ax, "ab"[col], name, species)
save(fig, "figure_panel_severity")

# ------------------------------------------------------------------------------------------------ numbers
for d, _, _ in DIS:
    N[d].pop("_arrays"); N[d].pop("_lids")
    N[d]["severity_plot_dodge"] = dict(swarm[d], marker_pt=MS_C, marker_mm=round(W_MM, 3), min_clearance_required_mm=CLEAR_MM,
                                       mm_per_grade_unit=round(UX, 3), mm_per_decade=round(float(UY), 3))
N["checks"] = {k: bool(v) for k, v in N["checks"].items()}
N["definitions"] = dict(primary_set="53 leaves not used for training (7 healthy rust training leaves excluded)",
                        rust_grade="abaxial face", leaf_spot_grade="mean of abaxial and adaxial faces",
                        cut_points="G0 <0.5%, G1 0.5-<5%, G2 5-<20%, G3 >=20%",
                        loo_reference="median of the other three experts",
                        evaluators="E1-E4 = expert columns of expert_consensus.csv, in file order")
json.dump(N, open(f"{OUT}/panel_figure_numbers.json", "w"), indent=1)
print(json.dumps(N["checks"], indent=0))
for d, _, _ in DIS:
    D = N[d]
    print(d, "n", D["n"], {k: (round(v, 3), [round(x, 3) for x in D["loo_qwk_ci95"][k]]) for k, v in D["loo_qwk"].items()})
    print("   closeness", {k: [round(x * 100, 1) for x in v] for k, v in D["closeness"].items()})
    print("   dodge", {k: v for k, v in D["severity_plot_dodge"].items() if k != "shifted"})
    print("   in range", D["segformer_in_expert_range"], "rho", round(D["spearman_segformer_severity_vs_mean_grade"], 3),
          "evaluators", {k: round(v, 3) for k, v in D["individual_evaluators_loo_qwk"].items()}, "dropped", D["bootstrap"]["dropped_undefined"])
