"""Journal-style expert-panel figures, FIVE evaluators (PG-26038 major revision, round 4, 28 Sep 2026).

Adapted from results/panel_figures/make_panel_figures.py (four-evaluator version; its outputs are backed up in
round4/before_round4/figures_panel_4experts/). Same style, colours, fonts, sizes and layout logic; what changed:
  * five evaluators; primary design = leave-one-evaluator-out, three-evaluator consensus: each evaluator i is scored
    against median(S) for every 3-subset S of the other four (20 tie-free reference sets per disease) and every model
    against the same 20 references (round4_plan.md, FIFTH EXPERT; results/r4_panel5/results.md, Design);
  * per-rater 95% intervals are taken from the verified results/r4_panel5/results.json (10,000 leaf resamples, seed
    20260928) and are re-derived here with the same resample stream as a check (the old figure used seed 7);
  * evaluators appear only as the 'typical evaluator' or as the panel (range, mean); no per-evaluator label or value is
    drawn or written, because one evaluator is affiliated with the authors' institution and the paper names the panel.

Outputs (this folder; PDFs also copied to the manuscript's figures/ folder under the same names):
  fig_panel_agreement.pdf/.png  2 x 2: (a, b) QWK vs the three-evaluator consensus with 95% CI (typical evaluator and
                                the three original models); (c, d) 100% stacked bars over the 20n comparisons with the
                                consensus: same grade / 1 grade off / >=2 grades off
  fig_panel_perleaf.pdf/.png    per-leaf range of the five panel grades vs the SegFormer-B5 grade
  fig_panel_severity.pdf/.png   SegFormer-B5 severity vs mean panel grade; overlapping markers are dodged by their real
                                outline in mm (asserted: no two triangles overlap)
  panel_figure_numbers.json     every plotted number and the consistency checks against results.json

Data (read only; nothing is copied):
  validation/private/form_export_5experts_raw.csv  raw form export, read IN PLACE; only the item names in the header of
        columns 8+ and the 60 grade cells of each row are used (identity columns are never read into a variable that is
        kept, printed or written; same parsing as results/r4_panel5/panel5_analysis.py)
  validation/form_image_key.csv, validation/model_predictions.csv (severities only; the *_grade columns are NOT used,
        because they grade rust from both faces), results/r4_panel5/results.json
Primary set = the 53 form leaves unseen in training (7 healthy rust training leaves excluded). Model grade: rust =
abaxial face; American leaf spot = mean of both faces; cut-points 0.5, 5, 20%.
"""
import os as _os  # (release) all paths below are relative to ROOT (repository root by default; see README)
ROOT = (_os.environ.get("PG26038_ROOT") or _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))).replace(chr(92), "/")
import csv, itertools, json, os, re, shutil
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerBase

SRC = (ROOT + "/validation")
R5 = (ROOT + "/reanalysis/results/r4_panel5")
OUT = os.path.dirname(os.path.abspath(__file__))
MAN = _os.environ.get("PG26038_MANUSCRIPT_FIGS", "")  # (release) optional copy target; empty = no copy
RES = json.load(open(f"{R5}/results.json", encoding="utf-8"))
K = 4
B, SEED = RES["meta"]["B"], RES["meta"]["seed"]
assert (B, SEED) == (10000, 20260928)
grade = lambda s: 0 if s < 0.5 else 1 if s < 5 else 2 if s < 20 else 3

# ------------------------------------------------------------------------------------------------ data
with open(f"{SRC}/private/form_export_5experts_raw.csv", encoding="utf-8-sig") as fh:
    _raw = list(csv.reader(fh))
assert len(_raw) == 6 and all(len(r) == 67 for r in _raw)
ITEMS = [re.match(r"^\s*\d+\)\s*(\w+)\s*-", h).group(1) for h in _raw[0][7:]]
assert ITEMS == [f"ojo{i:02d}" for i in range(1, 31)] + [f"roya{i:02d}" for i in range(1, 31)]
_cells = [[c.strip() for c in r[7:]] for r in _raw[1:]]
assert all(re.fullmatch("G[0-3]", c) for r in _cells for c in r)
GR = np.array([[int(c[1]) for c in r] for r in _cells])          # 5 evaluators x 60 items (grades only)
del _raw, _cells                                                  # nothing else from the export is kept
P = {r["leaf_id"]: r for r in csv.DictReader(open(f"{SRC}/model_predictions.csv", encoding="utf-8"))}
KEY = {r["form_id"]: r for r in csv.DictReader(open(f"{SRC}/form_image_key.csv", encoding="utf-8"))}
TRAIN = {f for f, r in KEY.items() if r["model_split"] == "train"}
assert sorted(TRAIN) == RES["meta"]["train_leaves_excluded"]


def severity(fid, arch="segformer"):
    p = P[fid]
    if fid.startswith("roya"):
        return float(p[f"{arch}_sev_ab"])                      # rust: abaxial face
    return (float(p[f"{arch}_sev_ab"]) + float(p[f"{arch}_sev_ad"])) / 2   # leaf spot: mean of both faces


def qwk(a, b):            # matrix form (as in the four-evaluator script)
    O = np.bincount(K * np.asarray(a) + np.asarray(b), minlength=K * K).reshape(K, K).astype(float)
    Ex = np.outer(O.sum(1), O.sum(0)) / O.sum()
    W = (np.subtract.outer(np.arange(K), np.arange(K)) ** 2) / (K - 1) ** 2
    den = (W * Ex).sum()
    return np.nan if den == 0 else 1 - (W * O).sum() / den


def qwk_v(a, b):          # same statistic, vectorised along the last axis (for the bootstrap re-derivation)
    a = np.asarray(a, float); b = np.asarray(b, float); n = a.shape[-1]
    num = ((a - b) ** 2).sum(-1)
    den = n * ((a * a).sum(-1) + (b * b).sum(-1)) - 2 * a.sum(-1) * b.sum(-1)
    with np.errstate(all="ignore"):
        return np.where(den > 0, 1 - n * num / np.where(den > 0, den, 1), np.nan)


def consensus_pairs(M):
    """(evaluator row, reference) for every evaluator i and every 3-subset of the other evaluators (20 with R = 5)."""
    R = M.shape[0]; pairs = []
    for i in range(R):
        for S in itertools.combinations([j for j in range(R) if j != i], 3):
            med = np.median(M[list(S)], axis=0); assert (med == np.round(med)).all()   # tie-free by construction
            pairs.append((i, med.astype(int)))
    return pairs


DIS = [("roya", "Coffee leaf rust", "Hemileia vastatrix"), ("ojo", "American leaf spot", "Mycena citricolor")]
ARCH = ["segformer", "unetpp", "deeplab"]
RKEY = {"segformer": "pub_segformer", "unetpp": "pub_unetpp", "deeplab": "pub_deeplab"}   # results.json rater keys

N = {"checks": {}}
for d, name, _ in DIS:
    fids = [f for f in ITEMS if f.startswith(d) and f not in TRAIN]
    PR = RES["primary"][d]
    N["checks"][f"{d}_leaf_set_equals_results_json"] = fids == PR["leaves"]
    M = GR[:, [ITEMS.index(f) for f in fids]]
    G = {a: np.array([grade(severity(f, a)) for f in fids]) for a in ARCH}
    n = len(fids)
    pairs = consensus_pairs(M)
    assert len(pairs) == PR["n_reference_sets"] == 20
    evi = np.array([p[0] for p in pairs]); refs = np.array([p[1] for p in pairs])

    # point values (typical evaluator = mean over the 20 pairs; model = mean over the same 20 references)
    point = {"expert": float(np.mean([qwk(M[i], r) for i, r in pairs]))}
    for a in ARCH:
        point[a] = float(np.mean([qwk(G[a], r) for _, r in pairs]))

    # per-rater 95% intervals: verified values from results.json, re-derived with the same resample stream
    rng = np.random.default_rng(SEED); idx = np.stack([rng.integers(0, n, n) for _ in range(B)])
    boot = {"expert": qwk_v(M[evi][:, idx], refs[:, idx]).mean(0)}
    for a in ARCH:
        boot[a] = qwk_v(np.broadcast_to(G[a][idx], refs[:, idx].shape), refs[:, idx]).mean(0)
    ci_here = {k: [float(x) for x in np.percentile(v[np.isfinite(v)], [2.5, 97.5])] for k, v in boot.items()}
    ci = {"expert": PR["evaluators"]["ci95"], **{a: PR["raters"][RKEY[a]]["ci95"] for a in ARCH}}
    diff = {a: point[a] - point["expert"] for a in ARCH}
    dboot = boot["segformer"] - boot["expert"]; dboot = dboot[np.isfinite(dboot)]
    diff_ci_seg = [float(x) for x in np.percentile(dboot, [2.5, 97.5])]

    # closeness: rater grade minus consensus grade over all 20 x n comparisons
    def dist(dev):
        dev = np.abs(np.concatenate(dev))
        return [float(np.mean(dev == 0)), float(np.mean(dev == 1)), float(np.mean(dev >= 2))]
    counts = {"expert": {str(v): int((np.concatenate([M[i] - r for i, r in pairs]) == v).sum()) for v in range(-3, 4)}}
    close = {"expert": dist([M[i] - r for i, r in pairs])}
    for a in ARCH:
        close[a] = dist([G[a] - r for _, r in pairs])
        counts[a] = {str(v): int((np.concatenate([G[a] - r for _, r in pairs]) == v).sum()) for v in range(-3, 4)}

    # consistency with the verified results.json
    C = N["checks"]
    C[f"{d}_expert_value"] = abs(point["expert"] - PR["evaluators"]["value"]) < 1e-9
    C[f"{d}_expert_ci_rederived"] = np.allclose(ci_here["expert"], ci["expert"], atol=1e-9)
    C[f"{d}_expert_diff_counts"] = counts["expert"] == PR["evaluators"]["diff_counts_vs_refs"]
    for a in ARCH:
        rr = PR["raters"][RKEY[a]]
        C[f"{d}_{a}_value"] = abs(point[a] - rr["value"]) < 1e-9
        C[f"{d}_{a}_ci_rederived"] = np.allclose(ci_here[a], ci[a], atol=1e-9)
        C[f"{d}_{a}_diff_counts"] = counts[a] == rr["diff_counts_vs_refs"]
        C[f"{d}_{a}_contrast"] = abs(diff[a] - PR["contrasts"][RKEY[a]]["estimate"]) < 1e-9
    C[f"{d}_segformer_contrast_ci_rederived"] = np.allclose(diff_ci_seg, PR["contrasts"]["pub_segformer"]["ci95"], atol=1e-9)

    # per leaf + severity (panel = all five evaluators; symmetric in the evaluators, no labels)
    lo, hi = M.min(0), M.max(0)
    sev = np.array([severity(f) for f in fids])
    mean_g = M.mean(0)
    med5 = np.median(M, axis=0).astype(int)
    inside = (G["segformer"] >= lo) & (G["segformer"] <= hi)
    DS = RES["descriptives"][d]["raters"]["pub_segformer"]
    C[f"{d}_segformer_inside_range"] = int(inside.sum()) == DS["inside_panel_range"]
    C[f"{d}_median5_grade_counts"] = np.bincount(med5, minlength=K).tolist() == RES["descriptives"][d]["median5_grade_counts"]
    outside = [dict(form_id=f, panel_grades_sorted=sorted(M[:, k].tolist()), segformer_grade=int(G["segformer"][k]),
                    segformer_severity=float(sev[k])) for k, f in enumerate(fids) if not inside[k]]

    N[d] = dict(n=n, design="leave-one-evaluator-out, three-evaluator consensus (20 reference sets)",
                qwk_vs_consensus=point, qwk_ci95=ci,
                evaluator_range=PR["evaluators"]["range"],
                diff_vs_typical_evaluator=diff,
                diff_vs_typical_evaluator_ci95_results_json={a: PR["contrasts"][RKEY[a]]["ci95"] for a in ARCH},
                p_two_sided_results_json={a: PR["contrasts"][RKEY[a]]["p_two_sided"] for a in ARCH},
                bootstrap=dict(B=B, seed=SEED, type="paired leaf resampling, percentile; evaluators fixed"),
                closeness=close, grade_difference_counts=counts, n_comparisons=int(len(pairs) * n),
                segformer_in_panel_range=[int(inside.sum()), n], segformer_outside_range=outside,
                spearman_segformer_severity_vs_mean_panel_grade=float(spearmanr(mean_g, sev).correlation),
                unanimous_leaves=int(np.sum(lo == hi)), leaves_all_within_one=int(np.sum(hi - lo <= 1)))
    N[d]["_arrays"] = dict(M=M, G=G, lo=lo, hi=hi, sev=sev, mean_g=mean_g, inside=inside)
    N[d]["_lids"] = fids

bad = [k for k, v in N["checks"].items() if not v]
assert not bad, bad

# ------------------------------------------------------------------------------------------------ style
CM = 1 / 2.54
INK, INK2, TICK, AXIS = "#1b1b1b", "#565656", "#8f8f8f", "#d2d2d2"
EXPERT_C = "#8c8c8c"
RATERS = [  # key, label, colour, marker, markersize
    ("expert", "Typical evaluator", EXPERT_C, "D", 5.8),
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
    if os.path.isdir(MAN) and not os.environ.get("PANEL_NOCOPY"):
        shutil.copyfile(f"{OUT}/{name}.pdf", f"{MAN}/{name}.pdf")
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


def pct_label(w):       # w in percent; small segments get one decimal so that 0.3% is not printed as "0%"
    return f"{w:.0f}%" if w >= 8 else f"{w:.1f}%"


# layout in cm: title block above each axes ~0.9 cm, x-label block below ~0.95 cm.
# LA and RA are wider than in the four-evaluator version: longer 'Typical evaluator' label, and the small '>=2 grades
# off' label of the leaf-spot typical evaluator now sits beyond the bar end of the right-hand panel.
WA, LA, GAPA, RA, HAX = 16.0, 2.65, 1.15, 0.95, 3.5
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
    ax.axvline(D["qwk_vs_consensus"]["expert"], color=EXPERT_C, lw=0.7, ls=(0, (2, 2)), zorder=1)
    for y, (k, lab, c, mk, ms) in zip(ypos, RATERS):
        v, (l, h) = D["qwk_vs_consensus"][k], D["qwk_ci95"][k]
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
for col, (d, name, species) in enumerate(DIS):
    D = N[d]
    ax = ax_at(1, col)
    for y, (k, lab, c, mk, ms) in zip(ypos, RATERS):
        fr = D["closeness"][k]; left = 0.0
        for s, (w, sc) in enumerate(zip(fr, seg_colours(c))):
            w *= 100
            if w <= 0:
                continue
            if w >= 2:
                ax.barh(y, w, left=left, height=0.62, color=sc, edgecolor="white", linewidth=0.8, zorder=2)
            else:     # sliver (e.g. 2 of 600 comparisons): no white edge, drawn on top so that it stays visible; its
                      # height is reduced by the 0.8 pt white edge so it is exactly as tall as the edged segments
                edge_units = 0.8 / (HAX / 2.54 * 72) * (len(RATERS) + 0.2)
                ax.barh(y, w, left=left, height=0.62 - edge_units, color=sc, edgecolor="none", linewidth=0, zorder=2.5)
            if w >= 8:
                ax.text(left + w / 2, y, pct_label(w), ha="center", va="center", fontsize=8, color=text_on(sc))
            else:     # small segment (>=2 grades off): label just beyond the bar end
                ax.text(left + w + 1.5, y, pct_label(w), ha="left", va="center", fontsize=8, color=INK, clip_on=False)
            left += w
    ax.set_ylim(-0.6, len(RATERS) - 0.4)
    ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks(ypos)
    if col == 0:
        ax.set_yticklabels([r[1] for r in RATERS], color=INK, fontsize=8.5)
    else:
        ax.set_yticklabels([])
    ax.set_xlabel("Comparisons with consensus (%)")   # unit = rater grade vs one three-evaluator consensus grade (20 x n)
    panel_title(ax, next(letters), name, species)
handles = [Swatch([seg_colours(r[2])[s] for r in RATERS]) for s in range(2)] + [Swatch([BEYOND])]
fig.legend(handles, SEG_LAB, handler_map={Swatch: SwatchHandler()}, loc="lower center", ncol=3,
           bbox_to_anchor=((LA + (WA - RA)) / 2 / WA, 0.08 / HA), handlelength=2.6, handleheight=0.9,
           columnspacing=2.2, borderaxespad=0, borderpad=0)
save(fig, "fig_panel_agreement")

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
    ax.set_xlabel("Leaves, ordered by mean panel grade")
    panel_title(ax, "ab"[col], name, species)
handles = [Rectangle((0, 0), 1, 1, facecolor=RANGE_C, edgecolor="none"),
           Line2D([], [], marker="^", ls="", ms=5.6, mfc=SEG_C, mec=SEG_C, mew=1.0),
           Line2D([], [], marker="^", ls="", ms=5.6, mfc="white", mec=SEG_C, mew=1.2)]
fig.legend(handles, ["Panel grade range", "SegFormer-B5 within range", "SegFormer-B5 outside range"],
           loc="lower center", ncol=3, bbox_to_anchor=((LB + WB - RB) / 2 / WB, 0.08 / HB), handlelength=1.4,
           columnspacing=2.2, borderaxespad=0, borderpad=0)
save(fig, "fig_panel_perleaf")

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
    ax.text(0.03, 0.97, f"Spearman \u03c1 = {N[d]['spearman_segformer_severity_vs_mean_panel_grade']:.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color=INK2)
    ax.set_yscale("log"); ax.set_ylim(*YLIM); ax.minorticks_off()
    ax.set_yticks([0.01, 0.1, 0.5, 1, 5, 20, 100])
    ax.set_yticklabels(["\u22640.01", "0.1", "0.5", "1", "5", "20", "100"] if col == 0 else [])
    ax.set_xlim(*XLIM); ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("Mean panel grade")
    if col == 0:
        ax.set_ylabel("SegFormer-B5 severity (%)")
    panel_title(ax, "ab"[col], name, species)
save(fig, "fig_panel_severity")

# ------------------------------------------------------------------------------------------------ numbers
for d, _, _ in DIS:
    N[d].pop("_arrays"); N[d].pop("_lids")
    N[d]["severity_plot_dodge"] = dict(swarm[d], marker_pt=MS_C, marker_mm=round(W_MM, 3), min_clearance_required_mm=CLEAR_MM,
                                       mm_per_grade_unit=round(UX, 3), mm_per_decade=round(float(UY), 3))
N["checks"] = {k: bool(v) for k, v in N["checks"].items()}
N["definitions"] = dict(primary_set="53 form leaves not used for training (7 healthy rust training leaves excluded)",
                        rust_grade="abaxial face", leaf_spot_grade="mean of abaxial and adaxial faces",
                        cut_points="G0 <0.5%, G1 0.5-<5%, G2 5-<20%, G3 >=20%",
                        consensus="median of each 3-subset of the other four evaluators (20 evaluator-subset pairs)",
                        typical_evaluator="mean over the 20 pairs of QWK(evaluator, consensus)",
                        panel="all five evaluators (grade range, mean grade); no per-evaluator values are drawn or stored",
                        intervals="per-rater 95% percentile intervals from results/r4_panel5/results.json, re-derived here")
json.dump(N, open(f"{OUT}/panel_figure_numbers.json", "w"), indent=1)
print(json.dumps(N["checks"], indent=0))
for d, _, _ in DIS:
    D = N[d]
    print(d, "n", D["n"], {k: (round(v, 3), [round(x, 3) for x in D["qwk_ci95"][k]]) for k, v in D["qwk_vs_consensus"].items()})
    print("   closeness %", {k: [round(x * 100, 2) for x in v] for k, v in D["closeness"].items()}, "comparisons", D["n_comparisons"])
    print("   dodge", {k: v for k, v in D["severity_plot_dodge"].items() if k != "shifted"})
    print("   in range", D["segformer_in_panel_range"], "rho", round(D["spearman_segformer_severity_vs_mean_panel_grade"], 4),
          "unanimous", D["unanimous_leaves"], "within one", D["leaves_all_within_one"], "outside", D["segformer_outside_range"])
