"""Rebuild the specificity figure (paper Figure 11; manuscript file figures/figure7.png).

Input: the PUBLISHED false-positive counts on the 54 healthy test leaves per disease
(a false positive = any predicted lesion pixel on a healthy leaf). No re-inference is
done here; only 21 of the 54 healthy rust-test leaves exist locally, so the published
counts are the only source that covers all 54.

Outputs (this folder): figure11.pdf (vector, TrueType fonts), figure11.png (600 dpi),
results.json (all intervals).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import beta, binomtest, norm

OUT = Path(__file__).resolve().parent

N_HEALTHY = 54
CONF = 0.95
Z = float(norm.ppf(1 - (1 - CONF) / 2))  # 1.959964...

# Published counts (manuscript Results, specificity paragraph).
COUNTS = {
    "rust": {"SegFormer-B5": 0, "UNet++": 1, "DeepLabV3+": 0},
    "leaf_spot": {"SegFormer-B5": 0, "UNet++": 0, "DeepLabV3+": 1},
}
PANELS = [
    ("rust", "(a) Coffee leaf rust", "Hemileia vastatrix"),
    ("leaf_spot", "(b) American leaf spot", "Mycena citricolor"),
]
MODELS = ["SegFormer-B5", "UNet++", "DeepLabV3+"]  # top to bottom
STYLE = {
    "SegFormer-B5": dict(color="#0072B2", marker="^", ms=7.0),
    "UNet++": dict(color="#009E73", marker="s", ms=6.2),
    "DeepLabV3+": dict(color="#E69F00", marker="o", ms=6.6),
}

INK = "#1a1a1a"      # titles, model names
GREY_TXT = "#555555"  # species names, axis label, point labels
GREY_TICK = "#8a8a8a"  # x tick labels
GREY_AXIS = "#c8c8c8"  # spines and ticks
RULE3 = 3.0 / N_HEALTHY


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
    """Wilson score interval, closed form (no continuity correction)."""
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo, hi = centre - half, centre + half
    # k = 0 gives lo = 0 analytically; clamp only floating-point noise.
    return max(0.0, lo), min(1.0, hi)


def clopper_pearson(k: int, n: int, conf: float = CONF) -> tuple[float, float]:
    a = 1 - conf
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def compute() -> dict:
    rows = []
    for disease, _, species in PANELS:
        for model in MODELS:
            k = COUNTS[disease][model]
            lo, hi = wilson(k, N_HEALTHY)
            # Independent check against scipy's implementation.
            ci = binomtest(k, N_HEALTHY).proportion_ci(CONF, method="wilson")
            assert abs(ci.low - lo) < 1e-12 and abs(ci.high - hi) < 1e-12, (disease, model)
            cp_lo, cp_hi = clopper_pearson(k, N_HEALTHY)
            rows.append(
                {
                    "disease": disease,
                    "species": species,
                    "model": model,
                    "false_positives": k,
                    "n_healthy": N_HEALTHY,
                    "fp_rate": k / N_HEALTHY,
                    "fp_rate_pct_1dp": round(100 * k / N_HEALTHY, 1),
                    "specificity": 1 - k / N_HEALTHY,
                    "wilson95_fp_rate": [lo, hi],
                    "wilson95_fp_rate_pct_1dp": [round(100 * lo, 1), round(100 * hi, 1)],
                    "wilson95_specificity": [1 - hi, 1 - lo],
                    "wilson95_specificity_pct_1dp": [round(100 * (1 - hi), 1), round(100 * (1 - lo), 1)],
                    "clopper_pearson95_fp_rate_pct_1dp_crosscheck": [round(100 * cp_lo, 1), round(100 * cp_hi, 1)],
                }
            )
    return {
        "source": "Published counts from the manuscript (54 healthy test leaves per disease); no re-inference.",
        "false_positive_definition": "healthy leaf with any predicted lesion pixel",
        "interval_method": "Wilson score interval, two-sided 95%, no continuity correction",
        "z": Z,
        "rule_of_three_upper_bound": {
            "value": RULE3,
            "pct_1dp": round(100 * RULE3, 1),
            "applies_to": "models with 0/54 false positives (approximate one-sided 95% upper bound, 3/n)",
            "vs_wilson_note": (
                "For 0/54 the two-sided 95% Wilson upper limit is 6.6%, which is effectively a one-sided "
                "97.5% bound; the rule of three (5.6%) is an approximate one-sided 95% bound, so it is lower. "
                "The dotted line in the figure applies only to the 0/54 models, not to the 1/54 rows it crosses."
            ),
        },
        "drawing_note": "Interval bars are drawn with butt line caps so their ends coincide exactly with the Wilson bounds.",
        "coverage_note": (
            "Only 21 of the 54 healthy rust-test leaves are present on the local computer; "
            "this figure uses the published counts for all 54 and was not recomputed locally."
        ),
        "clopper_pearson_note": "Exact (Clopper-Pearson) intervals are included only as a cross-check; the figure shows Wilson intervals.",
        "results": rows,
    }


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.edgecolor": GREY_AXIS,
            "axes.linewidth": 0.9,
            "xtick.color": GREY_AXIS,
            "ytick.color": GREY_AXIS,
            "xtick.labelcolor": GREY_TICK,
            "ytick.labelcolor": INK,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def draw(res: dict) -> plt.Figure:
    setup_style()
    cm = 1 / 2.54
    fig, axes = plt.subplots(1, 2, figsize=(16 * cm, 6.4 * cm))
    fig.subplots_adjust(left=0.155, right=0.975, bottom=0.175, top=0.83, wspace=0.5)

    by_key = {(r["disease"], r["model"]): r for r in res["results"]}
    ypos = {m: len(MODELS) - 1 - i for i, m in enumerate(MODELS)}  # SegFormer on top
    xmax = 13.6  # room for the right-hand point labels inside the figure

    for ax, (disease, title, species) in zip(axes, PANELS):
        # Rule-of-three bound (behind the data).
        ax.axvline(100 * RULE3, color="#7a7a7a", lw=1.0, ls=(0, (1, 2)), zorder=1)
        ax.annotate(
            f"rule of three (0/54 only)\n3/54 = {pct(RULE3)}, one-sided",
            xy=(100 * RULE3, 1.0), xycoords=("data", "axes fraction"),
            xytext=(3, -2), textcoords="offset points",
            ha="left", va="top", fontsize=6.3, color="#7a7a7a", linespacing=1.1,
        )

        for m in MODELS:
            r = by_key[(disease, m)]
            st = STYLE[m]
            y = ypos[m]
            lo, hi = (100 * v for v in r["wilson95_fp_rate"])
            x = 100 * r["fp_rate"]
            # Butt caps: the drawn bar ends exactly at the Wilson bounds (round or
            # projecting caps would extend it by lw/2 = 1.15 pt, ~0.11 pp, at each end).
            ax.plot([lo, hi], [y, y], color=st["color"], lw=2.3, solid_capstyle="butt", zorder=2)
            ax.plot(
                x, y, marker=st["marker"], ms=st["ms"], color=st["color"],
                mec="white", mew=1.1, ls="none", zorder=3, clip_on=False,
            )
            ax.annotate(
                f"{pct(r['fp_rate'])} ({r['false_positives']}/{r['n_healthy']})",
                xy=(hi, y), xytext=(6, 0), textcoords="offset points",
                ha="left", va="center", fontsize=7.2, color=GREY_TXT,
            )

        ax.set_yticks([ypos[m] for m in MODELS], MODELS, fontsize=8.6)
        ax.set_ylim(-0.6, len(MODELS) - 1 + 0.75)
        ax.set_xlim(-0.7, xmax)
        ax.set_xticks([0, 2, 4, 6, 8, 10])
        ax.tick_params(axis="x", labelsize=8.0, pad=2.5)
        ax.tick_params(axis="y", pad=3)
        ax.set_xlabel("False-positive rate (%, 95% Wilson CI)", fontsize=8.4, color=GREY_TXT, labelpad=3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_bounds(-0.6, len(MODELS) - 1 + 0.75)
        # Bottom spine stops just past the last tick (same 0.7 pad as on the left, as
        # in figure5); the extra x room up to xmax is only for the point labels.
        ax.spines["bottom"].set_bounds(-0.7, 10.7)

        # Panel title (bold) with the species name in grey italic beneath, left-aligned
        # with the y axis, as in the paper's other two-panel figures.
        ax.annotate(title, xy=(0, 1), xycoords="axes fraction", xytext=(0, 17),
                    textcoords="offset points", ha="left", va="bottom",
                    fontsize=10.2, fontweight="bold", color=INK)
        ax.annotate(species, xy=(0, 1), xycoords="axes fraction", xytext=(0, 5),
                    textcoords="offset points", ha="left", va="bottom",
                    fontsize=8.4, fontstyle="italic", color=GREY_TXT)
    return fig


def main() -> None:
    res = compute()
    (OUT / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    fig = draw(res)
    fig.savefig(OUT / "figure11.pdf")
    fig.savefig(OUT / "figure11.png", dpi=600)
    plt.close(fig)
    # Flatten to opaque RGB (Agg writes RGBA) so the PNG has a true white background.
    from PIL import Image
    with Image.open(OUT / "figure11.png") as im:
        rgb = Image.new("RGB", im.size, "white")
        rgb.paste(im, mask=im.convert("RGBA").getchannel("A"))
    rgb.save(OUT / "figure11.png", dpi=(600, 600))
    for r in res["results"]:
        lo, hi = r["wilson95_fp_rate_pct_1dp"]
        print(f"{r['disease']:9s} {r['model']:13s} {r['false_positives']}/{r['n_healthy']} "
              f"= {r['fp_rate_pct_1dp']:.1f}%  Wilson95 {lo:.1f}-{hi:.1f}%")
    print(f"rule of three: {100 * RULE3:.1f}%")


if __name__ == "__main__":
    main()
