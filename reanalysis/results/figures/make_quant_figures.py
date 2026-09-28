"""Reproduce the published leaf-level values, then rebuild paper Figure 4 (leaf-level lesion IoU with 95% bootstrap CI)
and paper Figure 12 (leaf-level lesion IoU by reference severity band). No re-inference.

Stops with an AssertionError if any published value (leaf IoU, MAE, bias, QWK, Bland-Altman LoA) is not reproduced.

Bootstrap: 10,000 resamples of leaves with replacement, percentile 95% CI, np.random.default_rng(20260927) created
fresh for each stratum (Figure 4: each disease; Figure 12: each disease x band); within a stratum the same resample
indices are used for the three models (paired), which also gives CIs for paired model differences.

Outputs (this folder): figure4.pdf/.png, figure12.pdf/.png, quant_results.json
"""
from __future__ import annotations

import json

import numpy as np
from scipy.stats import spearmanr

import common as C

C.setup_style()
import matplotlib.pyplot as plt  # noqa: E402

BANDS = [  # (key, tick label, lower inclusive, upper exclusive); aligned with the grade cut-points 0.5 / 5 / 20 %
    ("lt0.5", "<0.5", 0.0, 0.5),
    ("0.5-2", "0.5–<2", 0.5, 2.0),
    ("2-5", "2–<5", 2.0, 5.0),
    ("5-20", "5–<20", 5.0, 20.0),
    ("ge20", "≥20", 20.0, float("inf")),
]
MIN_N_CI = 3


def band_of(s: float) -> str:
    for key, _, lo, hi in BANDS:
        if lo <= s < hi:
            return key
    raise ValueError(s)


def r2(x):
    return float(f"{x:.2f}")


def r3(x):
    return float(f"{x:.3f}")


# ----------------------------------------------------------------------------------------------- compute
def compute() -> dict:
    rows = C.read_rows()
    res = {"reproduction": {}, "figure4": {}, "figure12": {}, "leaf_tables": {}}
    checks = []
    for d in C.DISEASES:
        tabs = {a: C.leaf_table(rows, d, a) for a in C.ARCHS}
        ids = [r["leaf_id"] for r in tabs["segformer"]]
        for a in C.ARCHS:
            assert [r["leaf_id"] for r in tabs[a]] == ids
            # the reference side must be identical across models (same images, same rebuild)
            assert [r["ref_sev"] for r in tabs[a]] == [r["ref_sev"] for r in tabs["segformer"]]
        n = len(ids)

        # ---------------- reproduction of the published values
        rep = {}
        for a in C.ARCHS:
            t = tabs[a]
            iou = np.array([r["iou"] for r in t])
            exact_mean = float(sum(r["iou_exact"] for r in t) / n)
            diff = np.array([r["pred_sev"] - r["ref_sev"] for r in t])
            sd = float(diff.std(ddof=1))
            bias = float(diff.mean())
            q = C.qwk([r["ref_grade"] for r in t], [r["pred_grade"] for r in t])
            assert [r["ref_grade"] for r in t] == [r["ref_grade_csv"] for r in t]
            m = dict(n_leaves=n, leaf_iou=float(iou.mean()), leaf_iou_exact_fraction_mean=exact_mean,
                     mae_pp=float(np.abs(diff).mean()), bias_pp=bias, sd_diff_pp=sd,
                     loa_pp=[bias - 1.96 * sd, bias + 1.96 * sd], qwk=q)
            pub = {k: C.PUBLISHED[k][d][a] for k in ("leaf_iou", "mae", "bias", "qwk", "loa")}
            ok = {
                "leaf_iou": r3(m["leaf_iou"]) == pub["leaf_iou"] and r3(exact_mean) == pub["leaf_iou"],
                "mae": r2(m["mae_pp"]) == pub["mae"],
                "bias": r2(m["bias_pp"]) == pub["bias"],
                "qwk": r2(q) == pub["qwk"],
                "loa": (r2(m["loa_pp"][0]), r2(m["loa_pp"][1])) == tuple(pub["loa"]),
            }
            m["published"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in pub.items()}
            m["reproduced"] = ok
            rep[a] = m
            for k, v in ok.items():
                checks.append((d, a, k, v))
        res["reproduction"][d] = rep

        # ---------------- Figure 4: leaf-level IoU with paired bootstrap CI
        idx = C.boot_indices(n)
        vals = {a: np.array([r["iou"] for r in tabs[a]]) for a in C.ARCHS}
        boot = {a: vals[a][idx].mean(1) for a in C.ARCHS}
        f4 = {}
        for a in C.ARCHS:
            lo, hi = C.percentile_ci(boot[a])
            f4[a] = dict(model=C.MODEL_NAME[a], n_leaves=n, mean=float(vals[a].mean()), label=f"{vals[a].mean():.3f}",
                         ci95=[lo, hi], ci95_3dp=[r3(lo), r3(hi)], sd=float(vals[a].std(ddof=1)),
                         median=float(np.median(vals[a])), min=float(vals[a].min()), max=float(vals[a].max()))
        pairs = [("segformer", "unetpp"), ("segformer", "deeplab"), ("unetpp", "deeplab")]
        f4["paired_differences"] = {
            f"{C.MODEL_NAME[x]} - {C.MODEL_NAME[y]}": dict(
                mean=float((vals[x] - vals[y]).mean()), ci95=list(C.percentile_ci(boot[x] - boot[y])),
                leaves_where_first_higher=int((vals[x] > vals[y]).sum()), n_leaves=n)
            for x, y in pairs}
        res["figure4"][d] = f4

        # ---------------- Figure 12: IoU by reference severity band
        bands_leaf = [band_of(r["ref_sev"]) for r in tabs["segformer"]]
        assert bands_leaf == [band_of(r["ref_sev_csv"]) for r in tabs["segformer"]], "band differs with rounded CSV"
        f12 = {}
        for key, label, lo_b, hi_b in BANDS:
            sel = np.array([b == key for b in bands_leaf])
            nb = int(sel.sum())
            entry = dict(label=label, lower_pct=lo_b, upper_pct=(None if hi_b == float("inf") else hi_b), n_leaves=nb,
                         leaf_ids=[i for i, s in zip(ids, sel) if s], models={})
            if d == "ojo":
                entry["annotator_grades"] = {g: int(sum(1 for r, s in zip(tabs["segformer"], sel) if s and r["ref_grade"] == int(g[1])))
                                             for g in ("G1", "G2", "G3")}
            else:
                entry["reference_grades"] = sorted({r["ref_grade"] for r, s in zip(tabs["segformer"], sel) if s})
            if nb:
                bidx = C.boot_indices(nb) if nb >= MIN_N_CI else None
                for a in C.ARCHS:
                    v = vals[a][sel]
                    e = dict(mean=float(v.mean()), label=f"{v.mean():.3f}", values=[float(x) for x in v])
                    if bidx is not None:
                        lo, hi = C.percentile_ci(v[bidx].mean(1))
                        e.update(ci95=[lo, hi], ci95_3dp=[r3(lo), r3(hi)])
                        e["distinct_bootstrap_means_possible"] = int(
                            __import__("math").comb(2 * nb - 1, nb))
                    else:
                        e.update(ci95=None, ci_note=f"CI omitted: n = {nb} < {MIN_N_CI}")
                    entry["models"][a] = e
            f12[key] = entry
        assert sum(e["n_leaves"] for e in f12.values()) == n
        # supplementary trend statistic (not plotted): Spearman rho between leaf reference severity and leaf IoU,
        # paired leaf bootstrap with the same indices as Figure 4
        sev = np.array([r["ref_sev"] for r in tabs["segformer"]])
        f12["spearman_severity_vs_iou"] = {}
        for a in C.ARCHS:
            rho = float(spearmanr(sev, vals[a]).statistic)
            bs = np.array([spearmanr(sev[i], vals[a][i]).statistic for i in idx])
            bs = bs[np.isfinite(bs)]
            f12["spearman_severity_vs_iou"][a] = dict(rho=rho, ci95=list(C.percentile_ci(bs)), n_leaves=n,
                                                      valid_resamples=int(bs.size))
        res["figure12"][d] = f12

        res["leaf_tables"][d] = [
            dict(leaf_id=i, ref_sev_pct=tabs["segformer"][k]["ref_sev"], band=bands_leaf[k],
                 ref_grade=tabs["segformer"][k]["ref_grade"],
                 **{f"iou_{a}": tabs[a][k]["iou"] for a in C.ARCHS},
                 **{f"pred_sev_{a}": tabs[a][k]["pred_sev"] for a in C.ARCHS})
            for k, i in enumerate(ids)]

    failed = [c for c in checks if not c[3]]
    res["reproduction"]["all_reproduced"] = not failed
    res["reproduction"]["failed_checks"] = [list(c[:3]) for c in failed]
    if failed:
        print(json.dumps(res["reproduction"], indent=1, default=str))
        raise AssertionError(f"published values NOT reproduced: {failed}")
    # Figure 4 labels must equal the published leaf-level means
    for d in C.DISEASES:
        for a in C.ARCHS:
            assert res["figure4"][d][a]["label"] == f"{C.PUBLISHED['leaf_iou'][d][a]:.3f}", (d, a)
    return res


# ----------------------------------------------------------------------------------------------- Figure 4
def draw_figure4(res: dict):
    fig, axes = plt.subplots(1, 2, figsize=(16 * C.CM, 6.4 * C.CM))
    fig.subplots_adjust(left=0.155, right=0.975, bottom=0.175, top=0.83, wspace=0.5)
    ypos = {a: len(C.ARCHS) - 1 - i for i, a in enumerate(C.ARCHS)}
    xspec = {"roya": dict(lim=(0.535, 0.905), ticks=np.arange(0.55, 0.851, 0.05), spine=(0.535, 0.865)),
             "ojo": dict(lim=(0.735, 1.005), ticks=np.arange(0.75, 0.951, 0.05), spine=(0.735, 0.965))}
    drawn = {}
    notes = []
    for ax, d in zip(axes, C.DISEASES):
        f4 = res["figure4"][d]
        drawn[d] = {}
        for a in C.ARCHS:
            st, y = C.STYLE[a], ypos[a]
            lo, hi = f4[a]["ci95"]
            m = f4[a]["mean"]
            ax.plot([lo, hi], [y, y], color=st["color"], lw=2.3, solid_capstyle="butt", zorder=2)
            ax.plot(m, y, marker=st["marker"], ms=st["ms"], color=st["color"], mec="white", mew=1.1, ls="none", zorder=3)
            ax.annotate(f4[a]["label"], xy=(hi, y), xytext=(6, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=7.6, color=C.GREY_TXT)
            drawn[d][a] = f4[a]["label"]
        ax.set_yticks([ypos[a] for a in C.ARCHS], [C.MODEL_NAME[a] for a in C.ARCHS], fontsize=8.6)
        ax.tick_params(axis="y", labelcolor=C.INK, pad=3)
        ax.set_ylim(-0.6, len(C.ARCHS) - 1 + 0.75)
        xs = xspec[d]
        ax.set_xlim(*xs["lim"])
        ax.set_xticks(xs["ticks"], [f"{t:.2f}" for t in xs["ticks"]])
        ax.tick_params(axis="x", labelsize=8.0, pad=2.5)
        ax.set_xlabel("Leaf-level lesion IoU (95% CI)", fontsize=8.4, color=C.GREY_TXT, labelpad=3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_bounds(-0.6, len(C.ARCHS) - 1 + 0.75)
        ax.spines["bottom"].set_bounds(*xs["spine"])
        n = res["figure4"][d]["segformer"]["n_leaves"]
        note = f"n = {n} test leaves" + (" (abaxial face)" if d == "roya" else " (mean of both faces)")
        # right-aligned to the visible right end of the x-axis (the spine end), not to the axes box, which is wider
        t = ax.annotate(note, xy=(xs["spine"][1], 0), xycoords=ax.get_xaxis_transform(), xytext=(0, 4),
                        textcoords="offset points", ha="right", va="bottom", fontsize=6.6, color=C.GREY_TICK)
        notes.append((d, ax, t, xs["spine"][1]))
        C.panel_title(ax, d)
    # rendered check: each n-annotation ends exactly at (not beyond) the right end of its x-axis
    renderer = fig.canvas.get_renderer()
    checks = {}
    for d, ax, t, x_end in notes:
        x_end_px = ax.transData.transform((x_end, 0))[0]
        right_px = t.get_window_extent(renderer).x1
        assert abs(right_px - x_end_px) <= 0.5, (t.get_text(), right_px, x_end_px)
        checks[d] = dict(text=t.get_text(), right_edge_minus_axis_end_px_at_100dpi=round(float(right_px - x_end_px), 3))
    return fig, drawn, checks


# ----------------------------------------------------------------------------------------------- Figure 12
def draw_figure12(res: dict):
    fig, axes = plt.subplots(1, 2, figsize=(16 * C.CM, 7.2 * C.CM), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.2, top=0.85, wspace=0.12)
    dodge = {"segformer": -0.22, "unetpp": 0.0, "deeplab": 0.22}
    ylo, yhi = 0.40, 1.0
    for ax, d in zip(axes, C.DISEASES):
        f12 = res["figure12"][d]
        for i, (key, label, _, _) in enumerate(BANDS):
            e = f12[key]
            if e["n_leaves"] == 0:
                ax.text(i, ylo + 0.015, "no leaves", ha="center", va="bottom", fontsize=6.6, color=C.GREY_TICK)
                continue
            for a in C.ARCHS:
                st, m = C.STYLE[a], e["models"][a]
                x = i + dodge[a]
                if m["ci95"] is not None:
                    ax.plot([x, x], m["ci95"], color=st["color"], lw=1.6, solid_capstyle="butt", zorder=2)
                    hollow = False
                else:
                    hollow = True
                ax.plot(x, m["mean"], marker=st["marker"], ms=st["ms"] * 0.8, ls="none", zorder=3,
                        color=st["color"], mec=st["color"], mew=(1.15 if hollow else 0.0),
                        mfc=("white" if hollow else st["color"]))
        ax.set_xticks(range(len(BANDS)),
                      [f"{lab}\nn = {f12[k]['n_leaves']}" for k, lab, _, _ in BANDS])
        ax.tick_params(axis="x", labelsize=7.8, pad=2.5)
        ax.set_xlim(-0.55, len(BANDS) - 0.45)
        ax.set_ylim(ylo, yhi)
        ax.set_yticks(np.arange(0.4, 1.001, 0.1), [f"{t:.1f}" for t in np.arange(0.4, 1.001, 0.1)])
        ax.tick_params(axis="y", labelsize=8.0)
        ax.set_xlabel("Reference lesion severity band (%)", fontsize=8.4, color=C.GREY_TXT, labelpad=3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_bounds(-0.55, len(BANDS) - 0.45)
        C.panel_title(ax, d)
    axes[0].set_ylabel("Leaf-level lesion IoU (95% CI)", fontsize=8.4, color=C.GREY_TXT, labelpad=4)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=C.STYLE[a]["color"], marker=C.STYLE[a]["marker"], ms=C.STYLE[a]["ms"] * 0.8,
                      mec=C.STYLE[a]["color"], mew=0.0, ls="none") for a in C.ARCHS]
    labels = [C.MODEL_NAME[a] for a in C.ARCHS]
    handles.append(Line2D([], [], color=C.GREY_TICK, marker="o", ms=5.4, mfc="white", mec=C.GREY_TICK, mew=1.2,
                          ls="none"))
    labels.append(f"open symbol: n < {MIN_N_CI}, CI omitted")
    axes[1].legend(handles, labels, loc="lower right", frameon=False, fontsize=7.4, handlelength=1.0,
                   handletextpad=0.5, borderaxespad=0.2, labelcolor=C.INK)
    return fig


def main():
    res = compute()
    fig, drawn, checks = draw_figure4(res)
    res["figure4"]["files"] = C.save(fig, "figure4")
    res["figure4"]["printed_labels"] = drawn
    res["figure4"]["n_annotation_alignment"] = checks
    plt.close(fig)
    fig = draw_figure12(res)
    res["figure12"]["files"] = C.save(fig, "figure12")
    plt.close(fig)
    res["figure12"]["bands"] = [dict(key=k, label=l, lower_pct=lo, upper_pct=(None if hi == float("inf") else hi))
                                for k, l, lo, hi in BANDS]
    res["figure12"]["min_n_for_ci"] = MIN_N_CI
    (C.OUT / "quant_results.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")

    for d in C.DISEASES:
        for a in C.ARCHS:
            m = res["reproduction"][d][a]
            print(f"{d:4s} {a:9s} IoU {m['leaf_iou']:.4f} MAE {m['mae_pp']:.3f} bias {m['bias_pp']:+.3f} "
                  f"QWK {m['qwk']:.3f} LoA {m['loa_pp'][0]:+.2f} {m['loa_pp'][1]:+.2f}")
    print("all reproduced:", res["reproduction"]["all_reproduced"])
    for d in C.DISEASES:
        for a in C.ARCHS:
            f = res["figure4"][d][a]
            print(f"Fig4 {d:4s} {a:9s} {f['label']} [{f['ci95'][0]:.3f}, {f['ci95'][1]:.3f}]")
        for k, v in res["figure4"][d]["paired_differences"].items():
            print(f"   {k}: {v['mean']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
    for d in C.DISEASES:
        for a, v in res["figure12"][d]["spearman_severity_vs_iou"].items():
            print(f"Spearman {d} {a}: {v['rho']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
        for key, e in res["figure12"][d].items():
            if key == "spearman_severity_vs_iou":
                continue
            s = f"Fig12 {d:4s} {e['label']:7s} n={e['n_leaves']:2d}"
            for a, m in e["models"].items():
                ci = "" if m["ci95"] is None else f" [{m['ci95'][0]:.3f},{m['ci95'][1]:.3f}]"
                s += f" | {a} {m['mean']:.3f}{ci}"
            print(s)


if __name__ == "__main__":
    main()
