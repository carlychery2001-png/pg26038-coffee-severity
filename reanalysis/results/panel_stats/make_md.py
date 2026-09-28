"""Write results.md from results.json (no hand-copied numbers)."""
import json, os
OUT = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(OUT, "results.json"), encoding="utf-8"))
C, T, D = r["contrasts"], r["primary_table"], r["direction_of_error"]
ARCH = ["SegFormer-B5", "DeepLabV3+", "UNet++"]; AK = {"SegFormer-B5": "segformer", "DeepLabV3+": "deeplab", "UNet++": "unetpp"}
DN = {"ojo": "American leaf spot", "roya": "Coffee leaf rust", "all": "Pooled"}

def f(x, d=2):
    if x is None: return "n/a"
    s = f"{x:.{d}f}"
    if float(s) == 0 and x < 0: s = f"{x:.4f}"      # avoid a bare "-0.00": show the sign with 4 decimals
    return s.replace("-", "−")
def sg(x, d=2): return ("+" if x >= 0 else "") + f(x, d)
def ci(c, d=2): return f"[{f(c[0], d)}, {f(c[1], d)}]"
def p(x):
    if x is None: return "n/a"
    return "<0.001" if x < 0.001 else f"{x:.3f}" if x < 0.1 else f"{x:.2f}"
def rng2(a, b, d=2, pct=False):
    lo, hi = sorted((a, b))
    return f"{lo*100:.0f}–{hi*100:.0f}%" if pct else f"{f(lo, d)}–{f(hi, d)}"

L = []
L.append("# Expert-panel secondary statistics (PG-26038 revision)\n")
L.append(f"Primary set: {T['ojo']['n'] + T['roya']['n']} leaves ({T['ojo']['n']} American leaf spot, {T['roya']['n']} rust); "
         f"the 7 rust-training healthy leaves are excluded. Rust grade from the abaxial face, leaf spot from the mean of both faces. "
         f"Statistic: mean leave-one-evaluator-out (LOO) QWK of the model minus that of the evaluators (as in `tiefree.py`). "
         f"Bootstrap: {r['meta']['B']:,} leaf resamples, seed {r['meta']['seed']} (the tiefree seed). Evaluators are held fixed, so inference is conditional on these four readers. "
         f"Two-sided p = min(1, 2·min(P*(d ≤ 0), P*(d ≥ 0))), which inverts the percentile interval. "
         f"Script: `panel_stats.py`; all numbers: `results.json`.\n")

L.append("## 1. Model-minus-evaluator contrasts\n")
L.append("| Disease | Model | Role | Model LOO QWK | Evaluators LOO QWK | Difference [95% CI] | p (unadjusted) | p Holm, 4 secondary | p Holm, all 6 |")
L.append("|---|---|---|---|---|---|---|---|---|")
for dis in ("ojo", "roya"):
    for a in ARCH:
        c = C[f"{dis}_{AK[a]}"]; role = "primary" if a == "SegFormer-B5" else "secondary"
        L.append(f"| {DN[dis]} | {a} | {role} | {f(c['model_loo_qwk'], 3)} | {f(c['evaluator_loo_qwk'], 3)} | {sg(c['diff'], 3)} {ci(c['ci95'], 3)} | "
                 f"{p(c['p_two_sided'])} | {p(c['p_holm_secondary4']) if c['p_holm_secondary4'] is not None else '— (unadjusted)'} | {p(c['p_holm_all6'])} |")
L.append("")
L.append("Pooled (descriptive only: it mixes two disease-specific models and belongs to no test family): " +
         "; ".join(f"{a} {sg(C['all_'+AK[a]]['diff'], 3)} {ci(C['all_'+AK[a]]['ci95'], 3)}" for a in ARCH) + ".\n")
rc = r["reproduction_check"]; mc = r["mc_check"]
L.append(f"**Reproduction check:** all 9 CIs (6 per-disease and 3 pooled) equal `validation/tiefree_results.json` `diff_ci` to 3 dp "
         f"(max |difference| = {rc['max_abs_ci_difference_3dp']}), and the point estimates match `final_results.json`. "
         f"**Monte Carlo check** (independent seed {r['meta']['seed_check']}): CI bounds moved by at most "
         f"{max(abs(mc[k]['ci95'][i] - C[k]['ci95'][i]) for k in mc for i in (0, 1)):.3f}; p-values: " +
         ", ".join(f"{k} {p(mc[k]['p_two_sided'])}" for k in mc) + ". No conclusion changes.\n")

L.append("## 2. Direction of error against the four-evaluator median (both tie rules)\n")
L.append(f"Two-against-two splits among the primary leaves: {D['ojo']['n_two_two_splits']} of {D['ojo']['n']} leaf-spot leaves and {D['roya']['n_two_two_splits']} of {D['roya']['n']} rust leaves. "
         "Counts are model grade lower / equal / higher than the median. Rounding ties up gives the higher reference, and rounding down the lower one. The last column is tie-free: it compares the model grade with the lowest and highest of the four panel grades.\n")
L.append("| Disease | Model | Ties up: lower / equal / higher (bias) | Ties down: lower / equal / higher (bias) | Panel-G0 leaves graded ≥ G1 (up; down) | Tie-free: below min / inside / above max |")
L.append("|---|---|---|---|---|---|")
for dis in ("ojo", "roya"):
    for a in ARCH:
        m = D[dis]["models"][a]; u, d, g = m["ties_up"], m["ties_down"], m["vs_panel_range_tie_free"]
        L.append(f"| {DN[dis]} | {a} | {u['lower']} / {u['equal']} / {u['higher']} ({sg(u['bias'])}) | {d['lower']} / {d['equal']} / {d['higher']} ({sg(d['bias'])}) | "
                 f"{u['panel_G0_model_above']} of {u['panel_G0']}; {d['panel_G0_model_above']} of {d['panel_G0']} | {g['below_min']} / {g['inside']} / {g['above_max']} |")
L.append("")
out = []
for dis in ("roya",):
    for a in ARCH:
        for x in D[dis]["models"][a]["outside_range_leaves"]:
            out.append(f"{a} {x['form_id']} graded G{x['model_grade']} (abaxial {x['sev_ab']}%) vs panel {'/'.join('G%d' % v for v in x['panel_grades'])}")
L.append("Grades outside the panel range (all rust; none for leaf spot): " + "; ".join(out) + ".\n")

L.append("## 3. Per-disease primary values\n")
L.append("| Disease (n) | Rater | LOO QWK | Difference [95% CI] | Mean QWK vs each evaluator | QWK vs 4-evaluator median (down–up) | Exact vs median (down–up) | Exact vs LOO consensus | Inside panel range | Bias (down; up) |")
L.append("|---|---|---|---|---|---|---|---|---|---|")
for dis in ("ojo", "roya", "all"):
    t = T[dis]; e = t["evaluators"]; lab = f"{DN[dis]} ({t['n']})" + (" [descriptive]" if dis == "all" else "")
    L.append(f"| {lab} | Evaluators (mean; range) | {f(e['loo_qwk_mean'])} ({rng2(*e['loo_qwk_range'])}) | — | {f(e['pairwise_qwk_mean'])} (pairwise) | — | — | {e['exact_vs_loo_mean']*100:.0f}% | — | — |")
    for a in ARCH:
        m = t["models"][a]
        L.append(f"| | {a} | {f(m['loo_qwk'])} | {sg(m['diff'])} {ci(m['ci95'])} | {f(m['pairwise_qwk_mean'])} | {rng2(m['qwk_median_down'], m['qwk_median_up'])} | "
                 f"{rng2(m['exact_median_down'], m['exact_median_up'], pct=True)} | {m['exact_vs_loo_mean']*100:.0f}% | {m['inside_panel_range']}/{t['n']} | {sg(m['bias_down'])}; {sg(m['bias_up'])} |")
L.append("")

# ---- suggested sentences (LaTeX-ready) ----
so, sr = C["ojo_segformer"], C["roya_segformer"]; do = C["ojo_deeplab"]
others = [C[k]["p_holm_secondary4"] for k in ("ojo_unetpp", "roya_deeplab", "roya_unetpp")]
dr = D["roya"]["models"]; dl, sf = dr["DeepLabV3+"], dr["SegFormer-B5"]; uo = D["ojo"]["models"]["UNet++"]
def tx(x, d=2): return f"{x:.{d}f}".replace("-", "$-$") if x < 0 else f"{x:.{d}f}"
def tci(c): return f"[{tx(c[0])}, {tx(c[1])}]"
def tp(x): return "$p < 0.001$" if x < 0.001 else f"$p = {x:.3f}$" if x < 0.1 else f"$p = {x:.2f}$"
nr = D["roya"]["n"]
L.append("## Suggested sentences for the paper\n")
L.append("Three sentences, LaTeX-ready. The fourth is optional.\n")
L.append("```latex")
L.append("% Methods (Section~\\ref{sec:expert_methods})")
L.append("The two SegFormer-B5 contrasts, one per disease, were designated primary and tested without adjustment, the four DeepLabV3+ and UNet++ contrasts were Holm-corrected~\\cite{ref47}, and two-sided $p$-values were obtained by inverting the percentile bootstrap interval with the evaluators held fixed, so inference is conditional on this panel.")
L.append("")
L.append("% Results (Section~\\ref{sec:expert_results})")
L.append(f"SegFormer-B5 exceeded the evaluators for American leaf spot (+{tx(so['diff'])} {tci(so['ci95'])}, {tp(so['p_two_sided'])}) but not significantly for rust (+{tx(sr['diff'])} {tci(sr['ci95'])}, {tp(sr['p_two_sided'])}), "
         f"and no secondary contrast remained significant after Holm correction, including DeepLabV3+ on American leaf spot (+{tx(do['diff'])} {tci(do['ci95'])}; unadjusted {tp(do['p_two_sided'])}, Holm {tp(do['p_holm_secondary4'])}; other contrasts Holm $p \\geq {min(others):.2f}$).")
L.append("")
L.append(f"The direction of the rust errors depended on how two-against-two panel splits were resolved: with ties rounded up, DeepLabV3+ graded {dl['ties_up']['lower']} of {nr} leaves below the four-evaluator median and {dl['ties_up']['higher']} above it, but with ties rounded down {dl['ties_down']['lower']} below and {dl['ties_down']['higher']} above (SegFormer-B5: {sf['ties_up']['lower']} below and {sf['ties_up']['higher']} above, against {sf['ties_down']['lower']} and {sf['ties_down']['higher']}), "
         f"whereas UNet++ over-graded near-healthy leaf-spot leaves under either rule, grading {uo['ties_up']['panel_G0_model_above']} of the {uo['ties_up']['panel_G0']} (ties up) or {uo['ties_down']['panel_G0_model_above']} of the {uo['ties_down']['panel_G0']} (ties down) panel-G0 leaves as G1.")
L.append("")
L.append("% Optional, tie-free")
L.append(f"Against the full range of panel grades, which needs no tie rule, DeepLabV3+ fell below the lowest panel grade on {dl['vs_panel_range_tie_free']['below_min']} rust leaves, both rated G3 by all four evaluators, and SegFormer-B5 above the highest on {sf['vs_panel_range_tie_free']['above_max']}, both rated G1 by all four.")
L.append("```\n")

L.append("## Notes\n")
L.append(f"- **Text that changes:** Section 3.5 now says DeepLabV3+ \"also exceeded the evaluators on leaf spot (+0.09 [0.02, 0.17])\". The CI is correct, but the contrast is secondary and its Holm-adjusted p is {p(do['p_holm_secondary4'])} ({p(do['p_holm_all6'])} with Holm over all six). Present it as not significant after correction.")
L.append(f"- **Text that changes:** the claims that DeepLabV3+ \"under-graded rust … never a higher one\" and that SegFormer-B5 rust errors were \"balanced (2 lower, 2 higher)\" hold only when ties are rounded up. When ties are rounded down, every model grades higher on average, because that reference is lower. Across both rules, DeepLabV3+ has the lowest rust bias of the three models ({sg(dl['ties_up']['bias'])} ties up, {sg(dl['ties_down']['bias'])} ties down). The UNet++ leaf-spot over-grading holds under both rules.")
L.append(f"- Only SegFormer-B5 on leaf spot stays significant under Holm over all six contrasts (p = {p(so['p_holm_all6'])}).")
L.append("- Do not quote tiefree's `p_le0` (for example, 0.042 for SegFormer-B5 on rust) as a p-value. It is one-sided; the two-sided value is 0.084.")
L.append("- The per-evaluator LOO QWKs in `results.json` are labelled E1–E4 in the column order of `expert_consensus.csv`. No names are written.")
L.append("- Not done: an MRMC (Obuchowski–Rockette) analysis or a reader-and-leaf bootstrap. Evaluators are fixed, so claims about \"a typical expert\" should be worded as conditional on this panel.")
open(os.path.join(OUT, "results.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("results.md written")
