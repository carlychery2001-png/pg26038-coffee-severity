"""Secondary statistics for the expert-panel comparison (PG-26038 major revision, reviewer requests).

Reads only from <ROOT>/validation and writes only to the folder that holds this script.

Definitions are those of validation/final_analysis.py and validation/tiefree.py:
  * qwk(), model_grade() and the data tables come from final_analysis.py, which is executed up to
    'res = {}' exactly as tiefree.py does, so the definitions are identical.
  * Rust model grade = abaxial face; American leaf spot = mean of both faces.
  * Primary set = 53 leaves: all 60 form leaves minus the 7 whose form_image_key.csv model_split == 'train'.
  * Leave-one-evaluator-out (LOO) statistic: for each evaluator j, reference = median of the other three
    (always an integer). Model value = mean_j QWK(model, ref_j); evaluator value = mean_j QWK(evaluator j, ref_j).
    Contrast = model value minus evaluator value.

Bootstrap: 10,000 resamples of leaves with replacement, numpy default_rng(seed=7) re-seeded for each
disease, the same scheme as tiefree.py (tiefree re-seeds per model, but n is the same within a disease, so
all three models see the same resamples there too). Evaluators are held fixed: inference is conditional on
these four readers. Resamples whose statistic is undefined (NaN) are dropped, as in tiefree.py.

Two-sided p-value = inversion of the percentile interval: p = min(1, 2 * min(P*(d <= 0), P*(d >= 0))).
The add-one version (1 + count)/(B + 1) is reported as a conservative check.
"""
import os as _os  # (release) all paths below are relative to ROOT (repository root by default; see README)
ROOT = (_os.environ.get("PG26038_ROOT") or _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))).replace(chr(92), "/")
import csv, itertools, json, os, platform, sys, time
import numpy as np

VAL = (ROOT + "/validation")
OUT = os.path.dirname(os.path.abspath(__file__))
B = 10000
SEED = 7                  # tiefree.py seed: reproduces the published diff_ci
SEED_CHECK = 20260927     # independent seed: Monte Carlo stability check
DIS = ("ojo", "roya")
DIS_NAME = {"ojo": "American leaf spot", "roya": "Coffee leaf rust", "all": "Pooled"}
ARCH_NAME = {"segformer": "SegFormer-B5", "deeplab": "DeepLabV3+", "unetpp": "UNet++"}

# ---- helper definitions, verbatim from final_analysis.py ----
_cwd = os.getcwd()
os.chdir(VAL)
try:
    ns = {}
    exec(open("final_analysis.py", encoding="utf-8").read().split("res = {}")[0], ns)
finally:
    os.chdir(_cwd)
qwk, model_grade, E, KEY, EXP, ARCH, P = (ns[k] for k in ("qwk", "model_grade", "E", "KEY", "EXP", "ARCH", "P"))
assert len(EXP) == 4 and ARCH == ["segformer", "deeplab", "unetpp"]
EVAL_LABEL = [f"E{i + 1}" for i in range(4)]          # column order of expert_consensus.csv; names not written out
TRAIN = sorted(f for f, r in KEY.items() if r["model_split"] == "train")
assert len(TRAIN) == 7

def subset(dis):
    rows = [r for r in E if (dis == "all" or r["leaf_id"].startswith(dis)) and r["leaf_id"] not in TRAIN]
    M = np.array([[int(r[e]) for r in rows] for e in EXP])                       # 4 x n
    G = {a: np.array([model_grade(r["leaf_id"], a) for r in rows]) for a in ARCH}  # abaxial rule for rust
    return rows, M, G

def loo_refs(M):
    return [np.median(np.delete(M, j, 0), axis=0).astype(int) for j in range(M.shape[0])]  # median of 3 = integer

def loo_stats(M, G):
    refs = loo_refs(M)
    fe = float(np.mean([qwk(M[j], refs[j]) for j in range(4)]))
    fm = {a: float(np.mean([qwk(G[a], refs[j]) for j in range(4)])) for a in G}
    return fm, fe, refs

def bootstrap(M, G, seed):
    rng = np.random.default_rng(seed); n = M.shape[1]; d = {a: [] for a in G}
    with np.errstate(all="ignore"):
        for _ in range(B):
            idx = rng.integers(0, n, n)          # one call per resample, as in tiefree.py
            fm, fe, _r = loo_stats(M[:, idx], {a: G[a][idx] for a in G})
            for a in G: d[a].append(fm[a] - fe)
    out = {}
    for a in G:
        x = np.array(d[a]); x = x[np.isfinite(x)]; nv = len(x)
        le, ge = int((x <= 0).sum()), int((x >= 0).sum())
        out[a] = dict(n_valid=nv, ci=[float(v) for v in np.percentile(x, [2.5, 97.5])],
                      p_le0=le / nv, p_ge0=ge / nv,
                      p_two_sided=min(1.0, 2 * min(le, ge) / nv),
                      p_two_sided_add1=min(1.0, 2 * min(le + 1, ge + 1) / (nv + 1)),
                      boot_mean=float(x.mean()), boot_sd=float(x.std(ddof=1)))
    return out

def holm(p):
    items = sorted(p.items(), key=lambda kv: kv[1]); m = len(items); out = {}; run = 0.0
    for i, (k, v) in enumerate(items):
        run = max(run, min(1.0, (m - i) * v)); out[k] = run
    return out

def tie_refs(M):
    med = np.median(M, axis=0)
    return {"up": (med + 0.5).astype(int), "down": np.ceil(med - 0.5).astype(int)}, med

t0 = time.time()
res = dict(
    meta=dict(script=os.path.abspath(__file__), data_dir=VAL, B=B, seed=SEED, seed_check=SEED_CHECK,
              python=platform.python_version(), numpy=np.__version__,
              primary_set="60 form leaves minus 7 rust-training healthy leaves (form_image_key.csv model_split == 'train')",
              train_leaves_excluded=TRAIN, evaluator_labels="E1-E4 = column order of expert_consensus.csv",
              rust_rule="abaxial face", leaf_spot_rule="mean of both faces",
              statistic="mean over 4 LOO references of QWK(model, median of other 3) minus mean of QWK(evaluator j, median of other 3)",
              p_value="two-sided percentile-bootstrap p = min(1, 2*min(P*(d<=0), P*(d>=0))); add-one version also given",
              inference="leaves resampled, evaluators fixed: conditional on these four readers"),
    contrasts={}, families={}, direction_of_error={}, primary_table={}, reproduction_check={}, mc_check={})
TF = json.load(open(os.path.join(VAL, "tiefree_results.json"), encoding="utf-8"))
FR = json.load(open(os.path.join(VAL, "final_results.json"), encoding="utf-8"))

data = {dis: subset(dis) for dis in DIS + ("all",)}
boot = {dis: bootstrap(data[dis][1], data[dis][2], SEED) for dis in DIS + ("all",)}
boot_chk = {dis: bootstrap(data[dis][1], data[dis][2], SEED_CHECK) for dis in DIS}
print(f"bootstrap done in {time.time() - t0:.0f} s", flush=True)

# ---- (1) contrasts, p-values, Holm ----
point = {}
for dis in DIS + ("all",):
    rows, M, G = data[dis]; fm, fe, _ = loo_stats(M, G); point[dis] = (fm, fe)
    for a in ARCH:
        b = boot[dis][a]
        c = dict(disease=DIS_NAME[dis], model=ARCH_NAME[a], n=len(rows), model_loo_qwk=fm[a], evaluator_loo_qwk=fe,
                 diff=fm[a] - fe, ci95=b["ci"], p_two_sided=b["p_two_sided"], p_two_sided_add1=b["p_two_sided_add1"],
                 p_le0=b["p_le0"], n_valid_resamples=b["n_valid"], boot_sd=b["boot_sd"])
        if dis == "all":
            c["role"] = "descriptive only (pooled kappa mixes two disease-specific models); not in any Holm family"
        else:
            c["role"] = "primary (unadjusted)" if a == "segformer" else "secondary (Holm over 4)"
        res["contrasts"][f"{dis}_{a}"] = c

keys6 = [f"{d}_{a}" for d in DIS for a in ARCH]
prim = [k for k in keys6 if k.endswith("segformer")]; sec = [k for k in keys6 if k not in prim]
h4 = holm({k: res["contrasts"][k]["p_two_sided"] for k in sec})
h6 = holm({k: res["contrasts"][k]["p_two_sided"] for k in keys6})
h4a = holm({k: res["contrasts"][k]["p_two_sided_add1"] for k in sec})
h6a = holm({k: res["contrasts"][k]["p_two_sided_add1"] for k in keys6})
for k in keys6:
    c = res["contrasts"][k]
    c["p_holm_secondary4"] = h4.get(k); c["p_holm_all6"] = h6[k]
    c["p_holm_secondary4_add1"] = h4a.get(k); c["p_holm_all6_add1"] = h6a[k]
    c["p_reported"] = c["p_two_sided"] if k in prim else h4[k]
    c["significant_0.05"] = bool(c["p_reported"] < 0.05)
    c["significant_0.05_holm_all6"] = bool(h6[k] < 0.05)
res["families"] = dict(primary=prim, primary_note="SegFormer-B5 per disease, unadjusted (two pre-designated primary contrasts)",
                       secondary=sec, secondary_note="DeepLabV3+ and UNet++ per disease, Holm step-down over these 4",
                       all_six=keys6, all_six_note="Holm step-down over all 6 contrasts (sensitivity)")

# ---- reproduction check against tiefree_results.json and final_results.json ----
maxd = 0.0
for dis in DIS + ("all",):
    for a in ARCH:
        k = f"{dis}_{a}"; c = res["contrasts"][k]; t = TF[k]
        dd = [abs(round(c["ci95"][i], 3) - t["diff_ci"][i]) for i in range(2)]
        maxd = max(maxd, *dd)
        res["reproduction_check"][k] = dict(ci_here=[round(v, 3) for v in c["ci95"]], ci_tiefree=t["diff_ci"],
                                            diff_here=round(c["diff"], 4), diff_tiefree=round(t["diff"], 4),
                                            p_le0_here=round(c["p_le0"], 4), p_le0_tiefree=t["p_le0"])
res["reproduction_check"]["max_abs_ci_difference_3dp"] = maxd
for dis in DIS:
    assert abs(point[dis][1] - FR[f"primary_{dis}"]["expert_fair"]) < 5e-4
    for a in ARCH: assert abs(point[dis][0][a] - FR[f"primary_{dis}"][a]["fair"]) < 5e-4

# ---- Monte Carlo check with an independent seed ----
for dis in DIS:
    for a in ARCH:
        b = boot_chk[dis][a]; k = f"{dis}_{a}"
        res["mc_check"][k] = dict(seed=SEED_CHECK, ci95=b["ci"], p_two_sided=b["p_two_sided"])

# ---- (2) direction of error vs the four-evaluator median, both tie rules; (3) primary table ----
for dis in DIS + ("all",):
    rows, M, G = data[dis]; n = len(rows); R, med = tie_refs(M); fm, fe, refs = loo_stats(M, G)
    ties = [rows[i]["leaf_id"] for i in range(n) if med[i] % 1]
    lo, hi = M.min(0), M.max(0)
    dd = dict(n=n, n_two_two_splits=len(ties), split_leaves=ties, models={})
    for a in ARCH:
        g = G[a]; m = {}
        for rule, ref in R.items():
            m[f"ties_{rule}"] = dict(lower=int((g < ref).sum()), equal=int((g == ref).sum()), higher=int((g > ref).sum()),
                                     bias=float((g - ref).mean()), max_abs_diff=int(np.abs(g - ref).max()),
                                     panel_G0=int((ref == 0).sum()), panel_G0_model_above=int(((ref == 0) & (g > 0)).sum()))
        m["vs_panel_range_tie_free"] = dict(below_min=int((g < lo).sum()), inside=int(((g >= lo) & (g <= hi)).sum()),
                                            above_max=int((g > hi).sum()))
        m["outside_range_leaves"] = [dict(form_id=rows[i]["leaf_id"], model_grade=int(g[i]), panel_grades=M[:, i].tolist(),
                                          sev_ab=float(P[rows[i]["leaf_id"]][f"{a}_sev_ab"]), sev_ad=float(P[rows[i]["leaf_id"]][f"{a}_sev_ad"]))
                                     for i in range(n) if g[i] < lo[i] or g[i] > hi[i]]
        dd["models"][ARCH_NAME[a]] = m
    res["direction_of_error"][dis] = dd

    ev_loo = [float(qwk(M[j], refs[j])) for j in range(4)]
    pw_e = [float(qwk(M[i], M[j])) for i, j in itertools.combinations(range(4), 2)]
    tab = dict(disease=DIS_NAME[dis], n=n,
               evaluators=dict(loo_qwk_mean=fe, loo_qwk_each=dict(zip(EVAL_LABEL, ev_loo)), loo_qwk_range=[min(ev_loo), max(ev_loo)],
                               pairwise_qwk_mean=float(np.mean(pw_e)), pairwise_qwk_range=[min(pw_e), max(pw_e)],
                               exact_vs_loo_mean=float(np.mean([(M[j] == refs[j]).mean() for j in range(4)]))),
               models={})
    if dis == "all": tab["note"] = "descriptive only; pooled kappa combines two disease-specific models"
    for a in ARCH:
        g = G[a]; c = res["contrasts"][f"{dis}_{a}"]
        tab["models"][ARCH_NAME[a]] = dict(
            loo_qwk=fm[a], diff=c["diff"], ci95=c["ci95"],
            p_two_sided=None if dis == "all" else c["p_two_sided"],
            p_holm_secondary4=c.get("p_holm_secondary4"), p_holm_all6=c.get("p_holm_all6"),
            pairwise_qwk_mean=float(np.mean([qwk(g, M[j]) for j in range(4)])),
            exact_vs_loo_mean=float(np.mean([(g == refs[j]).mean() for j in range(4)])),
            qwk_median_down=float(qwk(g, R["down"])), qwk_median_up=float(qwk(g, R["up"])),
            exact_median_down=float((g == R["down"]).mean()), exact_median_up=float((g == R["up"]).mean()),
            within1_both_rules=bool(((np.abs(g - R["up"]) <= 1) & (np.abs(g - R["down"]) <= 1)).all()),
            inside_panel_range=int(((g >= lo) & (g <= hi)).sum()),
            bias_down=float((g - R["down"]).mean()), bias_up=float((g - R["up"]).mean()))
    res["primary_table"][dis] = tab

res["meta"]["runtime_s"] = round(time.time() - t0, 1)
json.dump(res, open(os.path.join(OUT, "results.json"), "w", encoding="utf-8"), indent=1)

# ---- console summary ----
for k in keys6:
    c = res["contrasts"][k]
    print(f"{k:16s} diff {c['diff']:+.3f} CI [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] p {c['p_two_sided']:.4f} "
          f"holm4 {c['p_holm_secondary4']} holm6 {c['p_holm_all6']:.4f}  (tiefree {TF[k]['diff_ci']})")
print("max |CI - tiefree| (3 dp):", maxd)
for dis in DIS:
    for a, m in res["direction_of_error"][dis]["models"].items():
        print(dis, a, "up", m["ties_up"], "| down", m["ties_down"], "| range", m["vs_panel_range_tie_free"])
