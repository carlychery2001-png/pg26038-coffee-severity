"""Round 4 / A2: expert panel with matched models + panel reanalyses (PG-26038 major revision).

Inputs (read only):
  validation/expert_consensus.csv   the four evaluators' grades (column order -> labels E1-E4; names never written)
  validation/form_image_key.csv     form item -> leaf ID and model split
  validation/final_analysis.py      qwk() and model_grade() definitions (executed up to 'res = {}' as tiefree.py does)
  validation/model_predictions.csv  published-model form predictions (reproduced by run_forms.py)
  results/r4_panel/form_predictions.csv  published + matched (lrB, seeds 0-2) form predictions from run_forms.py
  reanalysis/out/per_image.csv, per_image_retrained.csv  scan predictions and reference masks
  datasets/*/split.csv, masks      reference grades

Design (as validation/tiefree.py and results/panel_stats):
  * primary set = 53 leaves (60 form leaves minus the 7 whose form_image_key model_split == 'train');
    30 American leaf spot, 23 rust. Matched models were trained on the same packs and splits, so the same 53 leaves
    are unseen by them as well (checked below).
  * rust grade = abaxial face; leaf spot = mean of both faces; cut-points 0.5 / 5 / 20 %.
  * LOO statistic: for each evaluator j the reference is the median of the other three (an integer). Rater value =
    mean_j QWK(rater, ref_j); evaluator value = mean_j QWK(evaluator j, ref_j); contrast = rater minus evaluators.
  * bootstrap: 10,000 resamples of leaves, numpy default_rng(20260928) (round-4 seed), one resample index vector per
    disease shared by every rater; evaluators fixed (inference conditional on these four readers); NaN resamples
    dropped per statistic. Two-sided p = min(1, 2 min(P*(d<=0), P*(d>=0))).
  * primary contrasts: published SegFormer-B5 per disease (unadjusted). Secondary: published DeepLabV3+ / UNet++ and
    matched DeepLabV3+ / UNet++ (seed-mean contrast = mean over seeds 0-2 of the model LOO QWK minus the evaluator
    LOO QWK). Holm over the 8 secondary contrasts (main), over the 4 matched only, and over the original 4 published.
  * the published CIs with the tiefree seed 7 are recomputed and must equal validation/tiefree_results.json.
Writes only into this folder: results.json.
"""
import os as _os  # (release) all paths below are relative to ROOT (repository root by default; see README)
ROOT = (_os.environ.get("PG26038_ROOT") or _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))).replace(chr(92), "/")
import csv, itertools, json, os, re, time
from collections import Counter
import numpy as np
from PIL import Image

OUT = os.path.dirname(os.path.abspath(__file__))
BASE = ROOT; VAL = f"{BASE}/validation"; RE = f"{BASE}/reanalysis"
B = 10000; SEED = 20260928; SEED_TIEFREE = 7
DIS = ("ojo", "roya"); DNAME = {"ojo": "American leaf spot", "roya": "Coffee leaf rust"}
ANAME = {"segformer": "SegFormer-B5", "deeplab": "DeepLabV3+", "unetpp": "UNet++"}
grade = lambda s: 0 if s < 0.5 else 1 if s < 5 else 2 if s < 20 else 3
W = (np.subtract.outer(np.arange(4), np.arange(4)) ** 2) / 9.0

# ---------------- definitions from final_analysis.py (identical to tiefree.py / panel_stats) ----------------
_cwd = os.getcwd(); os.chdir(VAL)
try:
    ns = {}; exec(open("final_analysis.py", encoding="utf-8").read().split("res = {}")[0], ns)
finally:
    os.chdir(_cwd)
qwk_ref, E, KEY, EXP = ns["qwk"], ns["E"], ns["KEY"], ns["EXP"]
assert len(EXP) == 4
EL = [f"E{i + 1}" for i in range(4)]     # column order of expert_consensus.csv; names are not written anywhere


def qwk(a, b):
    """Same statistic as final_analysis.qwk (asserted below), vectorised with bincount."""
    O = np.bincount(np.asarray(a) * 4 + np.asarray(b), minlength=16).reshape(4, 4).astype(float)
    Ex = np.outer(O.sum(1), O.sum(0)) / O.sum()
    with np.errstate(all="ignore"):
        return float(1 - (W * O).sum() / (W * Ex).sum())


TRAIN = sorted(f for f, r in KEY.items() if r["model_split"] == "train"); assert len(TRAIN) == 7
LEAF = {f: r["leaf_id"] for f, r in KEY.items()}

# ---------------- reference grades ----------------
PI = list(csv.DictReader(open(f"{RE}/out/per_image.csv")))
PR = list(csv.DictReader(open(f"{RE}/out/per_image_retrained.csv")))
ref_sev_face = {}      # (leaf_id, face) -> reference severity (letterbox space), from the published SegFormer-B5 rows
for r in PI:
    if r["arch"] == "segformer" and r["model_disease"] == r["image_pack"]:
        ref_sev_face[(r["leaf_id"], r["face"])] = float(r["ref_sev"])
split_rows = {"roya": list(csv.DictReader(open(f"{BASE}/datasets/pg26038-roya-real-pack/split.csv"))),
              "ojo": list(csv.DictReader(open(f"{BASE}/datasets/pg26038-ojo-real-pack/meta/split.csv")))}
visual = {r["leaf_id"]: r["grade"] for r in split_rows["ojo"]}
SPLIT = {dis: {r["leaf_id"]: r["split"] for r in split_rows[dis]} for dis in DIS}


def healthy_mask_empty(lid):
    """True if an empty reference lesion mask exists locally for a healthy leaf (both faces)."""
    for d in ("pg26038-roya-real-pack", "pg26038-ojo-lesions"):
        ok = []
        for face in ("abaxial", "adaxial"):
            ip, mp = f"{BASE}/datasets/{d}/images/{lid}__{face}.png", f"{BASE}/datasets/{d}/masks/{lid}__{face}.png"
            if not (os.path.exists(ip) and os.path.exists(mp)): break
            a = np.asarray(Image.open(ip).convert("RGBA"))[..., 3] > 8
            m = (np.asarray(Image.open(mp).convert("L")) > 127) & a
            ok.append(int(m.sum()) == 0)
        if len(ok) == 2: return all(ok), d
    return None, None


REF = {}               # form_id -> dict(mask=grade, visual=grade or None, sev=float, source=str)
for f, lid in LEAF.items():
    dis = "ojo" if f.startswith("ojo") else "roya"
    if "Healthy" in lid:
        empty, src = healthy_mask_empty(lid)
        assert empty, (f, lid)
        REF[f] = dict(mask=0, visual=0, sev=0.0, sev_ab=0.0, sev_ad=0.0, source=f"healthy leaf; empty reference mask in {src}")
    elif dis == "roya":
        s_ab, s_ad = ref_sev_face[(lid, "abaxial")], ref_sev_face[(lid, "adaxial")]
        REF[f] = dict(mask=grade(s_ab), visual=None, sev=s_ab, sev_ab=s_ab, sev_ad=s_ad, source="rust test mask, abaxial")
    else:
        s_ab, s_ad = ref_sev_face[(lid, "abaxial")], ref_sev_face[(lid, "adaxial")]
        s = (s_ab + s_ad) / 2
        REF[f] = dict(mask=grade(s), visual=int(visual[lid][1]), sev=s, sev_ab=s_ab, sev_ad=s_ad, source="leaf-spot test mask, mean of faces")

# ---------------- model grades on the form images ----------------
FP = list(csv.DictReader(open(f"{OUT}/form_predictions.csv")))
RUNS = sorted({r["run"] for r in FP})
FORM = {}              # run -> form_id -> (sev_ab, sev_ad)
for r in FP: FORM.setdefault(r["run"], {})[r["form_id"]] = (float(r["sev_ab"]), float(r["sev_ad"]))
RUNINFO = {r["run"]: dict(kind=r["kind"], disease=r["disease"], arch=r["arch"], seed=r["seed"]) for r in FP}
MP = {r["leaf_id"]: r for r in csv.DictReader(open(f"{VAL}/model_predictions.csv"))}
# published form predictions must equal validation/model_predictions.csv
repro = {}
for run, d in FORM.items():
    if RUNINFO[run]["kind"] != "published": continue
    a = RUNINFO[run]["arch"]
    diffs = [max(abs(v[0] - float(MP[f][f"{a}_sev_ab"])), abs(v[1] - float(MP[f][f"{a}_sev_ad"]))) for f, v in d.items()]
    repro[run] = dict(n=len(diffs), max_abs_diff=max(diffs))


def form_grade(run, f, rule="paper"):
    ab, ad = FORM[run][f]
    if f.startswith("roya") and rule == "paper": return grade(ab)
    return grade((ab + ad) / 2)


def matched_runs(dis, arch):
    return sorted([r for r in RUNS if RUNINFO[r]["kind"] == "matched" and RUNINFO[r]["disease"] == dis and RUNINFO[r]["arch"] == arch],
                  key=lambda r: int(RUNINFO[r]["seed"]))


# check: matched models were trained on the same pack/splits -> primary leaves unseen (form leaves in 'train' = the 7)
seen_check = {}
for dis in DIS:
    tr = sorted(f for f, lid in LEAF.items() if f.startswith(dis) and SPLIT[dis].get(lid) == "train")
    seen_check[dis] = dict(form_leaves_in_own_pack_train_split=tr)
assert seen_check["ojo"]["form_leaves_in_own_pack_train_split"] == [] and seen_check["roya"]["form_leaves_in_own_pack_train_split"] == TRAIN

# ---------------- rater matrices for the primary set ----------------
def subset(dis, primary=True):
    rows = [r for r in E if r["leaf_id"].startswith(dis) and (not primary or r["leaf_id"] not in TRAIN)]
    fids = [r["leaf_id"] for r in rows]
    M = np.array([[int(r[e]) for r in rows] for e in EXP])
    G = {}
    for a in ANAME: G[f"pub_{a}"] = np.array([form_grade(f"published_{dis}_{a}", f) for f in fids])
    for a in ("deeplab", "unetpp"):
        for run in matched_runs(dis, a): G[f"m_{a}_s{RUNINFO[run]['seed']}"] = np.array([form_grade(run, f) for f in fids])
    G["ref_mask"] = np.array([REF[f]["mask"] for f in fids])
    if dis == "ojo": G["ref_visual"] = np.array([REF[f]["visual"] for f in fids])
    return fids, M, G


def loo_refs(M): return [np.median(np.delete(M, j, 0), axis=0).astype(int) for j in range(4)]


def loo_all(M, G):
    refs = loo_refs(M)
    fe = float(np.mean([qwk(M[j], refs[j]) for j in range(4)]))
    fm = {k: float(np.mean([qwk(g, refs[j]) for j in range(4)])) for k, g in G.items()}
    return fm, fe


def derived(fm, fe):
    """Contrasts computed from one set of LOO values (point or bootstrap)."""
    d = {k: fm[k] - fe for k in fm}
    for a in ("deeplab", "unetpp"):
        ks = [k for k in fm if k.startswith(f"m_{a}_s")]
        d[f"mmean_{a}"] = float(np.mean([fm[k] for k in ks])) - fe                         # seed-mean contrast
        d[f"mmean_{a}_minus_pub_{a}"] = float(np.mean([fm[k] for k in ks])) - fm[f"pub_{a}"]  # matched vs published
        d[f"mmean_{a}_minus_pub_segformer"] = float(np.mean([fm[k] for k in ks])) - fm["pub_segformer"]
    for k in [k for k in fm if k.startswith("ref_")]:
        d[f"pub_segformer_minus_{k}"] = fm["pub_segformer"] - fm[k]
    return d


def pval(x):
    le, ge = int((x <= 0).sum()), int((x >= 0).sum()); n = len(x)
    return min(1.0, 2 * min(le, ge) / n)


def bootstrap(M, G, seed, keys=None):
    rng = np.random.default_rng(seed); n = M.shape[1]; acc = {}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        fm, fe = loo_all(M[:, idx], {k: g[idx] for k, g in G.items() if keys is None or k in keys})
        for k, v in derived(fm, fe).items() if keys is None else {k: fm[k] - fe for k in fm}.items():
            acc.setdefault(k, []).append(v)
    out = {}
    for k, v in acc.items():
        x = np.array(v); x = x[np.isfinite(x)]
        out[k] = dict(ci95=[float(q) for q in np.percentile(x, [2.5, 97.5])], p_two_sided=pval(x), n_valid=int(len(x)))
    return out


def holm(p):
    items = sorted(p.items(), key=lambda kv: kv[1]); m = len(items); out = {}; run = 0.0
    for i, (k, v) in enumerate(items):
        run = max(run, min(1.0, (m - i) * v)); out[k] = run
    return out


def describe(g, M):
    refs = loo_refs(M); med = np.median(M, axis=0); up, dn = (med + 0.5).astype(int), np.ceil(med - 0.5).astype(int)
    lo, hi = M.min(0), M.max(0)
    return dict(exact_vs_loo=float(np.mean([(g == r).mean() for r in refs])),
                within1_vs_loo=float(np.mean([(np.abs(g - r) <= 1).mean() for r in refs])),
                pairwise_qwk_mean=float(np.mean([qwk(g, M[j]) for j in range(4)])),
                qwk_median_up=qwk(g, up), qwk_median_down=qwk(g, dn),
                exact_median_up=float((g == up).mean()), exact_median_down=float((g == dn).mean()),
                bias_up=float((g - up).mean()), bias_down=float((g - dn).mean()),
                inside_panel_range=int(((g >= lo) & (g <= hi)).sum()), below_min=int((g < lo).sum()), above_max=int((g > hi).sum()),
                panel_G0_up=int((up == 0).sum()), panel_G0_up_rater_above=int(((up == 0) & (g > 0)).sum()),
                panel_G0_down=int((dn == 0).sum()), panel_G0_down_rater_above=int(((dn == 0) & (g > 0)).sum()))


t0 = time.time()
res = dict(meta=dict(script=os.path.abspath(__file__), B=B, seed=SEED, seed_tiefree_check=SEED_TIEFREE,
                     evaluator_labels="E1-E4 = column order of validation/expert_consensus.csv (no names written)",
                     primary_set="60 form leaves minus 7 rust-training healthy leaves", train_leaves_excluded=TRAIN,
                     rust_rule="abaxial face", leaf_spot_rule="mean of both faces", cut_points=[0.5, 5, 20],
                     statistic="mean_j QWK(rater, median of the other 3 evaluators) minus mean_j QWK(evaluator j, same reference)",
                     p_value="two-sided percentile-bootstrap inversion", inference="leaves resampled, evaluators fixed",
                     form_reproduction=repro, matched_training_overlap=seen_check),
           reference_grades={f: {k: v for k, v in REF[f].items()} for f in sorted(REF)},
           part1={}, part2={}, part3={}, part4={})

# ---------- PART 1 + PART 2 (same bootstrap) ----------
for dis in DIS:
    fids, M, G = subset(dis)
    # the vectorised qwk equals final_analysis.qwk
    for k, g in G.items(): assert abs(qwk(g, M[0]) - qwk_ref(g, M[0])) < 1e-12 or not np.isfinite(qwk(g, M[0]))
    fm, fe = loo_all(M, G); d = derived(fm, fe)
    bt = bootstrap(M, G, SEED)
    bt7 = bootstrap(M, {k: G[k] for k in ("pub_segformer", "pub_deeplab", "pub_unetpp")}, SEED_TIEFREE,
                    keys={"pub_segformer", "pub_deeplab", "pub_unetpp"})
    refs = loo_refs(M); ev = [qwk(M[j], refs[j]) for j in range(4)]
    part = dict(disease=DNAME[dis], n=len(fids), leaves=fids,
                evaluators=dict(loo_qwk_mean=fe, loo_qwk_each=dict(zip(EL, ev)), loo_qwk_range=[min(ev), max(ev)],
                                pairwise_qwk_mean=float(np.mean([qwk(M[i], M[j]) for i, j in itertools.combinations(range(4), 2)])),
                                exact_vs_loo_mean=float(np.mean([(M[j] == refs[j]).mean() for j in range(4)])),
                                within1_vs_loo_mean=float(np.mean([(np.abs(M[j] - refs[j]) <= 1).mean() for j in range(4)]))),
                raters={}, contrasts={}, tiefree_seed7_check={})
    for k, g in G.items():
        part["raters"][k] = dict(loo_qwk=fm[k], grades=g.tolist(), **describe(g, M))
    mk = [k for k in G if k.startswith(("pub_", "m_"))]
    Gm = np.array([G[k] for k in mk])
    part["model_disagreement_leaves"] = [dict(form_id=fids[i], grades={k: int(G[k][i]) for k in mk}, panel=M[:, i].tolist())
                                         for i in range(len(fids)) if len(set(Gm[:, i])) > 1]
    part["identical_grade_vectors"] = [[a, b] for i, a in enumerate(G) for b in list(G)[i + 1:] if (G[a] == G[b]).all()]
    for k, v in d.items():
        part["contrasts"][k] = dict(estimate=v, **bt[k])
    # seed summaries for matched
    for a in ("deeplab", "unetpp"):
        ks = [k for k in G if k.startswith(f"m_{a}_s")]
        vals = np.array([fm[k] for k in ks]); diffs = vals - fe
        part["contrasts"][f"mmean_{a}"].update(seed_loo_qwk=vals.tolist(), seed_loo_mean=float(vals.mean()), seed_loo_sd=float(vals.std(ddof=1)),
                                               seed_diff_sd=float(diffs.std(ddof=1)), seed_diff_range=[float(diffs.min()), float(diffs.max())])
        for m in ("exact_vs_loo", "pairwise_qwk_mean", "inside_panel_range", "bias_up", "bias_down", "qwk_median_up", "qwk_median_down"):
            xs = np.array([part["raters"][k][m] for k in ks], float)
            part["contrasts"][f"mmean_{a}"][f"seed_{m}"] = [float(xs.mean()), float(xs.std(ddof=1)), float(xs.min()), float(xs.max())]
    TF = json.load(open(f"{VAL}/tiefree_results.json"))
    for a in ANAME:
        part["tiefree_seed7_check"][a] = dict(ci_seed7=[round(x, 3) for x in bt7[f"pub_{a}"]["ci95"]], ci_tiefree=TF[f"{dis}_{a}"]["diff_ci"],
                                              diff=round(fm[f"pub_{a}"] - fe, 4), diff_tiefree=round(TF[f"{dis}_{a}"]["diff"], 4))
    res["part1"][dis] = part
    print(dis, "bootstrap done", round(time.time() - t0), "s", flush=True)

# Holm families
P = {f"{dis}:{k}": res["part1"][dis]["contrasts"][k]["p_two_sided"] for dis in DIS
     for k in ("pub_deeplab", "pub_unetpp", "mmean_deeplab", "mmean_unetpp")}
h8 = holm(P)
h4m = holm({k: v for k, v in P.items() if ":mmean_" in k})
h4p = holm({k: v for k, v in P.items() if ":pub_" in k})
for k, v in P.items():
    dis, c = k.split(":"); cc = res["part1"][dis]["contrasts"][c]
    cc["p_holm_secondary8"] = h8[k]; cc["p_holm_matched4"] = h4m.get(k); cc["p_holm_published4"] = h4p.get(k)
res["meta"]["holm_families"] = dict(primary=["ojo:pub_segformer", "roya:pub_segformer"], secondary8=sorted(P),
                                    note="matched contrasts are seed-mean contrasts; per-seed contrasts are descriptive and unadjusted")

# ---------- PART 2: reference as a rater, per-evaluator agreement, unanimous disagreements ----------
for dis in DIS:
    for primary in (True, False):
        fids, M, G = subset(dis, primary)
        tag = "primary" if primary else "all60"
        out = {}
        for rk in [k for k in G if k.startswith("ref_")]:
            ref = G[rk]; per = {}
            rng = np.random.default_rng(SEED); n = len(fids); boots = {e: [] for e in EL}
            for _ in range(B):
                idx = rng.integers(0, n, n)
                for j, e in enumerate(EL): boots[e].append(qwk(M[j][idx], ref[idx]))
            for j, e in enumerate(EL):
                x = np.array(boots[e]); x = x[np.isfinite(x)]
                per[e] = dict(exact=int((M[j] == ref).sum()), within1=int((np.abs(M[j] - ref) <= 1).sum()), n=n,
                              qwk=qwk(M[j], ref), qwk_ci95=[float(q) for q in np.percentile(x, [2.5, 97.5])],
                              mean_signed_diff_expert_minus_ref=float((M[j] - ref).mean()),
                              higher=int((M[j] > ref).sum()), lower=int((M[j] < ref).sum()))
            med = np.median(M, axis=0); up, dn = (med + 0.5).astype(int), np.ceil(med - 0.5).astype(int)
            allfour = [dict(form_id=fids[i], leaf_id=LEAF[fids[i]], ref_grade=int(ref[i]), ref_sev_used=round(REF[fids[i]]["sev"], 3),
                            ref_sev_ab=round(REF[fids[i]]["sev_ab"], 3), ref_sev_ad=round(REF[fids[i]]["sev_ad"], 3),
                            panel=M[:, i].tolist(), direction="all higher" if (M[:, i] > ref[i]).all() else "all lower" if (M[:, i] < ref[i]).all() else "mixed",
                            in_primary=fids[i] not in TRAIN,
                            pub_segformer_form_grade=form_grade(f"published_{dis}_segformer", fids[i]))
                       for i in range(n) if (M[:, i] != ref[i]).all()]
            out[rk] = dict(n=n, per_evaluator=per,
                           median_up=dict(exact=int((up == ref).sum()), within1=int((np.abs(up - ref) <= 1).sum()), qwk=qwk(up, ref)),
                           median_down=dict(exact=int((dn == ref).sum()), within1=int((np.abs(dn - ref) <= 1).sum()), qwk=qwk(dn, ref)),
                           ref_grade_counts=dict(sorted(Counter(int(x) for x in ref).items())),
                           panel_grade_counts={e: dict(sorted(Counter(int(x) for x in M[j]).items())) for j, e in enumerate(EL)},
                           all_four_disagree=allfour,
                           unanimous_panel_G3=[fids[i] for i in range(n) if (M[:, i] == 3).all()])
        res["part2"][f"{dis}_{tag}"] = out

# rust reference G3 check on the whole local rust pack (abaxial, native resolution; structure/ found grades identical)
g3 = {}
for r in split_rows["roya"]:
    if r["grade"] == "healthy" or "abaxial" not in r["file"]: continue
    a = np.asarray(Image.open(f"{BASE}/datasets/pg26038-roya-real-pack/images/{r['file']}").convert("RGBA"))[..., 3] > 8
    m = (np.asarray(Image.open(f"{BASE}/datasets/pg26038-roya-real-pack/masks/{r['file']}").convert("L")) > 127) & a
    s = 100 * m.sum() / a.sum(); g3.setdefault(r["split"], Counter())[grade(s)] += 1
    if grade(s) == 3: g3.setdefault("G3_leaves", []).append(dict(leaf_id=r["leaf_id"], split=r["split"], sev_ab=round(float(s), 2)))
res["part2"]["rust_pack_abaxial_grade_counts"] = {k: (dict(sorted(v.items())) if isinstance(v, Counter) else v) for k, v in g3.items()}

# ---------- PART 3: panel-selected vs unselected leaf-spot test leaves (scans, published models) ----------
sel_ids = {LEAF[f] for f in LEAF if f.startswith("ojo") and KEY[f]["model_split"] == "test"}
leaf = {}
for r in PI:
    if r["model_disease"] != "ojo" or r["image_pack"] != "ojo" or r["split"] != "test": continue
    leaf.setdefault((r["arch"], r["leaf_id"]), []).append(r)
ids = sorted({lid for (_a, lid) in leaf}); assert len(ids) == 55 and sel_ids <= set(ids) and len(sel_ids) == 27
sel = np.array([lid in sel_ids for lid in ids])


def perleaf(arch):
    rows = [leaf[(arch, lid)] for lid in ids]; assert all(len(x) == 2 for x in rows)
    rs = np.array([np.mean([float(x["ref_sev"]) for x in v]) for v in rows]); ps = np.array([np.mean([float(x["pred_sev"]) for x in v]) for v in rows])
    iou = np.array([np.mean([float(x["lesion_iou"]) for x in v]) for v in rows])
    vg = np.array([int(v[0]["grade_label"][1]) for v in rows]); mg = np.array([grade(s) for s in rs]); pg = np.array([grade(s) for s in ps])
    return dict(ref_sev=rs, pred_sev=ps, abs_err=np.abs(ps - rs), signed_err=ps - rs, iou=iou, visual=vg, maskg=mg, predg=pg,
                correct_mask=(pg == mg).astype(float), correct_visual=(pg == vg).astype(float))


def chi2(labels, groups, K):
    tab = np.zeros((2, K))
    for g, l in zip(groups, labels): tab[int(g), l] += 1
    exp = np.outer(tab.sum(1), tab.sum(0)) / tab.sum()
    with np.errstate(all="ignore"): return float(np.nansum(np.where(exp > 0, (tab - exp) ** 2 / exp, 0)))


def perm_test(x, stat, seed=SEED):
    rng = np.random.default_rng(seed); t = stat(x, sel); cnt = 0
    for _ in range(B):
        s = rng.permutation(sel)
        cnt += abs(stat(x, s)) >= abs(t) - 1e-12
    return t, (cnt + 1) / (B + 1)


def boot_diff(x, seed=SEED):
    rng = np.random.default_rng(seed); a, b = x[sel], x[~sel]; v = []
    for _ in range(B): v.append(rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean())
    return [float(q) for q in np.percentile(v, [2.5, 97.5])]


mean_diff = lambda x, s: float(x[s].mean() - x[~s].mean())
med_diff = lambda x, s: float(np.median(x[s]) - np.median(x[~s]))
p3 = dict(n_selected=int(sel.sum()), n_unselected=int((~sel).sum()), selected_leaf_ids=sorted(sel_ids), models={})
for arch in ANAME:
    L = perleaf(arch); m = {}
    for var in ("ref_sev", "abs_err", "signed_err", "iou", "correct_mask", "correct_visual") if arch == "segformer" else ("abs_err", "signed_err", "iou", "correct_mask", "correct_visual"):
        x = L[var]; t, p = perm_test(x, mean_diff)
        m[var] = dict(mean_selected=float(x[sel].mean()), mean_unselected=float(x[~sel].mean()),
                      median_selected=float(np.median(x[sel])), median_unselected=float(np.median(x[~sel])),
                      diff_mean=t, diff_mean_ci95=boot_diff(x), p_perm_mean=p, p_perm_median=perm_test(x, med_diff)[1])
    if arch == "segformer":
        for gv, K in (("visual", 4), ("maskg", 4)):
            x = L[gv]; t, p = perm_test(x, lambda xx, s: chi2(xx, s, K)); tm, pm = perm_test(x.astype(float), mean_diff)
            m[f"grade_mix_{gv}"] = dict(selected=dict(sorted(Counter(int(v) for v in x[sel]).items())),
                                        unselected=dict(sorted(Counter(int(v) for v in x[~sel]).items())),
                                        chi2=t, p_perm_chi2=p, mean_grade_diff=tm, p_perm_mean_grade=pm)
        m["ref_sev_range"] = dict(selected=[float(L["ref_sev"][sel].min()), float(L["ref_sev"][sel].max())],
                                  unselected=[float(L["ref_sev"][~sel].min()), float(L["ref_sev"][~sel].max())])
        dates = np.array([0 if l.startswith("20260621") else 1 for l in ids])
        m["collection_date"] = dict(selected=dict(Counter(l[:8] for l, s in zip(ids, sel) if s)),
                                    unselected=dict(Counter(l[:8] for l, s in zip(ids, sel) if not s)),
                                    p_perm_chi2=perm_test(dates, lambda xx, s: chi2(xx, s, 2))[1])
        # sensitivity: severity-range-matched comparison (drop selected leaves above the unselected maximum)
        keep = L["ref_sev"] <= L["ref_sev"][~sel].max() + 1e-9
        sel_full = sel.copy(); sel = sel_full[keep]
        sub = {}
        for var in ("ref_sev", "abs_err", "signed_err", "iou"):
            x = L[var][keep]; t, p = perm_test(x, mean_diff)
            sub[var] = dict(mean_selected=float(x[sel].mean()), mean_unselected=float(x[~sel].mean()), diff_mean=t, p_perm_mean=p)
        m["range_matched"] = dict(max_unselected_ref_sev=float(L["ref_sev"][~sel_full].max()), n_selected=int(sel.sum()),
                                  n_unselected=int((~sel).sum()),
                                  dropped=[ids[i] for i in range(len(ids)) if not keep[i]], **sub)
        sel = sel_full
        # error relative to reference severity (the Bland-Altman analysis found proportional error)
        rel = L["abs_err"] / np.maximum(L["ref_sev"], 1e-9); t, p = perm_test(rel, mean_diff)
        m["abs_err_over_ref_sev"] = dict(median_selected=float(np.median(rel[sel])), median_unselected=float(np.median(rel[~sel])),
                                         p_perm_median=perm_test(rel, med_diff)[1], p_perm_mean=p)
    p3["models"][ANAME[arch]] = m
res["part3"] = p3

# ---------- PART 4: form image vs original scan ----------
scan = {}   # (run, leaf_id) -> {face: pred_sev}
for r in PI:
    if r["model_disease"] == r["image_pack"] and r["split"] == "test":
        scan.setdefault((f"published_{r['model_disease']}_{r['arch']}", r["leaf_id"]), {})[r["face"]] = float(r["pred_sev"])
for r in PR:
    if r["split"] == "test": scan.setdefault((r["run"], r["leaf_id"]), {})[r["face"]] = float(r["pred_sev"])
p4 = dict(runs={}, not_available=[])
for f in sorted(LEAF):
    if not any((run, LEAF[f]) in scan for run in RUNS): p4["not_available"].append(dict(form_id=f, leaf_id=LEAF[f], split=KEY[f]["model_split"]))
for run in RUNS:
    dis = RUNINFO[run]["disease"]; fids = [f for f in sorted(LEAF) if f.startswith(dis) and (run, LEAF[f]) in scan]
    sf, ss, gf, gs = [], [], [], []
    for f in fids:
        s = scan[(run, LEAF[f])]; ab, ad = FORM[run][f]
        if dis == "roya": sf.append(ab); ss.append(s["abaxial"])
        else: sf.append((ab + ad) / 2); ss.append((s["abaxial"] + s["adaxial"]) / 2)
        gf.append(grade(sf[-1])); gs.append(grade(ss[-1]))
    sf, ss, gf, gs = map(np.array, (sf, ss, gf, gs)); n = len(fids); dsev = sf - ss
    rng = np.random.default_rng(SEED); ex, kq = [], []
    for _ in range(B):
        idx = rng.integers(0, n, n); ex.append((gf[idx] == gs[idx]).mean()); kq.append(qwk(gf[idx], gs[idx]))
    kq = np.array(kq); kq = kq[np.isfinite(kq)]
    rk = lambda x: np.argsort(np.argsort(x))
    p4["runs"][run] = dict(kind=RUNINFO[run]["kind"], disease=dis, arch=RUNINFO[run]["arch"], seed=RUNINFO[run]["seed"], n=n,
                           form_grade_counts=[int((gf == k).sum()) for k in range(4)], scan_grade_counts=[int((gs == k).sum()) for k in range(4)],
                           exact=int((gf == gs).sum()), exact_ci95=[float(q) for q in np.percentile(ex, [2.5, 97.5])],
                           within1=int((np.abs(gf - gs) <= 1).sum()), form_higher=int((gf > gs).sum()), form_lower=int((gf < gs).sum()),
                           qwk=qwk(gf, gs), qwk_ci95=[float(q) for q in np.percentile(kq, [2.5, 97.5])],
                           sev_diff_mean=float(dsev.mean()), sev_diff_sd=float(dsev.std(ddof=1)),
                           loa95=[float(dsev.mean() - 1.96 * dsev.std(ddof=1)), float(dsev.mean() + 1.96 * dsev.std(ddof=1))],
                           abs_sev_diff_median=float(np.median(np.abs(dsev))), spearman=float(np.corrcoef(rk(sf), rk(ss))[0, 1]),
                           disagreements=[dict(form_id=fids[i], leaf_id=LEAF[fids[i]], form_sev=round(float(sf[i]), 2), scan_sev=round(float(ss[i]), 3),
                                               form_grade=int(gf[i]), scan_grade=int(gs[i])) for i in range(n) if gf[i] != gs[i]])
# matched seed summaries for part 4
for dis in DIS:
    for a in ("deeplab", "unetpp"):
        rr = [p4["runs"][r] for r in matched_runs(dis, a)]
        p4[f"matched_{dis}_{a}_seedsummary"] = {m: [float(np.mean([x[m] for x in rr])), float(np.std([x[m] for x in rr], ddof=1)),
                                                    float(min(x[m] for x in rr)), float(max(x[m] for x in rr))]
                                                for m in ("exact", "within1", "qwk", "sev_diff_mean", "sev_diff_sd", "spearman")}
res["part4"] = p4
res["meta"]["runtime_s"] = round(time.time() - t0, 1)


def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.bool_): return bool(o)
    return o


json.dump(clean(res), open(f"{OUT}/results.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("done", res["meta"]["runtime_s"], "s")
