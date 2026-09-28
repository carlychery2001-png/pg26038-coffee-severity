"""Round 4, fifth expert: five-evaluator expert-panel reanalysis (PG-26038 major revision, 28 Sep 2026).

Inputs (read only; nothing is copied):
  validation/private/form_export_5experts_raw.csv  raw 5-row form export. ONLY the header of columns 8+ (item names) and
        the 60 grade cells ('G0'-'G3') of each row are used; identity columns 2-7 are never read into any variable that is
        written or printed. Row order = form-submission order -> labels E1-E5 (rows 1-4 = E1-E4 of results/r4_panel).
  validation/expert_consensus.csv   used only to assert that rows 1-4 equal the earlier E1-E4 grades (column order)
  validation/expert_grades_raw.csv  used only to assert that rows 1-4 of the new export are identical to the old export
  validation/form_image_key.csv     form item -> leaf ID, model split (7 rust-training healthy leaves excluded -> 53)
  validation/model_predictions.csv  original-model form predictions (rust grade = abaxial; leaf spot = mean of faces)
  validation/final_analysis.py      reference qwk / kripp / icc21 / fleiss (executed up to 'res = {}', assertions only)
  results/r4_panel/form_predictions.csv  matched-protocol (lrB, seeds 0-2) form predictions
  results/r4_panel/results.json     reference grades and 4-evaluator values (reproduction checks only)
  reanalysis/out/per_image.csv, per_image_retrained.csv  scan predictions / reference severities
  datasets/.../split.csv, masks     visual leaf-spot grade; empty masks of the healthy form leaves

Design (round4_plan.md, 'FIFTH EXPERT'):
  PRIMARY  'loo3': for each evaluator i and each 3-subset S of the other 4 evaluators, reference = median(S) (tie-free);
           evaluator value = mean over the 20 (i, S) of QWK(evaluator i, median S); model value = mean over the same 20
           of QWK(model, median S); contrast = model minus evaluators. With 4 evaluators this is the published design.
  (a)      'rest_up' / 'rest_down': reference = median of the other 4 evaluators, ties resolved up / down (5 references).
  (b)      'all5': model vs the 5-evaluator median (tie-free). The evaluators' values include their own grade (favours them).
  (c)      'four_E1_E4': the 4-evaluator design with E1-E4 (the panel of results/r4_panel)
           (= r4_panel; must reproduce it exactly).
  extra    'drop_k': 4-evaluator design dropping each evaluator in turn; pooled 53; all 60 leaves; rust from both faces.
Bootstrap: 10,000 resamples of leaves, numpy default_rng(20260928), one resample vector per leaf set shared by every
rater and design (drawn exactly as results/r4_panel did, so (c) reproduces its intervals); evaluators and seeds fixed
(inference conditional on these five readers). Two-sided p = min(1, 2 min(P*(d <= 0), P*(d >= 0))); NaN resamples dropped.
Primary contrast: original SegFormer-B5 per disease (unadjusted). Secondary family (Holm over 8): original DeepLabV3+ and
UNet++, matched DeepLabV3+ and UNet++ (seed-mean contrast) per disease.
Writes only into this folder: results.json (no names, no timestamps, no affiliations).
"""
import os as _os  # (release) all paths below are relative to ROOT (repository root by default; see README)
ROOT = (_os.environ.get("PG26038_ROOT") or _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))).replace(chr(92), "/")
import csv, itertools, json, os, re, time
from collections import Counter
import numpy as np
from PIL import Image

OUT = os.path.dirname(os.path.abspath(__file__))
BASE = ROOT; VAL = f"{BASE}/validation"; REA = f"{BASE}/reanalysis"
RAW5 = f"{VAL}/private/form_export_5experts_raw.csv"
B = 10000; SEED = 20260928; K = 4
DIS = ("ojo", "roya"); DNAME = {"ojo": "American leaf spot", "roya": "Coffee leaf rust", "all": "Pooled"}
ANAME = {"segformer": "SegFormer-B5", "deeplab": "DeepLabV3+", "unetpp": "UNet++"}
grade = lambda s: 0 if s < 0.5 else 1 if s < 5 else 2 if s < 20 else 3
t0 = time.time()

# ---------------- reference implementations (assertions only) ----------------
_cwd = os.getcwd(); os.chdir(VAL)
try:
    ns = {}; exec(open("final_analysis.py", encoding="utf-8").read().split("res = {}")[0], ns)
finally:
    os.chdir(_cwd)
qwk_ref, kripp_ref, icc_ref, fleiss_ref = ns["qwk"], ns["kripp"], ns["icc21"], ns["fleiss"]

# ---------------- grades of the five evaluators (grade cells only) ----------------
with open(RAW5, encoding="utf-8-sig") as fh:
    _raw = list(csv.reader(fh))
_hdr = _raw[0]
assert len(_raw) == 6 and len(_hdr) == 67 and all(len(r) == 67 for r in _raw[1:])
ITEMS = [re.match(r"^\s*\d+\)\s*(\w+)\s*-", h).group(1) for h in _hdr[7:]]
assert ITEMS == [f"ojo{i:02d}" for i in range(1, 31)] + [f"roya{i:02d}" for i in range(1, 31)]
_cells = [[c.strip() for c in r[7:]] for r in _raw[1:]]
assert all(re.fullmatch("G[0-3]", c) for r in _cells for c in r)
GR = np.array([[int(c[1]) for c in r] for r in _cells])          # 5 x 60
with open(f"{VAL}/expert_grades_raw.csv", encoding="utf-8-sig") as fh:
    _old = list(csv.reader(fh))
STRUCT = dict(rows=5, grade_columns=60, header_identical_to_4expert_export=_old[0] == _hdr,
              rows1to4_identical_to_4expert_export=all(_old[i] == _raw[i] for i in range(1, 5)))
del _raw, _old, _cells, _hdr                                        # identity columns never leave this block
_ec = list(csv.DictReader(open(f"{VAL}/expert_consensus.csv", encoding="utf-8")))
_cols = [k for k in _ec[0] if k not in ("leaf_id", "consensus_grade", "range")]
STRUCT["rows1to4_equal_r4_panel_E1_E4"] = all(GR[j, ITEMS.index(r["leaf_id"])] == int(r[c]) for j, c in enumerate(_cols) for r in _ec)
del _ec, _cols
assert all(STRUCT[k] for k in STRUCT if k.startswith(("header", "rows1")))
EL = [f"E{i + 1}" for i in range(5)]
FIFTH = 4          # row 5 (E5) = the fifth evaluator; sensitivity (c) uses E1-E4
ITIDX = {f: i for i, f in enumerate(ITEMS)}

# ---------------- form key ----------------
KEY = {r["form_id"]: r for r in csv.DictReader(open(f"{VAL}/form_image_key.csv", encoding="utf-8"))}
TRAIN = sorted(f for f, r in KEY.items() if r["model_split"] == "train"); assert len(TRAIN) == 7
LEAF = {f: KEY[f]["leaf_id"] for f in KEY}
dis_of = lambda f: "ojo" if f.startswith("ojo") else "roya"

# ---------------- reference grades (recomputed; checked against r4_panel) ----------------
PI = list(csv.DictReader(open(f"{REA}/out/per_image.csv")))
PR = list(csv.DictReader(open(f"{REA}/out/per_image_retrained.csv")))
ref_face = {(r["leaf_id"], r["face"]): float(r["ref_sev"]) for r in PI if r["arch"] == "segformer" and r["model_disease"] == r["image_pack"]}
visual = {r["leaf_id"]: r["grade"] for r in csv.DictReader(open(f"{BASE}/datasets/pg26038-ojo-real-pack/meta/split.csv"))}


def healthy_mask_empty(lid):
    for d in ("pg26038-roya-real-pack", "pg26038-ojo-lesions"):
        ok = []
        for face in ("abaxial", "adaxial"):
            ip, mp = f"{BASE}/datasets/{d}/images/{lid}__{face}.png", f"{BASE}/datasets/{d}/masks/{lid}__{face}.png"
            if not (os.path.exists(ip) and os.path.exists(mp)): break
            a = np.asarray(Image.open(ip).convert("RGBA"))[..., 3] > 8
            m = (np.asarray(Image.open(mp).convert("L")) > 127) & a
            ok.append(int(m.sum()) == 0)
        if len(ok) == 2: return all(ok)
    return None


REF = {}
for f, lid in LEAF.items():
    if "Healthy" in lid:
        assert healthy_mask_empty(lid)
        REF[f] = dict(mask=0, visual=0, sev=0.0, sev_ab=0.0, sev_ad=0.0)
    elif dis_of(f) == "roya":
        ab, ad = ref_face[(lid, "abaxial")], ref_face[(lid, "adaxial")]
        REF[f] = dict(mask=grade(ab), visual=None, sev=ab, sev_ab=ab, sev_ad=ad)
    else:
        ab, ad = ref_face[(lid, "abaxial")], ref_face[(lid, "adaxial")]
        REF[f] = dict(mask=grade((ab + ad) / 2), visual=int(visual[lid][1]), sev=(ab + ad) / 2, sev_ab=ab, sev_ad=ad)
R4 = json.load(open(f"{REA}/results/r4_panel/results.json", encoding="utf-8"))
assert all(REF[f]["mask"] == R4["reference_grades"][f]["mask"] and REF[f]["visual"] == R4["reference_grades"][f]["visual"] for f in REF)

# ---------------- model grades on the form images ----------------
MP = {r["leaf_id"]: r for r in csv.DictReader(open(f"{VAL}/model_predictions.csv"))}
FORM, RUNINFO = {}, {}
for r in csv.DictReader(open(f"{REA}/results/r4_panel/form_predictions.csv")):
    FORM.setdefault(r["run"], {})[r["form_id"]] = (float(r["sev_ab"]), float(r["sev_ad"]))
    RUNINFO[r["run"]] = dict(kind=r["kind"], disease=r["disease"], arch=r["arch"], seed=int(r["seed"]) if r["seed"] != "" else None)
MRUN = {(v["disease"], v["arch"], v["seed"]): k for k, v in RUNINFO.items() if v["kind"] == "matched"}
assert len(MRUN) == 12
orig_check = {}
for run, v in RUNINFO.items():
    if v["kind"] != "published": continue
    a = v["arch"]
    orig_check[run] = max(max(abs(ab - float(MP[f][f"{a}_sev_ab"])), abs(ad - float(MP[f][f"{a}_sev_ad"]))) for f, (ab, ad) in FORM[run].items())
assert max(orig_check.values()) < 1e-9


def g_rule(ab, ad, f, rule):
    if f.startswith("roya") and rule == "paper": return grade(ab)
    return grade((ab + ad) / 2)


def rater_grades(fids, rule="paper"):
    G = {}
    for a in ANAME:
        G[f"pub_{a}"] = np.array([g_rule(float(MP[f][f"{a}_sev_ab"]), float(MP[f][f"{a}_sev_ad"]), f, rule) for f in fids])
    for a in ("deeplab", "unetpp"):
        for s in (0, 1, 2):
            G[f"m_{a}_s{s}"] = np.array([g_rule(*FORM[MRUN[(dis_of(f), a, s)]][f], f, rule) for f in fids])
    G["ref_mask"] = np.array([REF[f]["mask"] for f in fids])
    if all(f.startswith("ojo") for f in fids): G["ref_visual"] = np.array([REF[f]["visual"] for f in fids])
    return G


def leafset(name):
    if name in DIS: return [f for f in ITEMS if f.startswith(name) and f not in TRAIN]
    if name == "pooled53": return [f for f in ITEMS if f not in TRAIN]
    if name in ("ojo_all60", "roya_all60"): return [f for f in ITEMS if f.startswith(name[:-6])]
    if name == "pooled60": return list(ITEMS)
    raise KeyError(name)


# ---------------- statistics ----------------
def qwk_v(a, b):
    """Quadratic-weighted kappa along the last axis (identical to final_analysis.qwk; asserted)."""
    a = np.asarray(a, float); b = np.asarray(b, float); n = a.shape[-1]
    num = ((a - b) ** 2).sum(-1)
    den = n * ((a * a).sum(-1) + (b * b).sum(-1)) - 2 * a.sum(-1) * b.sum(-1)     # exact for integer grades
    with np.errstate(all="ignore"):
        return np.where(den > 0, 1 - n * num / np.where(den > 0, den, 1), np.nan)


_BI = {}


def boot_idx(n):
    """10,000 resample index vectors drawn exactly as results/r4_panel (fresh generator, sequential draws)."""
    if n not in _BI:
        rng = np.random.default_rng(SEED); _BI[n] = np.stack([rng.integers(0, n, n) for _ in range(B)])
    return _BI[n]


def pval(x):
    x = np.asarray(x); x = x[np.isfinite(x)]; n = len(x)
    return min(1.0, 2 * min(int((x <= 0).sum()), int((x >= 0).sum())) / n)


def summarize(pt, bt):
    x = np.asarray(bt); x = x[np.isfinite(x)]
    return dict(estimate=float(pt), ci95=[float(q) for q in np.percentile(x, [2.5, 97.5])], p_two_sided=pval(x), n_valid=int(len(x)))


def holm(p):
    items = sorted(p.items(), key=lambda kv: kv[1]); m = len(items); out = {}; run = 0.0
    for i, (k, v) in enumerate(items):
        run = max(run, min(1.0, (m - i) * v)); out[k] = run
    return out


def design_pairs(M, kind):
    """List of (evaluator row, reference vector) pairs."""
    R = M.shape[0]; pairs = []
    if kind == "loo3":
        for i in range(R):
            for S in itertools.combinations([j for j in range(R) if j != i], 3):
                med = np.median(M[list(S)], axis=0); assert (med == np.round(med)).all(); pairs.append((i, med.astype(int), S))
    elif kind in ("rest_up", "rest_down"):
        for i in range(R):
            med = np.median(np.delete(M, i, 0), axis=0)
            ref = (med + 0.5).astype(int) if kind == "rest_up" else np.ceil(med - 0.5).astype(int)
            pairs.append((i, ref, tuple(j for j in range(R) if j != i)))
    elif kind == "all_median":
        med = np.median(M, axis=0); assert R % 2 == 1 and (med == np.round(med)).all()
        pairs = [(i, med.astype(int), tuple(range(R))) for i in range(R)]
    else:
        raise KeyError(kind)
    return pairs


SECONDARY = ("pub_deeplab", "pub_unetpp", "mmean_deeplab", "mmean_unetpp")


def run_design(M, G, kind, idx, labels):
    pairs = design_pairs(M, kind); P = len(pairs)
    evi = np.array([p[0] for p in pairs]); refs = np.array([p[1] for p in pairs])
    ev_pt = qwk_v(M[evi], refs); ev_bt = qwk_v(M[evi][:, idx], refs[:, idx])          # P ; P x B
    for (i, r, _S) in pairs: assert abs(qwk_v(M[i], r) - qwk_ref(M[i], r)) < 1e-12 or not np.isfinite(qwk_v(M[i], r))
    E_pt, E_bt = ev_pt.mean(), ev_bt.mean(0)
    per_ev = {labels[i]: dict(loo_qwk=float(ev_pt[evi == i].mean()), **{k: v for k, v in summarize(ev_pt[evi == i].mean(), ev_bt[evi == i].mean(0)).items() if k in ("ci95", "n_valid")},
                              exact_vs_refs=float(np.mean([(M[i] == r).mean() for (j, r, _S) in pairs if j == i])),
                              n_references=int((evi == i).sum())) for i in sorted(set(evi.tolist()))}
    Rpt, Rbt, Rper = {}, {}, {}
    for k, g in G.items():
        v_pt = qwk_v(np.broadcast_to(g, refs.shape), refs); v_bt = qwk_v(np.broadcast_to(g[idx], refs[:, idx].shape), refs[:, idx])
        Rpt[k], Rbt[k] = float(v_pt.mean()), v_bt.mean(0)
        Rper[k] = {labels[i]: float(v_pt[evi == i].mean()) for i in sorted(set(evi.tolist()))}
    for a in ("deeplab", "unetpp"):
        ks = [f"m_{a}_s{s}" for s in (0, 1, 2)]
        Rpt[f"mmean_{a}"] = float(np.mean([Rpt[k] for k in ks])); Rbt[f"mmean_{a}"] = np.mean([Rbt[k] for k in ks], axis=0)
    C = {}
    for k in Rpt:
        C[k] = summarize(Rpt[k] - E_pt, Rbt[k] - E_bt)
    for a in ("deeplab", "unetpp"):
        C[f"mmean_{a}_minus_pub_{a}"] = summarize(Rpt[f"mmean_{a}"] - Rpt[f"pub_{a}"], Rbt[f"mmean_{a}"] - Rbt[f"pub_{a}"])
        C[f"mmean_{a}_minus_pub_segformer"] = summarize(Rpt[f"mmean_{a}"] - Rpt["pub_segformer"], Rbt[f"mmean_{a}"] - Rbt["pub_segformer"])
    for r in [k for k in G if k.startswith("ref_")]:
        C[f"pub_segformer_minus_{r}"] = summarize(Rpt["pub_segformer"] - Rpt[r], Rbt["pub_segformer"] - Rbt[r])
    if "ref_visual" in G:
        C["ref_mask_minus_ref_visual"] = summarize(Rpt["ref_mask"] - Rpt["ref_visual"], Rbt["ref_mask"] - Rbt["ref_visual"])
    rng_ev = [min(v["loo_qwk"] for v in per_ev.values()), max(v["loo_qwk"] for v in per_ev.values())]
    raters = {}
    for k in Rpt:
        rb = np.asarray(Rbt[k]); rb = rb[np.isfinite(rb)]
        raters[k] = dict(value=Rpt[k], ci95=[float(q) for q in np.percentile(rb, [2.5, 97.5])],
                         inside_evaluator_range=bool(rng_ev[0] <= Rpt[k] <= rng_ev[1]), above_evaluator_max=bool(Rpt[k] > rng_ev[1]),
                         n_evaluators_below=int(sum(v["loo_qwk"] < Rpt[k] for v in per_ev.values())))
        if k in Rper:
            raters[k]["per_evaluator_refs"] = Rper[k]
            raters[k]["minus_each_evaluator"] = {e: Rper[k][e] - per_ev[e]["loo_qwk"] for e in Rper[k]}
            raters[k]["exact_vs_refs"] = float(np.mean([(G[k] == r).mean() for (_i, r, _S) in pairs]))
            dd = np.concatenate([G[k] - r for (_i, r, _S) in pairs])
            raters[k]["diff_counts_vs_refs"] = {str(v): int((dd == v).sum()) for v in range(-3, 4)}   # rater minus reference, over all P x n comparisons
            raters[k]["max_abs_diff_vs_refs"] = int(np.abs(dd).max())
    for a in ("deeplab", "unetpp"):
        vals = np.array([Rpt[f"m_{a}_s{s}"] for s in (0, 1, 2)])
        raters[f"mmean_{a}"].update(seed_values=vals.tolist(), seed_sd=float(vals.std(ddof=1)), seed_range=[float(vals.min()), float(vals.max())])
    return dict(kind=kind, n_reference_sets=P, reference_sets=[dict(evaluator=labels[i], subset=[labels[j] for j in S]) for (i, _r, S) in pairs],
                evaluators=dict(value=float(E_pt), ci95=[float(q) for q in np.percentile(E_bt[np.isfinite(E_bt)], [2.5, 97.5])],
                                range=rng_ev, per_evaluator=per_ev,
                                exact_vs_refs=float(np.mean([(M[i] == r).mean() for (i, r, _S) in pairs])),
                                diff_counts_vs_refs={str(v): int((np.concatenate([M[i] - r for (i, r, _S) in pairs]) == v).sum()) for v in range(-3, 4)}),
                raters=raters, contrasts=C)


def add_holm(dres, tag):
    """Holm over the 8 secondary contrasts (both diseases) of one design."""
    P = {f"{d}:{k}": dres[d]["contrasts"][k]["p_two_sided"] for d in DIS for k in SECONDARY}
    h = holm(P)
    for key, v in h.items():
        d, k = key.split(":"); dres[d]["contrasts"][k][f"p_holm_secondary8"] = v
    return sorted(P)


def descriptives(g, M):
    med = np.median(M, axis=0); lo, hi = M.min(0), M.max(0)
    out = dict(pairwise_qwk_mean=float(np.mean([qwk_v(g, M[j]) for j in range(M.shape[0])])),
               inside_panel_range=int(((g >= lo) & (g <= hi)).sum()), below_min=int((g < lo).sum()), above_max=int((g > hi).sum()), n=int(len(g)))
    if M.shape[0] % 2 == 1:
        m5 = med.astype(int)
        out.update(qwk_median5=float(qwk_v(g, m5)), exact_median5=int((g == m5).sum()), within1_median5=int((np.abs(g - m5) <= 1).sum()),
                   bias_median5=float((g - m5).mean()), panel_G0=int((m5 == 0).sum()), panel_G0_rater_above=int(((m5 == 0) & (g > 0)).sum()),
                   higher_than_median5=int((g > m5).sum()), lower_than_median5=int((g < m5).sum()))
    return out


# ---------------- inter-rater agreement (vectorised over bootstrap resamples) ----------------
def unit_counts(M):  # n x K
    return np.stack([(M == k).sum(0) for k in range(K)], axis=1)


def kripp_v(M, idx):
    R, n = M.shape; C = unit_counts(M).astype(float)
    Pu = np.einsum("uc,uk->uck", C, C) - np.einsum("uc,ck->uck", C, np.eye(K))           # n x K x K
    w = np.bincount((idx + n * np.arange(len(idx))[:, None]).ravel(), minlength=len(idx) * n).reshape(len(idx), n).astype(float)
    O = (w @ Pu.reshape(n, K * K)).reshape(-1, K, K) / (R - 1)
    nc = O.sum(2); N = nc.sum(1); cum = np.cumsum(nc, 1)
    D = np.zeros_like(O)
    for c in range(K):
        for k in range(K):
            lo_, hi_ = min(c, k), max(c, k)
            s = cum[:, hi_] - (cum[:, lo_ - 1] if lo_ > 0 else 0)
            D[:, c, k] = (s - (nc[:, c] + nc[:, k]) / 2) ** 2
    Do = (O * D).sum((1, 2)) / N; De = (np.einsum("bc,bk->bck", nc, nc) * D).sum((1, 2)) / (N * (N - 1))
    return 1 - Do / De


def icc_v(M, idx):
    X = M.T.astype(float)[idx]; n, k = X.shape[1], X.shape[2]            # B x n x k
    m = X.mean((1, 2), keepdims=True); rm = X.mean(2, keepdims=True); cm = X.mean(1, keepdims=True)
    MSR = k * ((rm - m) ** 2).sum((1, 2)) / (n - 1); MSC = n * ((cm - m) ** 2).sum((1, 2)) / (k - 1)
    MSE = ((X - rm - cm + m) ** 2).sum((1, 2)) / ((n - 1) * (k - 1))
    return (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n)


def fleiss_v(M, idx):
    R = M.shape[0]; C = unit_counts(M).astype(float)[idx]                  # B x n x K
    Pi = ((C ** 2).sum(2) - R) / (R * (R - 1)); p = C.sum(1) / (C.shape[1] * R); Pe = (p ** 2).sum(1)
    return (Pi.mean(1) - Pe) / (1 - Pe)


def irr(M, idx, labels):
    R, n = M.shape; ident = np.arange(n)[None, :]
    pairs = list(itertools.combinations(range(R), 2))
    def pw(idx_):
        q = np.stack([qwk_v(M[a][idx_], M[b][idx_]) for a, b in pairs]); ex = np.stack([(M[a][idx_] == M[b][idx_]).mean(-1) for a, b in pairs])
        w1 = np.stack([(np.abs(M[a][idx_] - M[b][idx_]) <= 1).mean(-1) for a, b in pairs]); return q, ex, w1
    q_pt, ex_pt, w1_pt = (x[:, 0] for x in pw(ident)); q_bt, ex_bt, w1_bt = pw(idx)
    stats = dict(alpha=(kripp_v(M, ident)[0], kripp_v(M, idx)), icc21=(icc_v(M, ident)[0], icc_v(M, idx)),
                 fleiss=(fleiss_v(M, ident)[0], fleiss_v(M, idx)), pairwise_qwk_mean=(q_pt.mean(), q_bt.mean(0)),
                 pairwise_exact_mean=(ex_pt.mean(), ex_bt.mean(0)), pairwise_within1_mean=(w1_pt.mean(), w1_bt.mean(0)))
    assert abs(stats["alpha"][0] - kripp_ref(M)) < 1e-9 and abs(stats["icc21"][0] - icc_ref(M)) < 1e-9 and abs(stats["fleiss"][0] - fleiss_ref(M)) < 1e-9
    out = {k: dict(estimate=float(v[0]), ci95=[float(x) for x in np.percentile(v[1][np.isfinite(v[1])], [2.5, 97.5])], n_valid=int(np.isfinite(v[1]).sum()))
           for k, v in stats.items()}
    maj = [int(np.bincount(M[:, i], minlength=K).max()) for i in range(n)]
    out.update(n=n, n_raters=R, unanimous=int(sum(len(set(M[:, i])) == 1 for i in range(n))),
               all_within_one=int(sum(np.ptp(M[:, i]) <= 1 for i in range(n))),
               max_spread_leaves=[dict(item=i, grades=sorted(M[:, i].tolist())) for i in range(n) if np.ptp(M[:, i]) > 1],
               largest_agreeing_group_counts=dict(sorted(Counter(maj).items())),
               pairwise_qwk_each={f"{labels[a]}-{labels[b]}": float(q_pt[p]) for p, (a, b) in enumerate(pairs)},
               pairwise_qwk_range=[float(q_pt.min()), float(q_pt.max())],
               mean_grade_each={labels[j]: float(M[j].mean()) for j in range(R)},
               grade_counts_each={labels[j]: [int((M[j] == k).sum()) for k in range(K)] for j in range(R)})
    if R % 2 == 1: out["median_grade_counts"] = [int((np.median(M, 0).astype(int) == k).sum()) for k in range(K)]
    if R == 4: out["two_against_two"] = int(sum(sorted(np.bincount(M[:, i], minlength=K).tolist())[-2:] == [2, 2] for i in range(n)))
    return out


# =====================================================================================================
res = dict(meta=dict(script=os.path.abspath(__file__), B=B, seed=SEED, structure_checks=STRUCT,
                     evaluator_labels="E1-E5 = form-submission order; E1-E4 identical to results/r4_panel; no names, timestamps or affiliations written",
                     excluded_in_c="E5 (the fifth evaluator); (c) = the four-evaluator panel E1-E4 of results/r4_panel",
                     primary_set="60 form leaves minus 7 rust-training healthy leaves (30 leaf spot, 23 rust)", train_leaves_excluded=TRAIN,
                     rust_rule="abaxial face", leaf_spot_rule="mean of both faces", cut_points=[0.5, 5, 20],
                     statistic_primary="mean over the 20 (i, S) of QWK(model, median S) minus mean over the same 20 of QWK(evaluator i, median S)",
                     p_value="two-sided percentile-bootstrap inversion", inference="leaves resampled, evaluators and seeds fixed",
                     original_form_predictions_equal_r4_panel=orig_check),
           irr={}, primary={}, sens_a_up={}, sens_a_down={}, sens_b_all5={}, sens_c_four_E1_E4={}, drop_one={}, extra={},
           descriptives={}, reference={}, model_disagreement={})

# ---------- 1. inter-rater agreement ----------
for setname, fids_fn in (("all60", lambda d: [f for f in ITEMS if d == "all" or f.startswith(d)]),
                         ("primary53", lambda d: [f for f in ITEMS if (d == "all" or f.startswith(d)) and f not in TRAIN])):
    for d in ("ojo", "roya", "all"):
        fids = fids_fn(d); M = GR[:, [ITIDX[f] for f in fids]]; idx = boot_idx(len(fids))
        r5 = irr(M, idx, EL); r4 = irr(M[:4], idx, EL[:4])
        for rr in (r5, r4):
            for x in rr["max_spread_leaves"]: x["item"] = fids[x["item"]]
        res["irr"][f"{setname}_{d}"] = dict(five=r5, four_E1_E4=r4)
    print("irr", setname, round(time.time() - t0), "s", flush=True)

# 4-evaluator IRR must reproduce validation/final_results.json (all 60)
FR = json.load(open(f"{VAL}/final_results.json"))
for d in ("ojo", "roya", "all"):
    r4 = res["irr"][f"all60_{d}"]["four_E1_E4"]; fr = FR[f"irr_{d}"]
    assert abs(round(r4["alpha"]["estimate"], 4) - fr["alpha"]) < 1e-4 and abs(round(r4["icc21"]["estimate"], 4) - fr["icc"]) < 1e-4
    assert abs(round(r4["fleiss"]["estimate"], 4) - fr["fleiss"]) < 1e-4 and r4["unanimous"] == fr["unanimous"] and r4["all_within_one"] == fr["within1"]

# ---------- 2-3. model-panel comparisons ----------
DESIGNS = dict(primary=("loo3", slice(0, 5)), sens_a_up=("rest_up", slice(0, 5)), sens_a_down=("rest_down", slice(0, 5)),
               sens_b_all5=("all_median", slice(0, 5)), sens_c_four_E1_E4=("loo3", [0, 1, 2, 3]))
assert FIFTH == 4
for d in DIS:
    fids = leafset(d); M5 = GR[:, [ITIDX[f] for f in fids]]; G = rater_grades(fids); idx = boot_idx(len(fids))
    for tag, (kind, rows) in DESIGNS.items():
        M = M5[rows]; labels = [EL[i] for i in (range(5) if isinstance(rows, slice) else rows)]
        res[tag][d] = dict(disease=DNAME[d], n=len(fids), leaves=fids, **run_design(M, G, kind, idx, labels))
    for k in range(5):
        keep = [j for j in range(5) if j != k]
        res["drop_one"].setdefault(f"drop_{EL[k]}", {})[d] = dict(n=len(fids), **run_design(M5[keep], G, "loo3", idx, [EL[j] for j in keep]))
    res["descriptives"][d] = dict(n=len(fids), evaluators_pairwise_qwk_mean=float(np.mean([qwk_v(M5[a], M5[b]) for a, b in itertools.combinations(range(5), 2)])),
                                  median5_grade_counts=[int((np.median(M5, 0).astype(int) == k).sum()) for k in range(K)],
                                  raters={k: descriptives(g, M5) for k, g in G.items()},
                                  evaluators_vs_median5={EL[j]: dict(qwk=float(qwk_v(M5[j], np.median(M5, 0).astype(int))),
                                                                     exact=int((M5[j] == np.median(M5, 0).astype(int)).sum())) for j in range(5)})
    for a in ("deeplab", "unetpp"):
        ks = [f"m_{a}_s{s}" for s in (0, 1, 2)]
        res["descriptives"][d][f"mmean_{a}"] = {m: [float(np.mean([res["descriptives"][d]["raters"][k][m] for k in ks])),
                                                    float(np.std([res["descriptives"][d]["raters"][k][m] for k in ks], ddof=1)),
                                                    float(min(res["descriptives"][d]["raters"][k][m] for k in ks)),
                                                    float(max(res["descriptives"][d]["raters"][k][m] for k in ks))]
                                                for m in ("pairwise_qwk_mean", "qwk_median5", "inside_panel_range", "exact_median5", "bias_median5", "panel_G0_rater_above")}
    # where the model runs differ (design-independent), with the 5 panel grades
    mk = [k for k in G if k.startswith(("pub_", "m_"))]; Gm = np.array([G[k] for k in mk])
    res["model_disagreement"][d] = dict(n_leaves=int(sum(len(set(Gm[:, i])) > 1 for i in range(len(fids)))),
                                        leaves=[dict(form_id=fids[i], panel_E1_E5=M5[:, i].tolist(), median5=int(np.median(M5[:, i])),
                                                     ref_mask=int(G["ref_mask"][i]), model_grades={k: int(G[k][i]) for k in mk})
                                                for i in range(len(fids)) if len(set(Gm[:, i])) > 1])
    print("designs", d, round(time.time() - t0), "s", flush=True)
for tag in list(DESIGNS) + [f"drop_{e}" for e in EL]:
    target = res[tag] if tag in DESIGNS else res["drop_one"][tag]
    target["holm_family"] = add_holm(target, tag)

# (c) must reproduce results/r4_panel exactly (same resamples)
repro = {}
for d in DIS:
    for k_new, k_old in (("pub_segformer", "pub_segformer"), ("pub_deeplab", "pub_deeplab"), ("pub_unetpp", "pub_unetpp"),
                         ("mmean_deeplab", "mmean_deeplab"), ("mmean_unetpp", "mmean_unetpp"), ("ref_mask", "ref_mask"),
                         ("ref_visual", "ref_visual"), ("ref_mask_minus_ref_visual", None), ("pub_segformer_minus_ref_mask", "pub_segformer_minus_ref_mask")):
        new = res["sens_c_four_E1_E4"][d]["contrasts"].get(k_new)
        if new is None: continue
        old = R4["part1"][d]["contrasts"].get(k_old) if k_old else None
        if old is None: continue
        repro[f"{d}:{k_new}"] = dict(new=[new["estimate"], *new["ci95"], new["p_two_sided"], new.get("p_holm_secondary8")],
                                     r4_panel=[old["estimate"], *old["ci95"], old["p_two_sided"], old.get("p_holm_secondary8")],
                                     identical=bool(np.allclose([new["estimate"], *new["ci95"], new["p_two_sided"]], [old["estimate"], *old["ci95"], old["p_two_sided"]], atol=1e-12)
                                                    and (old.get("p_holm_secondary8") is None or abs(new["p_holm_secondary8"] - old["p_holm_secondary8"]) < 1e-12)))
    repro[f"{d}:evaluators"] = dict(new=res["sens_c_four_E1_E4"][d]["evaluators"]["value"], r4_panel=R4["part1"][d]["evaluators"]["loo_qwk_mean"],
                                    identical=abs(res["sens_c_four_E1_E4"][d]["evaluators"]["value"] - R4["part1"][d]["evaluators"]["loo_qwk_mean"]) < 1e-12)
res["sens_c_four_E1_E4"]["reproduction_of_r4_panel"] = repro
assert all(v["identical"] for v in repro.values()), {k: v for k, v in repro.items() if not v["identical"]}

# ---------- extra sensitivities (primary 5-evaluator design): pooled 53, all 60 leaves, rust both faces ----------
for name, rule in (("pooled53", "paper"), ("ojo_all60", "paper"), ("roya_all60", "paper"), ("pooled60", "paper"), ("roya", "both_faces")):
    fids = leafset(name); M5 = GR[:, [ITIDX[f] for f in fids]]; G = rater_grades(fids, "paper" if rule == "paper" else "mean"); idx = boot_idx(len(fids))
    tag = name if rule == "paper" else "roya_bothfaces"
    res["extra"][tag] = dict(n=len(fids), five=run_design(M5, G, "loo3", idx, EL), four_E1_E4=run_design(M5[:4], G, "loo3", idx, EL[:4]))
# 4-evaluator values must reproduce validation/final_results.json (point estimates, 3 decimals)
for tag, frk in (("pooled60", "all60_all"), ("roya_all60", "all60_roya"), ("roya_bothfaces", "bothfaces_roya"), ("pooled53", "primary_all")):
    x = res["extra"][tag]["four_E1_E4"]
    assert abs(round(x["evaluators"]["value"], 3) - FR[frk]["expert_fair"]) < 1e-9, tag
    for a in ANAME: assert abs(round(x["raters"][f"pub_{a}"]["value"], 3) - FR[frk][a]["fair"]) < 1e-9, (tag, a)
print("extra", round(time.time() - t0), "s", flush=True)

# ---------- 4. reference as a rater: per-evaluator agreement, unanimous disagreements, scan grades ----------
scan = {}
for r in PI:
    if r["model_disease"] == r["image_pack"]:
        scan.setdefault((f"orig_{r['arch']}", r["leaf_id"]), {})[r["face"]] = float(r["pred_sev"])
for r in PR:
    m = re.match(r"pg26038-mbs?-(ojo|roya)-(deeplab|unetpp)-lrB(?:-seed(\d))?$", r["run"])
    if m: scan.setdefault((f"matched_{m.group(2)}_s{m.group(3) or 0}", r["leaf_id"]), {})[r["face"]] = float(r["pred_sev"])


def scan_grade(key, f):
    s = scan.get((key, LEAF[f]))
    if s is None: return None
    v = s["abaxial"] if f.startswith("roya") else (s["abaxial"] + s["adaxial"]) / 2
    return dict(sev=round(v, 2), grade=grade(v))


SCANKEYS = [f"orig_{a}" for a in ANAME] + [f"matched_{a}_s{s}" for a in ("deeplab", "unetpp") for s in (0, 1, 2)]
for d in DIS:
    for setname in ("primary", "all60"):
        fids = leafset(d) if setname == "primary" else leafset(f"{d}_all60")
        M5 = GR[:, [ITIDX[f] for f in fids]]; G = rater_grades(fids); n = len(fids); idx = boot_idx(n)
        out = {}
        for rk in [k for k in G if k.startswith("ref_")]:
            ref = G[rk]; per = {}
            for j, e in enumerate(EL):
                qb = qwk_v(M5[j][idx], ref[idx])
                per[e] = dict(n=n, exact=int((M5[j] == ref).sum()), within1=int((np.abs(M5[j] - ref) <= 1).sum()),
                              higher=int((M5[j] > ref).sum()), lower=int((M5[j] < ref).sum()), mean_diff=float((M5[j] - ref).mean()),
                              qwk=float(qwk_v(M5[j], ref)), qwk_ci95=[float(q) for q in np.percentile(qb[np.isfinite(qb)], [2.5, 97.5])])
            m5 = np.median(M5, 0).astype(int)
            def unan(Mx, need):
                rows_ = []
                for i in range(n):
                    dis_n = int((Mx[:, i] != ref[i]).sum())
                    if dis_n >= need:
                        f = fids[i]
                        rows_.append(dict(form_id=f, leaf_id=LEAF[f], ref_grade=int(ref[i]), ref_sev=round(REF[f]["sev"], 2),
                                          ref_sev_ab=round(REF[f]["sev_ab"], 2), ref_sev_ad=round(REF[f]["sev_ad"], 2),
                                          panel=Mx[:, i].tolist(), n_disagree=dis_n,
                                          direction="all higher" if (Mx[:, i] > ref[i]).all() else "all lower" if (Mx[:, i] < ref[i]).all() else "mixed",
                                          in_primary=f not in TRAIN,
                                          form_grades={k: int(G[k][i]) for k in G if not k.startswith("ref_")},
                                          scan={k: scan_grade(k, f) for k in SCANKEYS}))
                return rows_
            out[rk] = dict(n=n, per_evaluator=per,
                           median5=dict(exact=int((m5 == ref).sum()), within1=int((np.abs(m5 - ref) <= 1).sum()), qwk=float(qwk_v(m5, ref)),
                                        higher=int((m5 > ref).sum()), lower=int((m5 < ref).sum())),
                           all_five_disagree=unan(M5, 5), at_least_four_of_five_disagree=unan(M5, 4),
                           all_four_E1_E4_disagree=[x["form_id"] for x in unan(M5[:4], 4)],
                           ref_grade_counts=[int((ref == k).sum()) for k in range(K)])
            if setname == "primary":   # E1-E4 per-evaluator values must equal r4_panel Table 2b
                old = R4["part2"][f"{d}_primary"][rk]["per_evaluator"]
                for e in EL[:4]:
                    assert old[e]["exact"] == per[e]["exact"] and abs(old[e]["qwk"] - per[e]["qwk"]) < 1e-12
                    assert np.allclose(old[e]["qwk_ci95"], per[e]["qwk_ci95"], atol=1e-12)
        res["reference"][f"{d}_{setname}"] = out
res["meta"]["runtime_s"] = round(time.time() - t0, 1)


def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.ndarray): return clean(o.tolist())
    return o


json.dump(clean(res), open(f"{OUT}/results.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("done", res["meta"]["runtime_s"], "s")
