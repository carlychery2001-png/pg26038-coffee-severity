"""Independent cross-check of panel5_analysis.py (plain loops, the reference functions of validation/final_analysis.py,
separate parsing of the grade cells). Compares with results.json and writes crosscheck.json. No names are read or written:
only columns 8+ of the raw export are parsed."""
import os as _os  # (release) all paths below are relative to ROOT (repository root by default; see README)
ROOT = (_os.environ.get("PG26038_ROOT") or _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))).replace(chr(92), "/")
import csv, itertools, json, os
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__)); VAL = (ROOT + "/validation"); REA = (ROOT + "/reanalysis")
B = 10000; SEED = 20260928
ns = {}; _c = os.getcwd(); os.chdir(VAL); exec(open("final_analysis.py", encoding="utf-8").read().split("res = {}")[0], ns); os.chdir(_c)
qwk, kripp, icc21, fleiss = ns["qwk"], ns["kripp"], ns["icc21"], ns["fleiss"]
grade = lambda s: 0 if s < 0.5 else 1 if s < 5 else 2 if s < 20 else 3

rows = list(csv.reader(open(f"{VAL}/private/form_export_5experts_raw.csv", encoding="utf-8-sig")))
items = [h.split(")")[1].split("-")[0].strip() for h in rows[0][7:]]
G5 = np.array([[int(r[7 + k].strip()[1]) for k in range(60)] for r in rows[1:]]); del rows
key = {r["form_id"]: r for r in csv.DictReader(open(f"{VAL}/form_image_key.csv"))}
pred = {r["leaf_id"]: r for r in csv.DictReader(open(f"{VAL}/model_predictions.csv"))}
R = json.load(open(f"{OUT}/results.json", encoding="utf-8"))
ref4 = json.load(open(f"{REA}/results/r4_panel/results.json", encoding="utf-8"))["reference_grades"]
out = {}; ok = True


def mgrade(f, a):
    ab, ad = float(pred[f][f"{a}_sev_ab"]), float(pred[f][f"{a}_sev_ad"])
    return grade(ab) if f.startswith("roya") else grade((ab + ad) / 2)


def loo3(M, g):
    ev, mo = [], []
    for i in range(M.shape[0]):
        for S in itertools.combinations([j for j in range(M.shape[0]) if j != i], 3):
            ref = np.array([sorted(M[list(S), c])[1] for c in range(M.shape[1])])
            ev.append(qwk(M[i], ref)); mo.append(qwk(g, ref))
    return np.mean(mo), np.mean(ev)


def rest(M, g, up):
    ev, mo = [], []
    for i in range(M.shape[0]):
        o = np.delete(M, i, 0); med = np.median(o, 0)
        ref = np.floor(med + 0.5).astype(int) if up else np.ceil(med - 0.5).astype(int)
        ev.append(qwk(M[i], ref)); mo.append(qwk(g, ref))
    return np.mean(mo), np.mean(ev)


def close(a, b, tol=1e-9):
    global ok
    good = abs(a - b) <= tol; ok &= bool(good); return bool(good)


for d in ("ojo", "roya"):
    fl = [f for f in items if f.startswith(d) and key[f]["model_split"] != "train"]
    ix = [items.index(f) for f in fl]; M = G5[:, ix]; n = len(fl)
    g = np.array([mgrade(f, "segformer") for f in fl])
    rng = np.random.default_rng(SEED); draws = [rng.integers(0, n, n) for _ in range(B)]
    res = {}
    for tag, fn in (("primary", lambda MM, gg: loo3(MM, gg)), ("sens_a_up", lambda MM, gg: rest(MM, gg, True)),
                    ("sens_a_down", lambda MM, gg: rest(MM, gg, False))):
        m, e = fn(M, g); bs = []
        for idx in draws:
            mm, ee = fn(M[:, idx], g[idx]); bs.append(mm - ee)
        bs = np.array(bs); bs = bs[np.isfinite(bs)]
        lo, hi = np.percentile(bs, [2.5, 97.5]); p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
        c = R[tag][d]["contrasts"]["pub_segformer"]
        res[tag] = dict(model=m, evaluators=e, diff=m - e, ci=[lo, hi], p=p,
                        match=all([close(m - e, c["estimate"]), close(lo, c["ci95"][0]), close(hi, c["ci95"][1]), close(p, c["p_two_sided"])]))
    for a in ("deeplab", "unetpp"):
        ga = np.array([mgrade(f, a) for f in fl]); m, e = loo3(M, ga)
        res[f"primary_pub_{a}_point"] = dict(diff=m - e, match=close(m - e, R["primary"][d]["contrasts"][f"pub_{a}"]["estimate"]))
    med5 = np.median(M, 0).astype(int)
    res["sens_b_segformer_qwk_median5"] = dict(value=qwk(g, med5), match=close(qwk(g, med5), R["sens_b_all5"][d]["raters"]["pub_segformer"]["value"]))
    refm = np.array([ref4[f]["mask"] for f in fl])
    res["all_five_disagree_mask"] = [fl[i] for i in range(n) if (M[:, i] != refm[i]).all()]
    ok &= res["all_five_disagree_mask"] == [x["form_id"] for x in R["reference"][f"{d}_primary"]["ref_mask"]["all_five_disagree"]]
    res["E5_exact_with_mask_ref"] = int((M[4] == refm).sum())
    ok &= res["E5_exact_with_mask_ref"] == R["reference"][f"{d}_primary"]["ref_mask"]["per_evaluator"]["E5"]["exact"]
    out[d] = res
    print(d, {k: v.get("match", v) if isinstance(v, dict) else v for k, v in res.items()}, flush=True)

# IRR point values (all 60 leaves) and Krippendorff alpha / ICC CIs by loop
for d in ("ojo", "roya", "all"):
    ix = [k for k, f in enumerate(items) if d == "all" or f.startswith(d)]; M = G5[:, ix]; n = len(ix)
    rng = np.random.default_rng(SEED); a_bs, i_bs = [], []
    for _ in range(B):
        idx = rng.integers(0, n, n); a_bs.append(kripp(M[:, idx])); i_bs.append(icc21(M[:, idx]))
    x = R["irr"][f"all60_{d}"]["five"]
    aci, ici = np.percentile(a_bs, [2.5, 97.5]), np.percentile(i_bs, [2.5, 97.5])
    out[f"irr_{d}"] = dict(alpha=kripp(M), icc=icc21(M), fleiss=fleiss(M), alpha_ci=aci.tolist(), icc_ci=ici.tolist(),
                           match=all([close(kripp(M), x["alpha"]["estimate"]), close(icc21(M), x["icc21"]["estimate"]), close(fleiss(M), x["fleiss"]["estimate"]),
                                      close(aci[0], x["alpha"]["ci95"][0], 1e-8), close(aci[1], x["alpha"]["ci95"][1], 1e-8),
                                      close(ici[0], x["icc21"]["ci95"][0], 1e-8), close(ici[1], x["icc21"]["ci95"][1], 1e-8)]))
    print("irr", d, out[f"irr_{d}"]["match"], flush=True)
out["all_match"] = bool(ok)
json.dump(out, open(f"{OUT}/crosscheck.json", "w"), indent=1, default=float)
print("ALL MATCH:", ok)
