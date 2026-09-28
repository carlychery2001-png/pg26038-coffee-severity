"""Final expert-panel validation numbers for the thesis.
Model grade follows the thesis rule: rust = abaxial face only; American leaf spot = mean of both faces.
Primary set = leaves never used to train the deployed models (53); sensitivity = all 60 and rust both-face mean."""
import csv, itertools, json
from collections import Counter
import numpy as np
K = 4; B = 10000
E = list(csv.DictReader(open("expert_consensus.csv", encoding="utf-8")))
P = {r["leaf_id"]: r for r in csv.DictReader(open("model_predictions.csv"))}
KEY = {r["form_id"]: r for r in csv.DictReader(open("form_image_key.csv"))}
EXP = [k for k in E[0] if k not in ("leaf_id", "consensus_grade", "range")]
ARCH = ["segformer", "deeplab", "unetpp"]
grade = lambda s: 0 if s < 0.5 else 1 if s < 5 else 2 if s < 20 else 3

def model_grade(lid, arch, rust_rule="abaxial"):
    p = P[lid]
    if lid.startswith("roya") and rust_rule == "abaxial": return grade(float(p[f"{arch}_sev_ab"]))
    return grade((float(p[f"{arch}_sev_ab"]) + float(p[f"{arch}_sev_ad"])) / 2)

def qwk(a, b):
    O = np.zeros((K, K))
    for x, y in zip(a, b): O[x, y] += 1
    Ex = np.outer(O.sum(1), O.sum(0)) / O.sum()
    W = (np.subtract.outer(np.arange(K), np.arange(K)) ** 2) / (K - 1) ** 2
    return 1 - (W * O).sum() / (W * Ex).sum()

def ci(a, b, seed=1):
    rng = np.random.default_rng(seed); a, b = np.asarray(a), np.asarray(b); v = []
    for _ in range(B):
        i = rng.integers(0, len(a), len(a))
        if len(set(a[i]) | set(b[i])) > 1: v.append(qwk(a[i], b[i]))
    return [round(x, 4) for x in np.percentile(v, [2.5, 97.5])]

def kripp(M):   # M: raters x items, ordinal
    vals = M.T.tolist(); nk = Counter(v for vs in vals for v in vs); n = sum(nk.values()); R = M.shape[0]
    d = lambda c, k: (sum(nk[g] for g in range(min(c, k), max(c, k) + 1)) - (nk[c] + nk[k]) / 2) ** 2
    Do = sum(d(a, b) for vs in vals for a, b in itertools.permutations(vs, 2)) / (R - 1) / n
    De = sum(nk[c] * nk[k] * d(c, k) for c in range(K) for k in range(K)) / (n * (n - 1))
    return 1 - Do / De

def icc21(M):
    X = M.T.astype(float); n, k = X.shape; m = X.mean()
    MSR = k * ((X.mean(1) - m) ** 2).sum() / (n - 1); MSC = n * ((X.mean(0) - m) ** 2).sum() / (k - 1)
    MSE = ((X - X.mean(1, keepdims=True) - X.mean(0) + m) ** 2).sum() / ((n - 1) * (k - 1))
    return (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n)

def fleiss(M):
    R = M.shape[0]; cnt = [Counter(M[:, i]) for i in range(M.shape[1])]
    Pi = [(sum(c[k] ** 2 for k in range(K)) - R) / (R * (R - 1)) for c in cnt]
    p = [sum(c[k] for c in cnt) / (len(cnt) * R) for k in range(K)]; Pe = sum(x * x for x in p)
    return (np.mean(Pi) - Pe) / (1 - Pe)

res = {}
# ---- inter-rater agreement (all 60 leaves; independent of model training) ----
for dis in ("ojo", "roya", "all"):
    rows = [r for r in E if dis == "all" or r["leaf_id"].startswith(dis)]
    M = np.array([[int(r[e]) for r in rows] for e in EXP])
    pw = [qwk(M[a], M[b]) for a, b in itertools.combinations(range(len(EXP)), 2)]
    res[f"irr_{dis}"] = dict(n=len(rows), alpha=round(kripp(M), 4), icc=round(icc21(M), 4), fleiss=round(fleiss(M), 4),
        unanimous=int(sum(len(set(M[:, i])) == 1 for i in range(M.shape[1]))), within1=int(sum(np.ptp(M[:, i]) <= 1 for i in range(M.shape[1]))),
        pairwise_qwk=[round(min(pw), 4), round(max(pw), 4)], rater_mean=[round(float(x), 2) for x in M.mean(1)],
        consensus_dist=dict(sorted(Counter(int(r["consensus_grade"]) for r in rows).items())))

# ---- model vs panel ----
def evaluate(tag, keep, rust_rule):
    for dis in ("ojo", "roya", "all"):
        rows = [r for r in E if (dis == "all" or r["leaf_id"].startswith(dis)) and keep(r["leaf_id"])]
        cons = np.array([int(r["consensus_grade"]) for r in rows]); M = np.array([[int(r[e]) for r in rows] for e in EXP])
        loo_ref = [np.array([int(np.median(np.delete(M, j, 0)[:, i]) + 0.5) for i in range(len(rows))]) for j in range(len(EXP))]
        out = dict(n=len(rows), expert_fair=round(float(np.mean([qwk(M[j], loo_ref[j]) for j in range(len(EXP))])), 3),
                   expert_fair_range=[round(float(x), 4) for x in (min(qwk(M[j], loo_ref[j]) for j in range(len(EXP))), max(qwk(M[j], loo_ref[j]) for j in range(len(EXP))))])
        for a in ARCH:
            g = np.array([model_grade(r["leaf_id"], a, rust_rule) for r in rows])
            cm = np.zeros((K, K), int)
            for x, y in zip(cons, g): cm[x, y] += 1
            out[a] = dict(qwk=round(qwk(g, cons), 4), ci=ci(g, cons), acc=round(float((g == cons).mean()), 3),
                          within1=round(float((abs(g - cons) <= 1).mean()), 3),
                          in_range=round(float(np.mean((g >= M.min(0)) & (g <= M.max(0)))), 3),
                          bias=round(float((g - cons).mean()), 2),
                          fair=round(float(np.mean([qwk(g, loo_ref[j]) for j in range(len(EXP))])), 3),
                          per_expert=[round(qwk(g, M[j]), 4) for j in range(len(EXP))], confusion=cm.tolist())
        res[f"{tag}_{dis}"] = out

train = {f for f, r in KEY.items() if r["model_split"] == "train"}
evaluate("primary", lambda l: l not in train, "abaxial")
evaluate("all60", lambda l: True, "abaxial")
evaluate("bothfaces", lambda l: l not in train, "mean")
res["composition"] = Counter(f"{f[:3]}|{'healthy' if 'Healthy' in r['leaf_id'] else 'diseased'}|{r['model_split']}" for f, r in KEY.items())
res["train_leaves"] = sorted(train)
json.dump(res, open("final_results.json", "w"), indent=1)
for k, v in res.items():
    if isinstance(v, dict) and "segformer" in v:
        print(f"{k:16s} n={v['n']:2d} expert_fair={v['expert_fair']} " + " | ".join(f"{a}: qwk {v[a]['qwk']} {v[a]['ci']} acc {v[a]['acc']} fair {v[a]['fair']} inR {v[a]['in_range']} bias {v[a]['bias']}" for a in ARCH))
    else: print(k, v)
for dis in ("ojo", "roya"): print("SegFormer confusion", dis, res[f"primary_{dis}"]["segformer"]["confusion"], "| per-expert", res[f"primary_{dis}"]["segformer"]["per_expert"])
print("expert names order:", EXP)
