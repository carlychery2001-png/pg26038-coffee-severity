"""Model grades vs the 4-expert panel on the 60 form leaves."""
import csv, itertools
import numpy as np
E = list(csv.DictReader(open("expert_consensus.csv", encoding="utf-8")))
P = {r["leaf_id"]: r for r in csv.DictReader(open("model_predictions.csv"))}
experts = [k for k in E[0] if k not in ("leaf_id", "consensus_grade", "range")]
K = 4

def qwk(a, b):
    O = np.zeros((K, K))
    for x, y in zip(a, b): O[x, y] += 1
    Ex = np.outer(O.sum(1), O.sum(0)) / O.sum()
    W = np.array([[(i - j) ** 2 for j in range(K)] for i in range(K)]) / (K - 1) ** 2
    return 1 - (W * O).sum() / (W * Ex).sum()

def boot_ci(a, b, n=5000, seed=0):
    rng = np.random.default_rng(seed); a, b = np.array(a), np.array(b); v = []
    for _ in range(n):
        i = rng.integers(0, len(a), len(a))
        if len(set(a[i])) > 1 and len(set(b[i])) > 1: v.append(qwk(a[i], b[i]))
    return np.percentile(v, [2.5, 97.5])

out = []
for dis in ("ojo", "roya", "all"):
    rows = [r for r in E if dis == "all" or r["leaf_id"].startswith(dis)]
    cons = [int(r["consensus_grade"]) for r in rows]
    ex = {e: [int(r[e]) for r in rows] for e in experts}
    print(f"\n==================== {dis.upper()} (n={len(rows)}) ====================")
    # expert-vs-expert reference: each expert vs the median of the other three
    loo = []
    for e in experts:
        others = [np.median([ex[o][i] for o in experts if o != e]) for i in range(len(rows))]
        loo.append(qwk(ex[e], [int(x + 0.5) for x in others]))
    print(f"  reference: expert vs other-3 consensus  QWK mean {np.mean(loo):.3f} (range {min(loo):.3f}-{max(loo):.3f})")
    for arch in ("segformer", "deeplab", "unetpp"):
        g = [int(P[r["leaf_id"]][f"{arch}_grade"]) for r in rows]
        k = qwk(g, cons); lo, hi = boot_ci(g, cons)
        exact = np.mean(np.array(g) == cons); w1 = np.mean(abs(np.array(g) - cons) <= 1)
        inrange = np.mean([min(ex[e][i] for e in experts) <= g[i] <= max(ex[e][i] for e in experts) for i in range(len(rows))])
        per = [qwk(g, ex[e]) for e in experts]
        bias = np.mean(np.array(g) - cons)
        print(f"  {arch:9s} vs consensus: QWK {k:.3f} [{lo:.2f},{hi:.2f}]  exact {exact:.0%}  within-1 {w1:.0%}  "
              f"inside expert range {inrange:.0%}  bias {bias:+.2f}  | vs each expert {', '.join(f'{p:.2f}' for p in per)}")
        out.append([dis, arch, round(k, 3), round(lo, 3), round(hi, 3), round(exact, 3), round(w1, 3), round(inrange, 3), round(bias, 3), round(np.mean(loo), 3)])
        if dis != "all":
            M = np.zeros((K, K), int)
            for x, y in zip(cons, g): M[x, y] += 1
            print("      confusion (rows = expert consensus G0-G3, cols = model):", M.tolist())
with open("model_vs_experts_summary.csv", "w", newline="") as fo:
    w = csv.writer(fo); w.writerow(["disease", "model", "qwk_vs_consensus", "ci_lo", "ci_hi", "exact", "within1", "inside_expert_range", "bias", "expert_loo_qwk"]); w.writerows(out)

# Fair comparison: model and each expert scored against the SAME 3-expert consensus (leave that expert out)
print("\n==================== FAIR: vs 3-expert consensus (each expert left out in turn) ====================")
for dis in ("ojo", "roya", "all"):
    rows = [r for r in E if dis == "all" or r["leaf_id"].startswith(dis)]
    ex = {e: [int(r[e]) for r in rows] for e in experts}
    line = []
    for who in ["expert"] + ["segformer", "deeplab", "unetpp"]:
        ks = []
        for e in experts:
            ref = [int(np.median([ex[o][i] for o in experts if o != e]) + 0.5) for i in range(len(rows))]
            pred = ex[e] if who == "expert" else [int(P[r["leaf_id"]][f"{who}_grade"]) for r in rows]
            ks.append(qwk(pred, ref))
        line.append(f"{who} {np.mean(ks):.3f}")
    print(f"  {dis:5s}: " + " | ".join(line))
