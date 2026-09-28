import csv, glob, itertools, statistics as st
from collections import Counter
f = glob.glob("**/*.csv", recursive=True)[0]
rows = list(csv.reader(open(f, encoding="utf-8-sig")))
hdr, data = rows[0], rows[1:]
items = [h.split(") ")[1].split(" -")[0] for h in hdr[7:]]
names = [f"E{i + 1}" for i in range(len(data))]  # (release) evaluator labels; identity columns are not read
G = [[int(v.strip()[1]) for v in r[7:]] for r in data]   # raters x items
R, N, K = len(G), len(items), 4

def kappa_w(a, b, w="quadratic"):
    import numpy as np
    O = np.zeros((K, K));  [O.__setitem__((x, y), O[x, y] + 1) for x, y in zip(a, b)]
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    W = np.array([[((i - j) / (K - 1)) ** 2 if w == "quadratic" else abs(i - j) / (K - 1) for j in range(K)] for i in range(K)])
    return 1 - (W * O).sum() / (W * E).sum()

def fleiss(sub):
    n = len(sub); cnt = [Counter(G[r][i] for r in range(R)) for i in sub]
    P = [(sum(c[k] ** 2 for k in range(K)) - R) / (R * (R - 1)) for c in cnt]
    p = [sum(c[k] for c in cnt) / (n * R) for k in range(K)]
    Pe = sum(x * x for x in p); return (st.mean(P) - Pe) / (1 - Pe)

def kripp_ordinal(sub):
    import numpy as np
    vals = [[G[r][i] for r in range(R)] for i in sub]
    n_k = Counter(v for vs in vals for v in vs); n = sum(n_k.values())
    def d(c, k):
        lo, hi = min(c, k), max(c, k)
        return (sum(n_k[g] for g in range(lo, hi + 1)) - (n_k[c] + n_k[k]) / 2) ** 2
    Do = sum(d(a, b) for vs in vals for a, b in itertools.permutations(vs, 2)) / (R - 1) / n
    De = sum(n_k[c] * n_k[k] * d(c, k) for c in range(K) for k in range(K)) / (n * (n - 1))
    return 1 - Do / De

def icc21(sub):
    import numpy as np
    X = np.array([[G[r][i] for r in range(R)] for i in sub], float); n, k = X.shape; m = X.mean()
    MSR = k * ((X.mean(1) - m) ** 2).sum() / (n - 1); MSC = n * ((X.mean(0) - m) ** 2).sum() / (k - 1)
    MSE = ((X - X.mean(1, keepdims=True) - X.mean(0) + m) ** 2).sum() / ((n - 1) * (k - 1))
    return (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n)

for label, sub in [("ojo", range(0, 30)), ("roya", range(30, 60)), ("all", range(60))]:
    sub = list(sub)
    unan = sum(len({G[r][i] for r in range(R)}) == 1 for i in sub)
    within1 = sum(max(G[r][i] for r in range(R)) - min(G[r][i] for r in range(R)) <= 1 for i in sub)
    print(f"\n== {label} (n={len(sub)}) ==")
    print(f"  unanimous: {unan}/{len(sub)}   all within 1 grade: {within1}/{len(sub)}")
    print(f"  Fleiss kappa (nominal): {fleiss(sub):.3f}   Krippendorff alpha (ordinal): {kripp_ordinal(sub):.3f}   ICC(2,1): {icc21(sub):.3f}")
    print("  mean grade per rater:", ", ".join(f"{names[r].split()[0]} {st.mean(G[r][i] for i in sub):.2f}" for r in range(R)))

print("\n== pairwise quadratic-weighted kappa (all 60) ==")
for a, b in itertools.combinations(range(R), 2):
    print(f"  {names[a].split()[0]:8s} vs {names[b].split()[0]:8s}: {kappa_w(G[a], G[b]):.3f}")

print("\n== consensus (median, ties -> round half up) and spread ==")
out = []
for i, it in enumerate(items):
    v = sorted(G[r][i] for r in range(R)); med = (v[1] + v[2]) / 2; cons = int(med + 0.5)
    out.append((it, *[G[r][i] for r in range(R)], cons, v[-1] - v[0]))
with open("expert_consensus.csv", "w", newline="", encoding="utf-8") as fo:
    w = csv.writer(fo); w.writerow(["leaf_id"] + [n.split()[0] for n in names] + ["consensus_grade", "range"]); w.writerows(out)
dist = Counter(o[-2] for o in out[:30]), Counter(o[-2] for o in out[30:])
print("  consensus distribution ojo :", dict(sorted(dist[0].items())))
print("  consensus distribution roya:", dict(sorted(dist[1].items())))
print("  leaves with 2-grade disagreement:", [o[0] for o in out if o[-1] >= 2])
