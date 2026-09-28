"""Tie-free comparisons (leave-one-expert-out 3-rater consensus; pairwise with individual experts),
bootstrap CI of the model-minus-expert difference, and 4-rater consensus under both tie rules."""
import csv, itertools, json, numpy as np
exec(open("final_analysis.py").read().split("res = {}")[0])
train = {f for f, r in KEY.items() if r["model_split"] == "train"}
R = {}
def stats(rows, a):
    M = np.array([[int(r[e]) for r in rows] for e in EXP]); n = len(rows)
    g = np.array([model_grade(r["leaf_id"], a) for r in rows])
    refs = [np.array([int(np.median(np.delete(M, j, 0)[:, i])) for i in range(n)]) for j in range(4)]  # median of 3 = integer
    fair_m = np.mean([qwk(g, refs[j]) for j in range(4)]); fair_e = np.mean([qwk(M[j], refs[j]) for j in range(4)])
    pw_m = np.mean([qwk(g, M[j]) for j in range(4)]); pw_e = np.mean([qwk(M[a_], M[b]) for a_, b in itertools.combinations(range(4), 2)])
    return fair_m, fair_e, pw_m, pw_e, g, M
for dis in ("ojo", "roya", "all"):
    rows = [r for r in E if (dis == "all" or r["leaf_id"].startswith(dis)) and r["leaf_id"] not in train]
    for a in ARCH:
        fm, fe, pm, pe, g, M = stats(rows, a)
        rng = np.random.default_rng(7); d = []
        for _ in range(10000):
            idx = rng.integers(0, len(rows), len(rows)); sub = [rows[i] for i in idx]
            try:
                x = stats(sub, a); d.append(x[0] - x[1])
            except Exception: pass
        d = np.array(d); d = d[np.isfinite(d)]
        up = np.array([int(np.median(M[:, i]) + 0.5) for i in range(len(rows))]); dn = np.array([int(np.ceil(np.median(M[:, i]) - 0.5)) for i in range(len(rows))])
        R[f"{dis}_{a}"] = dict(n=len(rows), fair_model=round(fm, 3), fair_expert=round(fe, 3), diff=round(fm - fe, 3), diff_ci=[round(x, 3) for x in np.percentile(d, [2.5, 97.5])],
                               p_le0=round(float((d <= 0).mean()), 4), pairwise_model=round(pm, 3), pairwise_expert=round(pe, 3),
                               cons_up=round(qwk(g, up), 3), cons_down=round(qwk(g, dn), 3),
                               acc_up=round(float((g == up).mean()), 3), acc_down=round(float((g == dn).mean()), 3),
                               within1_both=bool(((abs(g - up) <= 1) & (abs(g - dn) <= 1)).all()))
        print(dis, a, R[f"{dis}_{a}"], flush=True)
json.dump(R, open("tiefree_results.json", "w"), indent=1)
