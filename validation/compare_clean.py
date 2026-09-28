"""Sensitivity analysis: drop form leaves that were in the deployed models' TRAINING split."""
import csv, numpy as np
exec(open("compare.py").read().split("out = []")[0])       # reuse qwk / boot_ci / data loading
K_ = {r["form_id"]: r for r in csv.DictReader(open("form_image_key.csv"))}
train = {f for f, r in K_.items() if r["model_split"] == "train"}
print("excluded (seen in training):", sorted(train))
for f in sorted(train): print(f"  {f}: consensus G{[r for r in E if r['leaf_id']==f][0]['consensus_grade']}, segformer G{P[f]['segformer_grade']} ({P[f]['segformer_sev']}%)")
for dis in ("roya", "all"):
    rows = [r for r in E if (dis == "all" or r["leaf_id"].startswith(dis)) and r["leaf_id"] not in train]
    cons = [int(r["consensus_grade"]) for r in rows]; ex = {e: [int(r[e]) for r in rows] for e in experts}
    print(f"\n== {dis} without training leaves (n={len(rows)}) ==")
    for who in ["expert", "segformer", "deeplab", "unetpp"]:
        ks = []
        for e in experts:
            ref = [int(np.median([ex[o][i] for o in experts if o != e]) + 0.5) for i in range(len(rows))]
            pred = ex[e] if who == "expert" else [int(P[r["leaf_id"]][f"{who}_grade"]) for r in rows]
            ks.append(qwk(pred, ref))
        extra = ""
        if who != "expert":
            g = [int(P[r["leaf_id"]][f"{who}_grade"]) for r in rows]; lo, hi = boot_ci(g, cons)
            extra = f" | vs 4-expert consensus QWK {qwk(g, cons):.3f} [{lo:.2f},{hi:.2f}], exact {np.mean(np.array(g)==cons):.0%}"
        print(f"  {who:9s} fair QWK {np.mean(ks):.3f}{extra}")
