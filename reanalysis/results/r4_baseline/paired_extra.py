"""A3 add-on: paired bootstrap differences involving the existing baseline with the validation-chosen erosion
(network minus existing-eroded; existing-eroded minus existing without erosion), on the SAME resamples as
r4_baseline.py (default_rng(20260928) re-seeded per disease, integers(0, n, (10000, n)) over leaves sorted by id).
Input: per_leaf_test.csv (6-decimal values). Check: the new baseline's IoU CI recomputed here must equal results.json
to 3 decimals. Output: paired_extra.json."""
import csv
import json
import os

import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(f"{OUT}/results.json"))
B, SEED = 10_000, 20260928
rows = list(csv.DictReader(open(f"{OUT}/per_leaf_test.csv", newline="")))
res = {}
for dis in ("roya", "ojo"):
    rr = sorted([r for r in rows if r["disease"] == dis], key=lambda r: r["leaf_id"])
    n = len(rr)
    IDX = np.random.default_rng(SEED).integers(0, n, size=(B, n))
    ref = np.array([float(r["ref_sev"]) for r in rr])
    get = lambda k: np.array([float(r[f"{k}_iou"]) for r in rr])
    sev = lambda k: np.array([float(r[f"{k}_sev"]) for r in rr])
    chk = np.percentile(get("new_primary")[IDX].mean(1), [2.5, 97.5])
    exp = R["test"][dis]["methods"]["new_primary"]["mask_grade"]["ci95"]["iou"]
    assert all(abs(round(a, 3) - round(b, 3)) < 1e-9 for a, b in zip(chk, exp)), (dis, chk, exp)
    out = {}
    pairs = [(a, "existing_fixed_thr_chosen_erosion") for a in ("segformer", "unetpp", "deeplab")] + \
            [("existing_fixed_thr_chosen_erosion", "existing_e0")]
    for a, b in pairs:
        dI = get(a) - get(b)
        dM = np.abs(sev(a) - ref) - np.abs(sev(b) - ref)
        bi = dI[IDX].mean(1); bm = dM[IDX].mean(1)
        ci_i = np.percentile(bi, [2.5, 97.5]).tolist(); ci_m = np.percentile(bm, [2.5, 97.5]).tolist()
        out[f"{a}_minus_{b}"] = dict(n=n, d_iou=float(dI.mean()), d_iou_ci95=ci_i, d_iou_ci_excludes_zero=bool(ci_i[0] > 0 or ci_i[1] < 0),
                                     leaves_first_higher=int((dI > 0).sum()), leaves_second_higher=int((dI < 0).sum()),
                                     d_mae=float(dM.mean()), d_mae_ci95=ci_m, d_mae_ci_excludes_zero=bool(ci_m[0] > 0 or ci_m[1] < 0))
        print(dis, a, b, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[f"{a}_minus_{b}"].items()})
    res[dis] = out
json.dump(dict(note="same resamples as results.json (checked); input per_leaf_test.csv", results=res),
          open(f"{OUT}/paired_extra.json", "w"), indent=1)
