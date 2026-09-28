"""A3 sensitivity (not a selection step): effect of each PRE-SPECIFIED petiole/cut-end exclusion option on the rust
detectors, at the thresholds and erosion fixed in results.json (validation chose 'none' for both indices).

Reads results.json for the fixed configurations; loads only the 21 healthy rust-pack test leaves (both faces) and the
21 diseased rust test leaves (abaxial). Reports, per exclusion option: healthy test leaves >= 0.5 % and > 0 % (either
face; abaxial), and the diseased test leaf IoU / MAE (point values only; this is a sensitivity table, not a result
used for any selection). Also reports the abaxial and adaxial healthy-face severity medians of the primary new
baseline. Output: exclusion_sensitivity.json.
"""
import csv
import importlib.util
import json
import os

import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("r4", f"{OUT}/r4_baseline.py")
r4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r4)
R = json.load(open(f"{OUT}/results.json"))


def main():
    cfg = R["test"]["roya"]["configs"]
    fixed = {"new_primary": (cfg["new_primary"]["index"], cfg["new_primary"]["threshold"], cfg["new_primary"]["erosion"]),
             "existing_fixed_thr_chosen_erosion": (cfg["existing_fixed_thr_chosen_erosion"]["index"],
                                                   cfg["existing_fixed_thr_chosen_erosion"]["threshold"],
                                                   cfg["existing_fixed_thr_chosen_erosion"]["erosion"])}
    rows = [r for r in csv.DictReader(open(r4.PACKS["roya"][1], newline="")) if r["split"] == "test"]
    recs = {}
    for r in rows:
        if r["grade"] != "healthy" and "adaxial" in r["file"]:
            continue
        rgb, lab = r4.load("roya", r["file"])
        leaf = lab >= 1; les = lab == 2
        cdt, obits, dend, _ = r4.geometry(leaf)
        recs[r["file"]] = dict(leaf_id=r["leaf_id"], healthy=r["grade"] == "healthy", face="abaxial" if "abaxial" in r["file"] else "adaxial",
                               n_leaf=int(leaf.sum()), n_les=int(les.sum()), les=les[leaf], cdt=cdt, obits=obits, dend=dend,
                               **r4.scores(rgb[leaf]))
    out = {}
    for name, (ix, t, e) in fixed.items():
        out[name] = dict(index=ix, threshold=t, erosion=e, options={})
        st = r4.t_to_score(ix, t)
        for p, (kind, val) in enumerate(r4.PETIOLE):
            hs = {}; ious = []; errs = []
            for f, rec in recs.items():
                pr = (rec[ix] >= st) & r4.eligible(rec, e, p)
                sev = 100.0 * pr.sum() / rec["n_leaf"]
                if rec["healthy"]:
                    hs.setdefault(rec["leaf_id"], {})[rec["face"]] = (sev, bool(pr.any()))
                else:
                    ious.append((pr & rec["les"]).sum() / (pr | rec["les"]).sum())
                    errs.append(sev - 100.0 * rec["n_les"] / rec["n_leaf"])
            assert len(hs) == 21 and len(ious) == 21
            ab = np.array([v["abaxial"][0] for v in hs.values()]); ad = np.array([v["adaxial"][0] for v in hs.values()])
            anyab = np.array([v["abaxial"][1] for v in hs.values()]); anyad = np.array([v["adaxial"][1] for v in hs.values()])
            out[name]["options"]["none" if kind == "none" else f"{kind} {val}"] = dict(
                healthy_either_ge_0_5=int((np.maximum(ab, ad) >= 0.5).sum()), healthy_abaxial_ge_0_5=int((ab >= 0.5).sum()),
                healthy_either_any=int((anyab | anyad).sum()), healthy_abaxial_any=int(anyab.sum()),
                healthy_abaxial_sev_median=float(np.median(ab)), healthy_abaxial_sev_max=float(ab.max()),
                healthy_adaxial_sev_median=float(np.median(ad)), healthy_adaxial_sev_max=float(ad.max()),
                diseased_test_iou=float(np.mean(ious)), diseased_test_mae=float(np.mean(np.abs(errs))),
                diseased_test_bias=float(np.mean(errs)))
            print(name, kind, val, out[name]["options"]["none" if kind == "none" else f"{kind} {val}"], flush=True)
    json.dump(dict(note="sensitivity only; thresholds and erosion fixed from results.json; n = 21 healthy test leaves, 21 diseased rust test leaves (abaxial)",
                   results=out), open(f"{OUT}/exclusion_sensitivity.json", "w"), indent=1)


if __name__ == "__main__":
    main()
