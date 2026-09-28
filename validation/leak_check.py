"""Cross the form->leaf key with the train/val/test split used by the deployed realpack models."""
import os as _os  # (release) all paths below are relative to ROOT (repository root by default; see README)
ROOT = (_os.environ.get("PG26038_ROOT") or _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))).replace(chr(92), "/")
import csv, collections
DS = (ROOT + "/datasets")
splits = {}
for dis, f in [("ojo", f"{DS}/pg26038-ojo-real-pack/meta/split.csv"), ("roya", f"{DS}/pg26038-roya-real-pack/split.csv")]:
    splits[dis] = {r["leaf_id"]: r["split"] for r in csv.DictReader(open(f))}
other = {}
for f in [f"{DS}/pg26038-ojo-balanced/meta/split.csv", f"{DS}/pg26038-roya-grade-pack-v2/split.csv", f"{DS}/pg26038-roya-train-pack/split.csv", f"{DS}/pg26038-ojo-pack/split.csv"]:
    for r in csv.DictReader(open(f)): other.setdefault(r["leaf_id"], set()).add(f.split("/")[-2 if "meta" not in f else -3] + ":" + r["split"])
key = list(csv.DictReader(open("form_image_key.csv")))
out = []
for r in key:
    dis = "ojo" if r["form_id"].startswith("ojo") else "roya"
    s = splits[dis].get(r["leaf_id"], "NOT IN PACK")
    out.append({**r, "model_split": s, "other_packs": ";".join(sorted(other.get(r["leaf_id"], [])))})
for dis in ("ojo", "roya"):
    c = collections.Counter(o["model_split"] for o in out if o["form_id"].startswith(dis))
    print(dis, dict(c))
print("duplicates in key:", [k for k, v in collections.Counter(o["leaf_id"] for o in out).items() if v > 1])
with open("form_image_key.csv", "w", newline="") as fo:
    w = csv.DictWriter(fo, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
for o in out: print(o["form_id"], o["leaf_id"], o["margin"], o["model_split"])
