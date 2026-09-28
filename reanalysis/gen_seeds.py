"""Phase 2: extra seeds (1, 2) for each disease x CNN at the learning rate chosen on VALIDATION in phase 1
(matched protocol: 60 epochs, patience 15, geometric augmentation only). Usage: python gen_seeds.py roya-deeplab:lrB ..."""
import json, os, sys
src = open(os.path.join(os.path.dirname(__file__), "make_kaggle_runs.py"), encoding="utf-8").read()
exec(src.split("runs = []")[0])            # defines make(), LRS, OUT without generating phase-1 kernels
queue_path = os.path.join(OUT, "queue_seeds.json")
queue = json.load(open(queue_path)) if os.path.exists(queue_path) else []
for spec in sys.argv[1:]:
    combo, lrk = spec.split(":"); dis, arch = combo.split("-")
    for seed in (1, 2):
        slug = f"pg26038-mbs-{dis}-{arch}-{lrk}-seed{seed}"
        if slug not in queue:
            make(slug, arch, dis, 60, 15, LRS[lrk], seed, photometric=False); queue.append(slug)
json.dump(queue, open(queue_path, "w"), indent=1)
print("seed queue:", *queue, sep="\n  ")
