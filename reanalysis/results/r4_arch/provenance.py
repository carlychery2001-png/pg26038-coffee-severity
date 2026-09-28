"""A1 part 1 - provenance of the published SegFormer-B5 training vs the matched-protocol CNN kernels.

Outputs (this folder): provenance.json, diffs/*.diff (unified, line-by-line, 0 context lines).
Reads only; writes only into results/r4_arch/.
"""
import os as _os  # (release) all paths below are relative to ROOT (repository root by default; see README)
ROOT = (_os.environ.get("PG26038_ROOT") or _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))).replace(chr(92), "/")
import csv, difflib, glob, hashlib, io, json, os, pickletools, re, zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
(HERE / "diffs").mkdir(exist_ok=True)
BASE = Path(ROOT)
MR = BASE / "modal-recovery"
KR = BASE / "reanalysis/kaggle_runs"
SEG = {"roya": MR / "kernels/lesion_roya_segformer_realpack_kernel.py",
       "ojo": MR / "kernels/lesion_ojo_segformer_realpack_kernel.py"}
SALV = MR / "kernels/salvage_roya_segformer_eval_export.py"
SIB = {"roya": BASE / "notebooks/pg26038-segformer-roya/pg26038-segformer-roya.py",
       "ojo": BASE / "notebooks/pg26038-segformer-ojo/pg26038-segformer-ojo.py"}


def P(p):
    return str(p).replace("\\", "/")


def cnn(dis, arch, seed):
    slug = f"pg26038-mb-{dis}-{arch}-lrB" if seed == 0 else f"pg26038-mbs-{dis}-{arch}-lrB-seed{seed}"
    return slug, KR / slug / f"{slug}.py"


def read(p):
    return Path(p).read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")


def udiff(a, b, name):
    la, lb = read(a).splitlines(keepends=True), read(b).splitlines(keepends=True)
    d = list(difflib.unified_diff(la, lb, fromfile=P(a), tofile=P(b), n=0))
    (HERE / "diffs" / name).write_text("".join(d), encoding="utf-8")
    rem = sum(1 for x in d if x.startswith("-") and not x.startswith("---"))
    add = sum(1 for x in d if x.startswith("+") and not x.startswith("+++"))
    return {"file": f"diffs/{name}", "lines_a": len(la), "lines_b": len(lb), "removed_lines": rem, "added_lines": add}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ settings extraction (regex, with line numbers)
PAT = {
    "loss_combination": r"return 0\.7 \* focal_tversky\(logits, y\) \+ 0\.3 \* F\.cross_entropy\(logits, y, weight=ce_w\)",
    "focal_tversky_params": r"def focal_tversky\(logits, y, a=([\d.]+), b=([\d.]+), gexp=([\d.]+)",
    "ce_class_weights": r"ce_w = torch\.tensor\(\[([^\]]+)\]",
    "optimizer": r"opt = torch\.optim\.(\w+)\(\[\{\"params\": enc, \"lr\": ([\de.-]+)\}, \{\"params\": dec, \"lr\": ([\de.-]+)\}\], weight_decay=([\d.]+)\)",
    "encoder_param_rule": r"^(enc = .*)$",
    "epochs": r"^EPOCHS = (\d+)",
    "scheduler": r"sched = torch\.optim\.lr_scheduler\.(\w+)\(opt, (\w+)\)",
    "scheduler_step_per_epoch": r"^    (sched\.step\(\))",
    "patience": r"patience, bad = (\d+), 0",
    "batch_accum": r"^BATCH, ACCUM = (\d+), (\d+)",
    "drop_last": r"(drop_last=True)",
    "hflip": r"if np\.random\.rand\(\) < ([\d.]+): rgb, lab = rgb\[:, ::-1\], lab\[:, ::-1\]",
    "vflip": r"if np\.random\.rand\(\) < ([\d.]+): rgb, lab = rgb\[::-1\], lab\[::-1\]",
    "rot90": r"k = np\.random\.randint\((\d)\)",
    "photometric_flag": r"^PHOTOMETRIC = (\w+)",
    "photometric_call": r"if PHOTOMETRIC and np\.random\.rand\(\) < ([\d.]+):",
    "glare_train": r"tr_dl = DataLoader\(LeafSet\(by\[\"train\"\], True(, glare=\w+)?\)",
    "input_size": r"^SIZE = (\d+)",
    "letterbox": r"(canvas\.paste\(im, \(\(size - im\.width\) // 2, \(size - im\.height\) // 2\)\))",
    "imagenet_mean": r"MEAN = np\.array\(\[([^\]]+)\]",
    "leaf_rule": r"(leaf = rgba\[\.\.\., 3\] > 8)",
    "lesion_rule": r"(> 127\) & leaf)",
    "selection_segformer": r"(if va\[\"lesion_iou\"\] > best:)",
    "selection_cnn_score": r"(score = va\[\"lesion_iou\"\] - [^#]*)",
    "val_set_cnn": r"(va = evaluate\(val_les\).*)",
    "val_set_segformer": r"(va = evaluate\(by\[\"val\"\]\))",
    "fp_definition": r"(float\(\(sevs >= 5\)\.mean\(\)\))",
    "wall_clock": r"if \(time\.time\(\) - t0\) / 3600 > ([\d.]+):",
    "amp_autocast": r"(torch\.cuda\.amp\.autocast\(enabled=\(DEV == \"cuda\"\)\))",
    "amp_scaler": r"(torch\.cuda\.amp\.GradScaler\(enabled=\(DEV == \"cuda\"\)\))",
    "seed_python_numpy": r"(_rnd\.seed\(SEED\); np\.random\.seed\(SEED\))",
    "seed_value": r"^SEED = (\d+)",
    "seed_torch": r"(torch\.manual_seed\(\d+\); torch\.cuda\.manual_seed_all\(\d+\))",
    "cudnn_deterministic": r"(cudnn\.deterministic|use_deterministic_algorithms)",
    "grad_checkpointing": r"(model\.gradient_checkpointing_enable\(\))",
    "pretrained_segformer": r"for src in \((.*)\):",
    "pretrained_cnn": r"smp\.(\w+)\(encoder_name=\"(\w+)\", encoder_weights=\"(\w+)\"",
    "init_ckpt_warmstart": r"(if init_ckpt:)",
    "onnx_opset": r"opset_version=(\d+)",
    "test_grade_order": r"(order = \{.*\})",
}


def extract(path):
    lines = read(path).split("\n")
    out = {}
    for k, pat in PAT.items():
        hits = []
        for i, ln in enumerate(lines, 1):
            m = re.search(pat, ln)
            if m:
                hits.append({"line": i, "groups": list(m.groups()), "text": ln.strip()[:200]})
        out[k] = hits
    return out


# ------------------------------------------------------------------ checkpoint pickle keys (no torch needed)
def ckpt_keys(p):
    z = zipfile.ZipFile(p)
    pk = [n for n in z.namelist() if n.endswith("data.pkl")][0]
    strs, floats = set(), []
    for op, arg, pos in pickletools.genops(io.BytesIO(z.read(pk))):
        if isinstance(arg, str) and "." not in arg and len(arg) < 60:
            strs.add(arg)
        if op.name == "BINFLOAT":
            floats.append(arg)
    return {"top_level_strings": sorted(s for s in strs if not s.isdigit()), "floats": floats,
            "n_zip_entries": len(z.namelist())}


# ------------------------------------------------------------------ history-based counterfactual (matched CNNs)
def simulate(hist, key, patience=15):
    best, sel, bad, stop = -2.0, None, 0, None
    for h in hist:
        v = h[key]
        if v > best:
            best, sel, bad = v, h["epoch"], 0
        else:
            bad += 1
            if bad >= patience:
                stop = h["epoch"]
                break
    return sel, stop


def main():
    res = {}
    # 1. line-by-line diffs
    D = {}
    for dis in ("roya", "ojo"):
        for arch in ("unetpp", "deeplab"):
            slug, p = cnn(dis, arch, 0)
            D[f"segformer_{dis}__vs__{slug}"] = udiff(SEG[dis], p, f"segformer_{dis}__vs__{slug}.diff")
    D["segformer_roya__vs__segformer_ojo"] = udiff(SEG["roya"], SEG["ojo"], "segformer_roya__vs__segformer_ojo.diff")
    D["segformer_roya__vs__salvage"] = udiff(SEG["roya"], SALV, "segformer_roya__vs__salvage.diff")
    for dis in ("roya", "ojo"):
        D[f"segformer_{dis}__vs__notebooks_sibling"] = udiff(SEG[dis], SIB[dis], f"segformer_{dis}__vs__notebooks_sibling.diff")
    ref_slug, ref = cnn("roya", "unetpp", 0)
    for dis in ("roya", "ojo"):
        for arch in ("unetpp", "deeplab"):
            for seed in (0, 1, 2):
                slug, p = cnn(dis, arch, seed)
                if slug == ref_slug:
                    continue
                D[f"{ref_slug}__vs__{slug}"] = udiff(ref, p, f"{ref_slug}__vs__{slug}.diff")
    res["diffs"] = D
    cnn_diff_lines = {}
    for k, v in D.items():
        if k.startswith(ref_slug):
            txt = (HERE / v["file"]).read_text(encoding="utf-8")
            cnn_diff_lines[k] = [ln.rstrip("\n")[:160] for ln in txt.splitlines()
                                 if (ln.startswith("+") or ln.startswith("-")) and not ln.startswith(("+++", "---"))]
    res["matched_cnn_kernel_differences"] = cnn_diff_lines

    # 2. settings extraction
    files = {"segformer_roya": SEG["roya"], "segformer_ojo": SEG["ojo"], "salvage_roya": SALV}
    for dis in ("roya", "ojo"):
        for arch in ("unetpp", "deeplab"):
            for seed in (0, 1, 2):
                slug, p = cnn(dis, arch, seed)
                files[slug] = p
    res["settings_extracted"] = {k: {"path": P(v), **extract(v)} for k, v in files.items()}

    # 3. SHA-256 of ONNX files and checkpoints
    H = {}
    for p in sorted(glob.glob(str(MR / "serving/*.onnx")) + glob.glob(str(MR / "out/*/*.onnx")) +
                    glob.glob(str(BASE / "datasets/pg26038-onnx-models/*.onnx")) + glob.glob(str(MR / "out/*/best.pt")) +
                    glob.glob(str(BASE / "datasets/pg26038-segformer-ckpt/*.pt"))):
        H[P(p)] = {"sha256": sha(p), "bytes": os.path.getsize(p)}
    res["sha256"] = H
    match = {}
    for dis in ("roya", "ojo"):
        a = P(MR / f"serving/{dis}_segformer_realpack_1024.onnx")
        b = P(MR / f"out/{dis}/{dis}_segformer_realpack_1024.onnx")
        c = P(BASE / f"datasets/pg26038-onnx-models/{dis}_segformer_realpack_1024.onnx")
        match[dis] = {"serving": a, "out": b, "serving_eq_out": H[a]["sha256"] == H[b]["sha256"],
                      "serving_eq_datasets_onnx_models": H[a]["sha256"] == H[c]["sha256"], "sha256": H[a]["sha256"],
                      "bytes": H[a]["bytes"]}
    res["segformer_onnx_hash_match"] = match

    # 4. checkpoint contents (does any record an epoch?)
    res["checkpoint_keys"] = {P(p): ckpt_keys(p) for p in
                              [MR / "out/roya/best.pt", MR / "out/ojo/best.pt", BASE / "datasets/pg26038-segformer-ckpt/best.pt"]}
    res["segformer_summaries"] = {d: json.loads(read(MR / f"out/{d}/summary.json").replace("NaN", "null")) for d in ("roya", "ojo")}

    # 5. search for any SegFormer epoch log
    epoch_pat = re.compile(r"epoch\s+\d+/60\s+loss=")                   # SegFormer kernel print format
    seg_pat = re.compile(r"segformer|mit-b5", re.I)
    roots = [MR, BASE / "datasets/pg26038-segformer-ckpt", KR, BASE / "notebooks", BASE / "kernel-outputs",
             BASE / "reanalysis", BASE / "datasets/pg26038-onnx-models", BASE / "validation",
             Path(_os.environ.get("PG26038_DOCS_DIR", ROOT))]  # (release) extra folder searched for logs
    scanned, seg_mention, epoch_hits = 0, [], []
    for r in roots:
        for p in r.rglob("*"):
            try:
                if not p.is_file() or p.suffix.lower() not in (".log", ".txt", ".json", ".md", ".out", ".ipynb", ".csv", ".tex", ".py"):
                    continue
                if p.stat().st_size > 50e6 or "r4_arch" in str(p):
                    continue
                t = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            scanned += 1
            if seg_pat.search(t) and p.suffix.lower() in (".log", ".txt", ".json", ".out", ".ipynb"):
                seg_mention.append(P(p))
            if epoch_pat.search(t) and p.suffix.lower() != ".py":
                epoch_hits.append({"file": P(p), "mentions_segformer_or_mit": bool(seg_pat.search(t)),
                                   "arch_in_sibling_summary": (json.loads(read(p.parent / "summary.json")).get("arch")
                                                               if (p.parent / "summary.json").exists() else None)})
    res["epoch_log_search"] = {"roots": [P(r) for r in roots], "text_files_scanned": scanned,
                               "log_like_files_mentioning_segformer": seg_mention,
                               "files_with_epoch_lines_in_segformer_print_format": epoch_hits,
                               "segformer_epoch_logs_found": [h["file"] for h in epoch_hits if h["mentions_segformer_or_mit"]],
                               "pattern": epoch_pat.pattern,
                               "manuscript_claim": "supplementary/note_methods.tex and the original draft (MDPI_rebuild/main.tex, body_draft.tex) state "
                                                   "'the rust run early-stopped at epoch 55 and the leaf-spot run used the full 60'; no log on disk supports it."}

    # 6. matched CNN histories: epochs run, selected epoch, FP term, IoU-only counterfactual
    CF = {}
    for dis in ("roya", "ojo"):
        for arch in ("unetpp", "deeplab"):
            for seed in (0, 1, 2):
                slug, p = cnn(dis, arch, seed)
                hist = json.load(open(KR / slug / "output/history.json"))
                summ = json.load(open(KR / slug / "output/summary.json"))
                log = read(KR / slug / "output" / f"{slug}.log")
                sel_s, stop_s = simulate(hist, "score")
                sel_i, stop_i = simulate(hist, "val_lesion_iou")
                last = hist[-1]["epoch"]
                fp_sel = next(h["val_fp"] for h in hist if h["epoch"] == sel_s)
                nz = [(h["epoch"], round(h["val_fp"], 4)) for h in hist if h["val_fp"] not in (None, 0.0)]
                iou_rule_needs_more = (stop_i is None and last < 60)
                m = re.search(r"init ckpt: ([^|]*)\|", log)
                CF[slug] = {"epochs_run": len(hist), "summary_best_epoch": summ.get("best_epoch"),
                            "selected_epoch_score_rule_sim": sel_s, "stop_epoch_score_rule_sim": stop_s,
                            "selected_epoch_iou_only_sim": sel_i, "stop_epoch_iou_only_sim": stop_i,
                            "same_checkpoint_under_iou_only": (sel_s == sel_i) and not iou_rule_needs_more,
                            "iou_only_would_need_unrecorded_epochs": iou_rule_needs_more,
                            "val_fp_at_selected_epoch": fp_sel, "epochs_with_nonzero_val_fp": nz,
                            "init_ckpt_logged": m.group(1).strip() if m else None,
                            "val_lesion_iou_at_selected": next(h["val_lesion_iou"] for h in hist if h["epoch"] == sel_s)}
    res["matched_cnn_selection_counterfactual"] = CF

    # 7. SegFormer selection set: recompute pooled lesion IoU with the served ONNX outputs (per_image.csv)
    rows = list(csv.DictReader(open(BASE / "reanalysis/out/per_image.csv")))
    pooled = {}
    f = lambda q: sum(int(r["inter_lesion"]) for r in q) / sum(int(r["union_lesion"]) for r in q)
    for dis in ("roya", "ojo"):
        for split in ("val", "test"):
            rr = [r for r in rows if r["model_disease"] == dis and r["image_pack"] == dis and r["arch"] == "segformer" and r["split"] == split]
            dd = [r for r in rr if r["grade_label"] != "healthy"]
            pooled[f"{dis}_{split}"] = {"all_images": f(rr), "n_all": len(rr), "diseased_only": f(dd), "n_diseased": len(dd)}
    pooled["ckpt_val_lesion_iou"] = {d: res["checkpoint_keys"][P(MR / f"out/{d}/best.pt")]["floats"] for d in ("roya", "ojo")}
    res["segformer_pooled_iou_check"] = pooled

    # 8. optimizer steps per epoch (drop_last + accumulation; the partial accumulation at epoch end is discarded
    #    by opt.zero_grad() at the start of the next epoch, in both kernels)
    ntrain = {"roya": 396, "ojo": 518}
    steps = {}
    for dis, n in ntrain.items():
        seg_it, cnn_it = n // 1, n // 2
        steps[dis] = {"n_train_images": n, "segformer_iters": seg_it, "segformer_optimizer_steps": seg_it // 32,
                      "segformer_images_used": (seg_it // 32) * 32, "cnn_iters": cnn_it, "cnn_optimizer_steps": cnn_it // 16,
                      "cnn_images_used": (cnn_it // 16) * 32}
    res["optimizer_steps_per_epoch"] = steps

    # 9. training data packs (split.csv of the local real packs)
    packs = {}
    for dis, zp in (("roya", MR / "roya-real-pack.zip"), ("ojo", MR / "ojo-real-pack.zip")):
        z = zipfile.ZipFile(zp)
        sp = [n for n in z.namelist() if n.endswith("split.csv")][0]
        rr = list(csv.DictReader(io.TextIOWrapper(z.open(sp), encoding="utf-8")))
        cnt = {}
        for r in rr:
            cnt[f"{r['split']}|{r['grade']}"] = cnt.get(f"{r['split']}|{r['grade']}", 0) + 1
        packs[dis] = {"zip": P(zp), "split_counts_images": cnt, "synthetic_rows": sum(r["file"].startswith("synth_") for r in rr),
                      "grade_labels": sorted({r["grade"] for r in rr})}
    res["packs"] = packs

    json.dump(res, open(HERE / "provenance.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(match, indent=1))
    for k, v in CF.items():
        print(k, {x: v[x] for x in ("epochs_run", "selected_epoch_score_rule_sim", "selected_epoch_iou_only_sim", "stop_epoch_iou_only_sim",
                                     "same_checkpoint_under_iou_only", "val_fp_at_selected_epoch", "summary_best_epoch", "init_ckpt_logged")})
    print(json.dumps(pooled, indent=1))
    print("epoch log hits:", epoch_hits, "| scanned", scanned, "| seg mentions", len(seg_mention))
    print(json.dumps(steps, indent=1))
    print(json.dumps(packs, indent=1))
    for k, v in D.items():
        print(k, v)


if __name__ == "__main__":
    main()
