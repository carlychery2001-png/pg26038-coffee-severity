# A1 Architecture comparison (round 4): published SegFormer-B5 vs the matched-protocol CNNs

> **Public-release copy.** The expert-identity section was removed. Lines naming panel experts or their affiliations were removed. All other text and every number are unchanged; local paths are written as `<ROOT>` (the working tree) and `<USER_HOME>`.

Scope: round-4 plan item A1 (`<ROOT>/reanalysis/round4/round4_plan.md`). No GPU training and no new inference: the provenance part reads kernels, logs, histories, checkpoints and ONNX files; the statistics use the existing per-image outputs (`reanalysis/out/per_image.csv` for the published SegFormer-B5, own-disease pack, test split; `reanalysis/out/per_image_retrained.csv` for the matched lrB runs, seeds 0–2) with the paper's evaluation protocol (leaf = alpha > 8; lesion = mask > 127 AND leaf; 1024 letterbox; ImageNet normalization; raw argmax; rust abaxial face; leaf spot mean of both faces; grades <0.5, 0.5–<5, 5–<20, ≥20%). Section 4 adds the expert-panel identities that the author asked for (name, occupation, affiliation, years of experience).

Scripts (this folder): `provenance.py` → `provenance.json`, `diffs/*.diff`; `paired_tests.py` → `paired_results.json`, `per_leaf_roya.csv`, `per_leaf_ojo.csv`; `experts.py` → `experts.json`, `experts_table.tex`, `experts_ack.tex`; `make_md.py` → this file and `results.json`. Logs: `provenance_run.log`, `paired_run.log`.

## 0. Key results

- **Provenance verified, with differences.** Line by line, the published SegFormer-B5 kernels use the same data packs, labels, 1024 letterbox input, loss (0.7·Focal Tversky with α = 0.3, β = 0.7, γ = 0.75 + 0.3·cross-entropy with class weights 0.25/0.5/3.0), AdamW (encoder 6e-5, decoder 6e-4, weight decay 0.01), cosine schedule over 60 epochs, patience 15, effective batch 32, geometric-only augmentation and FP16 AMP as the 12 matched lrB CNN kernels. They even take the same number of optimizer steps per epoch: 12 for rust and 16 for leaf spot. The differences are: micro-batch 1 × 32 with gradient checkpointing (CNNs 2 × 16); **the rust checkpoint criterion**; **no fixed seed**; **an 8-h wall-clock stop**; learning rates hand-set rather than chosen on validation (same values); rust test evaluation and ONNX export done afterwards by the salvage script; GPU/platform; ONNX opset (17 vs 16) (§1).
- **Rust selection criterion (verified numerically).** SegFormer-B5 kept the checkpoint with the best validation lesion IoU pooled over **all 84 rust-pack validation images, the 42 healthy ones included**. The checkpoint stores 0.6446. Recomputed from the served ONNX outputs, the pooled IoU is 0.6446 over all 84 images and 0.6569 over the 42 diseased images only. The rust CNNs selected on diseased-image lesion IoU minus the share of healthy validation images with predicted severity ≥5%. In all 6 matched rust runs that FP term was 0 at the selected epoch. An IoU-only rule would have selected the same checkpoint in all 12 matched runs (§1.4).
- **ONNX identity.** SHA-256 of `modal-recovery/serving/roya_segformer_realpack_1024.onnx` = `out/roya/…onnx` = `f3ffcae17dc0b59a…`. For leaf spot, `serving/ojo_…onnx` = `out/ojo/…onnx` = `c135695f7a1f4712…`. Both are also byte-identical to `datasets/pg26038-onnx-models/` (§1.5).
- **Epochs completed by SegFormer-B5: unknown.** No training log exists on disk (498 text files searched), and the checkpoints store no epoch. The manuscript's "rust early-stopped at epoch 55, leaf spot used the full 60" (Supplement, note_methods.tex) has no supporting file (§1.6).

- **Rust (n = 21).** Matched UNet++ (seed mean) vs SegFormer-B5: ΔIoU −0.009 (95% CI [−0.027, +0.009]; Wilcoxon p = 0.6333, Holm 0.6333). ΔMAE −0.16 pp [−0.58, +0.23]; ΔQWK +0.020 [0.000, +0.081]. Matched DeepLabV3+ had a lower leaf IoU than SegFormer-B5: ΔIoU −0.042 [−0.058, −0.026], Wilcoxon Holm p = 0.0002. Its severity error was also lower: ΔMAE −0.37 pp [−0.74, −0.07], Holm p = 0.013. This MAE difference is not robust to seed choice: the seed × leaf bootstrap gives [−1.04, +0.07]. UNet++ − DeepLabV3+: ΔIoU +0.033 [+0.020, +0.045], Wilcoxon Holm p = 0.0005; seed × leaf [−0.002, +0.064].
- **Leaf spot (n = 55).** Matched UNet++ vs SegFormer-B5: ΔIoU −0.006 [−0.015, +0.001] (Wilcoxon Holm p = 0.6385). ΔMAE −0.107 pp [−0.175, −0.044], Holm p <0.0001, and the MAE difference holds in the seed × leaf bootstrap, [−0.179, −0.041]. ΔQWK (mask-derived) +0.026 [−0.012, +0.079]. Matched DeepLabV3+ had a lower IoU than SegFormer-B5: −0.037 [−0.048, −0.028], Wilcoxon Holm p = 1.9e−13. It also had a lower IoU than matched UNet++: +0.031 [+0.025, +0.039], Holm p = 3.5e−14.
- **Equivalence** (TOST; the margins were fixed post hoc, after the primary results were known, and were not preregistered):
  - Rust UNet++ vs SegFormer-B5, leaf IoU (±0.03): **equivalent**. It is inconclusive in the seed × leaf sensitivity analysis. MAE and QWK are inconclusive.
  - Leaf spot UNet++ vs SegFormer-B5: leaf IoU and MAE are **equivalent** under both bootstrap methods; QWK is inconclusive.
  - DeepLabV3+ vs SegFormer-B5, leaf IoU: inconclusive for both diseases. The 90% CI lies entirely below 0 but crosses −0.03, so the result is also not "not equivalent".
  - No contrast met the "not equivalent" criterion (90% CI entirely outside the margin). Full table in §3.

## 1. Provenance of the published SegFormer-B5 training

### 1.1 Files compared

- SegFormer-B5 (published): `modal-recovery/kernels/lesion_roya_segformer_realpack_kernel.py` and `lesion_ojo_segformer_realpack_kernel.py`; rust evaluation/export: `salvage_roya_segformer_eval_export.py`.
- Matched CNNs: the 12 lrB kernels in `reanalysis/kaggle_runs/`: `pg26038-mb-{roya,ojo}-{unetpp,deeplab}-lrB` (seed 0) and `pg26038-mbs-…-lrB-seed{1,2}`. `reanalysis/make_kaggle_runs.py` generated them from the published rust CNN kernels by changing only the learning rate, the epoch cap and patience, photometric jitter (turned off), the seed and history logging.
- Unified diffs (0 context lines) are in `diffs/`. The two SegFormer kernels differ only in 6 lines (output names and the `arch` string). Relative to `pg26038-mb-roya-unetpp-lrB.py`, the other 11 matched kernels differ only in the architecture constructor (`smp.UnetPlusPlus` vs `smp.DeepLabV3Plus`), `SEED`/`torch.manual_seed`, output names and docstrings. Their data, loss, optimizer, schedule, augmentation and selection code is byte-identical (`provenance.json` → `matched_cnn_kernel_differences`). The kernel files are 261 lines (SegFormer) and 330 lines (CNN) long; `diffs/segformer_roya__vs__pg26038-mb-roya-unetpp-lrB.diff` is the line-by-line comparison behind the table below.
- The Kaggle notebooks `notebooks/pg26038-segformer-roya` and `pg26038-segformer-ojo` contain the same training code with Kaggle paths (12 differing lines, all paths and names; `diffs/*notebooks_sibling.diff`). [Verifier correction V1] The `arch` string stored in both published checkpoints (`SegFormer-MiT-B5-1024-roya-realpack`, `SegFormer-MiT-B5-1024-ojo-realpack`) is written only by the modal-recovery kernels (L209); the Kaggle notebook siblings write `SegFormer-MiT-B5-1024-roya` / `-ojo`, and the other rust notebook `notebooks/pg26038-roya-real-segformer-train` writes `SegFormer-MiT-B5-2048` (and has an 11.5-h guard). The published checkpoints were therefore produced by the modal-recovery kernel code (or a copy carrying the same `arch` string), so the 8-h stop applies. The platform is still not recorded: the salvage docstring calls the rust run "Kaggle-trained", but the kernel uses `/content/` paths.

### 1.2 Settings table (L = line number in the kernel file)

| Setting | SegFormer-B5 (published; rust and leaf-spot kernels identical) | Matched CNNs (lrB, 12 kernels) | Same? |
|---|---|---|---|
| Model / pretrained weights | `SegformerForSemanticSegmentation.from_pretrained` from `nvidia/mit-b5` (L115–120). Fallbacks: `nvidia/segformer-b5-finetuned-ade-640-640`, then random init (L121–127). Which one loaded is not logged anywhere on disk | `smp.UnetPlusPlus` / `smp.DeepLabV3Plus`, `mobilenet_v2`, `encoder_weights="imagenet"` (L193). Warm-start code exists (L210), but all 12 logs print `init ckpt: None` | differs (by design) |
| Data | roya/ojo real packs via `split.csv` (L56–63). Leaf-spot summary: n_train 518, n_val 110, n_test 110, 0 synthetic. Rust: salvage summary says "trained 100% real roya pack"; the val-IoU recomputation (§1.4) fits the same pack | Kaggle datasets `pg26038-{roya,ojo}-real-pack`. Logs: rust train 396 / val 84 / test 84 images (half healthy); leaf spot 518/110/110 | same (rust SegFormer inferred) |
| Labels | 3 classes; leaf = alpha > 8; lesion = mask > 127 AND leaf (L84–90) | same (L133–141) | same |
| Input | 1024 letterbox, aspect preserved, bilinear image / nearest label, centred black padding (L66–74); ImageNet mean/std (L51–52) | same (L86–91, L81–82) | same |
| Loss | 0.7·FocalTversky + 0.3·CE (L158) | same (L233) | same |
| Focal Tversky | lesion class; α = 0.3 (FP), β = 0.7 (FN), γ = 0.75 (L149–154) | same (L226–229) | same* |
| CE class weights | [0.25, 0.5, 3.0] for bg/leaf/lesion (L142) | same (L223) | same |
| Optimizer | AdamW; encoder 6e-5, decoder 6e-4; weight decay 0.01 on all parameters (L136–138) | same (L217–219) | same |
| How the LR was set | hand-set (no search) | best of {2e-5/2e-4, 6e-5/6e-4} on validation; 6e-5/6e-4 won in all 4 configurations | same values, different procedure |
| Schedule | CosineAnnealingLR(T_max = 60), stepped once per epoch (L140, L201) | same (L221, L287) | same |
| Epoch cap / patience | 60 / 15, strict improvement (L139, L190, L206–214) | 60 / 15 (L220, L278, L295–303) | same |
| Batch × accumulation | 1 × 32 plus gradient checkpointing (L109, L132) | 2 × 16 (L184) | effective 32 in both |
| Optimizer steps per epoch | rust 12, leaf spot 16 (drop_last; the partial accumulation is discarded) | rust 12, leaf spot 16 | same |
| Geometric augmentation | h-flip p = 0.5, v-flip p = 0.5, rot90 k ∈ {0,1,2,3} uniform (L97–100) | same (L148–152) | same |
| Photometric / glare | no such code | `PHOTOMETRIC = False` (L32), `glare=False` for training (L185) | same (off) |
| Checkpoint selection, leaf spot | val lesion IoU pooled over all 110 val images (L202, L206) | `score = IoU − FP`; FP is NaN (no healthy leaves) and is set to 0, so pooled val lesion IoU over 110 images (L288–289) | same |
| Checkpoint selection, rust | val lesion IoU pooled over **all 84 val images (42 rust + 42 healthy)** (L202, L206). Healthy images enter the union, a pixel-level FP penalty | lesion IoU pooled over the **42 rust val images** minus FP rate = share of the **42 healthy val images** with predicted severity ≥5% (L288–289, L272) | **differs** |
| Wall-clock stop | stop after any epoch once 8 h have elapsed (L215–216) | none | **differs** |
| AMP | FP16 autocast + GradScaler (L141, L195) | same (L222, L241) | same |
| Seeds | none (no `random`, NumPy or torch seed) | SEED 0/1/2 for `random`, NumPy, torch and CUDA (L30–31, L208) | **differs** |
| cuDNN determinism | not set | not set | same |
| Test evaluation / export | rust: salvage script re-evaluated `best.pt` and exported ONNX, opset 17 (docstring: Colab A100). Leaf spot: the kernel exported ONNX, opset 17 | kernel exports ONNX, opset 16 | differs (weights unaffected) |
| Platform | rust "Kaggle-trained" (salvage docstring); the kernel files use `/content/` paths; GPU per run not recorded | Kaggle, Tesla T4 (kernel-metadata and logs) | differs / unknown |
| Epochs completed | **unknown** (no log) | logged: 43–60 (see §1.4) | n/a |

\* The Focal Tversky sums run over the whole micro-batch. The term is therefore computed per image for SegFormer-B5 (batch 1) and jointly over 2 images for the CNNs. The formula and weights are identical, but the loss value is not strictly the same quantity. The CNNs' BatchNorm layers also see batches of 2.

### 1.3 Other observations from the files

- **Why the rust run needed salvage (code-level inference, no log).** After testing, the SegFormer kernel maps each leaf's grade through `order = {"G1": 0, "G2": 1, "G3": 2}` (L221, L230). The rust pack's grade labels are `roya` and `healthy`, so as written the rust kernel stops with a KeyError before it writes `summary.json` or the ONNX file. That is consistent with "died before test-eval/export". The leaf-spot pack uses G1–G3, so its kernel completed. Its kernel-internal grade metrics in `out/ojo/summary.json` (accuracy 1.0, QWK NaN, empty confusion matrix) are inconsistent and were not used anywhere here.
- The salvage docstring gives the rust run a validation IoU of 0.619. The checkpoint itself stores 0.6446, which the recomputation in §1.4 reproduces. The 0.619 in the docstring is not supported.
- The SegFormer kernel docstring still describes the older balanced dataset (`pg26038-ojo-balanced`, "train = real+synthetic"). The leaf-spot run's summary shows 0 synthetic images and 518 training images (the real pack), so that part of the docstring is stale.
- `datasets/pg26038-segformer-ckpt/best.pt` is **not** the published model. It has 650 zip entries against 1178, is 189 MB against 339 MB, has no `arch` key and stores val IoU 0.860. No checkpoint, published or not, stores an epoch.
- `modal-recovery/seeds_out/` holds seed 0–2 reruns of the published rust DeepLabV3+ and UNet++ configurations only. No SegFormer-B5 seed rerun exists on disk, so the Discussion's "SD 0.006 for SegFormer-B5" cannot be verified from files.

### 1.4 Rust checkpoint criterion: verification and counterfactual

- SegFormer-B5 rust: the checkpoint's `val_lesion_iou` = 0.644614. From the served ONNX (per_image.csv, val split), the pooled lesion IoU is 0.644638 over all 84 val images (|Δ| = 2.3e-05) and 0.656928 over the 42 diseased images. This confirms that the healthy validation images were inside SegFormer-B5's selection metric, and that the served ONNX is the selected checkpoint. The rust test figures match the salvage summary: 0.67391 against 0.67392 over 42 diseased images. For leaf spot, the checkpoint stores 0.881059; recomputed: 0.881058.
- Matched CNNs, from `history.json`. The selection rule is replayed with patience 15, first with the logged score and then with val lesion IoU alone. If the IoU-only rule would have needed epochs beyond the logged history, that is flagged.

| Run | Epochs run | Selected epoch (score) | Selected epoch (IoU only) | Same checkpoint | Val FP at selected epoch | Epochs with FP > 0 |
|---|---|---|---|---|---|---|
| pg26038-mb-roya-unetpp-lrB | 54 | 39 | 39 | yes | 0.000 | 1: 1.000, 2: 0.071, 3: 0.024, 6: 0.024 |
| pg26038-mbs-roya-unetpp-lrB-seed1 | 60 | 50 | 50 | yes | 0.000 | 1: 0.048, 2: 0.309 |
| pg26038-mbs-roya-unetpp-lrB-seed2 | 54 | 39 | 39 | yes | 0.000 | 1: 0.405, 3: 0.024, 4: 0.024 |
| pg26038-mb-roya-deeplab-lrB | 60 | 57 | 57 | yes | 0.000 | 1: 0.048 |
| pg26038-mbs-roya-deeplab-lrB-seed1 | 59 | 44 | 44 | yes | 0.000 | 1: 0.048 |
| pg26038-mbs-roya-deeplab-lrB-seed2 | 60 | 50 | 50 | yes | 0.000 | 25: 0.024, 28: 0.024, 29: 0.024, 30: 0.024, 31: 0.024 |
| pg26038-mb-ojo-unetpp-lrB | 57 | 42 | 42 | yes | n/a (no healthy val leaves) | none |
| pg26038-mbs-ojo-unetpp-lrB-seed1 | 60 | 53 | 53 | yes | n/a (no healthy val leaves) | none |
| pg26038-mbs-ojo-unetpp-lrB-seed2 | 60 | 54 | 54 | yes | n/a (no healthy val leaves) | none |
| pg26038-mb-ojo-deeplab-lrB | 43 | 28 | 28 | yes | n/a (no healthy val leaves) | none |
| pg26038-mbs-ojo-deeplab-lrB-seed1 | 60 | 46 | 46 | yes | n/a (no healthy val leaves) | none |
| pg26038-mbs-ojo-deeplab-lrB-seed2 | 60 | 45 | 45 | yes | n/a (no healthy val leaves) | none |

In every matched run, the replayed selected epoch equals `best_epoch` in summary.json. FP > 0 occurred only in the first few epochs (≤6), plus epochs 25 and 28–31 of rust DeepLabV3+ seed 2, never at a selected epoch. The "IoU minus FP" term therefore did not change any matched rust checkpoint. What cannot be computed is which epoch SegFormer-B5's pooled diseased + healthy IoU would have picked for the CNNs: their histories store diseased-only IoU.

### 1.5 ONNX identity (SHA-256)

| File | SHA-256 | Bytes | = out/ | = datasets/pg26038-onnx-models/ |
|---|---|---|---|---|
| `modal-recovery/serving/roya_segformer_realpack_1024.onnx` | `f3ffcae17dc0b59ab1639120f740c8428fd004765df8c489dabd6239771d783b` | 339818146 | yes | yes |
| `modal-recovery/serving/ojo_segformer_realpack_1024.onnx` | `c135695f7a1f4712b1473ba7587355c0798610994f483cf3fb9c75cca958865d` | 339818146 | yes | yes |

All six serving ONNX files are byte-identical to `datasets/pg26038-onnx-models/` (`provenance.json` → `sha256`). `out/` contains only the two SegFormer ONNX files.

### 1.6 Epochs completed by the SegFormer-B5 runs

The search covered 498 text files (.log/.txt/.json/.md/.out/.ipynb/.csv/.tex) under `modal-recovery/`, `datasets/pg26038-segformer-ckpt/`, `reanalysis/` (incl. `kaggle_runs/`), `notebooks/`, `kernel-outputs/`, `validation/`, `datasets/pg26038-onnx-models/` and the thesis folder. It looked for the SegFormer kernel's epoch print format (`epoch\s+\d+/60\s+loss=`). The only hits are two DeepLabV3+ logs in `kernel-outputs/` (arch recorded as DeepLabV3Plus-MobileNetV2-1024). No SegFormer training log exists. The checkpoints store only `state_dict`, `val_lesion_iou` and `arch`, with no epoch. **The number of epochs completed is unknown for both SegFormer-B5 runs**, and so is whether the 8-h stop, early stopping or the 60-epoch cap ended them. The manuscript's statement (Supplement `note_methods.tex`, inherited from the original draft `main.tex`/`body_draft.tex`) that "the rust run early-stopped at epoch 55 and the leaf-spot run used the full 60" has no supporting log. It should be removed or marked `% TODO(author)`.

### 1.7 Verdict

**The published SegFormer-B5 training is equivalent to the matched protocol except for the differences listed.** The data, labels, input pipeline, loss and its weights, optimizer, learning rates, weight decay, schedule, epoch cap, patience, effective batch size (with the same optimizer steps per epoch), geometric-only augmentation and AMP are the same. The differences:

1. The architecture and its pretrained encoder (by design). The pretrained source is not logged; the code tries `nvidia/mit-b5` first.
2. Micro-batch 1 × 32 with gradient checkpointing, against 2 × 16. The Focal Tversky term is therefore computed per image rather than over pairs.
3. **Rust checkpoint criterion.** SegFormer-B5 used pooled validation lesion IoU over diseased and healthy images. The CNNs used diseased-image IoU minus the healthy false-positive rate (≥5%). For the matched CNNs the FP term was 0 at every selected epoch. For leaf spot the criteria are identical.
4. **No fixed seed** (single run). The matched runs used seeds 0–2.
5. **8-h wall-clock stop.** The number of epochs actually completed is **unknown**.
6. The learning rates were hand-set for SegFormer-B5. For the CNNs they were chosen on validation, and the chosen pair equals SegFormer-B5's.
7. The rust run's test evaluation and ONNX export came from the salvage script, run on `best.pt`. The served ONNX is verified as the selected checkpoint by hash and by recomputing its validation IoU.
8. Platform/GPU (not recorded for SegFormer-B5) and ONNX opset (17 vs 16).

Suggested wording: "SegFormer-B5 was trained once, without a fixed seed, with the settings that the matched-protocol runs copy (loss, optimizer, learning rates, schedule, at most 60 epochs with patience 15, effective batch size 32, geometric augmentation only), plus an 8-h wall-clock limit; the number of epochs it completed was not logged. For rust, its checkpoint was selected on validation lesion IoU pooled over diseased and healthy leaves, whereas the matched convolutional models used diseased-leaf lesion IoU minus the healthy-leaf false-positive rate (predicted severity ≥5%), which was zero at every selected epoch." The paper should stop saying that SegFormer-B5 was "not retrained under the matched protocol".

## 2. Paired tests (test split)

### 2.1 Design

- Leaves: rust = the 21 diseased test leaves, abaxial face. Leaf spot = the 55 test leaves, mean of both faces (IoU = mean of the face IoUs; severity = mean of the face severities). The reference severity is identical across all 7 models (max |Δ| = 0 pp).
- Grade references. Rust: the grade of the reference-mask severity. Leaf spot: **mask-derived grade (primary**, per the round-4 plan) and the annotator's visual grade (secondary). QWK uses quadratic weights on grades 0–3. |Bias| is the absolute value of the model's mean signed error.
- Matched model, primary method (a) = **seed mean**. Each statistic is computed per seed on the same leaves and then averaged over seeds 0–2. For IoU, MAE, bias and accuracy this equals the statistic of the per-leaf seed means. |Bias| = |seed-mean bias|; QWK = mean of the 3 per-seed QWKs. The paired bootstrap resamples leaves (10,000 resamples, `default_rng(20260928)`, one index matrix per disease shared by all models and contrasts; percentile CIs). The Wilcoxon signed-rank test is two-sided, on the per-leaf IoU differences (per-leaf seed-mean IoU of the matched model), exact (no zero or tied differences occurred).
- Sensitivity method (b) = **seed × leaf bootstrap**. It uses the same leaf resamples, but in each resample one seed is drawn uniformly and independently for each matched model, and that seed's statistic is used.
- Bootstrap two-sided p = min(1, 2·min(P*(Δ ≤ 0), P*(Δ ≥ 0))). **Holm** is applied within each disease across the 3 contrasts, per endpoint (and separately for the Wilcoxon tests). Differences are the first model minus the second. ΔIoU > 0 favors the first model, ΔMAE < 0 and Δ|bias| < 0 favor the first, and ΔQWK > 0 and Δaccuracy > 0 favor the first.
- Resamples in which QWK was undefined (0/0) are dropped and counted. There were none here (`undefined_resamples` in paired_results.json).

### 2.2 Per-model values (point estimates; matched = mean ± SD (range) over 3 seeds, SD with ddof = 1)

**Coffee leaf rust (abaxial face, n = 21 leaves)**

| Model | Leaf IoU | MAE (pp) | Bias (pp) | |Bias| (pp) | Grade accuracy | QWK |
|---|---|---|---|---|---|---|
| SegFormer-B5 (published, single run) | 0.767 | 0.99 | +0.99 | 0.99 | 0.905 | 0.878 |
| UNet++ (matched, 3 seeds) | 0.759 ± 0.015 (0.746–0.776) | 0.83 ± 0.29 (0.50–1.02) | +0.82 ± 0.30 (+0.47–+1.01) | 0.82 ± 0.30 (0.47–1.01) | 0.921 ± 0.027 (0.905–0.952) | 0.898 ± 0.035 (0.878–0.938) |
| DeepLabV3+ (matched, 3 seeds) | 0.725 ± 0.010 (0.718–0.736) | 0.62 ± 0.25 (0.36–0.87) | +0.47 ± 0.32 (+0.13–+0.77) | 0.47 ± 0.32 (0.13–0.77) | 0.937 ± 0.027 (0.905–0.952) | 0.914 ± 0.032 (0.878–0.938) |

**American leaf spot (mean of both faces, n = 55 leaves)**

| Model | Leaf IoU | MAE (pp) | Bias (pp) | |Bias| (pp) | Grade accuracy (mask-derived) | QWK (mask-derived) | Grade accuracy, visual ref. | QWK, visual ref. |
|---|---|---|---|---|---|---|---|---|
| SegFormer-B5 (published, single run) | 0.884 | 0.220 | +0.164 | 0.164 | 0.945 | 0.950 | 0.836 | 0.825 |
| UNet++ (matched, 3 seeds) | 0.878 ± 0.004 (0.874–0.882) | 0.113 ± 0.011 (0.100–0.121) | +0.048 ± 0.008 (+0.039–+0.054) | 0.048 ± 0.008 (0.039–0.054) | 0.976 ± 0.010 (0.964–0.982) | 0.977 ± 0.010 (0.965–0.983) | 0.879 ± 0.010 (0.873–0.891) | 0.859 ± 0.012 (0.852–0.873) |
| DeepLabV3+ (matched, 3 seeds) | 0.847 ± 0.015 (0.830–0.856) | 0.209 ± 0.118 (0.134–0.345) | +0.090 ± 0.207 (−0.073–+0.323) | 0.138 ± 0.162 (0.019–0.323) | 0.958 ± 0.010 (0.945–0.964) | 0.961 ± 0.009 (0.950–0.966) | 0.861 ± 0.028 (0.836–0.891) | 0.845 ± 0.025 (0.825–0.873) |

For leaf-spot DeepLabV3+, the seed-level biases have mixed signs (+0.019, −0.073, +0.323 pp). The |Bias| column above is the mean ± SD of the per-seed |bias| (0.138 ± 0.162). The seed-mean |bias| used in the contrasts is |mean bias| = 0.090.

Cross-check with `pass3/seed_summary.json`: all per-seed IoU, MAE, bias, grade accuracy and QWK values match at 3 dp: **yes**. The leaf-spot grade accuracy and QWK in seed_summary.json use the visual reference and are compared with the visual-reference columns. SegFormer-B5 reproduces the published values: rust 0.767 / 0.99 / +0.99 / 0.90 / 0.88; leaf spot 0.884 / 0.22 / +0.16, with visual accuracy 0.84 and QWK 0.82, and mask-derived accuracy 0.95 and QWK 0.95 (as in `results/paired`). The mask-derived leaf-spot grade accuracy and QWK of the matched models are new here.

### 2.3 Paired differences

**Coffee leaf rust (abaxial face, n = 21 leaves)**

| Contrast | Endpoint | Δ (seed mean) | 95% CI (a) | p (a) | Holm p (a) | 95% CI (b, seed × leaf) | p (b) | Holm p (b) | Δ per matched seed |
|---|---|---|---|---|---|---|---|---|---|
| UNet++ (matched) − SegFormer-B5 | Leaf IoU | −0.009 | [−0.027, +0.009] | 0.339 | 0.339 | [−0.043, +0.019] | 0.680 | 0.680 | +0.008, −0.013, −0.022 |
| UNet++ (matched) − SegFormer-B5 | MAE (pp) | −0.16 | [−0.58, +0.23] | 0.451 | 0.451 | [−0.84, +0.42] | 0.684 | 1.000 | −0.49, −0.01, +0.03 |
| UNet++ (matched) − SegFormer-B5 | Bias (pp) | −0.17 | [−0.60, +0.21] | 0.407 | 0.407 | [−0.87, +0.41] | 0.672 | 0.779 | −0.52, −0.01, +0.02 |
| UNet++ (matched) − SegFormer-B5 | |Bias| (pp) | −0.17 | [−0.60, +0.21] | 0.407 | 0.407 | [−0.87, +0.41] | 0.672 | 0.780 | −0.52, −0.01, +0.02 |
| UNet++ (matched) − SegFormer-B5 | Grade accuracy | +0.016 | [0.000, +0.048] † | 0.697 | 1.000 | [0.000, +0.143] | 1.000 | 1.000 | +0.048, 0.000, 0.000 |
| UNet++ (matched) − SegFormer-B5 | QWK | +0.020 | [0.000, +0.081] † | 0.631 | 1.000 | [0.000, +0.189] | 1.000 | 1.000 | +0.060, 0.000, 0.000 |
| DeepLabV3+ (matched) − SegFormer-B5 | Leaf IoU | −0.042 | [−0.058, −0.026] | <0.0001 | <0.0001 | [−0.064, −0.017] | 0.0006 | 0.002 | −0.046, −0.049, −0.031 |
| DeepLabV3+ (matched) − SegFormer-B5 | MAE (pp) | −0.37 | [−0.74, −0.07] | 0.004 | 0.013 | [−1.04, +0.07] | 0.138 | 0.415 | −0.12, −0.35, −0.63 |
| DeepLabV3+ (matched) − SegFormer-B5 | Bias (pp) | −0.52 | [−0.90, −0.20] | <0.0001 | <0.0001 | [−1.31, −0.01] | 0.039 | 0.117 | −0.22, −0.47, −0.86 |
| DeepLabV3+ (matched) − SegFormer-B5 | |Bias| (pp) | −0.52 | [−0.90, −0.20] | <0.0001 | <0.0001 | [−1.29, −0.01] | 0.039 | 0.117 | −0.22, −0.47, −0.86 |
| DeepLabV3+ (matched) − SegFormer-B5 | Grade accuracy | +0.032 | [−0.063, +0.159] | 0.641 | 1.000 | [−0.095, +0.190] | 0.916 | 1.000 | +0.048, 0.000, +0.048 |
| DeepLabV3+ (matched) − SegFormer-B5 | QWK | +0.036 | [−0.096, +0.219] | 0.663 | 1.000 | [−0.176, +0.243] | 0.915 | 1.000 | +0.060, 0.000, +0.049 |
| UNet++ (matched) − DeepLabV3+ (matched) | Leaf IoU | +0.033 | [+0.020, +0.045] | <0.0001 | <0.0001 | [−0.002, +0.064] | 0.063 | 0.126 | +0.054, +0.037, +0.009 |
| UNet++ (matched) − DeepLabV3+ (matched) | MAE (pp) | +0.21 | [−0.02, +0.50] | 0.085 | 0.171 | [−0.49, +0.97] | 0.537 | 1.000 | −0.36, +0.34, +0.66 |
| UNet++ (matched) − DeepLabV3+ (matched) | Bias (pp) | +0.35 | [+0.09, +0.64] | 0.005 | 0.010 | [−0.44, +1.21] | 0.389 | 0.779 | −0.30, +0.46, +0.88 |
| UNet++ (matched) − DeepLabV3+ (matched) | |Bias| (pp) | +0.35 | [+0.09, +0.64] | 0.005 | 0.010 | [−0.44, +1.20] | 0.390 | 0.780 | −0.30, +0.46, +0.88 |
| UNet++ (matched) − DeepLabV3+ (matched) | Grade accuracy | −0.016 | [−0.111, +0.063] | 0.840 | 1.000 | [−0.143, +0.143] | 1.000 | 1.000 | 0.000, 0.000, −0.048 |
| UNet++ (matched) − DeepLabV3+ (matched) | QWK | −0.016 | [−0.142, +0.097] | 0.808 | 1.000 | [−0.226, +0.193] | 1.000 | 1.000 | 0.000, 0.000, −0.049 |

**American leaf spot (mean of both faces, n = 55 leaves)**

| Contrast | Endpoint | Δ (seed mean) | 95% CI (a) | p (a) | Holm p (a) | 95% CI (b, seed × leaf) | p (b) | Holm p (b) | Δ per matched seed |
|---|---|---|---|---|---|---|---|---|---|
| UNet++ (matched) − SegFormer-B5 | Leaf IoU | −0.006 | [−0.015, +0.001] | 0.119 | 0.119 | [−0.022, +0.004] | 0.259 | 0.259 | −0.010, −0.002, −0.006 |
| UNet++ (matched) − SegFormer-B5 | MAE (pp) | −0.107 | [−0.175, −0.044] | <0.0001 | <0.0001 | [−0.179, −0.041] | 0.0006 | 0.002 | −0.099, −0.101, −0.119 |
| UNet++ (matched) − SegFormer-B5 | Bias (pp) | −0.116 | [−0.190, −0.051] | <0.0001 | <0.0001 | [−0.192, −0.047] | 0.0002 | 0.0006 | −0.112, −0.110, −0.125 |
| UNet++ (matched) − SegFormer-B5 | |Bias| (pp) | −0.116 | [−0.190, −0.050] | 0.0006 | 0.002 | [−0.191, −0.046] | 0.001 | 0.003 | −0.112, −0.110, −0.125 |
| UNet++ (matched) − SegFormer-B5 | Grade accuracy (mask-derived) | +0.030 | [−0.012, +0.091] | 0.284 | 0.681 | [−0.018, +0.091] | 0.439 | 1.000 | +0.036, +0.018, +0.036 |
| UNet++ (matched) − SegFormer-B5 | QWK (mask-derived) | +0.026 | [−0.012, +0.079] | 0.289 | 0.738 | [−0.024, +0.082] | 0.439 | 1.000 | +0.032, +0.015, +0.032 |
| UNet++ (matched) − SegFormer-B5 | Grade accuracy, visual ref. | +0.042 | [0.000, +0.097] † | 0.071 | 0.212 | [0.000, +0.109] | 0.198 | 0.593 | +0.036, +0.055, +0.036 |
| UNet++ (matched) − SegFormer-B5 | QWK, visual ref. | +0.035 | [0.000, +0.082] † | 0.069 | 0.206 | [0.000, +0.098] | 0.198 | 0.593 | +0.028, +0.049, +0.028 |
| DeepLabV3+ (matched) − SegFormer-B5 | Leaf IoU | −0.037 | [−0.048, −0.028] | <0.0001 | <0.0001 | [−0.068, −0.021] | <0.0001 | <0.0001 | −0.029, −0.029, −0.054 |
| DeepLabV3+ (matched) − SegFormer-B5 | MAE (pp) | −0.011 | [−0.068, +0.044] | 0.716 | 0.716 | [−0.147, +0.158] | 0.688 | 0.688 | −0.086, −0.072, +0.125 |
| DeepLabV3+ (matched) − SegFormer-B5 | Bias (pp) | −0.074 | [−0.143, −0.015] | 0.012 | 0.012 | [−0.304, +0.195] | 0.661 | 1.000 | −0.145, −0.237, +0.159 |
| DeepLabV3+ (matched) − SegFormer-B5 | |Bias| (pp) | −0.074 | [−0.143, −0.015] | 0.012 | 0.024 | [−0.203, +0.195] | 0.719 | 1.000 | −0.145, −0.091, +0.159 |
| DeepLabV3+ (matched) − SegFormer-B5 | Grade accuracy (mask-derived) | +0.012 | [−0.012, +0.042] | 0.561 | 0.681 | [−0.018, +0.073] | 1.000 | 1.000 | +0.018, 0.000, +0.018 |
| DeepLabV3+ (matched) − SegFormer-B5 | QWK (mask-derived) | +0.010 | [−0.013, +0.039] | 0.557 | 0.738 | [−0.023, +0.060] | 1.000 | 1.000 | +0.016, 0.000, +0.015 |
| DeepLabV3+ (matched) − SegFormer-B5 | Grade accuracy, visual ref. | +0.024 | [0.000, +0.055] † | 0.071 | 0.212 | [0.000, +0.091] | 0.934 | 1.000 | +0.018, 0.000, +0.055 |
| DeepLabV3+ (matched) − SegFormer-B5 | QWK, visual ref. | +0.021 | [0.000, +0.050] † | 0.069 | 0.206 | [0.000, +0.093] | 0.934 | 1.000 | +0.013, 0.000, +0.049 |
| UNet++ (matched) − DeepLabV3+ (matched) | Leaf IoU | +0.031 | [+0.025, +0.039] | <0.0001 | <0.0001 | [+0.010, +0.060] | 0.004 | 0.008 | +0.019, +0.027, +0.048 |
| UNet++ (matched) − DeepLabV3+ (matched) | MAE (pp) | −0.095 | [−0.121, −0.074] | <0.0001 | <0.0001 | [−0.277, +0.003] | 0.086 | 0.171 | −0.013, −0.029, −0.244 |
| UNet++ (matched) − DeepLabV3+ (matched) | Bias (pp) | −0.042 | [−0.067, −0.019] | 0.0004 | 0.0008 | [−0.317, +0.150] | 0.694 | 1.000 | +0.033, +0.127, −0.284 |
| UNet++ (matched) − DeepLabV3+ (matched) | |Bias| (pp) | −0.042 | [−0.067, −0.016] | 0.021 | 0.024 | [−0.317, +0.056] | 0.779 | 1.000 | +0.033, −0.019, −0.284 |
| UNet++ (matched) − DeepLabV3+ (matched) | Grade accuracy (mask-derived) | +0.018 | [0.000, +0.048] † | 0.227 | 0.681 | [−0.018, +0.073] | 0.855 | 1.000 | +0.018, +0.018, +0.018 |
| UNet++ (matched) − DeepLabV3+ (matched) | QWK (mask-derived) | +0.016 | [0.000, +0.045] † | 0.246 | 0.738 | [−0.023, +0.068] | 0.855 | 1.000 | +0.016, +0.015, +0.017 |
| UNet++ (matched) − DeepLabV3+ (matched) | Grade accuracy, visual ref. | +0.018 | [0.000, +0.048] † | 0.241 | 0.241 | [−0.036, +0.091] | 0.923 | 1.000 | +0.018, +0.055, −0.018 |
| UNet++ (matched) − DeepLabV3+ (matched) | QWK, visual ref. | +0.014 | [0.000, +0.039] † | 0.243 | 0.243 | [−0.048, +0.084] | 0.923 | 1.000 | +0.014, +0.049, −0.021 |

† A percentile bound pinned at exactly 0: at least 2.5% of resamples showed no difference at all, because the few discordant leaves all favor one model. Such an interval understates that tail. "Δ per matched seed" gives the point differences for seeds 0, 1 and 2 of the matched model (for UNet++ − DeepLabV3+, the same seed index for both).

### 2.4 Wilcoxon signed-rank tests on per-leaf IoU (matched: per-leaf seed mean)

| Disease | Contrast | n | Leaves favoring the first / second model | Median Δ | W | p (exact) | Holm p |
|---|---|---|---|---|---|---|---|
| Rust | UNet++ (matched) − SegFormer-B5 | 21 | 10 / 11 | −0.0006 | 101 | 0.6333 | 0.6333 |
| Rust | DeepLabV3+ (matched) − SegFormer-B5 | 21 | 3 / 18 | −0.0399 | 11 | 5.2e−05 | 0.0002 |
| Rust | UNet++ (matched) − DeepLabV3+ (matched) | 21 | 17 / 4 | +0.0413 | 18 | 0.0002 | 0.0005 |
| Leaf spot | UNet++ (matched) − SegFormer-B5 | 55 | 26 / 29 | −0.0005 | 713 | 0.6385 | 0.6385 |
| Leaf spot | DeepLabV3+ (matched) − SegFormer-B5 | 55 | 2 / 53 | −0.0249 | 29 | 9.7e−14 | 1.9e−13 |
| Leaf spot | UNet++ (matched) − DeepLabV3+ (matched) | 55 | 53 / 2 | +0.0245 | 17 | 1.2e−14 | 3.5e−14 |

Reading:

- Rust IoU: matched UNet++ and SegFormer-B5 cannot be told apart (10 vs 11 leaves; p = 0.63). The three seed differences (+0.008, −0.013, −0.022) straddle 0. Matched DeepLabV3+ is below both SegFormer-B5 and matched UNet++ (Holm p < 0.001). The UNet++ − DeepLabV3+ contrast is not robust to seed choice: its seed × leaf CI crosses 0.
- Rust severity: in the seed-mean analysis DeepLabV3+ has a lower MAE and |bias| than SegFormer-B5 (ΔMAE −0.37 pp, Holm p = 0.013; Δ|bias| −0.52 pp, Holm p < 0.001). In the seed × leaf analysis no MAE contrast excludes 0; only the DeepLabV3+ − SegFormer-B5 bias difference still does (95% CI −1.31 to −0.01 pp). The rust MAE of the matched models varies by up to about 0.5 pp between seeds, so any rust severity ranking depends on the seed.
- Rust grades: no contrast is detectable (all Holm p = 1.00). The CIs are wide because 21 leaves give only a few discordant grades.
- Leaf spot: matched UNet++ has a slightly lower IoU than SegFormer-B5 (−0.006; CI includes 0; Wilcoxon p = 0.64) and about half its MAE (−0.107 pp; Holm p < 0.001 in (a), 0.002 in (b)). Matched DeepLabV3+ is below SegFormer-B5 and UNet++ in IoU (53 of 55 leaves). No grade contrast is significant with either reference.

## 3. Equivalence (TOST)

**The equivalence margins were fixed in the round-4 A1 task specification (they are not written in `round4_plan.md`) after the primary results were known. They are post hoc and were not preregistered.** Margins: leaf IoU ±0.03 (about twice the largest seed SD, 0.015); MAE ±0.5 pp (the G0/G1 cut-point); QWK ±0.05. No margin was fixed for bias, |bias| or grade accuracy, so no TOST is reported for them. Test: two one-sided tests at α = 0.05 via the 90% percentile bootstrap CI of the paired difference. **Equivalent**: the 90% CI lies inside (−m, +m). **Not equivalent**: the 90% CI lies entirely outside [−m, +m]. **Inconclusive**: otherwise. The bootstrap TOST p = max(P*(Δ ≤ −m), P*(Δ ≥ +m)); Holm is applied within disease across the 3 contrasts. Method (a), seed mean, is primary; method (b), seed × leaf, is the sensitivity analysis.

| Disease | Contrast | Endpoint (margin) | Δ | 90% CI (a) | TOST p (a) [Holm] | Verdict (a) | 90% CI (b) | TOST p (b) [Holm] | Verdict (b) |
|---|---|---|---|---|---|---|---|---|---|
| Rust | UNet++ (matched) − SegFormer-B5 | Leaf IoU (±0.03) | −0.009 | [−0.024, +0.006] | 0.011 [0.033] | equivalent | [−0.037, +0.016] | 0.105 [0.315] | inconclusive |
| Rust | UNet++ (matched) − SegFormer-B5 | MAE (±0.5 pp) | −0.16 | [−0.51, +0.18] | 0.055 [0.109] | inconclusive | [−0.73, +0.34] | 0.160 [0.479] | inconclusive |
| Rust | UNet++ (matched) − SegFormer-B5 | QWK (±0.05) | +0.020 | [0.000, +0.068] | 0.133 [0.399] | inconclusive | [0.000, +0.139] | 0.194 [0.582] | inconclusive |
| Rust | DeepLabV3+ (matched) − SegFormer-B5 | Leaf IoU (±0.03) | −0.042 | [−0.056, −0.029] | 0.930 [1.000] | inconclusive | [−0.061, −0.021] | 0.833 [1.000] | inconclusive |
| Rust | DeepLabV3+ (matched) − SegFormer-B5 | MAE (±0.5 pp) | −0.37 | [−0.67, −0.11] | 0.210 [0.210] | inconclusive | [−0.91, +0.02] | 0.298 [0.479] | inconclusive |
| Rust | DeepLabV3+ (matched) − SegFormer-B5 | QWK (±0.05) | +0.036 | [−0.072, +0.182] | 0.404 [0.521] | inconclusive | [−0.127, +0.205] | 0.469 [0.679] | inconclusive |
| Rust | UNet++ (matched) − DeepLabV3+ (matched) | Leaf IoU (±0.03) | +0.033 | [+0.022, +0.044] | 0.708 [1.000] | inconclusive | [+0.004, +0.061] | 0.573 [1.000] | inconclusive |
| Rust | UNet++ (matched) − DeepLabV3+ (matched) | MAE (±0.5 pp) | +0.21 | [+0.01, +0.44] | 0.024 [0.071] | equivalent (not after Holm) | [−0.40, +0.86] | 0.223 [0.479] | inconclusive |
| Rust | UNet++ (matched) − DeepLabV3+ (matched) | QWK (±0.05) | −0.016 | [−0.118, +0.074] | 0.260 [0.521] | inconclusive | [−0.187, +0.149] | 0.340 [0.679] | inconclusive |
| Leaf spot | UNet++ (matched) − SegFormer-B5 | Leaf IoU (±0.03) | −0.006 | [−0.013, 0.000] | <0.0001 [<0.0001] | equivalent | [−0.018, +0.002] | 0.003 [0.010] | equivalent |
| Leaf spot | UNet++ (matched) − SegFormer-B5 | MAE (±0.5 pp) | −0.107 | [−0.164, −0.053] | <0.0001 [<0.0001] | equivalent | [−0.166, −0.050] | <0.0001 [<0.0001] | equivalent |
| Leaf spot | UNet++ (matched) − SegFormer-B5 | QWK (±0.05), mask-derived | +0.026 | [−0.006, +0.068] | 0.155 [0.155] | inconclusive | [−0.017, +0.071] | 0.177 [0.177] | inconclusive |
| Leaf spot | UNet++ (matched) − SegFormer-B5 | QWK (±0.05), visual | +0.035 | [+0.006, +0.073] | 0.228 [0.228] | inconclusive | [0.000, +0.084] | 0.247 [0.356] | inconclusive |
| Leaf spot | DeepLabV3+ (matched) − SegFormer-B5 | Leaf IoU (±0.03) | −0.037 | [−0.046, −0.030] | 0.937 [1.000] | inconclusive | [−0.063, −0.023] | 0.592 [0.818] | inconclusive |
| Leaf spot | DeepLabV3+ (matched) − SegFormer-B5 | MAE (±0.5 pp) | −0.011 | [−0.058, +0.034] | <0.0001 [<0.0001] | equivalent | [−0.133, +0.149] | <0.0001 [<0.0001] | equivalent |
| Leaf spot | DeepLabV3+ (matched) − SegFormer-B5 | QWK (±0.05), mask-derived | +0.010 | [−0.009, +0.033] | 0.005 [0.015] | equivalent | [−0.016, +0.049] | 0.048 [0.143] | equivalent (not after Holm) |
| Leaf spot | DeepLabV3+ (matched) − SegFormer-B5 | QWK (±0.05), visual | +0.021 | [+0.004, +0.044] | 0.023 [0.046] | equivalent | [0.000, +0.080] | 0.158 [0.356] | inconclusive |
| Leaf spot | UNet++ (matched) − DeepLabV3+ (matched) | Leaf IoU (±0.03) | +0.031 | [+0.026, +0.038] | 0.616 [1.000] | inconclusive | [+0.014, +0.056] | 0.409 [0.818] | inconclusive |
| Leaf spot | UNet++ (matched) − DeepLabV3+ (matched) | MAE (±0.5 pp) | −0.095 | [−0.116, −0.077] | <0.0001 [<0.0001] | equivalent | [−0.263, −0.001] | <0.0001 [<0.0001] | equivalent |
| Leaf spot | UNet++ (matched) − DeepLabV3+ (matched) | QWK (±0.05), mask-derived | +0.016 | [0.000, +0.039] | 0.012 [0.025] | equivalent † | [−0.016, +0.058] | 0.082 [0.163] | inconclusive |
| Leaf spot | UNet++ (matched) − DeepLabV3+ (matched) | QWK (±0.05), visual | +0.014 | [0.000, +0.034] | 0.004 [0.013] | equivalent † | [−0.036, +0.068] | 0.119 [0.356] | inconclusive |

† The equivalence rests on a 90% CI with one bound pinned at exactly 0. Treat it with caution.

Reading:

- **Rust, UNet++ vs SegFormer-B5.** Leaf IoU is equivalent within ±0.03 under the seed-mean analysis (90% CI −0.024 to +0.006; survives Holm). It is inconclusive when seed choice is propagated (−0.037 to +0.016), because the three UNet++ seeds span 0.746–0.776. MAE and QWK are inconclusive: the rust MAE of UNet++ varies by about 0.5 pp across seeds, the size of the margin itself. The honest statement is "comparable, and equivalent in leaf IoU if seed variability is averaged; not shown when it is propagated".
- **Rust, DeepLabV3+ vs SegFormer-B5.** Leaf IoU is inconclusive: the 90% CI is −0.056 to −0.029, below 0 and mostly beyond −0.03 without being entirely outside it. MAE and QWK are inconclusive.
- **Leaf spot, UNet++ vs SegFormer-B5.** Leaf IoU (90% CI −0.013 to 0.000) and MAE (−0.16 to −0.05 pp) are equivalent under both methods. QWK is inconclusive with both grade references.
- **Leaf spot, DeepLabV3+ vs SegFormer-B5.** MAE and mask-derived QWK are equivalent. The mask-derived QWK is equivalent under both methods but does not survive Holm in method (b); the visual-reference QWK is equivalent in (a) only. Leaf IoU is inconclusive (90% CI −0.046 to −0.0295, just inside −0.03 at the upper end).
- **UNet++ vs DeepLabV3+.** Rust: MAE is equivalent in method (a) only and does not survive Holm; IoU and QWK are inconclusive. Leaf spot: MAE is equivalent under both methods; QWK is equivalent in (a) only, with the lower bound pinned at 0; IoU is inconclusive (+0.026 to +0.038).
- No contrast met "not equivalent". Every point estimate is smaller than its margin except the DeepLabV3+ IoU contrasts (−0.042 rust, −0.037 leaf spot, UNet++ − DeepLabV3+ +0.033 and +0.031), which lie just beyond ±0.03 with CIs that reach inside it.

## 4. [Section omitted from the public release]

This section listed the expert-panel identities (names, occupations, affiliations, years of experience). Identities are not part of the code release.

## 5. What could not be determined from the files

- The number of epochs each SegFormer-B5 run completed, and what ended each run (8-h stop, patience or the 60-epoch cap).
- Which pretrained source the SegFormer kernel actually loaded (`nvidia/mit-b5` is tried first; the fallbacks are not logged). Also the GPU and platform of each SegFormer-B5 run.
- ~~Which copy of the identical training code produced each published checkpoint.~~ Resolved by the verifier from the stored `arch` strings (see §1.1, V1): the modal-recovery kernel code. Where each run executed (Kaggle, Colab or Modal) remains unknown.
- SegFormer-B5's seed-to-seed variability: it was a single run with no seed set, and no SegFormer rerun exists on disk.
- Which checkpoints SegFormer-B5's rust criterion (pooled IoU with the healthy images included) would have picked for the CNNs: their histories store diseased-only IoU.

## Verifier corrections

Independent verifier, 28 Sep 2026. Everything below was re-derived with new code (`verifier_independent.py` → `verifier_independent.json`, `verifier_independent.log`); only the paper's grade/QWK definitions were reused.

What was re-checked and found correct:
- **Leaf sets and protocol.** Rust: 21 diseased test leaves, abaxial face; leaf spot: 55 test leaves, mean of both faces; the reference severity is identical across all 7 model/seed tables (max |Δ| = 0); grade cut-points 0.5/5/20%; mask-derived and visual leaf-spot references as described. Every per-seed IoU, MAE, bias, grade accuracy and QWK in §2.2 matches `pass3/seed_summary.json` at 3 dp, and the SegFormer-B5 values reproduce the published ones.
- **Paired statistics.** Point differences, 95% and 90% percentile CIs, bootstrap p, Holm (within disease across the 3 contrasts), TOST p and every equivalence verdict in §2.3 and §3 were reproduced. Method (a) rust CIs match to all printed digits (same RNG stream). Method (b) and the leaf-spot intervals differ only by Monte Carlo noise (≤0.005 on IoU and QWK, ≤0.03 pp on MAE and bias; one rust accuracy bound differs by one leaf, 1/21, because accuracy is discrete), because the verifier drew the resamples in a different order; no verdict or significance statement changes. Exact Wilcoxon p-values and Holm values in §2.4 match exactly (rust 0.6333 / 5.2e−05 / 2.4e−04; leaf spot 0.6385 / 9.7e−14 / 1.2e−14).
- **Provenance.** Line references in §1.2 were checked against both kernels; the 12 matched kernels differ only in architecture, seed and names; the SHA-256 values of the served, `out/` and `datasets/pg26038-onnx-models/` SegFormer ONNX files are identical (rust `f3ffcae1…`, leaf spot `c135695f…`); the checkpoint pickles store only `state_dict`, `val_lesion_iou` (0.644614, 0.881059) and `arch`, with no epoch; the pooled validation IoU recomputation (0.644638 over 84 images, 0.656928 over 42 diseased) and the checkpoint-selection replay of all 12 matched runs (selected epoch identical with and without the FP term; FP term 0 at every selected rust epoch) were reproduced; the rust pack's `split.csv` grades are only `roya`/`healthy` (198/198 train, 42/42 val, 42/42 test images), so the KeyError at L230 follows; no SegFormer training log or SegFormer seed rerun exists (`modal-recovery/seeds_out/` holds DeepLabV3+ and UNet++ only).
[Line omitted from the public release: expert-identity detail.]

Corrections made:
- **V1 (§1.1, §5).** The claim that the files do not record which code copy produced the published checkpoints was wrong. The stored `arch` strings identify the modal-recovery kernels (see §1.1). The platform/GPU remains unknown.
- **V2 (§3).** The equivalence margins are not in `round4_plan.md`. They were set in the round-4 A1 task specification. The wording was corrected; they remain post hoc and not preregistered.

Clarifications (no number changed):
- The leaf-spot mask-derived grade, now the primary leaf-spot reference, is not independent of the models, because all models were trained on these masks (`results/paired`). The high mask-derived accuracies (0.945–0.976) and QWKs (0.950–0.977) should be read with that in mind.
- Rust UNet++ bias SD is 0.3047. It is printed here as ±0.30 and as ±0.31 in `seed_facts.md` (rounding of 0.305). Use one form consistently (0.30 is the correct 2-dp rounding).
- Rust DeepLabV3+ − SegFormer-B5 leaf IoU is significantly below 0 but not "not equivalent" at ±0.03: the 90% CI upper bound (−0.029) lies just inside the margin. The paper should say "lower, and equivalence not shown", not "not equivalent".

### Second independent verifier (28 Sep 2026)

[Line omitted from the public release: expert-identity detail.]

Result: every point estimate, per-seed value, SD, Wilcoxon statistic/p and every equivalence verdict (both methods) in §2–§3 reproduced; CIs agree within Monte Carlo noise (≤0.008 on IoU and QWK, ≤0.03 pp on MAE and bias). The pooled rust validation IoU of the served SegFormer-B5 ONNX is 0.644638 over 84 images and 0.656928 over the 42 diseased images; leaf spot 0.881058.

Corrections made:
- **V3.** `paired_results.json` and `results.json` still said the margins were "fixed in round4_plan.md". Changed to the round-4 A1 task specification (post hoc, not preregistered), consistent with V2.

Clarifications (no number changed):
- **Bootstrap p-values printed as "<0.0001" are at the Monte Carlo resolution limit.** With the second generator, leaf-spot UNet++ − SegFormer-B5 MAE gave Holm p(a) = 0.0008 (reported <0.0001) and rust DeepLabV3+ − SegFormer-B5 MAE gave Holm p(a) = 0.019 (reported 0.013). For the paper use "p < 0.001" for the former and "Holm p < 0.05" (or the protocol-seed value 0.013) for the latter. All other significance statements are unaffected.
- **Near-threshold verdicts.** Leaf-spot DeepLabV3+ − SegFormer-B5 visual-reference QWK is equivalent in (a) with Holm TOST p = 0.046 (0.042 with the second generator), and rust UNet++ − SegFormer-B5 leaf IoU with Holm TOST p = 0.033 (0.030). Both hold, but both are close to 0.05 and rest on post hoc margins.
- **† in §3.** For the leaf-spot UNet++ − DeepLabV3+ QWK equivalence, the bound pinned at 0 is the lower one and lies far from −0.05; the verdict is decided by the upper bound (0.039 mask-derived, 0.034 visual), so the pinned bound does not weaken that equivalence. The caution applies to reading the 90% CI as excluding negative differences, not to the equivalence call.
[Line omitted from the public release: expert-identity detail.]
