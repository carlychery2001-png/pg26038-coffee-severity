# pg26038-coffee-severity

Code and derived data for the paper

> **Semantic Segmentation for Leaf-Level Severity Grading of Coffee Leaf Rust and American Leaf Spot on Flatbed Scans**
> Carly Chery and Victor Hugo Morales Pena. Manuscript in revision (MDPI *Agriculture*, PG-26038).

<!-- RELEASE-URL: https://github.com/carlychery2001-png/pg26038-coffee-severity (named in the paper's Data Availability statement); confirm after the author approves the push; add the Zenodo DOI of this repository when minted. -->

## Purpose

The paper compares three semantic-segmentation architectures (DeepLabV3+, UNet++ and SegFormer-B5) for leaf-level lesion
segmentation and G0–G3 severity grading of coffee leaf rust (*Hemileia vastatrix*) and American leaf spot
(*Mycena citricolor*) on flatbed scans of detached leaves (one site per disease). This repository holds:

- the training scripts of the original models and of the matched-protocol retraining;
- the evaluation code that re-runs the released ONNX models and writes one row per scan (the per-image tables);
- every analysis that turns the per-image tables into the paper's tables, with its verified `results.md` / `results.json`;
- the split manifests and the per-image and per-leaf prediction tables. Individual expert grades are not released (see Privacy).

Terms used throughout (as in the paper):

- **Original models**: the three models per disease trained for the first author's thesis at EARTH University and
  reported in the first submission; each is a single training run without a fixed seed.
- **Matched-protocol retraining**: DeepLabV3+ and UNet++ retrained for the revision with the SegFormer-B5 settings,
  seeds 0, 1 and 2.
- **Reanalysis**: every evaluation recomputed locally (CPU, ONNX Runtime) from the ONNX files with one protocol:
  leaf = alpha > 8; lesion = mask > 127 and leaf; aspect-preserving letterbox to 1024 × 1024; ImageNet normalization;
  raw argmax; rust graded on the abaxial face, American leaf spot on the mean of both faces; grades G0 < 0.5%,
  G1 0.5–<5%, G2 5–<20%, G3 ≥ 20% of leaf area.

## Repository map

The folder layout mirrors the working tree in which the analyses were run, so the scripts find the released tables at
the paths they expect (see *Paths* below).

```
modal-recovery/kernels/            SegFormer-B5 training kernels of the original models (rust, leaf spot) and the
                                   script that evaluated/exported the rust SegFormer-B5 checkpoint
reanalysis/
  kaggle_src/                      original rust CNN kernels (DeepLabV3+, UNet++; templates of the matched runs) and two
                                   further UNet++ kernels kept with them
  make_kaggle_runs.py              generates the matched-protocol kernels from the templates (phase 1: seed 0,
                                   learning-rate pairs lrA and lrB)
  gen_seeds.py                     generates the seed-1 and seed-2 kernels at the learning rate chosen on validation
  kaggle_scheduler.py              pushes queued kernels to Kaggle, polls them and downloads outputs
  kaggle_runs/<run>/<run>.py       the 16 matched-protocol kernels that were run (code only; lrB = chosen LR pair)
  run_inference.py                 re-runs the six original ONNX models -> out/per_image.csv
  eval_retrained.py                evaluates the matched-run ONNX models -> out/per_image_retrained.csv
  out/per_image.csv                original models: one row per (model, image pack, split, scan); pixel counts, IoU, severity
  out/per_image_retrained.csv      matched runs (lrA and lrB; seeds 0-2 for lrB): same columns plus the run name
  pass3/seed_summary.json          per-seed leaf-level summaries of the matched runs
  results/<analysis>/              analysis scripts + results.md + results.json (see the table below)
datasets/
  pg26038-roya-real-pack/split.csv         rust pack manifest (rust and healthy leaves; train/val/test)
  pg26038-ojo-real-pack/meta/split.csv     American leaf spot manifest (visual grade G1-G3; train/val/test)
validation/                        expert-panel scripts and tables
  form_image_key.csv               form item (ojo01-ojo30, roya01-roya30) -> leaf ID, matching score, model split
  model_predictions.csv            original models on the 60 form images (severity per face and mean)
```

Analysis folders under `reanalysis/results/`:

| Folder | What it computes |
|---|---|
| `agreement` | severity agreement of the original models (MAE, bias, Bland–Altman, error decomposition, boundary F1) |
| `paired` | paired comparisons between the original models (IoU, severity, grade) |
| `detection` | detection, specificity and cross-disease behavior |
| `structure` | validation-set results, data structure, date clustering, grade distributions, class weights |
| `baseline` | training-free color-threshold baseline (first version; sensitivity rows) |
| `figures`, `fig2_grades`, `fig11_specificity` | figure scripts and their numbers |
| `panel_stats`, `panel_figures` | earlier four-evaluator expert-panel statistics and figures |
| `r4_arch` | SegFormer-B5 provenance; matched-protocol models vs SegFormer-B5: paired tests and TOST equivalence |
| `r4_baseline` | detector-independent CIELAB a* color baseline with leaf-edge erosion; healthy-leaf specificity |
| `r4_grades` | American leaf spot grade agreement with the mask-derived (primary) and visual (secondary) grade |
| `r4_oppoint` | rust operating point (validation-calibrated lesion-probability threshold) |
| `r4_panel` | four-evaluator panel with matched models; selected vs unselected leaves; form image vs scan |
| `r4_panel5` | five-evaluator panel (primary panel analysis of the paper) |
| `r4_related` | related-work table (literature values) |

## What is not in this repository

- **Images and masks.** The held-out test images are not public until peer review is complete. A demonstration subset
  (50 validation leaves, 100 scans with lesion masks) is on Hugging Face (see *Links*).
- **Models.** The six original ONNX models are on Kaggle (see *Links*). Checkpoints and ONNX files of the matched runs,
  their training logs and per-epoch histories are not included.
- **Expert-panel identities and raw form data.** The raw form export (names, timestamps, affiliations) and every file
  that contains or processes expert identities are excluded: `r4_arch/experts*`, `r4_arch/make_md.py`,
  `r4_grades/verify_experts.py`, `r4_grades/experts_verified.json`, `r4_grades/make_md.py`, `r4_panel/make_md.py`,
  `r4_related/build_table.py`, `validation/facts.py`, and the working file with the evaluators' first names. The
  report generator `r4_panel5/make_md.py` and `r4_panel5/results.json` are also withheld (per-evaluator detail).
- **Verifier scratch.** Independent verification scripts, their outputs and backups (`verifier*/`, `verify/`,
  `verifier*.py`), intermediate `work/` folders (except the two per-leaf tables of `agreement/work/`), figures and binaries.
- **The form images** of the expert panel (`validation/imagenes/`).

## Environment (versions actually used)

Reanalysis and all statistics (local CPU, Windows 11; ONNX Runtime CPU execution provider):

| Package | Version |
|---|---|
| Python | 3.13.15 |
| numpy | 2.5.3 |
| scipy | 1.18.1 |
| onnxruntime | 1.30.0 |
| Pillow | 12.3.0 |
| matplotlib | 3.11.2 (figures) |
| PyMuPDF | 1.28.2 (only `results/figures/make_qualitative_figures.py`) |
| python-docx | 1.2.0 (only `validation/build_section.py`) |

Python, numpy and scipy versions are also recorded in several `results.json` files.

```
pip install numpy==2.5.3 scipy==1.18.1 onnxruntime==1.30.0 pillow==12.3.0 matplotlib==3.11.2
```

Training:

- Matched-protocol retraining: Kaggle, NVIDIA Tesla T4; PyTorch 2.10.0+cu128 (printed at the start of every run log).
  `segmentation-models-pytorch` was installed at run time without a version pin, and its version was not logged.
- Original models: the SegFormer-B5 kernels use Hugging Face `transformers` (`SegformerForSemanticSegmentation`);
  no training log of the original runs exists, so library versions, completed epochs and the platform are not recorded.
  The original CNN runs likewise have no recorded library versions.

<!-- TODO(author): confirm which scripts produced the original American leaf spot CNN models, and the role of
reanalysis/kaggle_src/pg26038-unetpp-ojo and pg26038-unetpp-roya (their docstrings are inherited from a fine-tuning kernel). -->

## Paths

Every script defines `ROOT`, the root of the working tree. It is read from the environment variable `PG26038_ROOT`
and defaults to the root of this repository. Inputs that are not released must be placed under `ROOT` at the paths the
scripts expect:

| Input | Path under `ROOT` |
|---|---|
| scans and masks | `datasets/pg26038-roya-real-pack/{images,masks}/`, `datasets/pg26038-ojo-real-pack/{images,masks}/` |
| original ONNX models (file names in `run_inference.py`) | `modal-recovery/serving/` |
| matched-run ONNX models | `reanalysis/kaggle_runs/<run>/output/<run>.onnx` |
| predicted class maps (written by `run_inference.py`) | `reanalysis/out/preds/` |
| expert-panel form images | `validation/imagenes/` |

Other optional variables: `KAGGLE_CLI` (Kaggle command-line tool), `PG26038_APP_DIR` (the prototype app, used only by
`validation/run_models.py` and a hash check in `r4_panel/run_forms.py`), `PG26038_OUTDIR` (`validation/build_mdpi.py`),
`PG26038_MANUSCRIPT_FIGS` (optional copy target of `r4_panel5/figures/make_panel_figures.py`), `PG26038_DOCS_DIR`
(extra folder searched for logs by `r4_arch/provenance.py`).

In the released `results.md` / `results.json` files, local paths are written as `<ROOT>` and `<USER_HOME>`. Apart from
that, and from the privacy redactions listed at the top of the affected `results.md` files, the result files are the
verified versions used for the paper. Several `results.md` files were corrected by hand after verification; re-running
a `make_md.py` overwrites those corrections, so compare numbers against `results.json`.

## Reproducing the tables from the per-leaf tables

The pipeline has two steps. (1) `reanalysis/run_inference.py` and `reanalysis/eval_retrained.py` run the ONNX models on
the scans and write `reanalysis/out/per_image.csv` and `per_image_retrained.csv`; this needs the images and models.
(2) Every table is computed from those per-image tables (plus the manifests), mostly without images. A leaf's value is
the abaxial-face row for rust and the mean of the two face rows for American leaf spot; leaf IoU is taken from the
pixel counts (`inter_lesion`, `union_lesion`) and severity from `ref_sev` / `pred_sev`.

Most scripts write `results.json` (and, via `make_md.py`, `results.md`) into their own folder, so re-running them
overwrites the released copies; use `git diff` to compare.

| Paper float (LaTeX label) | Script(s) | Inputs in this repository | Also needs |
|---|---|---|---|
| Dataset composition (`tab1`) | counts from the manifests; per-split tables in `results/structure/structure_analysis.py` | both `split.csv` | masks, for class pixel frequencies |
| Original models: segmentation and severity (`tab2`/`tab4`), Bland–Altman (`fig5`) | leaf IoU, MAE and bias directly from `per_image.csv` (test split, own-disease pack); agreement statistics in `results/agreement/agreement.py` | `per_image.csv`, `agreement/work/*.csv` | nothing (regenerating `agreement/work/` with `pixel_metrics.py` needs masks and class maps). The 95% CIs printed in `tab2` are those of the original submission, whose bootstrap seed was not recorded |
| Grade agreement (`tab5`) | leaf spot: `results/r4_grades/grades_analysis.py`; rust QWK intervals: `results/r4_baseline` | `per_image.csv`, `per_image_retrained.csv`, `seed_summary.json`, `results/baseline/per_leaf_test.csv`, `results/paired/*` | `r4_baseline.py` as a whole needs images |
| Color thresholds, matched models, equivalence (`tab:baseline`/`tab:retrain`/`tab:equiv`) | (a) color rows: `results/r4_baseline/r4_baseline.py`, paired intervals re-derived by `paired_extra.py` from `per_leaf_test.csv`; matched rows and (b) paired differences and TOST: `results/r4_arch/paired_tests.py` | `per_image.csv`, `per_image_retrained.csv`, `seed_summary.json`, `r4_baseline/per_leaf_*.csv` | images, only to recompute the color thresholds |
| Rust operating point (Results text; Supplement `tab:r4_oppoint`, `tab:r4_oppoint_models`) | `results/r4_oppoint/infer_prob.py`, `oppoint.py` | `per_image_retrained.csv`, manifests | images and ONNX models (lesion probabilities are recomputed) |
| Expert panel (`tab:expert_models`, `fig:expert_forest`, `fig:expert_perleaf`, `fig:app_panel_severity`; Supplement `tab:expert_irr`, `tab:r4_panel`, `tab:r4_panel_sens`, `tab:r4_panel_ref`) | `results/r4_panel5/panel5_analysis.py`, `crosscheck.py`, `figures/make_panel_figures.py` | individual expert grades (not released; see Privacy), `form_image_key.csv`, `model_predictions.csv`, `results/r4_panel/form_predictions.csv`, `per_image*.csv`, leaf-spot `split.csv` | the (empty) masks of the healthy form leaves; see the note below |
| Selected vs unselected leaves, form image vs scan (Supplement `tab:r4_panel_checks`) | `results/r4_panel/panel_analysis.py` | `per_image*.csv`, `model_predictions.csv`, `form_predictions.csv`, manifests | masks; the form-image predictions come from `run_forms.py` (form images) |
| Related work (`tab:related`) | literature values; `results/r4_related/results.md` gives every value with its quote and URL | — | — |
| Supplement `tab:r4_paired`, `tab:r4_tost` | `results/r4_arch/paired_tests.py` | as above | — |
| Supplement `tab:r4_grades` | `results/r4_grades/grades_analysis.py` | as above | — |
| Supplement `tab:r4_baseline`, `tab:r4_healthy` | `results/r4_baseline/*.py` | `per_image.csv`, `r4_baseline/per_leaf_*.csv` | images |

The other Supplement tables come from the pass-2 folders (`agreement`, `paired`, `detection`, `structure`, `baseline`,
`figures`); each `results.md` describes its tables and inputs.

**Expert-panel note.** `panel5_analysis.py` and `crosscheck.py` read the grade cells of the private form export, and
the earlier four-evaluator scripts (`validation/*.py`, `results/panel_stats`, `results/panel_figures`,
`results/r4_panel`) read a private four-evaluator grade file. Individual expert grades are not part of this release
(see Privacy), so these scripts document the analysis but cannot be rerun from the repository alone; their verified
outputs are in the corresponding `results.md` files.

**Data notes.**

- The manifests list one row per scan (`file`, `leaf_id`, `grade`, `source`, `split`); both faces of a leaf share
  `leaf_id` and partition. In the rust pack, `grade` is the category (`roya` or `healthy`); in the leaf-spot pack it is
  the annotator's visual grade (G1–G3). They list 282 rust-pack leaves and 369 leaf-spot leaves; the paper's 684 leaves
  also include the 33 healthy test leaves whose images were not retained.
- The `*_grade` columns of `validation/model_predictions.csv` use the mean of both faces for rust as well. The paper's
  rust grade uses the abaxial face: recompute it from `*_sev_ab` (the analyses do this).

## Randomness and seeds

- Bootstrap: 10,000 resamples of leaves. Seed 20260927 for the pass-2 analyses (`agreement`, `baseline`, `detection`,
  `figures`, `paired`, `structure`) and 20260928 for all round-4 analyses (`r4_*`), including all expert-panel inference.
  The earlier four-evaluator panel scripts (`validation/tiefree.py`, `results/panel_stats`, `results/panel_figures`)
  used seed 7.
- Matched-protocol retraining: seeds 0, 1 and 2 (Python `random`, NumPy and `torch.manual_seed` /
  `torch.cuda.manual_seed_all` are set in each kernel). cuDNN determinism is not enforced, so a rerun with the same seed
  need not reproduce the weights exactly.
- Original models: one training run each, no fixed seed.

## Privacy

The names of the panel experts appear only in the paper (expert table and Acknowledgments), never in this repository.
The raw form export is not included, and neither is any table of individual grades. One panel member joined after the
four-evaluator analysis and works at the authors' institution, and the paper reports results with and without this
evaluator. With a per-evaluator grade table, even with shuffled labels, that evaluator's grades could be recovered by
matching the published four-evaluator statistics. Released `results.md` / `results.json` copies had their
expert-identity sections removed, and the panel folders (`r4_panel5`, `r4_panel`, `panel_stats`) also had
per-evaluator values and all per-leaf panel grades (sorted or not) removed. Each affected file says so at the top.
<!-- TODO(author): release individual grades (e.g. validation/expert_grades_anonymized.csv) only if every evaluator,
in particular the evaluator affiliated with the authors' institution, consents to their grades being identifiable. -->

## Links

Links given in the paper's Data Availability statement:

- Zenodo record of the released artifacts (version 1.0.0): https://doi.org/10.5281/zenodo.22701350
- Demonstration subset of the annotated corpus (Hugging Face, CC BY 4.0):
  https://huggingface.co/datasets/cchery2001/pg26038-coffee-leaf-samples
- The six original ONNX models (Kaggle, CC0 1.0): https://www.kaggle.com/datasets/carlychery2001/pg26038-onnx-models

## License and citation

Code: MIT License (see `LICENSE`). **The author still has to confirm the license**, including the terms for the data
tables in this repository (manifests, prediction tables).
<!-- TODO(author): confirm MIT for the code and choose the license for the data tables. -->

Citation metadata are in `CITATION.cff`. No DOI has been assigned to this repository yet.
