# symbol_sanity

Reproducible code for concept bottleneck symbol-swap experiments.

## Installation

Create an isolated environment and install the dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
export PYTHONPATH=src
```

`requirements.txt` contains the runtime and verification dependencies.
`requirements-tested.txt` records the exact versions used for the verified
publication environment.

## Current Scope

The first implemented milestone is a deterministic synthetic colored-shape
dataset for concept bottleneck symbol-swap sanity checks. It defines a fixed
concept order, renders images, and exports task labels for multiple tasks over
the same concept vocabulary.

## Experiment Suite

The paper is organized around the claim that a CBM bottleneck should expose
symbols that are stable enough to be reused across detectors, classifier heads,
tasks, and random seeds. The repository therefore separates the experiments by
which part of the CBM is held fixed and which part is exchanged.

### Experiment 0: Synthetic Symbol Sanity Check

Purpose: verify the protocol on a dataset where the true symbols are known.

Setup:

- Generate colored-shape images with a fixed binary concept schema.
- Train or instantiate concept detectors over the same concept order.
- Train task heads for symbolic tasks such as shape/color classification.
- Swap detectors while keeping the head fixed.

Interpretation:

- Oracle detectors should be perfect.
- Noisy detectors should produce controlled swap drops.
- Concept-order mismatches should fail hard.

### Experiment 1: CUB `koh112` Manifest Reconstruction

Purpose: construct the real CUB benchmark used by the original CBM-style
experiments.

Setup:

- Read raw `CUB_200_2011`.
- Use the official Koh et al. `class_attr_data_10` 112 concept indices.
- Aggregate concepts to class-level binary labels with the original
majority/tie/not-visible policy.
- Export train/val/test manifests with fixed concept order and metadata.

Interpretation:

- This is a reproducible raw-data reconstruction of the original 112-concept
CUB setting, not a dependency on downloaded processed pickle files.

### Experiment 2: Detector Swap With Fixed Classification Head

Purpose: test whether independently trained concept detectors produce compatible
symbols for the same downstream classifier.

Setup:

- Train N independent official `C -> Y` heads on oracle concepts, one per seed.
- Train N independent official `X -> C` detectors with the same seed set.
- Evaluate each matched pair `(h_i, g_i)` as one independent CBM.
- Evaluate every detector through a primary frozen head for the classic detector
swap matrix.
- Report detector accuracy, macro-F1, concept agreement, swap accuracy, swap
drop, and relative retention.

Output:

- `official/summary.json`.
- `figures/model_results_table.csv`.
- `swap_accuracy_heatmap.{png,pdf}`.
- `swap_drop_heatmap.{png,pdf}`.
- `detector_metrics_bar.{png,pdf}`.

Interpretation:

- Low swap drop means detectors agree on a usable concept interface.
- High swap drop means the named concepts are not being represented compatibly,
even when the concept order is identical.

### Experiment 3: Classification-Head Swap Matrix

Purpose: test whether the same detector outputs remain useful for different
classification heads trained on the same concept vocabulary.

Setup:

- Train a family of heads `h_i`, currently one per random seed over the same
label task and concept schema.
- Train a family of detectors `g_j`, currently one per random seed over the same
input/concept schema.
- Evaluate the full matrix `h_i(g_j(x))`.
- Compare matched independent CBMs `(h_i, g_i)`, same-head detector swaps,
same-detector head swaps, and fully crossed detector/head swaps.

Recommended CUB variants:

- Same CUB `koh112` concepts, different species subsets.
- Same CUB `koh112` concepts, coarse superclass/group labels if defined.
- Same detector evaluated through heads trained on 5-class, 20-class, and
full-200-class tasks when label spaces are compatible with the evaluation split.

Interpretation:

- Detector compatibility asks whether `g_j` emits the expected symbols.
- Head compatibility asks whether different classifiers use those symbols in
consistent ways.
- Fully crossed matrices test whether failures come from detector semantics,
head semantics, or their interaction.

Implementation status:

- The current runner implements same-task, same-schema head-by-detector swaps
across random seeds.
- Different-task and different-class-subset head swaps are the next extension.

Output:

- `official/summary.json` with `model_rows` and `head_detector_rows`.
- `figures/head_detector_accuracy_heatmap.{png,pdf}`.
- `figures/head_detector_matrix.csv`.

### Experiment 4: Concept Identifiability

Purpose: estimate which concepts are reliably detectable and which are likely
ambiguous, rare, or artifact-prone.

Setup:

- Evaluate each trained detector against oracle concept labels.
- Aggregate per-concept metrics across detector seeds.
- Report prevalence, accuracy, balanced accuracy, F1, AUROC, AUPRC, and
permutation p-values.

Output:

- `statistical_report/concept_reliability.csv`.
- `concept_balanced_accuracy_hist.{png,pdf}`.
- `concept_prevalence_vs_balanced_accuracy.{png,pdf}`.
- `concept_reliability_top_bottom.csv`.

Interpretation:

- Raw concept accuracy is not sufficient for sparse CUB concepts.
- Balanced accuracy, AUROC, and AUPRC are the main reliability diagnostics.
- Concepts with no positives or no negatives in a split are marked undefined for
balanced accuracy rather than being treated as reliable.

### Experiment 5: Head-Compatibility Null Tests

Purpose: quantify whether a classification head uses the intended concept
interface rather than arbitrary or prevalence-preserving shortcuts.

Setup:

- Evaluate observed head accuracy on the true concept matrix.
- Generate null distributions by randomly permuting concept dimensions.
- Generate null distributions by shuffling each concept across samples while
preserving concept prevalence.
- Compute empirical p-values using plus-one correction.

Output:

- `statistical_report/head_compatibility_nulls.csv`.
- `head_compatibility_nulls.{png,pdf}`.

Interpretation:

- If observed accuracy is far above the null distributions, the head depends on
the aligned concept interface.
- If null distributions are close to observed accuracy, the task may be solvable
through concept prevalence, leakage, or class-level signatures.

### Experiment 6: Real CUB Preset Runs

Purpose: standardize the actual runs used for smoke tests, pilots, and
paper-scale experiments.

Presets:

- `cub_koh112_5cls_smoke`: local CPU wiring check.
- `cub_koh112_20cls_pilot`: GPU pilot to check training dynamics.
- `cub_koh112_full`: main full-CUB paper-scale run.

Each preset writes:

- `environment.json`.
- `manifest/`.
- `official/summary.json`.
- `statistical_report/`.
- `figures/`.

The paper figures intentionally omit the oracle-head reference line because the
oracle concept head can be trivially perfect in class-level CUB subsets and can
visually dominate the detector/swap comparisons.


## Usage

Run tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Generate a small synthetic dataset:

```bash
PYTHONPATH=src python3 -m symbol_sanity.cli generate-synthetic \
  --output-dir /tmp/symbol_sanity_smoke \
  --num-examples 100 \
  --seed 0 \
  --image-size 64
```

The generated directory contains:

- `schema.json`: concept order, concept groups, task definitions, and detector/head interface metadata.
- `metadata.jsonl`: one row per image with attributes, concept vector, and task labels.
- `images/`: rendered PNG files.

Create detector and head checkpoints:

```bash
PYTHONPATH=src python3 -m symbol_sanity.cli create-oracle-detector \
  --dataset-dir /tmp/symbol_sanity_smoke \
  --output-path /tmp/symbol_sanity_oracle.json \
  --name oracle

PYTHONPATH=src python3 -m symbol_sanity.cli create-noisy-detector \
  --dataset-dir /tmp/symbol_sanity_smoke \
  --output-path /tmp/symbol_sanity_noisy.json \
  --name noisy \
  --flip-probability 0.25 \
  --seed 1

PYTHONPATH=src python3 -m symbol_sanity.cli train-lookup-head \
  --dataset-dir /tmp/symbol_sanity_smoke \
  --task shape_color \
  --output-path /tmp/symbol_sanity_shape_color_head.json \
  --name shape_color_lookup
```

Evaluate original and swapped detector/head pairs:

```bash
PYTHONPATH=src python3 -m symbol_sanity.cli evaluate \
  --dataset-dir /tmp/symbol_sanity_smoke \
  --detector /tmp/symbol_sanity_oracle.json \
  --head /tmp/symbol_sanity_shape_color_head.json

PYTHONPATH=src python3 -m symbol_sanity.cli evaluate-swap \
  --dataset-dir /tmp/symbol_sanity_smoke \
  --original-detector /tmp/symbol_sanity_oracle.json \
  --swap-detector /tmp/symbol_sanity_noisy.json \
  --head /tmp/symbol_sanity_shape_color_head.json
```

Check the neural CBM backend:

```bash
PYTHONPATH=src python3 -m symbol_sanity.cli official-cbm-status
```

The Koh et al. experiment interfaces are exposed through
`symbol_sanity.official_cbm`. The Inception v3 implementation is provided by
the declared torchvision dependency; no torchvision or ConceptBottleneck
source code is copied into this repository. Neural model construction requires:

```bash
python3 -m pip install -r requirements.txt
```

The rest of this repository remains importable without those dependencies.

Run torch-backed official CBM adapter tests with:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_official_cbm -v
```

## Neural Synthetic Experiment

Run a full official-CBM synthetic experiment:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 \
  -m symbol_sanity.cli run-official-synthetic-experiment \
  --output-dir /tmp/symbol_sanity_runner_smoke \
  --num-examples 16 \
  --data-seed 21 \
  --task shape_color \
  --detector-seeds 1 2 \
  --detector-epochs 1 \
  --head-epochs 10 \
  --batch-size 4 \
  --detector-lr 0.001 \
  --head-lr 0.05 \
  --device cpu \
  --detector-image-size 299
```

This writes:

- `data/`: generated synthetic images and metadata.
- `head.pt`: frozen official `C -> Y` MLP head.
- `detectors/detector_seed_*.pt`: official `X -> C` detector checkpoints.
- `summary.json`: per-detector evaluation and full detector-swap matrix.

CPU smoke runs with one detector epoch are expected to be undertrained. Use them
to validate the pipeline, not to draw scientific conclusions.

## Real CUB Dataset

Build train/val/test manifests from a local raw `CUB_200_2011` folder:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 \
  -m symbol_sanity.cli build-cub-manifest \
  --cub-root /path/to/CUB_200_2011 \
  --output-dir /tmp/symbol_sanity_cub_manifest_smoke \
  --num-attributes 32 \
  --num-classes 5 \
  --class-start 1 \
  --val-fraction 0.15 \
  --seed 0
```

By default this uses `--attribute-policy balanced`, which selects image-level
attributes with train prevalence closest to 0.5. For paper-comparable CBM runs,
use the official Koh et al. 112 class-level concept subset:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 \
  -m symbol_sanity.cli build-cub-manifest \
  --cub-root /path/to/CUB_200_2011 \
  --output-dir /tmp/symbol_sanity_cub_koh112 \
  --attribute-policy koh112 \
  --val-fraction 0.2 \
  --seed 0
```

`koh112` uses the exact 112 attribute indices documented in the official
ConceptBottleneck CUB `class_attr_data_10` preprocessing and class-level
majority labels with the upstream tie and not-visible handling. On the full
local CUB dataset this yields 4,794 train rows, 1,200 val rows, 5,794 test rows,
200 species classes, and 112 concepts.

Train an official `C -> Y` head on CUB concepts:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 \
  -m symbol_sanity.cli train-official-synthetic-head \
  --dataset-dir /tmp/symbol_sanity_cub_manifest_smoke/train \
  --task species \
  --output-path /tmp/symbol_sanity_cub_manifest_smoke/species_head.pt \
  --epochs 5 \
  --batch-size 16 \
  --lr 0.01 \
  --seed 0 \
  --device cpu
```

The same official detector/head training and evaluation commands used for the
synthetic experiment work on these CUB manifests. CPU Inception detector training
is slow; use small class subsets for smoke tests and GPU-backed runs for
scientific results.

Run a complete official-CBM detector/head/swap experiment on an existing CUB
manifest:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 \
  -m symbol_sanity.cli run-official-manifest-experiment \
  --train-dir results/cub_koh112_5cls_seed0/manifest/train \
  --eval-dir results/cub_koh112_5cls_seed0/manifest/test \
  --output-dir results/cub_koh112_5cls_seed0/official_smoke \
  --task species \
  --detector-seeds 0 1 \
  --detector-epochs 1 \
  --head-epochs 50 \
  --batch-size 8 \
  --detector-lr 0.001 \
  --head-lr 0.01 \
  --device cpu \
  --detector-image-size 75 \
  --freeze
```

This writes `summary.json` with the oracle `C -> Y` head result, each detector's
test result, and the full detector-swap table. `--freeze` is useful for fast CPU
smoke tests only; remove it and use a GPU-backed run for paper-quality detector
training.

Compute concept-identifiability and head-compatibility statistics from a run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 \
  -m symbol_sanity.cli statistical-report \
  --summary-path results/cub_koh112_5cls_seed0/official_smoke/summary.json \
  --output-dir results/cub_koh112_5cls_seed0/statistical_report \
  --num-permutations 1000 \
  --seed 0 \
  --batch-size 16 \
  --device cpu
```

This writes `concept_reliability.csv`, `head_compatibility_nulls.csv`, and
`statistical_report.json`. Smoke tests can use fewer permutations; paper runs
should use enough permutations to resolve the desired p-value threshold.

Generate paper figures from a completed run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 \
  -m symbol_sanity.cli plot-report \
  --summary-path results/cub_koh112_5cls_seed0/official_smoke/summary.json \
  --statistical-report-dir results/cub_koh112_5cls_seed0/statistical_report_smoke \
  --output-dir results/cub_koh112_5cls_seed0/figures \
  --formats png pdf
```

This writes:

- `swap_accuracy_heatmap.{png,pdf}`.
- `swap_drop_heatmap.{png,pdf}`.
- `head_detector_accuracy_heatmap.{png,pdf}` when the run contains multiple
heads.
- `detector_metrics_bar.{png,pdf}`.
- `concept_balanced_accuracy_hist.{png,pdf}`.
- `concept_prevalence_vs_balanced_accuracy.{png,pdf}`.
- `head_compatibility_nulls.{png,pdf}`.
- `concept_reliability_top_bottom.csv`.
- `model_results_table.csv`.
- `head_detector_matrix.csv` when the run contains multiple heads.

## Real CUB Experiment Presets

The preset file [configs/cub_real_experiments.json](configs/cub_real_experiments.json)
defines the standard CUB runs:

- `cub_koh112_5cls_smoke`: local CPU wiring check.
- `cub_koh112_20cls_pilot`: GPU pilot before committing to the full run.
- `cub_koh112_full`: main full-CUB paper-scale run.

Run a preset end-to-end:

```bash
DEVICE=cpu \
scripts/run_cub_real_experiment.sh cub_koh112_5cls_smoke
```

Preview the full-CUB GPU command without running it:

```bash
DRY_RUN=1 \
DEVICE=cuda \
scripts/run_cub_real_experiment.sh cub_koh112_full
```

Launch the main full-CUB run on a GPU machine:

```bash
DEVICE=cuda \
scripts/run_cub_real_experiment.sh cub_koh112_full
```

The script writes a run directory under `results/<preset>_seed<seed>/` with:

- `environment.json`: Python, Torch, CUDA, official-CBM backend, and git metadata.
- `manifest/`: generated CUB train/val/test manifests.
- `official/summary.json`: independent model rows, head/detector matrix, detector
evaluations, and swap matrix.
- `statistical_report/`: concept reliability and head-compatibility null reports.
- `figures/`: paper-oriented PNG/PDF plots and a top/bottom concept table.

## Original CBM Training Regimes

The official CBM paper evaluates several training regimes:

- Independent: train `X -> C` and `C -> Y` separately.
- Sequential: train `X -> C`, then train `C -> Y` on predicted concepts.
- Joint: train `X -> C -> Y` end-to-end with both concept and task losses.
- Joint sigmoid: joint model with sigmoid between concept logits and the task
head.

The current experiment runner implements the independent regime for the swap
study: N separately trained `X -> C` detectors and N separately trained `C -> Y`
heads. This is the cleanest setting for modular detector/head swapping.

The preset launchers also run a shared-extractor multi-head fix: one official
`X -> C` detector is trained jointly with several `C -> Y` heads using a concept
loss plus the average task loss across heads. This writes results under
`shared_multihead/`, `shared_multihead_statistical_report/`, and
`shared_multihead_figures/`. Use `--skip-shared` or `RUN_SHARED=0` for
baseline-only reruns.

Joint CBMs are supported by the torchvision-backed architecture adapter. The implemented
paper-facing fix is the shared-extractor multi-head regime because it directly
targets detector/head symbol misalignment while preserving the original CBM
detector/head decomposition.

The full and pilot presets use `--pretrained`, which may require the Inception
weights to be available in the Torch cache or downloadable on the GPU machine.
If network access is unavailable, pre-populate the cache or set `pretrained` to
`false` in the preset.

Experiment commands print progress lines prefixed with `[symbol_sanity]`,
including phase changes, detector seeds, epoch losses, evaluation metrics, report
steps, and written file paths. Set `SYMBOL_SANITY_QUIET=1` only when you need to
suppress these logs, for example in automated tests.



## HPC Paper Runs

GPU qsub scripts are provided for the final CUB paper runs:

- [scripts/hpc/setup_datasets.sh](scripts/hpc/setup_datasets.sh)
- [scripts/hpc/setup_datasets.qsub](scripts/hpc/setup_datasets.qsub)
- [scripts/hpc/run_cub_full_gpu.qsub](scripts/hpc/run_cub_full_gpu.qsub)
- [scripts/hpc/submit_paper_gpu_jobs.sh](scripts/hpc/submit_paper_gpu_jobs.sh)

First download and validate the datasets on the cluster:

```bash
mkdir -p logs
qsub -N cbm_setup_data \
  -o logs/cbm_setup_data.out \
  -e logs/cbm_setup_data.err \
  -v DATA_ROOT=/path/to/datasets/symbol_sanity \
  scripts/hpc/setup_datasets.qsub
```

The setup job writes `configs/hpc_datasets.env` with `CUB_ROOT`. The GPU qsub scripts source this file automatically.

Submit the default paper matrix, which runs CUB and full presets for seeds
`0 1 2`:

```bash
scripts/hpc/submit_paper_gpu_jobs.sh
```

Each submitted job runs both the independent baseline and the shared-extractor
multi-head fix. Set `RUN_SHARED=0` to submit baseline-only reruns.

Preview submissions without calling `qsub`:

```bash
DRY_RUN=1 scripts/hpc/submit_paper_gpu_jobs.sh
```

Override seeds and paths with environment variables:

```bash
SEEDS="0 1 2 3 4" \
CUB_ROOT=/path/to/CUB_200_2011 \
RESULTS_ROOT=/path/to/results \
scripts/hpc/submit_paper_gpu_jobs.sh
```

## License

The project is released under the MIT License. Dataset licenses and access
conditions remain those of CUB-200-2011 and Animals with Attributes 2.
See [THIRD_PARTY.md](THIRD_PARTY.md) for dependency and provenance details.
