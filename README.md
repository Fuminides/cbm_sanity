# Concet Bottleneck Model Sanity Analysis

Code for testing whether concept bottleneck model (CBM) symbols are stable
across independently trained detectors, classification heads, tasks, and random
seeds.

The main diagnostic is component swapping: train several `X -> C` detectors and
`C -> Y` heads over the same named concept interface, then evaluate crossed
pairs `h_i(g_j(x))`. Large swap drops indicate that nominally identical
concepts are not functionally interchangeable.

## Quick Start

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
export PYTHONPATH=src

python3 -m unittest discover -s tests -v
python3 -m symbol_sanity.cli official-cbm-status
```

Dependencies are declared in `requirements.txt`.

## Experiments

| Experiment | Question | Main output |
| --- | --- | --- |
| Synthetic sanity check | Does the swap protocol recover known symbol corruption? | Detector/head swap metrics |
| Detector swap | Are independently trained detectors compatible with one frozen head? | Swap accuracy and drop matrices |
| Head-detector matrix | Are independently trained heads and detectors mutually compatible? | Full `h_i(g_j(x))` matrix |
| Concept reliability | Which concepts are detectable and stable across seeds? | Per-concept balanced accuracy, F1, AUROC, AUPRC |
| Head null tests | Does the head use the aligned concept interface? | Permutation and prevalence-preserving null distributions |
| Reliability mitigation | Do multi-head and reliability-aware objectives improve compatibility? | Joint vs multi-head vs reliability-aware comparison |

Low swap drop means detectors expose a compatible concept interface. High swap
drop means that identical concept names and ordering do not guarantee compatible
representations.

## Datasets

### Synthetic

Deterministic colored-shape images with known binary concepts and symbolic
classification tasks. Use this to validate the pipeline before running neural
experiments:

```bash
python3 -m symbol_sanity.cli generate-synthetic \
  --output-dir /tmp/symbol_sanity_smoke \
  --num-examples 100 \
  --seed 0
```

### CUB-200-2011

The `koh112` policy reconstructs the 112 class-level concepts used by Koh et al.
from a raw `CUB_200_2011` directory. It applies the original majority, tie, and
not-visible handling rather than relying on redistributed processed pickle
files.

### Animals with Attributes 2

AwA2 provides 85 class-level predicates for 50 animal classes. Predicate
vectors are replicated across images of each class, so concept reliability
measures class-signature detectability, not per-image attribute annotation
quality.

Datasets are downloaded separately and are not included in this repository.

## Preset Runs

The launchers build manifests, train models, evaluate swaps, generate
statistical reports, and produce paper figures.

### CUB

```bash
CUB_ROOT=/path/to/CUB_200_2011 \
DEVICE=cpu \
scripts/run_cub_real_experiment.sh cub_koh112_5cls_smoke
```

Available presets:

- `cub_koh112_5cls_smoke`: fast CPU wiring check, not a paper result.
- `cub_koh112_20cls_pilot`: GPU pilot.
- `cub_koh112_full`: full paper-scale CUB run.

### AwA2

```bash
AWA2_ROOT=/path/to/Animals_with_Attributes2 \
DEVICE=cuda \
scripts/run_awa2_real_experiment.sh awa2_full
```

Available presets:

- `awa2_20cls_pilot`: attribute-transfer pilot.
- `awa2_full`: full paper-scale AwA2 run.

### Reliability Comparison

Compare joint, multi-head, and reliability-aware training:

```bash
CUB_ROOT=/path/to/CUB_200_2011 \
DEVICE=cpu \
scripts/run_reliability_experiment.sh cub_reliability_smoke
```

Paper-scale presets are `cub_reliability_full` and
`awa2_reliability_full`. See
[configs/reliability_experiments.json](configs/reliability_experiments.json)
for all settings.

Use `DRY_RUN=1` with any launcher to inspect generated commands without running
the experiment.

## Outputs

Each preset writes to `results/<preset>_seed<seed>/`:

- `environment.json`: Python, dependency, device, and source metadata.
- `manifest/`: normalized train/validation/test metadata.
- `official/summary.json`: detector, head, and swap results.
- `statistical_report/`: concept reliability and head-compatibility null tests.
- `figures/`: PNG/PDF plots and CSV tables.
- `shared_multihead/`: shared-detector multi-head mitigation results.

Primary paper artifacts include:

- `swap_accuracy_heatmap.{png,pdf}`
- `swap_drop_heatmap.{png,pdf}`
- `head_detector_accuracy_heatmap.{png,pdf}`
- `concept_reliability.csv`
- `head_compatibility_nulls.csv`
- `model_results_table.csv`

CPU smoke presets validate execution only. Use GPU paper presets and multiple
seeds for scientific conclusions.

## HPC

Cluster scripts are under [scripts/hpc](scripts/hpc). A typical workflow is:

```bash
mkdir -p logs
qsub -N cbm_setup_data \
  -o logs/cbm_setup_data.out \
  -e logs/cbm_setup_data.err \
  -v DATA_ROOT=/path/to/datasets/symbol_sanity \
  scripts/hpc/setup_datasets.qsub

scripts/hpc/submit_paper_gpu_jobs.sh
scripts/hpc/submit_reliability_jobs.sh cub 0
```

Dataset setup writes `configs/hpc_datasets.env`, which the qsub scripts load
automatically. Set `DRY_RUN=1` where supported to preview submissions.

## Implementation Notes

- The neural backbone uses torchvision's maintained Inception v3
  implementation; third-party source is not copied into this repository.
- The independent CBM regime trains separate `X -> C` and `C -> Y` models.
- The mitigation experiments train a shared detector with multiple heads and an
  optional reliability-aware penalty.
- Pretrained runs may need network access or pre-populated torchvision weights.
- Commands log progress with the `[symbol_sanity]` prefix. Set
  `SYMBOL_SANITY_QUIET=1` to suppress it.

For individual commands and options:

```bash
python3 -m symbol_sanity.cli --help
python3 -m symbol_sanity.cli <command> --help
```

Preset definitions are in [configs](configs). The model interfaces follow Koh
et al., "Concept Bottleneck Models" (ICML 2020). Dataset files are downloaded
from their original providers and are not redistributed.

## License

Code is released under the [MIT License](LICENSE). CUB-200-2011 and AwA2 retain
their own licenses, access conditions, and citation requirements.
