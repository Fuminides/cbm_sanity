#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-$HOME/datasci/bin/python}"
DATASET_ENV_PATH="${DATASET_ENV_PATH:-configs/hpc_datasets.env}"
if [[ -f "$DATASET_ENV_PATH" ]]; then
  source "$DATASET_ENV_PATH"
fi
CUB_ROOT="${CUB_ROOT:-$PWD/data/hpc_datasets/CUB_200_2011}"
AWA2_ROOT="${AWA2_ROOT:-$PWD/data/hpc_datasets/Animals_with_Attributes2}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
CONFIG_PATH="${CONFIG_PATH:-configs/reliability_experiments.json}"
DEVICE="${DEVICE:-cpu}"
SEED="${SEED:-0}"
DRY_RUN="${DRY_RUN:-0}"
ARMS="${ARMS:-joint multihead reliability}"
MANIFEST_ONLY="${MANIFEST_ONLY:-0}"
EPOCHS_OVERRIDE="${EPOCHS_OVERRIDE:-}"
RELIABILITY_EPOCHS_OVERRIDE="${RELIABILITY_EPOCHS_OVERRIDE:-}"
EXPERIMENT_NAME="cub_reliability_smoke"

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_reliability_experiment.sh [preset] [options]

Runs the joint vs multi-head vs reliability-aware comparison experiment
(run-reliability-experiment) plus a statistical report per arm.

Options:
  --device DEVICE          Device passed to Torch, e.g. cpu or cuda.
  --seed SEED              Manifest seed and output suffix.
  --cub-root PATH          Path to CUB_200_2011 (cub presets).
  --awa2-root PATH         Path to Animals_with_Attributes2 (awa2 presets).
  --results-root PATH      Root directory for experiment outputs.
  --config-path PATH       Preset JSON path.
  --python-bin PATH        Python executable for the CLI.
  --arms LIST              Space-separated arms to run (default: all three).
                           Arms share the run directory, so they can run as
                           separate jobs; reliability reuses multihead
                           checkpoints when they already exist.
  --epochs N               Override the preset's phase-1 epochs.
  --reliability-epochs N   Override the preset's reliability epochs.
  --manifest-only          Build the manifest and environment file, then exit.
  --dry-run                Print commands without running them.
  -h, --help               Show this help.

Environment fallbacks:
  PYTHON_BIN, CUB_ROOT, AWA2_ROOT, RESULTS_ROOT, CONFIG_PATH, DEVICE, SEED,
  DRY_RUN, ARMS, MANIFEST_ONLY, EPOCHS_OVERRIDE, RELIABILITY_EPOCHS_OVERRIDE
USAGE
}

require_value() {
  local option="$1"
  local value="${2:-}"
  if [[ -z "$value" || "$value" == --* ]]; then
    printf 'Missing value for %s\n\n' "$option" >&2
    usage >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)
      require_value "$1" "${2:-}"
      DEVICE="$2"
      shift 2
      ;;
    --seed)
      require_value "$1" "${2:-}"
      SEED="$2"
      shift 2
      ;;
    --cub-root)
      require_value "$1" "${2:-}"
      CUB_ROOT="$2"
      shift 2
      ;;
    --awa2-root)
      require_value "$1" "${2:-}"
      AWA2_ROOT="$2"
      shift 2
      ;;
    --results-root)
      require_value "$1" "${2:-}"
      RESULTS_ROOT="$2"
      shift 2
      ;;
    --config-path)
      require_value "$1" "${2:-}"
      CONFIG_PATH="$2"
      shift 2
      ;;
    --python-bin)
      require_value "$1" "${2:-}"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --arms)
      require_value "$1" "${2:-}"
      ARMS="$2"
      shift 2
      ;;
    --epochs)
      require_value "$1" "${2:-}"
      EPOCHS_OVERRIDE="$2"
      shift 2
      ;;
    --reliability-epochs)
      require_value "$1" "${2:-}"
      RELIABILITY_EPOCHS_OVERRIDE="$2"
      shift 2
      ;;
    --manifest-only)
      MANIFEST_ONLY="1"
      shift
      ;;
    --dry-run)
      DRY_RUN="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      printf 'Unknown option: %s\n\n' "$1" >&2
      usage >&2
      exit 2
      ;;
    *)
      EXPERIMENT_NAME="$1"
      shift
      ;;
  esac
done

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

read_preset() {
  "$PYTHON_BIN" - "$CONFIG_PATH" "$EXPERIMENT_NAME" <<'PY'
import json
import sys

config_path, experiment_name = sys.argv[1], sys.argv[2]
with open(config_path, "r", encoding="utf-8") as handle:
    config = json.load(handle)
for experiment in config["experiments"]:
    if experiment["name"] == experiment_name:
        for key, value in experiment.items():
            shell_key = key.upper()
            if isinstance(value, bool):
                shell_value = "1" if value else "0"
            elif isinstance(value, list):
                shell_value = " ".join(str(item) for item in value)
            elif value is None:
                shell_value = ""
            else:
                shell_value = str(value)
            print(f"{shell_key}={shell_value!r}")
        break
else:
    raise SystemExit(f"Unknown experiment preset: {experiment_name}")
PY
}

eval "$(read_preset)"

if [[ -n "$EPOCHS_OVERRIDE" ]]; then
  EPOCHS="$EPOCHS_OVERRIDE"
fi
if [[ -n "$RELIABILITY_EPOCHS_OVERRIDE" ]]; then
  RELIABILITY_EPOCHS="$RELIABILITY_EPOCHS_OVERRIDE"
fi

RUN_DIR="${RESULTS_ROOT}/${EXPERIMENT_NAME}_seed${SEED}"
MANIFEST_DIR="${RUN_DIR}/manifest"
EXPERIMENT_DIR="${RUN_DIR}/reliability"
ENV_PATH="${RUN_DIR}/environment.json"

mkdir -p "$RUN_DIR"

log() {
  printf '[symbol_sanity] %s\n' "$*"
}

if [[ "$DATASET" == "cub" ]]; then
  build_manifest_cmd=(
    "$PYTHON_BIN" -m symbol_sanity.cli build-cub-manifest
    --cub-root "$CUB_ROOT"
    --output-dir "$MANIFEST_DIR"
    --attribute-policy "$ATTRIBUTE_POLICY"
    --val-fraction "$VAL_FRACTION"
    --seed "$SEED"
  )
elif [[ "$DATASET" == "awa2" ]]; then
  build_manifest_cmd=(
    "$PYTHON_BIN" -m symbol_sanity.cli build-awa2-manifest
    --awa2-root "$AWA2_ROOT"
    --output-dir "$MANIFEST_DIR"
    --val-fraction "$VAL_FRACTION"
    --test-fraction "$TEST_FRACTION"
    --seed "$SEED"
  )
else
  printf 'Unknown dataset in preset: %s\n' "$DATASET" >&2
  exit 2
fi
if [[ -n "${NUM_CLASSES}" ]]; then
  build_manifest_cmd+=(--num-classes "$NUM_CLASSES" --class-start "$CLASS_START")
fi

run_experiment_cmd=(
  "$PYTHON_BIN" -m symbol_sanity.cli run-reliability-experiment
  --train-dir "$MANIFEST_DIR/train"
  --val-dir "$MANIFEST_DIR/val"
  --eval-dir "$MANIFEST_DIR/test"
  --output-dir "$EXPERIMENT_DIR"
  --task species
  --seeds $SEEDS
  --num-heads "$NUM_HEADS"
  --lambda-concept "$LAMBDA_CONCEPT"
  --eta "$ETA"
  --beta "$BETA"
  --concept-dropout "$CONCEPT_DROPOUT"
  --epochs "$EPOCHS"
  --reliability-epochs "$RELIABILITY_EPOCHS"
  --batch-size "$BATCH_SIZE"
  --detector-lr "$DETECTOR_LR"
  --head-lr "$HEAD_LR"
  --device "$DEVICE"
  --detector-image-size "$DETECTOR_IMAGE_SIZE"
  --arms $ARMS
)
if [[ "$FREEZE" == "1" ]]; then
  run_experiment_cmd+=(--freeze)
fi
if [[ "$PRETRAINED" == "1" ]]; then
  run_experiment_cmd+=(--pretrained)
fi

env_cmd=(
  "$PYTHON_BIN" -m symbol_sanity.cli experiment-environment
  --output-path "$ENV_PATH"
)

run_command() {
  local label="$1"
  shift
  log "$label"
  printf '  %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

run_command "record environment" "${env_cmd[@]}"
if [[ "$DRY_RUN" != "1" && -f "$MANIFEST_DIR/train/schema.json" ]]; then
  log "manifest already exists at $MANIFEST_DIR, skipping build"
else
  run_command "build manifest" "${build_manifest_cmd[@]}"
fi
if [[ "$MANIFEST_ONLY" == "1" ]]; then
  log "manifest-only mode, done run_dir=$RUN_DIR"
  exit 0
fi
run_command "run reliability comparison" "${run_experiment_cmd[@]}"

for arm in $ARMS; do
  arm_summary="$EXPERIMENT_DIR/arms/$arm/summary.json"
  if [[ "$DRY_RUN" == "1" || -f "$arm_summary" ]]; then
    report_cmd=(
      "$PYTHON_BIN" -m symbol_sanity.cli statistical-report
      --summary-path "$arm_summary"
      --output-dir "$EXPERIMENT_DIR/arms/$arm/statistical_report"
      --num-permutations "$STATISTICAL_PERMUTATIONS"
      --seed "$SEED"
      --batch-size "$BATCH_SIZE"
      --device "$DEVICE"
    )
    run_command "statistical report arm=$arm" "${report_cmd[@]}"
  fi
done

log "done run_dir=$RUN_DIR"
