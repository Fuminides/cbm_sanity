#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
AWA2_ROOT="${AWA2_ROOT:-$PWD/data/hpc_datasets/Animals_with_Attributes2}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
CONFIG_PATH="${CONFIG_PATH:-configs/awa2_real_experiments.json}"
DEVICE="${DEVICE:-cpu}"
SEED="${SEED:-0}"
DRY_RUN="${DRY_RUN:-0}"
RUN_SHARED="${RUN_SHARED:-1}"
EXPERIMENT_NAME="awa2_20cls_pilot"

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_awa2_real_experiment.sh [preset] [options]

Options:
  --device DEVICE          Device passed to Torch, e.g. cpu or cuda.
  --seed SEED              Experiment seed and output suffix.
  --awa2-root PATH         Path to Animals_with_Attributes2.
  --results-root PATH      Root directory for experiment outputs.
  --config-path PATH       Preset JSON path.
  --python-bin PATH        Python executable for the CLI.
  --dry-run                Print commands without running them.
  --skip-shared            Skip the shared-extractor multi-head fix experiment.
  -h, --help               Show this help.

Environment fallbacks:
  PYTHON_BIN, AWA2_ROOT, RESULTS_ROOT, CONFIG_PATH, DEVICE, SEED, DRY_RUN, RUN_SHARED
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
    --dry-run)
      DRY_RUN="1"
      shift
      ;;
    --skip-shared)
      RUN_SHARED="0"
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

preset_vars="$(read_preset)"
eval "$preset_vars"

RUN_DIR="${RESULTS_ROOT}/${EXPERIMENT_NAME}_seed${SEED}"
MANIFEST_DIR="${RUN_DIR}/manifest"
OFFICIAL_DIR="${RUN_DIR}/official"
REPORT_DIR="${RUN_DIR}/statistical_report"
FIGURE_DIR="${RUN_DIR}/figures"
SHARED_DIR="${RUN_DIR}/shared_multihead"
SHARED_REPORT_DIR="${RUN_DIR}/shared_multihead_statistical_report"
SHARED_FIGURE_DIR="${RUN_DIR}/shared_multihead_figures"
ENV_PATH="${RUN_DIR}/environment.json"

mkdir -p "$RUN_DIR"

log() {
  printf '[symbol_sanity] %s\n' "$*"
}

build_manifest_cmd=(
  "$PYTHON_BIN" -m symbol_sanity.cli build-awa2-manifest
  --awa2-root "$AWA2_ROOT"
  --output-dir "$MANIFEST_DIR"
  --attribute-kind "$ATTRIBUTE_KIND"
  --val-fraction "$VAL_FRACTION"
  --test-fraction "$TEST_FRACTION"
  --seed "$SEED"
)
if [[ -n "${NUM_CLASSES}" ]]; then
  build_manifest_cmd+=(--num-classes "$NUM_CLASSES" --class-start "$CLASS_START")
fi

run_experiment_cmd=(
  "$PYTHON_BIN" -m symbol_sanity.cli run-official-manifest-experiment
  --train-dir "$MANIFEST_DIR/train"
  --eval-dir "$MANIFEST_DIR/test"
  --output-dir "$OFFICIAL_DIR"
  --task species
  --detector-seeds $DETECTOR_SEEDS
  --detector-epochs "$DETECTOR_EPOCHS"
  --head-epochs "$HEAD_EPOCHS"
  --batch-size "$BATCH_SIZE"
  --detector-lr "$DETECTOR_LR"
  --head-lr "$HEAD_LR"
  --device "$DEVICE"
  --detector-image-size "$DETECTOR_IMAGE_SIZE"
)
if [[ "$FREEZE" == "1" ]]; then
  run_experiment_cmd+=(--freeze)
fi
if [[ "$PRETRAINED" == "1" ]]; then
  run_experiment_cmd+=(--pretrained)
fi

report_cmd=(
  "$PYTHON_BIN" -m symbol_sanity.cli statistical-report
  --summary-path "$OFFICIAL_DIR/summary.json"
  --output-dir "$REPORT_DIR"
  --num-permutations "$STATISTICAL_PERMUTATIONS"
  --seed "$SEED"
  --batch-size "$BATCH_SIZE"
  --device "$DEVICE"
)

if [[ "$RUN_SHARED" == "1" ]]; then
  shared_cmd=(
    "$PYTHON_BIN" -m symbol_sanity.cli run-shared-extractor-multihead-manifest-experiment
    --train-dir "$MANIFEST_DIR/train"
    --eval-dir "$MANIFEST_DIR/test"
    --output-dir "$SHARED_DIR"
    --task species
    --head-seeds $DETECTOR_SEEDS
    --epochs "$SHARED_EPOCHS"
    --batch-size "$BATCH_SIZE"
    --lr "$SHARED_LR"
    --concept-loss-weight "$SHARED_CONCEPT_LOSS_WEIGHT"
    --task-loss-weight "$SHARED_TASK_LOSS_WEIGHT"
    --device "$DEVICE"
    --detector-image-size "$DETECTOR_IMAGE_SIZE"
    --seed "$SEED"
  )
  if [[ "$FREEZE" == "1" ]]; then
    shared_cmd+=(--freeze)
  fi
  if [[ "$PRETRAINED" == "1" ]]; then
    shared_cmd+=(--pretrained)
  fi

  shared_report_cmd=(
    "$PYTHON_BIN" -m symbol_sanity.cli statistical-report
    --summary-path "$SHARED_DIR/summary.json"
    --output-dir "$SHARED_REPORT_DIR"
    --num-permutations "$STATISTICAL_PERMUTATIONS"
    --seed "$SEED"
    --batch-size "$BATCH_SIZE"
    --device "$DEVICE"
  )

  shared_plot_cmd=(
    "$PYTHON_BIN" -m symbol_sanity.cli plot-report
    --summary-path "$SHARED_DIR/summary.json"
    --statistical-report-dir "$SHARED_REPORT_DIR"
    --output-dir "$SHARED_FIGURE_DIR"
    --formats png pdf
  )
fi

plot_cmd=(
  "$PYTHON_BIN" -m symbol_sanity.cli plot-report
  --summary-path "$OFFICIAL_DIR/summary.json"
  --statistical-report-dir "$REPORT_DIR"
  --output-dir "$FIGURE_DIR"
  --formats png pdf
)

env_cmd=(
  "$PYTHON_BIN" -m symbol_sanity.cli experiment-environment
  --output-path "$ENV_PATH"
)

print_command() {
  local label="$1"
  shift
  printf '%s:\n  ' "$label"
  printf '%q ' "$@"
  printf '\n'
}

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'Run directory: %s\n' "$RUN_DIR"
  print_command "Build manifest command" "${build_manifest_cmd[@]}"
  print_command "Run experiment command" "${run_experiment_cmd[@]}"
  print_command "Statistical report command" "${report_cmd[@]}"
  print_command "Plot command" "${plot_cmd[@]}"
  if [[ "$RUN_SHARED" == "1" ]]; then
    print_command "Shared multi-head command" "${shared_cmd[@]}"
    print_command "Shared statistical report command" "${shared_report_cmd[@]}"
    print_command "Shared plot command" "${shared_plot_cmd[@]}"
  fi
  print_command "Environment command" "${env_cmd[@]}"
  exit 0
fi

log "starting preset=${EXPERIMENT_NAME} run_dir=${RUN_DIR} device=${DEVICE} seed=${SEED}"
log "writing environment metadata"
"${env_cmd[@]}"
log "building AwA2 manifest"
"${build_manifest_cmd[@]}"
log "running official CBM detector/head experiment"
"${run_experiment_cmd[@]}"
log "running statistical report"
"${report_cmd[@]}"
log "generating paper figures"
"${plot_cmd[@]}"
if [[ "$RUN_SHARED" == "1" ]]; then
  log "running shared-extractor multi-head fix experiment"
  "${shared_cmd[@]}"
  log "running shared-extractor statistical report"
  "${shared_report_cmd[@]}"
  log "generating shared-extractor paper figures"
  "${shared_plot_cmd[@]}"
fi
log "completed preset=${EXPERIMENT_NAME} run_dir=${RUN_DIR}"
