#!/bin/bash
# Local (single-GPU) synthetic reliability experiment + lambda leakage sweep.
# Mirrors scripts/hpc/run_synth_reliability_gpu.qsub but targets a local CUDA
# device and a Python environment with requirements.txt installed. Generates its own data
# (no dataset download needed), then runs the joint / multi-head / reliability
# arms over a lambda sweep and writes a statistical report per arm.
#
# Usage:  bash scripts/run_synth_reliability_local.sh
# Resumable-ish: re-running regenerates data and overwrites lambda_* outputs.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
export SYMBOL_SANITY_NUM_WORKERS="${SYMBOL_SANITY_NUM_WORKERS:-4}"

SEED="${SEED:-0}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
LAMBDAS="${LAMBDAS:-0.01 0.1 1.0}"
SEEDS="${SEEDS:-0 1 2 3 4}"
NUM_HEADS="${NUM_HEADS:-5}"
CONCEPT_DROPOUT="${CONCEPT_DROPOUT:-0.2}"
EPOCHS="${EPOCHS:-30}"
RELIABILITY_EPOCHS="${RELIABILITY_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TASK="${TASK:-shape_color}"
DEVICE="${DEVICE:-cuda}"
ARMS="${ARMS:-joint multihead reliability}"

RUN_DIR="${RESULTS_ROOT}/synth_reliability_seed${SEED}"
DATA_DIR="${RUN_DIR}/data"

echo "[local] python: $PYTHON_BIN"
"$PYTHON_BIN" -c 'import torch; print("[local] torch", torch.__version__, "cuda", torch.cuda.is_available())'

"$PYTHON_BIN" -m symbol_sanity.cli experiment-environment \
  --output-path "$RUN_DIR/environment.json"

# All four five-class tasks are written into the metadata (default); the sweep
# below trains on $TASK. Re-run with TASK=color_size (etc.) to cover the others.
"$PYTHON_BIN" -m symbol_sanity.cli generate-synthetic \
  --output-dir "$DATA_DIR/train" --num-examples 2000 --seed "$((SEED * 10 + 0))"
"$PYTHON_BIN" -m symbol_sanity.cli generate-synthetic \
  --output-dir "$DATA_DIR/val" --num-examples 500 --seed "$((SEED * 10 + 1))"
"$PYTHON_BIN" -m symbol_sanity.cli generate-synthetic \
  --output-dir "$DATA_DIR/test" --num-examples 1000 --seed "$((SEED * 10 + 2))"

for LAM in $LAMBDAS; do
  OUT_DIR="${RUN_DIR}/lambda_${LAM}"
  echo "[local] === lambda ${LAM} -> ${OUT_DIR} ==="
  "$PYTHON_BIN" -m symbol_sanity.cli run-reliability-experiment \
    --train-dir "$DATA_DIR/train" \
    --val-dir "$DATA_DIR/val" \
    --eval-dir "$DATA_DIR/test" \
    --output-dir "$OUT_DIR" \
    --task "$TASK" \
    --seeds $SEEDS \
    --num-heads "$NUM_HEADS" \
    --lambda-concept "$LAM" \
    --concept-dropout "$CONCEPT_DROPOUT" \
    --eta 0.1 --beta 1.0 \
    --epochs "$EPOCHS" \
    --reliability-epochs "$RELIABILITY_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --detector-lr 0.0001 --head-lr 0.001 \
    --device "$DEVICE" \
    --detector-image-size 299 \
    --pretrained \
    --arms $ARMS

  for arm in $ARMS; do
    "$PYTHON_BIN" -m symbol_sanity.cli statistical-report \
      --summary-path "$OUT_DIR/arms/$arm/summary.json" \
      --output-dir "$OUT_DIR/arms/$arm/statistical_report" \
      --num-permutations 500 \
      --seed "$SEED" \
      --batch-size "$BATCH_SIZE" \
      --device "$DEVICE"
  done
done

echo "[local] done. Outputs under ${RUN_DIR}/lambda_*/"
