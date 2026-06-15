#!/usr/bin/env bash
set -euo pipefail

SEEDS="${SEEDS:-0 1 2}"
DATASET_ENV_PATH="${DATASET_ENV_PATH:-configs/hpc_datasets.env}"
if [[ -f "$DATASET_ENV_PATH" ]]; then
  source "$DATASET_ENV_PATH"
fi

RESULTS_ROOT="${RESULTS_ROOT:-results}"
CUB_ROOT="${CUB_ROOT:-$PWD/data/hpc_datasets/CUB_200_2011}"
AWA2_ROOT="${AWA2_ROOT:-$PWD/data/hpc_datasets/Animals_with_Attributes2}"
RUN_SHARED="${RUN_SHARED:-1}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p logs

submit_job() {
  local dataset="$1"
  local seed="$2"
  local script_path="$3"
  local root_var="$4"
  local root_value="$5"
  local job_name="cbm_${dataset}_s${seed}"
  local stdout_path="logs/${job_name}.out"
  local stderr_path="logs/${job_name}.err"
  local command=(
    qsub
    -N "$job_name"
    -o "$stdout_path"
    -e "$stderr_path"
    -v "SEED=${seed},RESULTS_ROOT=${RESULTS_ROOT},CONDA_ENV=gpuenv,RUN_SHARED=${RUN_SHARED},${root_var}=${root_value}"
    "$script_path"
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
  else
    "${command[@]}"
  fi
}

for seed in $SEEDS; do
  submit_job "cub" "$seed" "scripts/hpc/run_cub_full_gpu.qsub" "CUB_ROOT" "$CUB_ROOT"
  submit_job "awa2" "$seed" "scripts/hpc/run_awa2_full_gpu.qsub" "AWA2_ROOT" "$AWA2_ROOT"
done
