#!/usr/bin/env bash
# Submit the reliability comparison as chained per-arm qsub jobs.
#
# Job graph (per dataset):
#   manifest (CPU)  ->  joint (GPU)
#                   ->  multihead (GPU)  ->  reliability (GPU)
#
# joint and multihead run in parallel on separate GPUs; reliability waits for
# multihead because it warm-starts from its checkpoints (shared run directory).
# Submits 4 jobs per dataset — far below the cluster's 100-job limit.
#
# Usage:
#   scripts/hpc/submit_reliability_jobs.sh cub   [SEED]
#   scripts/hpc/submit_reliability_jobs.sh awa2  [SEED]

DATASET="${1:-cub}"
SEED="${2:-${SEED:-0}}"

case "$DATASET" in
  cub)
    PRESET="${PRESET:-cub_reliability_full}"
    ARM_QSUB="scripts/hpc/run_cub_reliability_gpu.qsub"
    ;;
  awa2)
    PRESET="${PRESET:-awa2_reliability_full}"
    ARM_QSUB="scripts/hpc/run_awa2_reliability_gpu.qsub"
    ;;
  *)
    echo "Unknown dataset: $DATASET (use cub or awa2)" >&2
    exit 2
    ;;
esac

mkdir -p logs

TAG="${DATASET}_rel_s${SEED}"

qsub -N "${TAG}_manifest" \
  -o "logs/${TAG}_manifest.out" -e "logs/${TAG}_manifest.err" \
  -v "PRESET=${PRESET},SEED=${SEED},CONDA_ENV=datasci" \
  scripts/hpc/prepare_reliability_manifest_cpu.qsub

qsub -N "${TAG}_joint" -hold_jid "${TAG}_manifest" \
  -o "logs/${TAG}_joint.out" -e "logs/${TAG}_joint.err" \
  -v "PRESET=${PRESET},SEED=${SEED},CONDA_ENV=gpuenv,ARMS=joint" \
  "$ARM_QSUB"

qsub -N "${TAG}_multihead" -hold_jid "${TAG}_manifest" \
  -o "logs/${TAG}_multihead.out" -e "logs/${TAG}_multihead.err" \
  -v "PRESET=${PRESET},SEED=${SEED},CONDA_ENV=gpuenv,ARMS=multihead" \
  "$ARM_QSUB"

qsub -N "${TAG}_reliability" -hold_jid "${TAG}_multihead" \
  -o "logs/${TAG}_reliability.out" -e "logs/${TAG}_reliability.err" \
  -v "PRESET=${PRESET},SEED=${SEED},CONDA_ENV=gpuenv,ARMS=reliability" \
  "$ARM_QSUB"

echo "Submitted ${TAG}: manifest -> {joint, multihead} -> reliability"
echo "Monitor with: qstat | grep ${TAG}"
