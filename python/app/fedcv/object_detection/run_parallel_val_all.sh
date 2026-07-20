#!/usr/bin/env bash
# Parallel evaluation: 3x v8s + 2x v8m per GPU across all GPUs.
set -euo pipefail

ROOT="/Shoaib/FedML_MSIS_Shoaib/python/app/fedcv/object_detection"
GPUS="${GPUS:-0,1,2,3,4,5,6}"
SLOTS_S="${SLOTS_S:-3}"
SLOTS_M="${SLOTS_M:-2}"
LOG="$ROOT/runs/val_eval/parallel_master.log"

mkdir -p "$ROOT/runs/val_eval"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

echo "Starting parallel val eval at $(date)" | tee "$LOG"
echo "GPUs=$GPUS slots_s=$SLOTS_S slots_m=$SLOTS_M" | tee -a "$LOG"

conda run -n shoaib python "$ROOT/scripts/run_parallel_val_eval.py" \
  --gpus "$GPUS" \
  --slots-s "$SLOTS_S" \
  --slots-m "$SLOTS_M" \
  2>&1 | tee -a "$LOG"

echo "Finished at $(date)" | tee -a "$LOG"
