#!/usr/bin/env bash
# Run inference + recovery + mAP for all BDD val sequences with one weight file.
set -euo pipefail

WEIGHTS="${1:?Usage: $0 <weights.pt> [weight_label] [device]}"
WEIGHT_LABEL="${2:-$(basename "$(dirname "$(dirname "$WEIGHTS")")")}"
DEVICE="${3:-0}"
VAL_ROOT="/datasets/BDDComplete/val"
ROOT="/Shoaib/FedML_MSIS_Shoaib/python/app/fedcv/object_detection"
OUT_ROOT="$ROOT/runs/val_eval/${WEIGHT_LABEL}"
LOG="$OUT_ROOT/batch.log"

mkdir -p "$OUT_ROOT"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

echo "Weights: $WEIGHTS"
echo "Label:   $WEIGHT_LABEL"
echo "Output:  $OUT_ROOT"
echo "Device:  $DEVICE"

mapfile -t SEQS < <(ls -1 "$VAL_ROOT")
total=${#SEQS[@]}
done=0
fail=0

for seq in "${SEQS[@]}"; do
  done=$((done + 1))
  echo "==> [$done/$total] $seq" | tee -a "$LOG"
  if conda run -n shoaib python "$ROOT/scripts/run_sequence_eval.py" \
      --weights "$WEIGHTS" \
      --seq-dir "$VAL_ROOT/$seq" \
      --output-root "$OUT_ROOT" \
      --device "$DEVICE" \
      --skip-if-done 2>&1 | tee -a "$LOG"; then
    :
  else
    fail=$((fail + 1))
    echo "FAILED: $seq" | tee -a "$LOG"
  fi
done

CSV="$OUT_ROOT/summary_all_sequences.csv"
conda run -n shoaib python "$ROOT/scripts/aggregate_val_results.py" \
  --results-root "$OUT_ROOT" \
  --output-csv "$CSV" \
  --improved-list "$OUT_ROOT/sequences_improved_map50.txt"

echo "Done. failures=$fail csv=$CSV" | tee -a "$LOG"
