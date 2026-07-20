#!/usr/bin/env bash
# mAP evaluation only (recovery/inference already finished)
set -euo pipefail

ROOT="/Shoaib/FedML_MSIS_Shoaib/python/app/fedcv/object_detection"
RECOVERY="$ROOT/Yolov5_DeepSORT_PseudoLabels"
YOLO="$RECOVERY/yolov5"
SEQ_DIR="/datasets/BDDComplete/val/b1c66a42-6f7d68ca"
NAME="b1c66a42-6f7d68ca"
HC="0.5"
DEVICE="${DEVICE:-0}"
DATA="data/bdd_b1c66a42.yaml"

OUT="$ROOT/runs/sequence_eval/${NAME}"
GT_3CLASS="$OUT/ground_truth_3class"
HC_Labels="$OUT/pseudo_high_${HC}"
Merged="$OUT/Recover-Merged"
REPORT="$OUT/report"

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
mkdir -p "$REPORT"

echo "==> [6/6] mAP evaluation"
cd "$YOLO"
run_map () {
  local preds="$1" run_name="$2"
  conda run -n shoaib python val_v3.py \
    --data "$DATA" --name "$run_name" --gt "$GT_3CLASS" \
    --saved_predictions "$preds" --exist-ok --device "$DEVICE" \
    2>&1 | tee "$REPORT/metrics_${run_name}.log"
}
run_map "$HC_Labels" "${NAME}-YOLO"
run_map "$Merged" "${NAME}-Recovered"

mkdir -p "$OUT/ground_truth"
for img in $(sort "$YOLO/b1c66a42_1seq.txt"); do
  base=$(basename "$img" .jpg)
  cp "$SEQ_DIR/${base}.txt" "$OUT/ground_truth/${base}.txt" 2>/dev/null || true
  cp "$img" "$OUT/ground_truth/${base}.jpg"
done

conda run -n shoaib python "$ROOT/scripts/summarize_sequence_metrics.py" \
  --report-dir "$REPORT" --output "$REPORT/summary.json"

echo "==> Done. Summary: $REPORT/summary.json"
