#!/usr/bin/env bash
# Resume pipeline from recovery step (after inference completed)
set -euo pipefail

ROOT="/Shoaib/FedML_MSIS_Shoaib/python/app/fedcv/object_detection"
RECOVERY="$ROOT/Yolov5_DeepSORT_PseudoLabels"
YOLO="$RECOVERY/yolov5"
WEIGHTS="$ROOT/best.pt"
SEQ_DIR="/datasets/BDDComplete/val/b1c66a42-6f7d68ca"
NAME="b1c66a42-6f7d68ca"
HC="0.5"
IoU="0.6"
DEVICE="${DEVICE:-0}"
DS="resnet50_MSMT17"
DATA="data/bdd_b1c66a42.yaml"
DATA_list="./yolov5/b1c66a42_1seq.txt"

OUT="/Shoaib/FedML_MSIS_Shoaib/python/app/fedcv/object_detection/runs/sequence_eval/${NAME}"
GT_3CLASS="$OUT/ground_truth_3class"
LC_Labels="$OUT/pseudo_low_0.1"
HC_Labels="$OUT/pseudo_high_${HC}"
FW_Labels="$OUT/Recover-FW"
BW_Labels="$OUT/Recover-BW"
Merged="$OUT/Recover-Merged"
REPORT="$OUT/report"

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
cd "$RECOVERY"

echo "==> Device: $DEVICE"
conda run -n shoaib python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count()); print('Current:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')" 2>&1 || true

echo "==> [3/6] Forward recovery"
rm -rf "$FW_Labels"
conda run -n shoaib python trackv2_from_file.py \
  --source "$LC_Labels" --output "$FW_Labels" \
  --conf-thres "$HC" --low-conf-thres 0.1 --high-conf-thres "$HC" --iou-thres "$IoU" --save-txt --exist-ok \
  --deep_sort_model "$DS" --yolo_model "$WEIGHTS" --device "$DEVICE"

echo "==> [4/6] Backward recovery"
rm -rf "$BW_Labels"
conda run -n shoaib python trackv2_from_file.py \
  --source "$LC_Labels" --output "$BW_Labels" \
  --conf-thres "$HC" --low-conf-thres 0.1 --high-conf-thres "$HC" --iou-thres "$IoU" --save-txt --exist-ok \
  --deep_sort_model "$DS" --yolo_model "$WEIGHTS" --device "$DEVICE" --reverse

echo "==> [5/6] Merge forward + backward"
rm -rf "$Merged"
conda run -n shoaib python merge_forward_backward_v2.py \
  --forward "$FW_Labels" --backward "$BW_Labels" --merged "$Merged"

xargs -a "$DATA_list" -I{} cp {} "$Merged/"
cp -r "$Merged" "$OUT/recovered_pseudo_labels"

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

echo "==> Done. Artifacts in $OUT"
