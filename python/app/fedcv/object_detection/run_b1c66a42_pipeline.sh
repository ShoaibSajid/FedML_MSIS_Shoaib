#!/usr/bin/env bash
# Standalone inference -> recovery -> mAP for BDD sequence b1c66a42-6f7d68ca
set -euo pipefail

ROOT="/Shoaib/FedML_MSIS_Shoaib/python/app/fedcv/object_detection"
RECOVERY="$ROOT/Yolov5_DeepSORT_PseudoLabels"
YOLO="$RECOVERY/yolov5"
WEIGHTS="$ROOT/best.pt"
SEQ_DIR="/datasets/BDDComplete/val/b1c66a42-6f7d68ca"
NAME="b1c66a42-6f7d68ca"
LC="0.1"
HC="0.5"
IoU="0.6"
DEVICE="${DEVICE:-0}"
DS="resnet50_MSMT17"
DATA="data/bdd_b1c66a42.yaml"
DATA_list="./yolov5/b1c66a42_1seq.txt"

OUT="/Shoaib/FedML_MSIS_Shoaib/python/app/fedcv/object_detection/runs/sequence_eval/${NAME}"
GT_3CLASS="$OUT/ground_truth_3class"
LC_Labels="$OUT/pseudo_low_${LC}"
HC_Labels="$OUT/pseudo_high_${HC}"
FW_Labels="$OUT/Recover-FW"
BW_Labels="$OUT/Recover-BW"
Merged="$OUT/Recover-Merged"
REPORT="$OUT/report"

mkdir -p "$OUT" "$REPORT" "$GT_3CLASS"

echo "==> Prepare 3-class GT + images for inference dataloader"
conda run -n shoaib python "$ROOT/scripts/remap_gt_3class.py" \
  --src "$SEQ_DIR" --dst "$GT_3CLASS" --copy-images
rm -f "$YOLO/b1c66a42_1seq.cache"
ls "$GT_3CLASS"/*.jpg | sort > "$YOLO/b1c66a42_1seq.txt"
cp "$YOLO/b1c66a42_1seq.txt" "$REPORT/image_list.txt"

echo "==> Using weights: $WEIGHTS"
echo "==> Sequence GT:   $SEQ_DIR"
echo "==> Output:        $OUT"
echo "==> Device:        $DEVICE"
conda run -n shoaib python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count()); print('Current:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')" 2>&1 || true

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

cd "$RECOVERY"

echo "==> [1/6] YOLO inference - low confidence pseudo labels (${LC})"
conda run -n shoaib python "$YOLO/val.py" \
  --weights "$WEIGHTS" --device "$DEVICE" --name "${NAME}-LC" --save-txt \
  --data "$YOLO/$DATA" --conf-thres "$LC" --iou-thres "$IoU" --save-conf --exist-ok \
  --batch-size 8 --workers 4

echo "==> [2/6] YOLO inference - high confidence pseudo labels (${HC})"
conda run -n shoaib python "$YOLO/val.py" \
  --weights "$WEIGHTS" --device "$DEVICE" --name "${NAME}-HC" --save-txt \
  --data "$YOLO/$DATA" --conf-thres "$HC" --iou-thres "$IoU" --save-conf --exist-ok \
  --batch-size 8 --workers 4

rm -rf "$LC_Labels" "$HC_Labels"
mv "$YOLO/runs/val/${NAME}-LC/labels" "$LC_Labels"
mv "$YOLO/runs/val/${NAME}-HC/labels" "$HC_Labels"

echo "==> Copy images into low-conf label folder for DeepSORT recovery"
xargs -a "$DATA_list" -I{} cp {} "$LC_Labels/"

cp -r "$LC_Labels" "$OUT/generated_pseudo_labels"
cp -r "$HC_Labels" "$OUT/generated_pseudo_high"

echo "==> [3/6] Forward recovery"
rm -rf "$FW_Labels"
conda run -n shoaib python trackv2_from_file.py \
  --source "$LC_Labels" --output "$FW_Labels" \
  --conf-thres "$HC" --low-conf-thres "$LC" --high-conf-thres "$HC" --iou-thres "$IoU" --save-txt --exist-ok \
  --deep_sort_model "$DS" --yolo_model "$WEIGHTS" --device "$DEVICE"

echo "==> [4/6] Backward recovery"
rm -rf "$BW_Labels"
conda run -n shoaib python trackv2_from_file.py \
  --source "$LC_Labels" --output "$BW_Labels" \
  --conf-thres "$HC" --low-conf-thres "$LC" --high-conf-thres "$HC" --iou-thres "$IoU" --save-txt --exist-ok \
  --deep_sort_model "$DS" --yolo_model "$WEIGHTS" --device "$DEVICE" --reverse

echo "==> [5/6] Merge forward + backward"
rm -rf "$Merged"
conda run -n shoaib python merge_forward_backward_v2.py \
  --forward "$FW_Labels" --backward "$BW_Labels" --merged "$Merged"

xargs -a "$DATA_list" -I{} cp {} "$Merged/"
cp -r "$Merged" "$OUT/recovered_pseudo_labels"

echo "==> Remap 13-class GT to 3-class for mAP (already at $GT_3CLASS)"

echo "==> [6/6] mAP evaluation (3-class GT vs generated / recovered)"
cd "$YOLO"

run_map () {
  local label="$1"
  local preds="$2"
  local run_name="$3"
  echo "--- mAP: $label ---"
  conda run -n shoaib python val_v3.py \
    --data "$DATA" --name "$run_name" --gt "$GT_3CLASS" \
    --saved_predictions "$preds" --exist-ok --device "$DEVICE" \
    2>&1 | tee "$REPORT/metrics_${run_name}.log"
}

run_map "Generated (high conf ${HC})" "$HC_Labels" "${NAME}-YOLO"
run_map "Recovered (merged)" "$Merged" "${NAME}-Recovered"

echo "==> Copy raw 13-class ground truth"
mkdir -p "$OUT/ground_truth"
for img in $(sort "$YOLO/b1c66a42_1seq.txt"); do
  base=$(basename "$img" .jpg)
  cp "$SEQ_DIR/${base}.txt" "$OUT/ground_truth/${base}.txt" 2>/dev/null || true
  cp "$img" "$OUT/ground_truth/${base}.jpg"
done

echo "==> Summarize metrics"
conda run -n shoaib python "$ROOT/scripts/summarize_sequence_metrics.py" \
  --report-dir "$REPORT" --output "$REPORT/summary.json"

echo "==> Done. Artifacts in $OUT"
echo "    image_list.txt          -> $REPORT/image_list.txt"
echo "    generated_pseudo_labels -> $OUT/generated_pseudo_labels"
echo "    recovered_pseudo_labels -> $OUT/recovered_pseudo_labels"
echo "    ground_truth            -> $OUT/ground_truth"
echo "    metrics logs            -> $REPORT/"
