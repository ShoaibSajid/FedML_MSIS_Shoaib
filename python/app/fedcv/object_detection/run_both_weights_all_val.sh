#!/usr/bin/env bash
set -euo pipefail
ROOT="/Shoaib/FedML_MSIS_Shoaib/python/app/fedcv/object_detection"
bash "$ROOT/run_all_val_sequences.sh" \
  "/Shoaib/BDD100k_yolov8s/full_v8s/weights/best.pt" \
  "full_v8s" \
  "${DEVICE:-0}"

bash "$ROOT/run_all_val_sequences.sh" \
  "/Shoaib/BDD100k_yolv8m/full_v8m/weights/best.pt" \
  "full_v8m" \
  "${DEVICE:-0}"

echo "Both weight evaluations complete."
