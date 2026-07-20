#!/usr/bin/env bash
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export DEVICE="${DEVICE:-0}"
ROOT="/Shoaib/FedML_MSIS_Shoaib/python/app/fedcv/object_detection"
conda run -n shoaib python "$ROOT/scripts/run_export_improved_all.py" "$@"
