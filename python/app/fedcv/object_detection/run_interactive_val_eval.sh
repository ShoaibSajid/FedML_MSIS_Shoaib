#!/usr/bin/env bash
# Interactive eval: one sequence at a time, both v8s + v8m, prompt to continue.
set -euo pipefail

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export DEVICE="${DEVICE:-0}"

ROOT="/Shoaib/FedML_MSIS_Shoaib/python/app/fedcv/object_detection"
conda run -n shoaib python "$ROOT/scripts/run_interactive_val_eval.py"
