#!/usr/bin/env bash
# Initialize the Yolov5_DeepSORT_PseudoLabels git submodule and install its dependencies.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SUBMODULE_PATH="python/app/fedcv/object_detection/Yolov5_DeepSORT_PseudoLabels"
SUBMODULE_DIR="$SCRIPT_DIR/Yolov5_DeepSORT_PseudoLabels"

echo "==> Initializing Yolov5_DeepSORT_PseudoLabels submodule..."

if [ -f "$REPO_ROOT/.gitmodules" ]; then
  cd "$REPO_ROOT"
  git submodule update --init --recursive "$SUBMODULE_PATH"
else
  echo "WARNING: .gitmodules not found at $REPO_ROOT — run this script from a full git clone."
fi

if [ ! -d "$SUBMODULE_DIR" ] || [ -z "$(ls -A "$SUBMODULE_DIR" 2>/dev/null)" ]; then
  echo ""
  echo "ERROR: Submodule directory is empty."
  echo "  Clone the parent repo with submodules:"
  echo "    git clone --recurse-submodules https://github.com/ShoaibSajid/FedML_MSIS_Shoaib.git"
  echo "  Or, from an existing clone:"
  echo "    git submodule update --init --recursive $SUBMODULE_PATH"
  echo ""
  echo "  (The submodule repo may be private — ensure your GitHub credentials are configured.)"
  exit 1
fi

echo "==> Installing submodule Python dependencies..."
cd "$SUBMODULE_DIR"
pip install -r requirements.txt

# The submodule ships a patched YOLOv5 fork. If the bundled copy is missing, fetch it.
if [ ! -d "yolov5" ] || [ ! -f "yolov5/val.py" ]; then
  echo "==> Fetching patched YOLOv5 fork into submodule..."
  rm -rf yolov5
  git clone --recursive https://github.com/ShoaibSajid/yolov5.git
fi

echo "==> Installing FedML app dependencies..."
cd "$SCRIPT_DIR"
pip install fedml --upgrade
pip install -r requirements.txt

echo "==> Setup complete."
