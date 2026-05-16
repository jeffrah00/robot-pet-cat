#!/usr/bin/env bash
# Isolated venv for DLC SuperAnimal pose extraction.
#
# Why a separate venv:
#   DLC 2.3.x needs tensorflow<2.13 + numpy<2.0. JAX 0.4.30+ (in our [motion]
#   extras) needs numpy>=2.0. These are mutually exclusive in one venv. So we
#   keep .venv (motion/brain/training) and .venv-pose (one-shot pose extraction)
#   side by side; they exchange data via JSON files in data/motion_clips_raw/.
#
# Run after scripts/setup_cloud.sh has created the main .venv.

set -euo pipefail

PY="${PY:-python3.11}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "ERROR: $PY not found. Run setup_cloud.sh first to install it."
  exit 1
fi

VENV_DIR=".venv-pose"

echo ">>> creating $VENV_DIR with $PY ($($PY --version))"
"$PY" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Pin setuptools<70 in this venv too (pkg_resources).
pip install --upgrade pip wheel "setuptools<70"

# Install order matters: numpy + TF before DLC so the TF binary wheel is
# matched against the right numpy ABI.
echo ">>> numpy < 2.0 (DLC/TF compat)"
pip install "numpy<2.0"

echo ">>> tensorflow 2.10-2.12 (Keras 2; avoids 2.13+ Keras 3 issues)"
pip install "tensorflow>=2.10,<2.13"

echo ">>> DLC hidden runtime deps"
pip install tf_keras tensorpack tf_slim

echo ">>> deeplabcut + opencv"
pip install "deeplabcut>=2.3.11" "opencv-python>=4.10"

# Verify the stack actually imports.
echo ">>> verifying"
TF_USE_LEGACY_KERAS=1 python <<'PY'
import deeplabcut, tensorflow as tf, numpy as np, pandas as pd
print(f"deeplabcut {deeplabcut.__version__}")
print(f"tensorflow {tf.__version__}")
print(f"numpy {np.__version__}")
print(f"pandas {pd.__version__}")
PY

echo
echo "============================================================"
echo "Pose venv ready at $VENV_DIR"
echo "Use it for pose extraction:"
echo "  bash scripts/extract_keypoints.sh path/to/clip.mp4"
echo "============================================================"
