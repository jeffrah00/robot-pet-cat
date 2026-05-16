#!/usr/bin/env bash
# Set up the DLC pose extraction environment via conda (micromamba), using
# DLC's own pinned environment file. This sidesteps the dep-hell we hit
# trying to build DLC's runtime with pip on Python 3.11.
#
# Why micromamba (not full conda/anaconda):
#   - Single static binary, no Python interpreter dependency for itself.
#   - Same solver as mamba, much faster than classic conda.
#   - One curl + tar to install.

set -euo pipefail

ENV_NAME="${ENV_NAME:-dlc}"

# --- 1. Install micromamba if missing ---------------------------------------
if ! command -v micromamba >/dev/null 2>&1; then
    echo ">>> installing micromamba"
    cd /tmp
    curl -fLs https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba
    sudo mv bin/micromamba /usr/local/bin/micromamba
    rm -rf /tmp/bin
    cd - >/dev/null
fi
echo ">>> using micromamba $(micromamba --version)"

# --- 2. Fetch DLC's official conda env file ---------------------------------
DLC_YAML=/tmp/DEEPLABCUT.yaml
echo ">>> fetching DLC's environment file"
curl -fL -o "$DLC_YAML" \
    https://raw.githubusercontent.com/DeepLabCut/DeepLabCut/main/conda-environments/DEEPLABCUT.yaml

# DLC's file may use 'name: DEEPLABCUT' -- override to our name.
sed -i "s/^name:.*/name: $ENV_NAME/" "$DLC_YAML"

# --- 3. Create the env ------------------------------------------------------
echo ">>> creating conda env '$ENV_NAME' (this takes 5-10 min)"
# -y to skip prompts; --root-prefix isolates state under our project so it's
# easy to nuke later
micromamba env create --name "$ENV_NAME" --file "$DLC_YAML" --yes

# --- 4. Verify --------------------------------------------------------------
echo ">>> verifying"
micromamba run -n "$ENV_NAME" python <<'PY'
import deeplabcut, tensorflow as tf, numpy as np, pandas as pd
print(f"deeplabcut {deeplabcut.__version__}")
print(f"tensorflow {tf.__version__}")
print(f"numpy {np.__version__}")
print(f"pandas {pd.__version__}")
PY

echo
echo "============================================================"
echo "Pose conda env '$ENV_NAME' ready."
echo "Run pose extraction with:"
echo "  bash scripts/extract_keypoints.sh <clip.mp4>"
echo "============================================================"
