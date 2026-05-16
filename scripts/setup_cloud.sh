#!/usr/bin/env bash
# Run this on a fresh RunPod / Lambda Labs Ubuntu 22.04+ box.
# Assumes you've SSH'd in and cloned the repo.
#
# This script is intended to be one-shot: end-to-end setup that lands you
# in a venv with everything installed and the smoke test green.
#
# Why python3.11:
#   - mujoco_playground requires Python >= 3.11.
#   - openpifpaf 0.13 pins torch==1.13.1 which only has 3.10/3.11 wheels.
#   - deeplabcut 2.3.x (if you ever swap pose backends) is also <=3.11.
#
# Why setuptools<70:
#   - setuptools >=80 (early 2025) stopped shipping pkg_resources by default.
#   - openpifpaf's setup.py imports pkg_resources at build time.
#   - PIP_CONSTRAINT enforces this pin in pip's PEP 517 isolated build envs.

set -euo pipefail

# --- 1. Pick Python 3.11 -----------------------------------------------------
detect_py() {
  for p in python3.11 python3.12; do
    if command -v "$p" >/dev/null 2>&1; then echo "$p"; return; fi
  done
}

PY="$(detect_py)"

if [[ -z "$PY" || "$PY" != "python3.11" ]]; then
  if [[ "$PY" == "python3.12" ]]; then
    echo ">>> only python3.12 found; installing 3.11 from deadsnakes"
  else
    echo ">>> no python3.11 found, installing it from deadsnakes"
  fi
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends software-properties-common
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev
  PY="python3.11"
fi
echo ">>> using $PY ($("$PY" --version))"

# --- 2. System deps ---------------------------------------------------------
echo ">>> apt deps (build tools, ffmpeg, GL)"
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  git git-lfs python3-pip \
  ffmpeg libgl1 libglib2.0-0 libegl1 \
  build-essential cmake

# --- 3. Venv ----------------------------------------------------------------
echo ">>> python venv"
"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

# Make the setuptools pin persist across re-activations.
CONSTRAINT_PATH="$(pwd)/.venv/pip-constraints.txt"
cat > "$CONSTRAINT_PATH" <<'CON'
setuptools<70
CON
# Append a one-liner to the activate script so re-sourcing it sets PIP_CONSTRAINT.
if ! grep -q "PIP_CONSTRAINT" .venv/bin/activate; then
    echo "export PIP_CONSTRAINT=$CONSTRAINT_PATH" >> .venv/bin/activate
fi
export PIP_CONSTRAINT="$CONSTRAINT_PATH"

# --- 4. Pip + Python deps ---------------------------------------------------
echo ">>> upgrade pip / pin setuptools<70"
pip install --upgrade pip wheel "setuptools<70"

echo ">>> project (motion + brain + pose + retargeting + dev) -- all extras"
pip install -e ".[motion,brain,pose,retargeting,dev]"

# --- 5. External service logins ---------------------------------------------
echo ">>> HF + W&B login (if .env has the keys)"
if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  set -a; source .env; set +a
  echo "${HF_TOKEN:-}" | huggingface-cli login --token "${HF_TOKEN:-}" --add-to-git-credential || true
  wandb login "${WANDB_API_KEY:-}" || true
else
  echo "No .env found - log in manually with huggingface-cli login && wandb login"
fi

# --- 6. Submodule + smoke test ----------------------------------------------
echo ">>> mujoco_menagerie submodule"
git submodule add https://github.com/google-deepmind/mujoco_menagerie third_party/mujoco_menagerie || true
git submodule update --init --recursive

echo ">>> smoke test"
python -m robot_pet_cat.cli sim --steps 100

echo
echo "============================================================"
echo "Setup complete. Next steps:"
echo "  python scripts/fetch_clip.py 34482817 --then-extract"
echo "  rpc retarget --clips data/motion_clips_raw --out data/motion_clips"
echo "============================================================"
