#!/usr/bin/env bash
# Run this on a fresh RunPod / Lambda Labs Ubuntu 22.04+ box.
# Assumes you've SSH'd in and cloned the repo.
#
# Why python3.11 specifically:
#   - mujoco_playground requires Python >= 3.11.
#   - deeplabcut 2.3.x (the SuperAnimal entry point) is only tested against
#     Python 3.10/3.11. On 3.12, its pinned tables==3.8.0 tries to compile
#     blosc2 from source and fails. So we cap at 3.11 until DLC 3.x ships.

set -euo pipefail

# Find a usable Python: prefer 3.11. Empty if none found.
detect_py() {
  for p in python3.11 python3.12; do
    if command -v "$p" >/dev/null 2>&1; then echo "$p"; return; fi
  done
}

PY="$(detect_py)"

if [[ -z "$PY" || "$PY" != "python3.11" ]]; then
  if [[ "$PY" == "python3.12" ]]; then
    echo ">>> only python3.12 found; DLC 2.3.x doesn't build on 3.12, installing 3.11"
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

echo ">>> apt deps (build tools, ffmpeg, GL)"
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  git git-lfs python3-pip \
  ffmpeg libgl1 libglib2.0-0 libegl1 \
  build-essential cmake

echo ">>> python venv"
"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip wheel && pip install "setuptools<70"

echo ">>> project (motion + brain + dev extras)"
pip install -e ".[motion,brain,dev]"

echo ">>> HF + W&B login"
if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  set -a; source .env; set +a
  echo "${HF_TOKEN:-}" | huggingface-cli login --token "${HF_TOKEN:-}" --add-to-git-credential || true
  wandb login "${WANDB_API_KEY:-}" || true
else
  echo "No .env found - log in manually with huggingface-cli login && wandb login"
fi

echo ">>> mujoco_menagerie submodule"
git submodule add https://github.com/google-deepmind/mujoco_menagerie third_party/mujoco_menagerie || true
git submodule update --init --recursive

echo ">>> smoke test"
python -m robot_pet_cat.cli sim --steps 100

echo "Done. Next: bash scripts/train_motion.sh"
echo "       (or: pip install -e \".[pose]\"  for the pose extractor)"
