#!/usr/bin/env bash
# Run this on a fresh RunPod / Lambda Labs Ubuntu 22.04 box to set things up.
# Assumes you've SSH'd in and cloned the repo.

set -euo pipefail

echo ">>> apt deps"
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  git git-lfs python3.10-venv python3-pip \
  ffmpeg libgl1 libglib2.0-0 libegl1 \
  build-essential cmake

echo ">>> python venv"
python3.10 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip wheel

echo ">>> project (locomotion + dev extras)"
pip install -e ".[locomotion,dev]"

echo ">>> HF + W&B login (interactive)"
if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  set -a; source .env; set +a
  echo "${HF_TOKEN:-}" | huggingface-cli login --token "${HF_TOKEN:-}" --add-to-git-credential || true
  wandb login "${WANDB_API_KEY:-}" || true
else
  echo "No .env found — log in manually with huggingface-cli login && wandb login"
fi

echo ">>> mujoco_menagerie submodule"
git submodule add https://github.com/google-deepmind/mujoco_menagerie third_party/mujoco_menagerie || true
git submodule update --init --recursive

echo ">>> smoke test"
python -m robot_pet_cat.cli sim --steps 100

echo "Done. Next: bash scripts/train_locomotion.sh"
