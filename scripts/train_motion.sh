#!/usr/bin/env bash
# Kicks off the Tier 1 AMP run. Assumes you've run setup_cloud.sh, which
# creates .venv/. We resolve to the venv's python directly so this works
# whether or not the parent shell has the venv activated.
set -euo pipefail

CONFIG="${CONFIG:-configs/motion/cat_amp.yaml}"

# shellcheck disable=SC1091
[[ -f .env ]] && { set -a; source .env; set +a; }

PY="python"
if [[ -x .venv/bin/python ]]; then
  PY=".venv/bin/python"
fi

"$PY" -m robot_pet_cat.cli train-motion --config "$CONFIG"
