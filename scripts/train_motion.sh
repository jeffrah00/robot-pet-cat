#!/usr/bin/env bash
# Kicks off the Tier 1 AMP run. Assumes you're on a GPU box and have run setup_cloud.sh.
set -euo pipefail

CONFIG="${CONFIG:-configs/motion/cat_amp.yaml}"

# shellcheck disable=SC1091
[[ -f .env ]] && { set -a; source .env; set +a; }

python -m robot_pet_cat.cli train-motion --config "$CONFIG"
