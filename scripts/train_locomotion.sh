#!/usr/bin/env bash
# Kicks off a Phase 2 PPO run. Assumes you're on a GPU box and have run setup_cloud.sh.
set -euo pipefail

CONFIG="${CONFIG:-configs/locomotion/unitree_go2_ppo.yaml}"

# shellcheck disable=SC1091
[[ -f .env ]] && { set -a; source .env; set +a; }

python -m robot_pet_cat.cli train-locomotion --config "$CONFIG"
