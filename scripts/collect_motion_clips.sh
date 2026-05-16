#!/usr/bin/env bash
# Phase 2 helper: pull ~10-30 short CC-licensed cat clips, run pose extraction,
# retarget to Go2 reference trajectories. Output: data/motion_clips/*.npz.
set -euo pipefail

# shellcheck disable=SC1091
[[ -f .env ]] && { set -a; source .env; set +a; }

PY="python"
if [[ -x .venv/bin/python ]]; then
  PY=".venv/bin/python"
fi

RAW_DIR="${RAW_DIR:-data/motion_clips_raw}"
mkdir -p "$RAW_DIR" data/motion_clips

echo "Place CC-licensed cat clips (.mp4) into $RAW_DIR before continuing."
echo "Sources: Pexels, Pixabay, Wikimedia Commons, or your own footage."
read -rp "Press enter once the directory has clips, or Ctrl-C to abort. "

"$PY" -m robot_pet_cat.cli retarget --clips "$RAW_DIR"

echo "Done. Retargeted clips in data/motion_clips/."
