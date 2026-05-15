#!/usr/bin/env bash
# Phase 2: pull ~10-30 short CC-licensed cat clips, run pose extraction, retarget
# to Go2 reference trajectories. Output: data/motion_clips/*.npz consumed by AMP.
set -euo pipefail

# shellcheck disable=SC1091
[[ -f .env ]] && { set -a; source .env; set +a; }

RAW_DIR="${RAW_DIR:-data/motion_clips_raw}"

mkdir -p "$RAW_DIR" data/motion_clips

# Step 1 — manual or scripted clip download into $RAW_DIR (Pexels / Pixabay).
# We deliberately do not automate the source ToS-checking; you curate ~30 clips
# from licensed sources before running this script.
echo "Place CC-licensed cat clips (.mp4) into $RAW_DIR before continuing."
echo "Sources: Pexels, Pixabay, Wikimedia Commons, or your own footage."
read -rp "Press enter once the directory has clips, or Ctrl-C to abort. "

# Step 2 — pose extraction + retargeting via the CLI.
python -m robot_pet_cat.cli retarget --clips "$RAW_DIR"

echo "Done. Retargeted clips in data/motion_clips/."
