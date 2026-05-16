#!/usr/bin/env bash
# Run scripts/extract_keypoints.py inside the dedicated pose venv (.venv-pose).
# This wrapper exists so callers don't have to remember to activate the right
# venv -- pose extraction always uses .venv-pose, never the main .venv.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <clip.mp4> [--out OUT] [--pcutoff N] [--body-length-m N] [--floor-y N]"
  exit 2
fi

if [[ ! -x .venv-pose/bin/python ]]; then
  echo "ERROR: .venv-pose not found. Run: bash scripts/setup_pose_venv.sh"
  exit 1
fi

# shellcheck disable=SC1091
[[ -f .env ]] && { set -a; source .env; set +a; }

.venv-pose/bin/python scripts/extract_keypoints.py "$@"
