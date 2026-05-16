#!/usr/bin/env bash
# Run scripts/extract_keypoints.py inside the DLC conda env (micromamba).
# Created by scripts/setup_pose_conda.sh. Users don't activate the env
# manually -- this wrapper does it for them.

set -euo pipefail

ENV_NAME="${ENV_NAME:-dlc}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <clip.mp4> [--out OUT] [--pcutoff N] [--body-length-m N] [--floor-y N]"
  exit 2
fi

if ! command -v micromamba >/dev/null 2>&1; then
  echo "ERROR: micromamba not installed. Run: bash scripts/setup_pose_conda.sh"
  exit 1
fi

if ! micromamba env list | grep -q "^[[:space:]]*$ENV_NAME[[:space:]]"; then
  echo "ERROR: conda env '$ENV_NAME' not found. Run: bash scripts/setup_pose_conda.sh"
  exit 1
fi

# shellcheck disable=SC1091
[[ -f .env ]] && { set -a; source .env; set +a; }

micromamba run -n "$ENV_NAME" python scripts/extract_keypoints.py "$@"
