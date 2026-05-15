#!/usr/bin/env bash
# Phase 3: scrape + filter + pose-extract cat clips.
set -euo pipefail

# shellcheck disable=SC1091
[[ -f .env ]] && { set -a; source .env; set +a; }

SOURCE="${SOURCE:-pexels}"

python -m robot_pet_cat.cli collect-data --source "$SOURCE"
