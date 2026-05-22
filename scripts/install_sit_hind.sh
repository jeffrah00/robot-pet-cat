#!/usr/bin/env bash
# Idempotent installer for the sit_hind task scaffold.
# Copies scripts/_sit_hind_scaffold/ -> unitree_rl_mjlab/src/tasks/sit_hind/
# so the task registers as Unitree-Go2-Sit-Hind.
#
# The scaffold uses the get_up-13k checkpoint's terminal sit pose as a fixed
# reset state (see _sit_hind_scaffold/mdp/events.py). The pose was extracted
# with scripts/probe_get_up_13k_pose.py.
set -euo pipefail
SRC=/workspace/robot-pet-cat/scripts/_sit_hind_scaffold
DST=/workspace/unitree_rl_mjlab/src/tasks/sit_hind
if [ ! -d "$SRC" ]; then echo "ERROR: scaffold not found at $SRC"; exit 1; fi
rm -rf "$DST"
cp -r "$SRC" "$DST"
find "$DST" -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
cd /workspace/unitree_rl_mjlab && python3 -c "import src.tasks; from mjlab.tasks.registry import load_env_cfg; cfg = load_env_cfg('Unitree-Go2-Sit-Hind', play=True); print('sit_hind registered OK')"
