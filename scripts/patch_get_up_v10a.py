#!/usr/bin/env python3
"""
get_up v10a -- v7 trained longer (15k iters via launch flag).

Hypothesis: v7 at 4k iters failed; v7.1 at 8k showed +42% base_height_recovery
with more body lift but back legs still splayed. Maybe v7's architecture is
fine and prior runs were under-trained. Cheapest test before adding any new
design lever -- rules out "needed more compute" first.

This patcher is identical to patch_get_up_v7.py; the only difference is the
launch flag (max-iterations 15000). The patcher exists so the v10 chain has a
uniform "patch -> train -> render" shape per variant.

Usage on a runpod:
  python3 scripts/patch_get_up_v10a.py
  cd /workspace/unitree_rl_mjlab && source /workspace/mjlab_venv/bin/activate
  set -a && source /workspace/robot-pet-cat/.env && set +a
  nohup python3 scripts/train.py Unitree-Go2-GetUp --agent.max-iterations 15000 \
      > /tmp/getup_v10a_train.log 2>&1 &
"""

from __future__ import annotations
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V7 = HERE / "patch_get_up_v7.py"


def main() -> int:
    rc = subprocess.run(["python3", str(V7)])
    if rc.returncode != 0:
        print(f"ERROR: v7 patcher failed (rc={rc.returncode})", file=sys.stderr)
        return rc.returncode
    print("[ok] v10a patch applied (== v7, train 15k iters)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
