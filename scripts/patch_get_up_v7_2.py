#!/usr/bin/env python3
"""
get_up v7.2 (rev 2) -- v7 FR-Net + reward/termination tweaks.

Changes vs v7:
  * stand_fully_bonus weight 15 -> 30
  * fell_over limit_angle 70 deg -> 100 deg
  * all_feet_contact weight 1.0 -> 4.0

Earlier rev added a feet_under_body reward that broke because mjlab body
names use a "robot/" prefix that I didn't account for. Reverting to the
three reward-shape changes only -- still the meaningful signal.
"""

from __future__ import annotations
import subprocess
import sys
from pathlib import Path

V3 = Path("/workspace/_v3_baseline/get_up")
ACTIVE = Path("/workspace/unitree_rl_mjlab/src/tasks/get_up")
V7 = Path("/workspace/robot-pet-cat/scripts/patch_get_up_v7.py")


def apply_v7_then_tweak() -> int:
    rc = subprocess.run(["python3", str(V7)], capture_output=True, text=True)
    if rc.returncode != 0:
        print("ERROR: v7 patcher failed:\n" + rc.stderr, file=sys.stderr)
        return rc.returncode
    print(rc.stdout)

    cfg_path = ACTIVE / "get_up_env_cfg.py"
    txt = cfg_path.read_text()

    def must_replace(old, new, label):
        nonlocal txt
        if old not in txt:
            raise RuntimeError("anchor not found: " + label)
        if txt.count(old) > 1:
            raise RuntimeError("anchor not unique: " + label)
        txt = txt.replace(old, new)
        print("[edit] " + label)

    must_replace(
        "func=get_up_mdp.stand_fully_bonus,\n            weight=15.0,",
        "func=get_up_mdp.stand_fully_bonus,\n            weight=30.0,",
        "stand_fully_bonus 15 -> 30",
    )
    must_replace(
        "func=get_up_mdp.all_feet_contact,\n            weight=1.0,",
        "func=get_up_mdp.all_feet_contact,\n            weight=4.0,",
        "all_feet_contact 1.0 -> 4.0",
    )
    must_replace(
        "math.radians(70.0)",
        "math.radians(100.0)",
        "fell_over 70deg -> 100deg",
    )

    cfg_path.write_text(txt)

    rc = subprocess.run(
        ["python3", "-c", "import ast; ast.parse(open('" + str(cfg_path) + "').read())"],
        capture_output=True, text=True,
    )
    if rc.returncode != 0:
        print("ERROR: syntax check failed:\n" + rc.stderr, file=sys.stderr)
        return 8

    print("[ok] v7.2 (rev 2) patch applied")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(apply_v7_then_tweak())
    except RuntimeError as e:
        print("ERROR: " + str(e), file=sys.stderr)
        sys.exit(9)
