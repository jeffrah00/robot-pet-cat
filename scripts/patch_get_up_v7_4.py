#!/usr/bin/env python3
"""
get_up v7.4 -- v7 FR-Net + reward shaping + hip-abduction penalty.

Failure mode addressed: across v7, v7.1, v7.3, the cat ends in a "dog sitting"
pose -- body lifted, front legs tucked, but hind legs splayed wide to the
sides (or extended back). The v3 hind_legs_extended reward targets hip_pitch
and knee angles but NOT hip abduction (the joint that controls sideways
splay), so the policy can hit "extended" while the legs point outward.

Changes vs v7:
  * stand_fully_bonus weight 15 -> 30 (from v7.2 -- stronger stand pull)
  * fell_over limit_angle 70 deg -> 100 deg (from v7.2 -- longer rollouts)
  * NEW: all_hips_aligned reward -- penalize non-zero hip abduction on all
    four legs (joints 0, 3, 6, 9). Drives legs to be UNDER the body.
"""

from __future__ import annotations
import subprocess
import sys
from pathlib import Path

V3 = Path("/workspace/_v3_baseline/get_up")
ACTIVE = Path("/workspace/unitree_rl_mjlab/src/tasks/get_up")
V7 = Path("/workspace/robot-pet-cat/scripts/patch_get_up_v7.py")


def apply() -> int:
    rc = subprocess.run(["python3", str(V7)], capture_output=True, text=True)
    if rc.returncode != 0:
        print("ERROR: v7 patcher failed:\n" + rc.stderr, file=sys.stderr)
        return rc.returncode
    print(rc.stdout)

    cfg_path = ACTIVE / "get_up_env_cfg.py"
    rewards_path = ACTIVE / "mdp" / "rewards.py"
    txt = cfg_path.read_text()

    def must_replace(old, new, label):
        nonlocal txt
        if old not in txt:
            raise RuntimeError("anchor not found: " + label)
        if txt.count(old) > 1:
            raise RuntimeError("anchor not unique: " + label)
        txt = txt.replace(old, new)
        print("[edit] " + label)

    # v7.2-style reward boosts and fell_over relaxation
    must_replace(
        "func=get_up_mdp.stand_fully_bonus,\n            weight=15.0,",
        "func=get_up_mdp.stand_fully_bonus,\n            weight=30.0,",
        "stand_fully_bonus 15 -> 30",
    )
    must_replace(
        "math.radians(70.0)",
        "math.radians(100.0)",
        "fell_over 70deg -> 100deg",
    )

    # NEW reward: all_hips_aligned. Insert before joint_velocity_penalty
    # (same insertion pattern as v7's hind_legs_extended).
    insert_anchor = '"joint_velocity_penalty": RewardTermCfg('
    insertion = (
        '"all_hips_aligned": RewardTermCfg(\n'
        '            func=get_up_mdp.all_hips_aligned,\n'
        '            weight=4.0,\n'
        '        ),\n'
        '        '
    )
    if insert_anchor not in txt:
        raise RuntimeError("joint_velocity_penalty anchor not found")
    txt = txt.replace(insert_anchor, insertion + insert_anchor, 1)
    print("[edit] insert all_hips_aligned before joint_velocity_penalty")

    cfg_path.write_text(txt)

    # Append reward function definition.
    extra = '\n\n# === v7.4 ============================================================\n'
    extra += '_ABDUCTION_INDICES_V74 = (0, 3, 6, 9)\n\n\n'
    extra += 'def all_hips_aligned(env, asset_cfg=_DEFAULT_ASSET_CFG):\n'
    extra += '    """Reward all four hip-abduction joints being near 0.\n\n'
    extra += '    Non-zero abduction = legs splayed out to the sides (dog-sitting,\n'
    extra += '    play-bow). Targeting 0 drives the legs UNDER the body, which is\n'
    extra += '    the missing constraint that v3/v7 left unspecified.\n'
    extra += '    """\n'
    extra += '    asset = env.scene[asset_cfg.name]\n'
    extra += '    jp = asset.data.joint_pos\n'
    extra += '    err = sum(jp[:, i].abs() for i in _ABDUCTION_INDICES_V74)\n'
    extra += '    return torch.exp(-2.0 * err)\n'
    rewards_path.write_text(rewards_path.read_text() + extra)
    print("[edit] mdp/rewards.py: all_hips_aligned")

    # Syntax check.
    rc = subprocess.run(
        ["python3", "-c",
         "import ast;"
         "ast.parse(open('" + str(cfg_path) + "').read());"
         "ast.parse(open('" + str(rewards_path) + "').read())"],
        capture_output=True, text=True,
    )
    if rc.returncode != 0:
        print("ERROR: syntax check failed:\n" + rc.stderr, file=sys.stderr)
        return 8

    print("[ok] v7.4 patch applied")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(apply())
    except RuntimeError as e:
        print("ERROR: " + str(e), file=sys.stderr)
        sys.exit(9)
