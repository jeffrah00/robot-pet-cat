#!/usr/bin/env python3
"""
get_up v7.2 -- v7 FR-Net + reward/termination tweaks.

Hypothesis: v7 settled in sphynx because (a) stand_fully_bonus weight 15
wasn't enough to pull through, and (b) fell_over termination at 70-deg
limit_angle truncates rollouts before the policy can demonstrate recovery
through wobbly intermediate poses.

Changes vs v7:
  * stand_fully_bonus weight 15 -> 30 (stronger pull toward goal)
  * fell_over limit_angle 70 deg -> 100 deg (tolerate intermediate wobble)
  * all_feet_contact weight 1.0 -> 4.0 (penalize splayed sphynx)
  * Add feet_under_body reward (penalizes feet that splay out wide of
    the base xy projection -- exactly the v7 sphynx failure mode)

Everything else (FR-Net 30->64->64->4 head, pose jitter, hind_legs_extended,
v4c smoothness) is identical to v7.
"""

from __future__ import annotations
import shutil
import subprocess
import sys
from pathlib import Path

V3 = Path("/workspace/_v3_baseline/get_up")
ACTIVE = Path("/workspace/unitree_rl_mjlab/src/tasks/get_up")
V7 = Path("/workspace/robot-pet-cat/scripts/patch_get_up_v7.py")


def apply_v7_then_tweak() -> int:
    # Run the v7 patcher first so we inherit all of its edits.
    rc = subprocess.run(["python3", str(V7)], capture_output=True, text=True)
    if rc.returncode != 0:
        print("ERROR: v7 patcher failed:\n" + rc.stderr, file=sys.stderr)
        return rc.returncode
    print(rc.stdout)

    cfg_path = ACTIVE / "get_up_env_cfg.py"
    rewards_path = ACTIVE / "mdp" / "rewards.py"
    txt = cfg_path.read_text()

    # 1. Boost stand_fully_bonus weight 15 -> 30.
    old = (
        '"stand_fully_bonus": RewardTermCfg(\n'
        '            func=get_up_mdp.stand_fully_bonus,\n'
        '            weight=15.0,\n'
        '        ),'
    )
    new = (
        '"stand_fully_bonus": RewardTermCfg(\n'
        '            func=get_up_mdp.stand_fully_bonus,\n'
        '            weight=30.0,\n'
        '        ),'
    )
    if old not in txt:
        print("ERROR: stand_fully_bonus anchor not found", file=sys.stderr)
        return 4
    txt = txt.replace(old, new)

    # 2. Boost all_feet_contact weight 1.0 -> 4.0.
    afc_old = (
        '"all_feet_contact": RewardTermCfg(\n'
        '            func=get_up_mdp.all_feet_contact,\n'
        '            weight=1.0,\n'
        '            params={"sensor_name": "feet_ground_contact"},\n'
        '        ),'
    )
    afc_new = afc_old.replace("weight=1.0,", "weight=4.0,")
    if afc_old not in txt:
        print("ERROR: all_feet_contact anchor not found", file=sys.stderr)
        return 5
    txt = txt.replace(afc_old, afc_new)

    # 3. Relax fell_over limit_angle 70 -> 100 deg.
    fo_old = (
        '"fell_over": TerminationTermCfg(\n'
        '    func=mdp.bad_orientation,\n'
        '    params={"limit_angle": math.radians(70.0)},\n'
        '  ),'
    )
    fo_new = fo_old.replace("math.radians(70.0)", "math.radians(100.0)")
    if fo_old not in txt:
        print("ERROR: fell_over anchor not found", file=sys.stderr)
        return 6
    txt = txt.replace(fo_old, fo_new)

    # 4. Insert feet_under_body reward term before joint_velocity_penalty.
    insert_anchor = '"joint_velocity_penalty": RewardTermCfg('
    insertion = (
        '"feet_under_body": RewardTermCfg(\n'
        '            func=get_up_mdp.feet_under_body,\n'
        '            weight=3.0,\n'
        '        ),\n'
        '        '
    )
    if insert_anchor not in txt:
        print("ERROR: joint_velocity_penalty anchor not found", file=sys.stderr)
        return 7
    txt = txt.replace(insert_anchor, insertion + insert_anchor, 1)

    cfg_path.write_text(txt)
    print("[edit] get_up_env_cfg.py: v7.2 weights + feet_under_body term")

    # 5. Append the feet_under_body reward function to mdp/rewards.py.
    extra = '''

# === v7.2 ============================================================
_FOOT_BODY_NAMES_V72 = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")


def feet_under_body(env, asset_cfg=_DEFAULT_ASSET_CFG):
    """Reward feet whose xy is close to the base xy.

    Penalizes the sphynx/splay failure mode observed in get_up v7.
    For each foot, compute horizontal distance from base; sum, negate,
    then exp so values live in [0, 1].
    """
    asset = env.scene[asset_cfg.name]
    base_xy = asset.data.root_link_pos_w[:, :2]                    # (N, 2)
    body_names = asset.data.body_names if hasattr(asset.data, "body_names") else asset.body_names
    feet_ids = [body_names.index(n) for n in _FOOT_BODY_NAMES_V72]
    foot_pos = asset.data.body_link_pos_w[:, feet_ids, :2]         # (N, 4, 2)
    dxy = foot_pos - base_xy.unsqueeze(1)                          # (N, 4, 2)
    dist = torch.linalg.norm(dxy, dim=-1)                          # (N, 4)
    # Tucked feet have dist ~ 0.15-0.20 m; sphynx splay ~ 0.30+. Use a
    # 0.20 m soft target.
    err = (dist - 0.20).clamp(min=0.0).sum(dim=-1)                 # (N,)
    return torch.exp(-2.0 * err)
'''
    rewards_path.write_text(rewards_path.read_text() + extra)
    print("[edit] mdp/rewards.py: feet_under_body")

    # 6. Syntax check.
    rc = subprocess.run(
        ["python3", "-c",
         f"import ast;"
         f"ast.parse(open('{cfg_path}').read());"
         f"ast.parse(open('{rewards_path}').read())"],
        capture_output=True, text=True,
    )
    if rc.returncode != 0:
        print(f"ERROR: syntax check failed:\n{rc.stderr}", file=sys.stderr)
        return 8

    print("[ok] v7.2 patch applied")
    return 0


if __name__ == "__main__":
    sys.exit(apply_v7_then_tweak())
