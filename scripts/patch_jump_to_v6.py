#!/usr/bin/env python3
"""
jump_to v6 — "Phase-Gated" (Sun + Yang inspired).

Goal: keep the couch as the target, but restructure the reward so that
every reward term that encourages motion toward the couch is GATED by
uprightness. The v3/v4 failure mode (topple toward couch) becomes
unprofitable.

Design notes (post-lit-review 2026-05-22):
- Sun 2023 splits jumping into takeoff vs flight-landing phases with
  separate reward formulas per phase.
- Wang 2025 uses interaction-aware rewards (gated by contact state).
- v3/v4 had `velocity_toward_couch` always-on, which credits any motion
  toward the couch regardless of posture. v6 multiplies every
  approach/height/landing reward by an upright indicator so toppling
  earns zero credit.
- Phase detection is foot-contact-based (stateless):
    loading  := root_z < rest AND all 4 feet in contact
    airborne := 0 feet in contact
    landing  := >=3 feet in contact after airborne (handled via
                landed_on_couch contact gating)

Reward shape:
- crouch_load: clamp(rest - z, 0, 0.08) * all_feet_contact, weight 3
- airborne_height: (z - rest) * (1 - n_feet/4), weight 10  (only counts
                   height while feet are off the ground)
- approach_couch_upright: exp(-dist) * upright_indicator, weight 3
- on_couch_xy_upright: on_couch * upright_indicator, weight 5
- landed_on_couch: (existing, sparse terminal) weight 15
- body_orientation_l2 doubled (-2.0)
- action_rate_l2 doubled (-0.1)
- REMOVED: velocity_toward_couch, approach_couch, height_apex,
           on_couch_xy, joint_velocity_penalty, time_penalty (the
           always-on terms that biased v3/v4 toward the failure mode)
"""

from __future__ import annotations
import shutil
import subprocess
import sys
from pathlib import Path

V3 = Path("/workspace/_v3_baseline/jump_to")
ACTIVE = Path("/workspace/unitree_rl_mjlab/src/tasks/jump_to")

def main() -> int:
    if not V3.exists():
        print(f"ERROR: {V3} missing.", file=sys.stderr)
        return 1
    if ACTIVE.exists():
        shutil.rmtree(ACTIVE)
    shutil.copytree(V3, ACTIVE)
    print(f"[restore] {V3} -> {ACTIVE}")

    # 1) Keep scene_extras.py as-is (couch stays in scene).

    # 2) Append v6 reward functions to mdp/rewards.py.
    extra = '''

# === v6 additions: phase-gated rewards ================================

def _upright_indicator(env, asset_cfg=_DEFAULT_ASSET_CFG):
    """1.0 when nearly upright, 0.0 when toppled. Smooth transition near edge."""
    asset = env.scene[asset_cfg.name]
    gravity_z = asset.data.projected_gravity_b[:, 2]
    # gravity_z is -1 fully upright, +1 upside-down. Map -1 -> 1, -0.85 -> 0.
    return torch.clamp((-gravity_z - 0.85) / 0.15, 0.0, 1.0)

def _all_feet_contact_ratio(env, sensor_name):
    contact_sensor: ContactSensor = env.scene[sensor_name]
    in_contact = (contact_sensor.data.found > 0).float()
    return torch.sum(in_contact, dim=1) / 4.0

def crouch_load(env, sensor_name, asset_cfg=_DEFAULT_ASSET_CFG):
    """Reward crouching (z below standing rest) while all 4 feet are grounded.
    Encourages the leg-loading phase of a jump.
    """
    asset = env.scene[asset_cfg.name]
    z = asset.data.root_link_pos_w[:, 2]
    crouch = torch.clamp(0.27 - z, 0.0, 0.08)  # only reward up to 8cm crouch
    contact_ratio = _all_feet_contact_ratio(env, sensor_name)
    is_grounded = (contact_ratio >= 0.99).float()
    return crouch * is_grounded

def airborne_height(env, sensor_name, asset_cfg=_DEFAULT_ASSET_CFG):
    """Reward height ABOVE rest, but ONLY when at least one foot is off ground.
    Toppling sideways doesn't gain height, so this can only be earned by jumping.
    """
    asset = env.scene[asset_cfg.name]
    z = asset.data.root_link_pos_w[:, 2]
    height = torch.clamp(z - 0.27, min=0.0, max=0.6)
    contact_ratio = _all_feet_contact_ratio(env, sensor_name)
    airborne_factor = torch.clamp(1.0 - contact_ratio, 0.0, 1.0)
    return height * airborne_factor

def approach_couch_upright(env, asset_cfg=_DEFAULT_ASSET_CFG):
    """exp(-dist) * upright. Toppling toward couch earns zero credit."""
    asset = env.scene[asset_cfg.name]
    xy = asset.data.root_link_pos_w[:, :2]
    cx = torch.full_like(xy[:, 0], COUCH_XY[0])
    cy = torch.full_like(xy[:, 1], COUCH_XY[1])
    dist = torch.sqrt((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2 + 1e-6)
    upright = _upright_indicator(env, asset_cfg)
    return torch.exp(-dist) * upright

def on_couch_xy_upright(env, asset_cfg=_DEFAULT_ASSET_CFG):
    """Cat root xy is inside couch footprint AND cat is upright."""
    asset = env.scene[asset_cfg.name]
    xy = asset.data.root_link_pos_w[:, :2]
    dx = torch.abs(xy[:, 0] - COUCH_XY[0])
    dy = torch.abs(xy[:, 1] - COUCH_XY[1])
    in_xy = ((dx < COUCH_HALF_X) & (dy < COUCH_HALF_Y)).float()
    upright = _upright_indicator(env, asset_cfg)
    return in_xy * upright
'''
    rewards_path = ACTIVE / "mdp" / "rewards.py"
    rewards_path.write_text(rewards_path.read_text() + extra)
    print("[edit] mdp/rewards.py: appended v6 phase-gated reward functions")

    # 3) Rewrite jump_to_env_cfg.py reward block.
    cfg_path = ACTIVE / "jump_to_env_cfg.py"
    txt = cfg_path.read_text()

    txt = txt.replace(
        '"body_orientation_l2": RewardTermCfg(\n      func=mdp.body_orientation_l2,\n      weight=-1.0,',
        '"body_orientation_l2": RewardTermCfg(\n      func=mdp.body_orientation_l2,\n      weight=-2.0,',
    )
    txt = txt.replace(
        '"action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),',
        '"action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),',
    )

    import re
    old_block_re = re.compile(
        r'"approach_couch":.*?"time_penalty":\s*RewardTermCfg\([^)]*\),',
        re.DOTALL,
    )
    new_block = '''"crouch_load": RewardTermCfg(
            func=jump_to_mdp.crouch_load,
            weight=3.0,
            params={"sensor_name": "feet_ground_contact"},
        ),
        "airborne_height": RewardTermCfg(
            func=jump_to_mdp.airborne_height,
            weight=10.0,
            params={"sensor_name": "feet_ground_contact"},
        ),
        "approach_couch_upright": RewardTermCfg(
            func=jump_to_mdp.approach_couch_upright,
            weight=3.0,
        ),
        "on_couch_xy_upright": RewardTermCfg(
            func=jump_to_mdp.on_couch_xy_upright,
            weight=5.0,
        ),
        "landed_on_couch": RewardTermCfg(
            func=jump_to_mdp.landed_on_couch,
            weight=15.0,
            params={"sensor_name": "feet_ground_contact"},
        ),'''
    if not old_block_re.search(txt):
        print("ERROR: could not find jump-specific reward block to replace", file=sys.stderr)
        return 2
    txt = old_block_re.sub(new_block, txt)

    cfg_path.write_text(txt)
    print("[edit] jump_to_env_cfg.py: bumped upright/action_rate, swapped reward block to phase-gated")

    rc = subprocess.run(
        ["python3", "-c", "import ast; ast.parse(open('{}').read()); ast.parse(open('{}').read())".format(
            cfg_path, ACTIVE / 'mdp' / 'rewards.py')],
        capture_output=True, text=True,
    )
    if rc.returncode != 0:
        print(f"ERROR: syntax check failed:\n{rc.stderr}", file=sys.stderr)
        return 3
    print("[ok] v6 patch applied successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
