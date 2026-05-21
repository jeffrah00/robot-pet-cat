#!/usr/bin/env python3
"""
jump_to v5 — "Jump-Up-First" (Atanassov-style curriculum stage 1).

Goal: teach the cat to push off all 4 legs and get airborne while staying
upright. No lateral target. The couch is REMOVED from the scene entirely
so the policy can't learn the v3/v4 failure mode of toppling toward it.

Design notes (post-lit-review 2026-05-22):
- Atanassov 2024 (arXiv:2401.16337) trains `go1_upwards` from scratch for
  3000 iters before introducing forward-jump variants. We replicate that
  stage-1 design here.
- v3/v4 had `velocity_toward_couch` at weight 20, which the policy
  maximizes most cheaply by falling sideways toward the couch.
- v5 removes the couch entirely, so the only positive-reward strategies
  involve pushing off the ground.

Reward shape (replaces v3 jump-specific terms):
- vertical_height: clamp(z - rest_z, 0, 0.5), weight 10
- airborne_bonus:  1.0 if all 4 feet off the ground, weight 5
- landed_after_air: 1.0 on the first step where >=3 feet return to
                    contact AFTER being fully airborne, weight 5
- body_orientation_l2 doubled (-2.0) so upright is strongly preferred
- action_rate_l2 doubled (-0.1) for smoothness

Run after `python3 scripts/patch_jump_to_v5.py`:
  set -a && source /workspace/robot-pet-cat/.env && set +a && \\
  cd /workspace/unitree_rl_mjlab && source /workspace/mjlab_venv/bin/activate && \\
  nohup python3 scripts/train.py Unitree-Go2-Jump-To \\
    --agent.max-iterations 5000 \\
    > /tmp/jump_to_v5_train.log 2>&1 &
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
        print(f"ERROR: {V3} missing. Restore v3 baseline first.", file=sys.stderr)
        return 1
    if ACTIVE.exists():
        shutil.rmtree(ACTIVE)
    shutil.copytree(V3, ACTIVE)
    print(f"[restore] {V3} -> {ACTIVE}")

    # 1) Replace scene_extras.py: empty couch (zero-size, far away) so the
    #    spec_fn callback is still satisfied but nothing intersects the cat.
    (ACTIVE / "mdp" / "scene_extras.py").write_text('''import mujoco
# v5: couch removed (tiny invisible geom far away to satisfy spec_fn signature)
COUCH_POS=(50.0,50.0,-10.0)
COUCH_HALF=(0.01,0.01,0.01)
COUCH_RGBA=(0.0,0.0,0.0,0.0)
def add_couch_spec_fn(spec):
    body=spec.worldbody.add_body(name="couch",pos=list(COUCH_POS))
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,size=list(COUCH_HALF),rgba=list(COUCH_RGBA))
''')
    print("[edit] mdp/scene_extras.py: couch removed (far away invisible geom)")

    # 2) Append v5 reward functions to mdp/rewards.py
    extra = '''

# === v5 additions ====================================================
# All v5 rewards are stateless module-level functions matching mjlab's
# RewardTermCfg signature: func(env, **params) -> Tensor[num_envs].

_v5_was_airborne = {}  # per-env memory for landed_after_air detection
_v5_first_landed = {}

CAT_REST_Z = 0.27  # standing height of Go2

def vertical_height(env, asset_cfg=_DEFAULT_ASSET_CFG):
    """Reward height above standing rest. Capped to discourage divergence."""
    asset = env.scene[asset_cfg.name]
    z = asset.data.root_link_pos_w[:, 2]
    return torch.clamp(z - CAT_REST_Z, min=0.0, max=0.5)

def airborne_bonus(env, sensor_name, asset_cfg=_DEFAULT_ASSET_CFG):
    """1.0 when all 4 feet are off the ground simultaneously."""
    contact_sensor: ContactSensor = env.scene[sensor_name]
    in_contact = (contact_sensor.data.found > 0).float()
    n_feet = torch.sum(in_contact, dim=1)
    return (n_feet < 0.5).float()  # 0 feet in contact

def landed_after_air(env, sensor_name, asset_cfg=_DEFAULT_ASSET_CFG):
    """1.0 on the first step where >=3 feet return to ground after being airborne.

    Encourages controlled landings rather than continued thrashing.
    """
    contact_sensor: ContactSensor = env.scene[sensor_name]
    in_contact = (contact_sensor.data.found > 0).float()
    n_feet = torch.sum(in_contact, dim=1)
    is_grounded = (n_feet >= 3.0)
    is_airborne_now = (n_feet < 0.5)
    key = id(env)
    was_air = _v5_was_airborne.get(key)
    if was_air is None or was_air.shape != is_airborne_now.shape:
        _v5_was_airborne[key] = is_airborne_now.detach()
        return torch.zeros_like(n_feet)
    just_landed = (was_air & is_grounded).float()
    _v5_was_airborne[key] = (was_air | is_airborne_now).detach() & (~is_grounded)
    return just_landed
'''
    rewards_path = ACTIVE / "mdp" / "rewards.py"
    rewards_path.write_text(rewards_path.read_text() + extra)
    print("[edit] mdp/rewards.py: appended v5 reward functions")

    # 3) Rewrite jump_to_env_cfg.py reward block.
    cfg_path = ACTIVE / "jump_to_env_cfg.py"
    txt = cfg_path.read_text()

    # Replace the entire jump-specific reward block (approach_couch through time_penalty)
    # with the v5 reward block. Also bump body_orientation_l2 and action_rate_l2.
    import re

    # Replace the body_orientation_l2 weight (-1.0) and action_rate_l2 weight (-0.05).
    txt = txt.replace(
        '"body_orientation_l2": RewardTermCfg(\n      func=mdp.body_orientation_l2,\n      weight=-1.0,',
        '"body_orientation_l2": RewardTermCfg(\n      func=mdp.body_orientation_l2,\n      weight=-2.0,',
    )
    txt = txt.replace(
        '"action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),',
        '"action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),',
    )

    # Replace the jump-specific block.
    old_block_re = re.compile(
        r'"approach_couch":.*?"time_penalty":\s*RewardTermCfg\([^)]*\),',
        re.DOTALL,
    )
    new_block = '''"vertical_height": RewardTermCfg(
            func=jump_to_mdp.vertical_height,
            weight=10.0,
        ),
        "airborne_bonus": RewardTermCfg(
            func=jump_to_mdp.airborne_bonus,
            weight=5.0,
            params={"sensor_name": "feet_ground_contact"},
        ),
        "landed_after_air": RewardTermCfg(
            func=jump_to_mdp.landed_after_air,
            weight=5.0,
            params={"sensor_name": "feet_ground_contact"},
        ),'''
    if not old_block_re.search(txt):
        print("ERROR: could not find jump-specific reward block to replace", file=sys.stderr)
        return 2
    txt = old_block_re.sub(new_block, txt)

    # Also drop couch_relative_pos from actor_terms (the obs becomes meaningless).
    txt = txt.replace(
        '''"couch_relative_pos": ObservationTermCfg(
      func=jump_to_mdp.couch_relative_pos,
    ),
    ''',
        '',
    )

    cfg_path.write_text(txt)
    print("[edit] jump_to_env_cfg.py: bumped upright/action_rate, swapped reward block, removed couch_relative_pos")

    # 4) Sanity compile.
    rc = subprocess.run(
        ["python3", "-c", "import ast; ast.parse(open('{}').read()); ast.parse(open('{}').read())".format(
            cfg_path, ACTIVE / 'mdp' / 'rewards.py')],
        capture_output=True, text=True,
    )
    if rc.returncode != 0:
        print(f"ERROR: syntax check failed:\n{rc.stderr}", file=sys.stderr)
        return 3
    print("[ok] v5 patch applied successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
