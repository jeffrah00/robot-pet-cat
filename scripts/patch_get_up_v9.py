#!/usr/bin/env python3
"""
get_up v9 -- v7 + "keep trying until upright perfectly" mechanics.

Builds on v7 (FR-Net + v4c structure: pose jitter, hind_legs_extended +3,
stand_fully_bonus +15, action_rate_l2 -0.5, target_height 0.30). Layers four
changes on top:

  1. Remove `fell_over` termination. Episode no longer ends on >70 deg tilt;
     it runs the full 20 s regardless.
  2. Add `soft_reset_fallen` interval event (fires every 1.0-2.0 s). For each
     env currently tilted >70 deg at fire-time, re-roll its joint + root pose
     from the same fallen-pose distribution used at episode start. Episode
     counter and reward accumulator continue. This gives the cat ~1.5 s per
     recovery attempt and a fresh pose draw if it doesn't succeed.
  3. Add `passivity_penalty` reward (weight -2.0). Returns 1.0 per step when
     tilted >70 deg, 0.0 otherwise. Sustained lying-still becomes strictly
     more costly than failed recovery attempts.
  4. Drop `is_terminated` reward (was -10). With fell_over removed, the only
     terminator left is `time_out`, and penalising that doesn't make sense.

Rationale: v8_retry removed fell_over but added no anti-passivity signal, so
the optimal policy converged to "lie still" (cheap reward, no termination
penalty). v9 keeps the "no termination" property -- the cat keeps trying --
but makes lying still strictly costly, and gives the policy fresh pose draws
mid-episode so it learns to recover from variety rather than from exactly
the splayed-out pose it ended up in.

Usage on a runpod:
  python3 scripts/patch_get_up_v9.py
  cd /workspace/unitree_rl_mjlab && source /workspace/mjlab_venv/bin/activate
  set -a && source /workspace/robot-pet-cat/.env && set +a
  nohup python3 scripts/train.py Unitree-Go2-GetUp --agent.max-iterations 10000 \
      > /tmp/getup_v9_train.log 2>&1 &
"""

from __future__ import annotations
import shutil
import subprocess
import sys
from pathlib import Path

V3 = Path("/workspace/_v3_baseline/get_up")
ACTIVE = Path("/workspace/unitree_rl_mjlab/src/tasks/get_up")
HERE = Path(__file__).resolve().parent


V9_EVENT_BLOCK = '''

# === v9 (keep-trying soft-reset) ====================================
import math as _math_v9


def soft_reset_fallen(env, env_ids=None,
                       limit_angle_deg: float = 70.0,
                       asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG):
    """v9: re-roll pose for envs currently tilted beyond limit_angle.

    Called as a periodic interval event (every 1.0-2.0 s). The event manager
    passes env_ids for the envs the event applies to this fire; we filter
    those down to only the ones still fallen and call the existing
    fallen-pose reset on that subset.

    Note: episode counters and reward accumulators are NOT reset, so this is
    a *soft* reset that lets the cat keep trying within one episode.
    """
    asset = env.scene[asset_cfg.name]
    pg_z = asset.data.projected_gravity_b[:, 2]
    # projected_gravity_b z is -1.0 when perfectly upright, ~0 on side, +1 upside-down.
    cos_lim = _math_v9.cos(_math_v9.radians(limit_angle_deg))
    fallen_mask = pg_z > -cos_lim

    if env_ids is None:
        candidate = torch.arange(env.num_envs, device=fallen_mask.device)
    else:
        candidate = env_ids
    sel = fallen_mask[candidate].nonzero(as_tuple=True)[0]
    if sel.numel() == 0:
        return
    fallen_ids = candidate[sel]
    reset_to_random_fallen_pose(env, env_ids=fallen_ids, asset_cfg=asset_cfg)
'''


V9_REWARD_BLOCK = '''

# === v9 (passivity penalty) =========================================

def passivity_penalty(env, asset_cfg=_DEFAULT_ASSET_CFG,
                       limit_angle_deg: float = 70.0):
    """v9: returns 1.0 per env-step when tilted beyond limit_angle, else 0.0.

    Combined with a negative reward weight, this makes lying still strictly
    costly. The weight is -2.0 by default in the env config.
    """
    import math as _m
    asset = env.scene[asset_cfg.name]
    pg_z = asset.data.projected_gravity_b[:, 2]
    cos_lim = _m.cos(_m.radians(limit_angle_deg))
    return (pg_z > -cos_lim).float()
'''


def remove_fell_over_termination(cfg_path: Path) -> None:
    """Strip the fell_over termination term entirely."""
    txt = cfg_path.read_text()
    old = (
        '    "fell_over": TerminationTermCfg(\n'
        '      func=mdp.bad_orientation,\n'
        '      params={"limit_angle": math.radians(70.0)},\n'
        '    ),\n'
    )
    if old not in txt:
        raise RuntimeError("fell_over termination block not found")
    cfg_path.write_text(txt.replace(old, "", 1))


def remove_is_terminated_reward(cfg_path: Path) -> None:
    """Strip the is_terminated reward (was -10)."""
    txt = cfg_path.read_text()
    old = '    "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-10.0),\n'
    if old not in txt:
        raise RuntimeError("is_terminated reward block not found")
    cfg_path.write_text(txt.replace(old, "", 1))


def register_v9_event(cfg_path: Path) -> None:
    """Insert soft_reset_fallen into the events dict (before foot_friction)."""
    txt = cfg_path.read_text()
    anchor = '    "foot_friction": EventTermCfg('
    insertion = (
        '    "soft_reset_fallen": EventTermCfg(\n'
        '      func=get_up_mdp.soft_reset_fallen,\n'
        '      mode="interval",\n'
        '      interval_range_s=(1.0, 2.0),\n'
        '      params={"limit_angle_deg": 70.0},\n'
        '    ),\n'
    )
    if anchor not in txt:
        raise RuntimeError("foot_friction event anchor not found")
    cfg_path.write_text(txt.replace(anchor, insertion + anchor, 1))


def register_v9_reward(cfg_path: Path) -> None:
    """Insert passivity_penalty into the rewards dict (before hind_legs_extended,
    which v7 inserts before joint_velocity_penalty)."""
    txt = cfg_path.read_text()
    anchor = '"hind_legs_extended": RewardTermCfg('
    insertion = (
        '"passivity_penalty": RewardTermCfg(\n'
        '            func=get_up_mdp.passivity_penalty,\n'
        '            weight=-2.0,\n'
        '        ),\n'
        '        '
    )
    if anchor not in txt:
        raise RuntimeError("hind_legs_extended anchor not found (was v7 applied?)")
    cfg_path.write_text(txt.replace(anchor, insertion + anchor, 1))


def append_v9_events(events_path: Path) -> None:
    events_path.write_text(events_path.read_text() + V9_EVENT_BLOCK)


def append_v9_rewards(rewards_path: Path) -> None:
    rewards_path.write_text(rewards_path.read_text() + V9_REWARD_BLOCK)


def main() -> int:
    # 1. Apply v7 patch first (restores from v3_baseline and layers v7 + v4c).
    rc = subprocess.run(
        ["python3", str(HERE / "patch_get_up_v7.py")],
        capture_output=False,
    )
    if rc.returncode != 0:
        print(f"ERROR: v7 patch failed (rc={rc.returncode})", file=sys.stderr)
        return rc.returncode

    if not ACTIVE.exists():
        print(f"ERROR: {ACTIVE} missing after v7 patch", file=sys.stderr)
        return 1

    cfg_path = ACTIVE / "get_up_env_cfg.py"
    events_path = ACTIVE / "mdp" / "events.py"
    rewards_path = ACTIVE / "mdp" / "rewards.py"

    # 2. Layer v9 changes onto v7.
    remove_fell_over_termination(cfg_path)
    print("[edit] removed fell_over termination")
    remove_is_terminated_reward(cfg_path)
    print("[edit] removed is_terminated reward (-10)")

    append_v9_events(events_path)
    print("[edit] appended soft_reset_fallen event function")
    append_v9_rewards(rewards_path)
    print("[edit] appended passivity_penalty reward function")

    register_v9_event(cfg_path)
    print("[edit] registered soft_reset_fallen event in cfg")
    register_v9_reward(cfg_path)
    print("[edit] registered passivity_penalty reward in cfg")

    # 3. Syntax check.
    rc = subprocess.run(
        ["python3", "-c",
         "import ast;"
         f"ast.parse(open('{cfg_path}').read());"
         f"ast.parse(open('{events_path}').read());"
         f"ast.parse(open('{rewards_path}').read())"],
        capture_output=True, text=True,
    )
    if rc.returncode != 0:
        print(f"ERROR: syntax check failed:\n{rc.stderr}", file=sys.stderr)
        return 3

    print("[ok] v9 patch applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
