#!/usr/bin/env python3
"""
get_up v6 -- dynamic reward shaping curriculum on top of v4c.

Method reproduced (not copied): the two-phase reward-shaping curriculum from
Learning to Recover (2025), arxiv 2506.05516. The paper's claim is that the
fall-recovery task has a sparse-reward / unstable-convergence problem, and
that a curriculum that starts loose and tightens fixes it. Two phases:

  Phase A (exploration):  heavy positives, light penalties. Goal is to make
                          the policy *discover* the upright posture without
                          being squashed by smoothness or fall penalties.
  Phase B (strict):       full penalty weights. Goal is to refine the motion
                          quality once the upright attractor has been found.

We train them as two sequential runs:
  1. patch_get_up_v6.py --phase A   then  train 5000 iters from scratch.
  2. patch_get_up_v6.py --phase B   then  resume from phase A's model_4999.pt
                                   for another 5000 iters.

Reused unchanged from v4c (see scripts/patch_get_up_v4c.py for derivation):
  - continuous pose jitter (+/- 0.3 rad joints, +/- 0.2 rad rpy on top of the
    6 discrete fallen anchor poses).
  - hind_legs_extended reward (Go2 hind thigh/calf indices 7, 8, 10, 11).
  - stand_fully_bonus sparse reward (z > 0.29 AND projected_gravity_b[2] <
    -0.996, i.e. nearly level).
  - base_height_recovery target raised 0.27 -> 0.30.

v6 changes vs v4c (per-phase weight tables below):
  - Phase A weakens penalties (action_rate_l2, joint_acc_l2, joint_pos_limits,
    joint_velocity_penalty, is_terminated, body_orientation_l2, body_ang_vel,
    angular_momentum, time_penalty) and roughly doubles positives.
  - Phase B restores v4c's strict penalty weights.

Usage on a runpod:
  python3 scripts/patch_get_up_v6.py --phase A
  cd /workspace/unitree_rl_mjlab && source /workspace/mjlab_venv/bin/activate
  set -a && source /workspace/robot-pet-cat/.env && set +a
  nohup python3 scripts/train.py Unitree-Go2-GetUp --agent.max-iterations 5000 \
      > /tmp/getup_v6_A_train.log 2>&1 &
  # ... wait for phase A to finish, note its run dir, then:
  python3 scripts/patch_get_up_v6.py --phase B
  nohup python3 scripts/train.py Unitree-Go2-GetUp --agent.max-iterations 5000 \
      --agent.resume True --agent.load-run <phaseA_run_dir> \
      --agent.load-checkpoint model_4999.pt \
      > /tmp/getup_v6_B_train.log 2>&1 &
"""

from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

V3 = Path("/workspace/_v3_baseline/get_up")
ACTIVE = Path("/workspace/unitree_rl_mjlab/src/tasks/get_up")

# --- Reward weight tables ---------------------------------------------------
# Each table maps the env_cfg reward term name to the (new) weight for the
# given phase. Anything not listed is left at the v3-baseline default.

PHASE_A_WEIGHTS = {
    # positives -- boosted to encourage exploration toward standing
    "base_height_recovery": 16.0,   # v4c: 8.0
    "upright_orientation": 10.0,    # v4c: 5.0
    "hind_legs_extended": 6.0,      # v4c: 3.0
    "stand_fully_bonus": 25.0,      # v4c: 15.0
    "all_feet_contact": 2.0,        # v4c: 1.0
    # penalties -- softened so exploration is not punished
    "is_terminated": -2.0,          # v4c: -10.0
    "action_rate_l2": -0.05,        # v4c: -0.5
    "joint_acc_l2": -2.5e-8,        # v4c: -2.5e-7
    "joint_pos_limits": -2.0,       # v4c: -10.0
    "joint_velocity_penalty": -0.0001,  # v4c: -0.001
    "body_orientation_l2": -0.2,    # v4c: -1.0
    "body_ang_vel": -0.005,         # v4c: -0.05
    "angular_momentum": -0.0025,    # v4c: -0.025
    "time_penalty": -0.01,          # v4c: -0.05
}

# Phase B is exactly v4c's strict weights.
PHASE_B_WEIGHTS = {
    "base_height_recovery": 8.0,
    "upright_orientation": 5.0,
    "hind_legs_extended": 3.0,
    "stand_fully_bonus": 15.0,
    "all_feet_contact": 1.0,
    "is_terminated": -10.0,
    "action_rate_l2": -0.5,
    "joint_acc_l2": -2.5e-7,
    "joint_pos_limits": -10.0,
    "joint_velocity_penalty": -0.001,
    "body_orientation_l2": -1.0,
    "body_ang_vel": -0.05,
    "angular_momentum": -0.025,
    "time_penalty": -0.05,
}


def apply_pose_jitter(events_path: Path) -> None:
    """v4b/v4c pose jitter -- continuous noise on top of the discrete poses."""
    txt = events_path.read_text()
    old = (
        "    #  Joint state \n"
        "    joint_pos = pose_joints_t[pose_idx]                 # [num_resets, 12]\n"
        "    joint_vel = torch.zeros_like(joint_pos)\n"
        "    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)"
    )
    new = (
        "    #  v6 (inherited from v4b/v4c): continuous pose jitter on top of\n"
        "    #  the 6 discrete fallen poses. +/- 0.3 rad per joint, +/- 0.2 rad\n"
        "    #  per euler axis on the body quaternion.\n"
        "    joint_pos = pose_joints_t[pose_idx]                 # [num_resets, 12]\n"
        "    joint_pos = joint_pos + (torch.rand_like(joint_pos) - 0.5) * 0.6\n"
        "    joint_vel = torch.zeros_like(joint_pos)\n"
        "    rpy_noise = (torch.rand((num_resets, 3), device=device) - 0.5) * 0.4\n"
        "    from mjlab.utils.lab_api.math import quat_mul as _quat_mul_v6\n"
        "    qn = quat_from_euler_xyz(rpy_noise[:, 0], rpy_noise[:, 1], rpy_noise[:, 2])\n"
        "    current_quat = root_states[:, 3:7]\n"
        "    jittered_quat = _quat_mul_v6(current_quat, qn)\n"
        "    root_states[:, 3:7] = jittered_quat\n"
        "    asset.write_root_state_to_sim(root_states, env_ids=env_ids)\n"
        "    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)"
    )
    combined_old = (
        "    asset.write_root_state_to_sim(root_states, env_ids=env_ids)\n\n"
        + old
    )
    if combined_old in txt:
        txt = txt.replace(combined_old, new)
    elif old in txt:
        txt = txt.replace(old, new)
    else:
        raise RuntimeError("pose-jitter anchor not found in events.py")
    events_path.write_text(txt)


def append_v4c_rewards(rewards_path: Path) -> None:
    """Append v4c's hind_legs_extended and stand_fully_bonus reward funcs."""
    extra = '''

# === v6 (inherited from v4c) =========================================
# Standing pose for Go2: thigh 0.9 rad, calf -1.8 rad.
# Hind joint indices: RL_thigh=7, RL_calf=8, RR_thigh=10, RR_calf=11.
_HIND_TARGETS_V6 = (0.9, -1.8, 0.9, -1.8)
_HIND_INDICES_V6 = (7, 8, 10, 11)

def hind_legs_extended(env, asset_cfg=_DEFAULT_ASSET_CFG):
    """exp(-L2 distance) reward for hind thigh/calf near the standing pose.

    Counteracts the v4b "sitting" failure mode where base height saturates
    but hind legs stay bent on the ground.
    """
    asset = env.scene[asset_cfg.name]
    jp = asset.data.joint_pos
    err = torch.zeros(jp.shape[0], device=jp.device)
    for idx, target in zip(_HIND_INDICES_V6, _HIND_TARGETS_V6):
        err = err + (jp[:, idx] - target) ** 2
    return torch.exp(-err)


def stand_fully_bonus(env, asset_cfg=_DEFAULT_ASSET_CFG):
    """Sparse 0/1 bonus for being tall (z > 0.29) AND level (~< 5 deg tilt)."""
    asset = env.scene[asset_cfg.name]
    z = asset.data.root_link_pos_w[:, 2]
    gz = asset.data.projected_gravity_b[:, 2]
    is_tall = (z > 0.29).float()
    is_level = (gz < -0.996).float()
    return is_tall * is_level
'''
    rewards_path.write_text(rewards_path.read_text() + extra)


def edit_env_cfg(cfg_path: Path, weights: dict) -> None:
    """Apply v4c structural changes (height 0.30, new reward terms) and the
    phase-specific weight table to env_cfg."""
    txt = cfg_path.read_text()

    # base_height_recovery target 0.27 -> 0.30 (v4c)
    txt = txt.replace('"target_height": 0.27', '"target_height": 0.30')

    # Insert v4c reward term entries before joint_velocity_penalty.
    insert_anchor = '"joint_velocity_penalty": RewardTermCfg('
    insertion = (
        '"hind_legs_extended": RewardTermCfg(\n'
        '            func=get_up_mdp.hind_legs_extended,\n'
        '            weight=0.0,\n'  # placeholder, overwritten below
        '        ),\n'
        '        "stand_fully_bonus": RewardTermCfg(\n'
        '            func=get_up_mdp.stand_fully_bonus,\n'
        '            weight=0.0,\n'  # placeholder, overwritten below
        '        ),\n'
        '        '
    )
    if insert_anchor not in txt:
        raise RuntimeError(f"anchor '{insert_anchor}' not found in env_cfg")
    txt = txt.replace(insert_anchor, insertion + insert_anchor, 1)

    # Now apply per-term weight overrides.
    # The reward terms in env_cfg.py are all on lines of the pattern:
    #    "<name>": RewardTermCfg(func=..., weight=<val>, ...)
    # or with multi-line cfg:
    #    "<name>": RewardTermCfg(
    #        func=...,
    #        weight=<val>,
    #        ...
    # so we replace the *literal* weight values one term at a time.
    # We do the substitution on a per-term basis by locating the term name
    # then the next "weight=" after it.
    for name, new_w in weights.items():
        marker = f'"{name}": RewardTermCfg('
        idx = txt.find(marker)
        if idx < 0:
            # This term may not exist (we just inserted hind_legs_extended /
            # stand_fully_bonus above with placeholders, so they should be
            # found). Skip silently if truly absent.
            print(f"  [warn] term {name} not found in env_cfg; skipping",
                  file=sys.stderr)
            continue
        w_idx = txt.find("weight=", idx)
        if w_idx < 0:
            raise RuntimeError(f"no weight= after term {name}")
        # Find end of weight literal (next comma or close-paren).
        end = w_idx + len("weight=")
        while end < len(txt) and txt[end] not in (",", ")", "\n"):
            end += 1
        txt = txt[:w_idx] + f"weight={new_w!r}" + txt[end:]

    cfg_path.write_text(txt)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["A", "B"], required=True,
                        help="Phase A = exploration, Phase B = strict")
    args = parser.parse_args()

    if not V3.exists():
        print(f"ERROR: {V3} missing -- run from a pod with the baseline.",
              file=sys.stderr)
        return 1

    if ACTIVE.exists():
        shutil.rmtree(ACTIVE)
    shutil.copytree(V3, ACTIVE)
    print(f"[restore] {V3} -> {ACTIVE}")

    apply_pose_jitter(ACTIVE / "mdp" / "events.py")
    print("[edit] mdp/events.py: applied v4b/v4c pose jitter")

    append_v4c_rewards(ACTIVE / "mdp" / "rewards.py")
    print("[edit] mdp/rewards.py: appended hind_legs_extended + stand_fully_bonus")

    weights = PHASE_A_WEIGHTS if args.phase == "A" else PHASE_B_WEIGHTS
    edit_env_cfg(ACTIVE / "get_up_env_cfg.py", weights)
    print(f"[edit] get_up_env_cfg.py: phase {args.phase} weight table applied")

    # Syntax-check the three edited files.
    rc = subprocess.run(
        ["python3", "-c",
         "import ast;"
         f"ast.parse(open('{ACTIVE / 'get_up_env_cfg.py'}').read());"
         f"ast.parse(open('{ACTIVE / 'mdp' / 'rewards.py'}').read());"
         f"ast.parse(open('{ACTIVE / 'mdp' / 'events.py'}').read())"],
        capture_output=True, text=True,
    )
    if rc.returncode != 0:
        print(f"ERROR: syntax check failed:\n{rc.stderr}", file=sys.stderr)
        return 3

    print(f"[ok] v6 phase {args.phase} patch applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
