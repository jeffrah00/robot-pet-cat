#!/usr/bin/env python3
"""
get_up v4c — fix the "sitting cat" failure mode where v4b @13k recovers the
front half but leaves hind legs bent on the ground.

Diagnosis (per Jeff's review of get_up_13k render):
  - v3/v4b `base_height_recovery` clamps `min(height/0.27, 1.0)`, so as soon
    as the front torso reaches ~0.27m the reward saturates — no incentive to
    extend the hind legs further.
  - `upright_orientation` is gravity-only and is happy with a sitting pose
    where the torso tilts back ~30deg.
  - `all_feet_contact` actually REWARDS the sit (all 4 paws on ground).

Three combined changes (Jeff approved all three; v4c stacks them):
  1. Raise `base_height_recovery` target from 0.27 to 0.30 (full standing).
  2. NEW `hind_legs_extended` reward: exp(-L2 distance of hind thigh/calf
     joint positions from the standing pose target (0.9, -1.8)). Pushes
     RL/RR knees toward extension regardless of body height.
  3. NEW `stand_fully` bonus: large (+15) sparse reward fires only when
     z > 0.29 AND nearly level (projected_gravity_b[:,2] < -0.996, ~5deg).

Inherits v4b's continuous pose jitter (±0.3 rad joints, ±0.2 rad rpy on top
of the 6 discrete fallen poses) and v4b's strong action_rate_l2 penalty
(-0.5 vs the v3 baseline's -0.05).

Train: --agent.max-iterations 15000, fresh from random init (no resume).
Render at model_5000, model_10000, model_15000 (well, 14999) to track progress.

Go2 joint order (verified in events.py):
  0 FL_hip,  1 FL_thigh,  2 FL_calf,
  3 FR_hip,  4 FR_thigh,  5 FR_calf,
  6 RL_hip,  7 RL_thigh,  8 RL_calf,    <-- hind left  indices 7, 8
  9 RR_hip, 10 RR_thigh, 11 RR_calf.    <-- hind right indices 10, 11
"""

from __future__ import annotations
import shutil
import subprocess
import sys
from pathlib import Path

V3 = Path("/workspace/_v3_baseline/get_up")
ACTIVE = Path("/workspace/unitree_rl_mjlab/src/tasks/get_up")

def main() -> int:
    if not V3.exists():
        print(f"ERROR: {V3} missing.", file=sys.stderr)
        return 1
    if ACTIVE.exists():
        shutil.rmtree(ACTIVE)
    shutil.copytree(V3, ACTIVE)
    print(f"[restore] {V3} -> {ACTIVE}")

    # ---- 1. events.py: add v4b's continuous pose jitter ------------------
    events_path = ACTIVE / "mdp" / "events.py"
    events_txt = events_path.read_text()
    # Replace the joint_pos / orientation assignment block with a jittered version.
    old_reset_block = (
        "    #  Joint state \n"
        "    joint_pos = pose_joints_t[pose_idx]                 # [num_resets, 12]\n"
        "    joint_vel = torch.zeros_like(joint_pos)\n"
        "    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)"
    )
    new_reset_block = (
        "    #  v4b/v4c: continuous pose jitter on top of the 6 discrete fallen poses.\n"
        "    #  Adds U(-0.3, +0.3) rad noise per joint and U(-0.2, +0.2) rad noise per\n"
        "    #  euler axis on the body orientation. Same source as the v4b/v4c memo.\n"
        "    joint_pos = pose_joints_t[pose_idx]                 # [num_resets, 12]\n"
        "    joint_pos = joint_pos + (torch.rand_like(joint_pos) - 0.5) * 0.6  # +/-0.3 rad\n"
        "    joint_vel = torch.zeros_like(joint_pos)\n"
        "    # Re-jitter orientation by applying a small extra rpy noise via quat_mul.\n"
        "    rpy_noise = (torch.rand((num_resets, 3), device=device) - 0.5) * 0.4  # +/-0.2 rad\n"
        "    from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul\n"
        "    qn = quat_from_euler_xyz(rpy_noise[:, 0], rpy_noise[:, 1], rpy_noise[:, 2])\n"
        "    current_quat = root_states[:, 3:7]\n"
        "    jittered_quat = quat_mul(current_quat, qn)\n"
        "    root_states[:, 3:7] = jittered_quat\n"
        "    asset.write_root_state_to_sim(root_states, env_ids=env_ids)\n"
        "    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)"
    )
    if old_reset_block not in events_txt:
        print("ERROR: could not find old joint state block to replace", file=sys.stderr)
        print("First 400 chars of events.py:", file=sys.stderr)
        print(events_txt[:400], file=sys.stderr)
        return 2
    # Also need to remove the original write_root_state_to_sim call so we don't
    # write twice. The original write_root_state is right before the joint block.
    events_txt = events_txt.replace(
        "    asset.write_root_state_to_sim(root_states, env_ids=env_ids)\n\n"
        "    #  Joint state \n"
        "    joint_pos = pose_joints_t[pose_idx]                 # [num_resets, 12]\n"
        "    joint_vel = torch.zeros_like(joint_pos)\n"
        "    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)",
        new_reset_block,
    )
    if old_reset_block in events_txt:
        # fallback replacement (in case the combined replace didn't catch it).
        events_txt = events_txt.replace(old_reset_block, new_reset_block)
    events_path.write_text(events_txt)
    print("[edit] mdp/events.py: added v4b pose jitter (+/-0.3 rad joints, +/-0.2 rad rpy)")

    # ---- 2. rewards.py: add hind_legs_extended + stand_fully -------------
    rewards_path = ACTIVE / "mdp" / "rewards.py"
    extra = '''

# === v4c additions: hind-leg extension + full-stand bonus =============

# Standing pose for Go2: thigh 0.9 rad, calf -1.8 rad.
# Hind joint indices: RL_thigh=7, RL_calf=8, RR_thigh=10, RR_calf=11.
_HIND_TARGETS = (0.9, -1.8, 0.9, -1.8)  # RL_thigh, RL_calf, RR_thigh, RR_calf
_HIND_INDICES = (7, 8, 10, 11)

def hind_legs_extended(env, asset_cfg=_DEFAULT_ASSET_CFG):
    """exp(-L2 distance) reward for hind thigh/calf being near the standing pose.
    Pushes the rear legs to extend even when body height already saturates the
    base_height reward, fixing the v4b "sitting" failure mode.
    """
    asset = env.scene[asset_cfg.name]
    jp = asset.data.joint_pos
    err = torch.zeros(jp.shape[0], device=jp.device)
    for idx, target in zip(_HIND_INDICES, _HIND_TARGETS):
        err = err + (jp[:, idx] - target) ** 2
    return torch.exp(-err)  # 1.0 at target, decays away

def stand_fully_bonus(env, asset_cfg=_DEFAULT_ASSET_CFG):
    """Sparse high-weight bonus: 1.0 only when height > 0.29 AND body is nearly
    level (projected_gravity_b[:,2] < -0.996 corresponds to <~5 deg total tilt).
    """
    asset = env.scene[asset_cfg.name]
    z = asset.data.root_link_pos_w[:, 2]
    gz = asset.data.projected_gravity_b[:, 2]
    is_tall = (z > 0.29).float()
    is_level = (gz < -0.996).float()
    return is_tall * is_level
'''
    rewards_path.write_text(rewards_path.read_text() + extra)
    print("[edit] mdp/rewards.py: appended hind_legs_extended + stand_fully_bonus")

    # ---- 3. env_cfg: bump target_height, add new terms, v4b action_rate ---
    cfg_path = ACTIVE / "get_up_env_cfg.py"
    txt = cfg_path.read_text()

    # action_rate_l2 weight -0.05 -> -0.5 (v4b inheritance)
    txt = txt.replace(
        '"action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),',
        '"action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.5),',
    )

    # base_height_recovery target 0.27 -> 0.30 (v4c)
    txt = txt.replace(
        '"target_height": 0.27',
        '"target_height": 0.30',
    )

    # Insert v4c reward terms right before joint_velocity_penalty.
    insert_anchor = '"joint_velocity_penalty": RewardTermCfg('
    insertion = (
        '"hind_legs_extended": RewardTermCfg(\n'
        '            func=get_up_mdp.hind_legs_extended,\n'
        '            weight=3.0,\n'
        '        ),\n'
        '        "stand_fully_bonus": RewardTermCfg(\n'
        '            func=get_up_mdp.stand_fully_bonus,\n'
        '            weight=15.0,\n'
        '        ),\n'
        '        '
    )
    if insert_anchor not in txt:
        print(f"ERROR: anchor '{insert_anchor}' not found in env_cfg", file=sys.stderr)
        return 2
    txt = txt.replace(insert_anchor, insertion + insert_anchor, 1)

    cfg_path.write_text(txt)
    print("[edit] get_up_env_cfg.py: target_height 0.27->0.30, action_rate -0.05->-0.5, added hind_legs_extended + stand_fully_bonus")

    # ---- 4. Sanity compile ------------------------------------------------
    rc = subprocess.run(
        ["python3", "-c", "import ast; ast.parse(open('{}').read()); ast.parse(open('{}').read()); ast.parse(open('{}').read())".format(
            cfg_path, rewards_path, events_path)],
        capture_output=True, text=True,
    )
    if rc.returncode != 0:
        print(f"ERROR: syntax check failed:\n{rc.stderr}", file=sys.stderr)
        return 3
    print("[ok] v4c patch applied successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
