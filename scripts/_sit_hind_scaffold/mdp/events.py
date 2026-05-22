"""sit_hind events: reset to the get_up-13k sitting pose.

Probed from get_up v4b @ model_13000.pt via
/workspace/robot-pet-cat/scripts/probe_get_up_13k_pose.py on 2026-05-21.
The robot's front legs are extended, hind legs folded with calves curled
under, body pitched back ~20 deg, root_z ~0.27. This is the visually correct
hind-leg sit Jeff approved.
"""
from __future__ import annotations
import math
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# Joint order: FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
#              RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf
SIT_HIND_JOINT_POS = (
    -0.3118, 0.7580, -1.4716,   # FL
     0.1567, 0.7967, -1.4300,   # FR
     0.1516, 0.6712, -1.9245,   # RL (deeply folded calf)
    -0.1164, 0.3073, -1.5552,   # RR
)
SIT_HIND_ROOT_Z = 0.2704
# Slight backward pitch (~20 deg) to match the upright-front sit posture.
SIT_HIND_PITCH_RAD = -0.34


def reset_to_sit_hind_pose(env, env_ids, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG):
    """Set robot root xy at env origin, root z to SIT_HIND_ROOT_Z, body pitched
    back, and joints at the probed 13k sit pose. No jitter (for now).
    """
    asset = env.scene[asset_cfg.name]
    num_resets = len(env_ids)
    device = env.device

    pose_t = torch.tensor(SIT_HIND_JOINT_POS, dtype=torch.float32, device=device)
    joint_pos = pose_t.unsqueeze(0).expand(num_resets, -1).clone()
    joint_vel = torch.zeros_like(joint_pos)

    root_states = asset.data.default_root_state[env_ids].clone()
    root_states[:, :3] += env.scene.env_origins[env_ids]
    root_states[:, 2] = SIT_HIND_ROOT_Z
    r = torch.zeros(num_resets, device=device)
    p = torch.full((num_resets,), SIT_HIND_PITCH_RAD, device=device)
    y = torch.zeros(num_resets, device=device)
    root_states[:, 3:7] = quat_from_euler_xyz(r, p, y)
    root_states[:, 7:] = 0.0

    asset.write_root_state_to_sim(root_states, env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
