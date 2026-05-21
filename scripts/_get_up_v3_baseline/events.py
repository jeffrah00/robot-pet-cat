from __future__ import annotations
import math
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# Go2 joint order: FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
#                  RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf
# fmt: off
_POSE_JOINTS = [
    # lie_down: body flat on ground, legs splayed
    [0.0,  1.5, -2.6,  0.0,  1.5, -2.6,  0.0,  1.5, -2.6,  0.0,  1.5, -2.6],
    # sit: front up, haunches down
    [0.0,  0.5, -1.0,  0.0,  0.5, -1.0,  0.0,  2.2, -2.6,  0.0,  2.2, -2.6],
    # crouch: all legs bent deep
    [0.0,  1.8, -2.7,  0.0,  1.8, -2.7,  0.0,  1.8, -2.7,  0.0,  1.8, -2.7],
    # stretch: front legs extended forward
    [0.0, -0.6, -0.5,  0.0, -0.6, -0.5,  0.0,  1.5, -2.0,  0.0,  1.5, -2.0],
    # upside-down: same joints as crouch, but body flipped roll=pi
    [0.0,  0.9, -1.8,  0.0,  0.9, -1.8,  0.0,  0.9, -1.8,  0.0,  0.9, -1.8],
    # sideways: rolled 90 degrees onto side
    [0.0,  0.9, -1.8,  0.0,  0.9, -1.8,  0.0,  0.9, -1.8,  0.0,  0.9, -1.8],
]
# fmt: on

# (roll, pitch, yaw) for each pose in radians
_POSE_EULER = [
    (0.0,   0.0,  0.0),   # lie_down: flat
    (0.0,   0.3,  0.0),   # sit: slight pitch back
    (0.0,   0.0,  0.0),   # crouch: level
    (0.0,  -0.3,  0.0),   # stretch: slight pitch forward
    (math.pi, 0.0, 0.0),  # upside-down: rolled 180
    (math.pi / 2, 0.0, 0.0),  # sideways: rolled 90
]

# Ground height offset for each pose (z of base when starting)
_POSE_HEIGHT = [0.06, 0.18, 0.10, 0.12, 0.12, 0.18]

_NUM_POSES = len(_POSE_JOINTS)


def reset_to_random_fallen_pose(env, env_ids, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG):
    """Reset robot to one of 6 randomly selected fallen poses."""
    asset = env.scene[asset_cfg.name]
    num_resets = len(env_ids)
    device = env.device

    # Build pose tensors once per call
    pose_joints_t = torch.tensor(_POSE_JOINTS, dtype=torch.float32, device=device)  # [6, 12]
    pose_heights_t = torch.tensor(_POSE_HEIGHT, dtype=torch.float32, device=device)  # [6]

    # Pre-compute quaternions for each pose
    quats = []
    for roll, pitch, yaw in _POSE_EULER:
        r = torch.tensor([roll], device=device)
        p = torch.tensor([pitch], device=device)
        y = torch.tensor([yaw], device=device)
        q = quat_from_euler_xyz(r, p, y)  # [1, 4]
        quats.append(q)
    pose_quats_t = torch.cat(quats, dim=0)  # [6, 4]

    # Randomly select a pose index for each resetting env
    pose_idx = torch.randint(0, _NUM_POSES, (num_resets,), device=device)

    #  Root state 
    root_states = asset.data.default_root_state[env_ids].clone()
    # xy from env origins
    root_states[:, :3] += env.scene.env_origins[env_ids]
    # Override z height
    root_states[:, 2] = pose_heights_t[pose_idx]
    # Override orientation (quaternion columns 3:7)
    root_states[:, 3:7] = pose_quats_t[pose_idx]
    # Zero velocities
    root_states[:, 7:] = 0.0
    asset.write_root_state_to_sim(root_states, env_ids=env_ids)

    #  Joint state 
    joint_pos = pose_joints_t[pose_idx]                 # [num_resets, 12]
    joint_vel = torch.zeros_like(joint_pos)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
