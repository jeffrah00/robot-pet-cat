from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

def base_height_recovery(env, target_height=0.27, asset_cfg=_DEFAULT_ASSET_CFG):
    """PRIMARY: reward for reaching standing height. Heavily weighted."""
    asset = env.scene[asset_cfg.name]
    height = asset.data.root_link_pos_w[:, 2]
    return torch.clamp(height / target_height, max=1.0)

def upright_orientation(env, asset_cfg=_DEFAULT_ASSET_CFG):
    """SECONDARY: reward for body returning to upright (gravity aligned with world up).
    projected_gravity_b[:,2] is -1 when fully upright, +1 when upside-down."""
    asset = env.scene[asset_cfg.name]
    gravity_z = asset.data.projected_gravity_b[:, 2]
    # Map: -1 (upright) -> 1.0 reward, +1 (upside-down) -> 0.0
    return (-gravity_z + 1.0) * 0.5

def all_feet_contact(env, sensor_name, asset_cfg=_DEFAULT_ASSET_CFG):
    """TERTIARY: reward when all 4 feet touch the ground."""
    contact_sensor: ContactSensor = env.scene[sensor_name]
    in_contact = (contact_sensor.data.found > 0).float()
    num_in_contact = torch.sum(in_contact, dim=1)
    return num_in_contact / 4.0

def joint_velocity_penalty(env, asset_cfg=_DEFAULT_ASSET_CFG):
    """PENALTY: penalize excessive joint velocity (prevent thrashing)."""
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel), dim=1)

def time_penalty(env):
    """PENALTY: constant per-step penalty to encourage recovering quickly."""
    return torch.ones(env.num_envs, device=env.device)

def base_height_target(env, target_height=0.30, asset_cfg=_DEFAULT_ASSET_CFG):
    """Penalize deviation from standing height once near target."""
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_link_pos_w[:, 2] - target_height)

def feet_in_contact(env, sensor_name, asset_cfg=_DEFAULT_ASSET_CFG):
    contact_sensor: ContactSensor = env.scene[sensor_name]
    in_contact = (contact_sensor.data.found > 0).float()
    return torch.sum(in_contact, dim=1)
