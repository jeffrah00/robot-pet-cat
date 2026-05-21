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


COUCH_XY=(1.0,0.0)
COUCH_TOP_Z=0.40
COUCH_HALF_X=0.40
COUCH_HALF_Y=0.60
CAT_STAND_HEIGHT=0.27

def approach_couch(env, asset_cfg=_DEFAULT_ASSET_CFG):
    asset = env.scene[asset_cfg.name]
    xy = asset.data.root_link_pos_w[:, :2]
    cx = torch.full_like(xy[:, 0], COUCH_XY[0])
    cy = torch.full_like(xy[:, 1], COUCH_XY[1])
    dist = torch.sqrt((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2 + 1e-6)
    return torch.exp(-dist)

def height_apex(env, asset_cfg=_DEFAULT_ASSET_CFG):
    asset = env.scene[asset_cfg.name]
    z = asset.data.root_link_pos_w[:, 2]
    return torch.clamp(z - COUCH_TOP_Z, min=0.0)

def on_couch_xy(env, asset_cfg=_DEFAULT_ASSET_CFG):
    asset = env.scene[asset_cfg.name]
    xy = asset.data.root_link_pos_w[:, :2]
    dx = torch.abs(xy[:, 0] - COUCH_XY[0])
    dy = torch.abs(xy[:, 1] - COUCH_XY[1])
    return ((dx < COUCH_HALF_X) & (dy < COUCH_HALF_Y)).float()

def landed_on_couch(env, sensor_name, asset_cfg=_DEFAULT_ASSET_CFG):
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    xy = asset.data.root_link_pos_w[:, :2]
    z = asset.data.root_link_pos_w[:, 2]
    dx = torch.abs(xy[:, 0] - COUCH_XY[0])
    dy = torch.abs(xy[:, 1] - COUCH_XY[1])
    in_xy = ((dx < COUCH_HALF_X) & (dy < COUCH_HALF_Y)).float()
    target_z = COUCH_TOP_Z + CAT_STAND_HEIGHT
    z_close = (torch.abs(z - target_z) < 0.10).float()
    in_contact = (contact_sensor.data.found > 0).float()
    n_feet = torch.sum(in_contact, dim=1)
    has_contact = (n_feet >= 3).float()
    return in_xy * z_close * has_contact


# Track previous distance per env to compute delta (closing speed).
_prev_dist = {}

def velocity_toward_couch(env, asset_cfg=_DEFAULT_ASSET_CFG):
    """Reward positive when the cat's xy moves closer to couch this step."""
    asset = env.scene[asset_cfg.name]
    xy = asset.data.root_link_pos_w[:, :2]
    cx = torch.full_like(xy[:, 0], COUCH_XY[0])
    cy = torch.full_like(xy[:, 1], COUCH_XY[1])
    dist = torch.sqrt((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2 + 1e-6)
    key = id(env)
    prev = _prev_dist.get(key)
    _prev_dist[key] = dist.detach()
    if prev is None or prev.shape != dist.shape:
        return torch.zeros_like(dist)
    # closing = prev - cur. Positive when cat got closer.
    closing = (prev - dist).clamp(min=-0.1, max=0.1)
    return closing
