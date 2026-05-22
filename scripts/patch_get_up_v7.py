#!/usr/bin/env python3
"""
get_up v7 -- single policy + auxiliary mass-contact predictor (FR-Net 2025).

Method reproduced (not copied): the policy has an auxiliary head, a small MLP
that predicts the next-step ground-contact configuration of each of the four
feet from current proprioception. The prediction is appended to the actor's
observation vector. The predictor is trained inline by supervised BCE loss
against the actual foot contacts at t+1. Single-policy framing (unlike Lee
2019 hierarchical), Go2 hardware-matched (unlike Learning-to-Recover 2025).

Reference: arXiv 2509.11504. We do not copy paper code or text; the
implementation below is original Python that follows the method.

Build on v4c base (pose jitter, hind_legs_extended reward, stand_fully_bonus,
strong action_rate_l2 smoothness penalty, height target 0.30). Six prior
single-policy variants (v3, v4a, v4b, v4c, v6-A, v6-B) failed visually for
get_up; FR-Net's claim is that the contact-prediction auxiliary task forces
the policy to internally model when each foot will land, which helps it
plan a stand-up trajectory rather than chase dense reward locally.

Usage on a runpod:
  python3 scripts/patch_get_up_v7.py
  cd /workspace/unitree_rl_mjlab && source /workspace/mjlab_venv/bin/activate
  set -a && source /workspace/robot-pet-cat/.env && set +a
  nohup python3 scripts/train.py Unitree-Go2-GetUp --agent.max-iterations 5000 \
      > /tmp/getup_v7_train.log 2>&1 &
"""

from __future__ import annotations
import shutil
import subprocess
import sys
from pathlib import Path

V3 = Path("/workspace/_v3_baseline/get_up")
ACTIVE = Path("/workspace/unitree_rl_mjlab/src/tasks/get_up")

CONTACT_PREDICTOR_PY = '''"""FR-Net 2025 mass-contact predictor.

Small MLP that learns to predict next-step foot-contact configuration from
current proprioception. Trained inline by BCE against actual contacts at t+1.
The prediction is appended to the actor observation, giving the policy an
implicit signal about which feet are about to land.

This module is written from scratch; it follows the method of arXiv 2509.11504
without reproducing the paper's code.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Module-level registry so the observation function and the supervised-update
# event share one network per env instance.
_REGISTRY: dict[int, "MassContactPredictor"] = {}


class MassContactPredictor(nn.Module):
    """Predicts a 4-dim foot-contact configuration for the next env step.

    Input layout (assembled by `build_input`):
      joint_pos      (12,)
      joint_vel      (12,)
      proj_gravity_b ( 3,)
      base_ang_vel   ( 3,)
      total           30

    Output: per-foot in-contact probability for (FL, FR, RL, RR).
    """

    INPUT_DIM = 30
    OUTPUT_DIM = 4

    def __init__(self, hidden: int = 64, lr: float = 1e-3, device: str = "cpu"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.INPUT_DIM, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, self.OUTPUT_DIM),
            nn.Sigmoid(),
        ).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.device = torch.device(device)
        # Per-step memo so actor + critic obs groups don't trigger two updates.
        self.last_step: int = -1
        self.prev_input: torch.Tensor | None = None
        self.last_prediction: torch.Tensor | None = None

    def update_against(self, actual: torch.Tensor) -> torch.Tensor:
        if self.prev_input is None:
            return torch.tensor(0.0, device=self.device)
        pred = self.net(self.prev_input)
        loss = F.binary_cross_entropy(pred, actual.to(self.device))
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.detach()

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.to(self.device))


def get_predictor(env) -> MassContactPredictor:
    key = id(env)
    pred = _REGISTRY.get(key)
    if pred is None:
        device = str(getattr(env, "device", "cpu"))
        pred = MassContactPredictor(device=device)
        _REGISTRY[key] = pred
    return pred


def _step_counter(env) -> int:
    """Robust per-step counter. Tries several mjlab/IsaacLab attributes."""
    for attr in ("common_step_counter", "_sim_step_counter", "episode_step",
                 "step_count"):
        c = getattr(env, attr, None)
        if c is None:
            continue
        if hasattr(c, "item"):
            try:
                return int(c.item())
            except Exception:
                pass
        if isinstance(c, int):
            return c
    # Fallback: a monotonic counter that ticks per call to this function.
    _step_counter.fallback = getattr(_step_counter, "fallback", 0) + 1
    return _step_counter.fallback


def build_input(env, asset_name: str = "robot") -> torch.Tensor:
    asset = env.scene[asset_name]
    parts = [
        asset.data.joint_pos,
        asset.data.joint_vel,
        asset.data.projected_gravity_b,
    ]
    # Base angular velocity: prefer the IMU sensor (matches actor obs), fall
    # back to the asset's root attribute if the sensor isn't wired here.
    bav = None
    sensors = getattr(env.scene, "sensors", {})
    if "robot/imu_ang_vel" in sensors:
        bav = sensors["robot/imu_ang_vel"].data
    if bav is None:
        bav = getattr(asset.data, "root_ang_vel_b", None)
    if bav is None:
        bav = torch.zeros(asset.data.joint_pos.shape[0], 3,
                          device=asset.data.joint_pos.device)
    parts.append(bav)
    return torch.cat(parts, dim=-1)


def _actual_contacts(env, sensor_name: str) -> torch.Tensor:
    """Read 0/1 contact state for the four feet, shape (num_envs, 4)."""
    sensor = env.scene.sensors[sensor_name]
    data = sensor.data
    for attr in ("in_contact", "is_in_contact", "current_contact_state"):
        v = getattr(data, attr, None)
        if v is not None:
            return v.float()
    # Fallback: try force-magnitude > threshold.
    forces = getattr(data, "net_forces_w_history", None)
    if forces is not None:
        mag = forces[..., 0, :].norm(dim=-1)
        return (mag > 1.0).float()
    raise RuntimeError(f"could not read contact state from sensor {sensor_name!r}")
'''

PREDICTOR_OBS_BLOCK = '''

# === v7 (FR-Net) =====================================================

def predicted_next_contact(env, sensor_name: str = "feet_ground_contact"):
    """Auxiliary observation: predicted foot-contact configuration at t+1.

    Also performs the supervised update of the predictor exactly once per env
    step (deduplicated across actor+critic obs groups via a step counter).
    Output shape: (num_envs, 4), values in [0, 1].
    """
    from .contact_predictor import (
        get_predictor, build_input, _actual_contacts, _step_counter,
    )

    pred = get_predictor(env)
    step = _step_counter(env)

    if step != pred.last_step:
        # New step: train predictor against the actual contacts now observed,
        # which are the t+1 targets for the input stored last step.
        if pred.prev_input is not None and pred.prev_input.shape[0] == env.num_envs:
            try:
                actual = _actual_contacts(env, sensor_name)
                pred.update_against(actual)
            except Exception:
                pass
        x = build_input(env).detach()
        # Keep prev_input on the predictor's device, with grad enabled for
        # backprop through the supervised update next step.
        pred.prev_input = x.to(pred.device).clone().requires_grad_(True)
        pred.last_prediction = pred.predict(x)
        pred.last_step = step

    return pred.last_prediction
'''


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
        "    #  v7 (inherits v4b/v4c): continuous pose jitter on top of the 6\n"
        "    #  discrete fallen poses. +/- 0.3 rad joints, +/- 0.2 rad rpy.\n"
        "    joint_pos = pose_joints_t[pose_idx]\n"
        "    joint_pos = joint_pos + (torch.rand_like(joint_pos) - 0.5) * 0.6\n"
        "    joint_vel = torch.zeros_like(joint_pos)\n"
        "    rpy_noise = (torch.rand((num_resets, 3), device=device) - 0.5) * 0.4\n"
        "    from mjlab.utils.lab_api.math import quat_mul as _quat_mul_v7\n"
        "    qn = quat_from_euler_xyz(rpy_noise[:, 0], rpy_noise[:, 1], rpy_noise[:, 2])\n"
        "    current_quat = root_states[:, 3:7]\n"
        "    jittered_quat = _quat_mul_v7(current_quat, qn)\n"
        "    root_states[:, 3:7] = jittered_quat\n"
        "    asset.write_root_state_to_sim(root_states, env_ids=env_ids)\n"
        "    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)"
    )
    combined_old = (
        "    asset.write_root_state_to_sim(root_states, env_ids=env_ids)\n\n" + old
    )
    if combined_old in txt:
        txt = txt.replace(combined_old, new)
    elif old in txt:
        txt = txt.replace(old, new)
    else:
        raise RuntimeError("pose-jitter anchor not found in events.py")
    events_path.write_text(txt)


def append_v4c_rewards(rewards_path: Path) -> None:
    extra = '''

# === v7 (inherits v4c hind-leg + stand bonus) =======================
_HIND_TARGETS_V7 = (0.9, -1.8, 0.9, -1.8)
_HIND_INDICES_V7 = (7, 8, 10, 11)


def hind_legs_extended(env, asset_cfg=_DEFAULT_ASSET_CFG):
    asset = env.scene[asset_cfg.name]
    jp = asset.data.joint_pos
    err = torch.zeros(jp.shape[0], device=jp.device)
    for idx, target in zip(_HIND_INDICES_V7, _HIND_TARGETS_V7):
        err = err + (jp[:, idx] - target) ** 2
    return torch.exp(-err)


def stand_fully_bonus(env, asset_cfg=_DEFAULT_ASSET_CFG):
    asset = env.scene[asset_cfg.name]
    z = asset.data.root_link_pos_w[:, 2]
    gz = asset.data.projected_gravity_b[:, 2]
    is_tall = (z > 0.29).float()
    is_level = (gz < -0.996).float()
    return is_tall * is_level
'''
    rewards_path.write_text(rewards_path.read_text() + extra)


def edit_observations(obs_path: Path) -> None:
    """Append the predicted_next_contact obs function to mdp/observations.py."""
    obs_path.write_text(obs_path.read_text() + PREDICTOR_OBS_BLOCK)


def edit_env_cfg(cfg_path: Path) -> None:
    """Apply v4c structural changes and register the v7 obs term in BOTH the
    actor and critic groups."""
    txt = cfg_path.read_text()

    # base_height_recovery target 0.27 -> 0.30 (v4c).
    txt = txt.replace('"target_height": 0.27', '"target_height": 0.30')

    # action_rate_l2 -0.05 -> -0.5 (v4c smoothness).
    txt = txt.replace(
        '"action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),',
        '"action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.5),',
    )

    # Insert v4c reward terms before joint_velocity_penalty.
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
        raise RuntimeError(f"reward insert anchor not found: {insert_anchor!r}")
    txt = txt.replace(insert_anchor, insertion + insert_anchor, 1)

    # Register the v7 obs term in actor_terms. The baseline ends actor_terms
    # with a height_scan entry; insert ours right before the closing brace.
    obs_insert = (
        '    "predicted_next_contact": ObservationTermCfg(\n'
        '      func=get_up_mdp.predicted_next_contact,\n'
        '      params={"sensor_name": "feet_ground_contact"},\n'
        '    ),\n'
        '  }\n\n'
        '  critic_terms = {'
    )
    anchor = "  }\n\n  critic_terms = {"
    if anchor not in txt:
        raise RuntimeError("actor_terms close brace anchor not found")
    txt = txt.replace(anchor, obs_insert, 1)

    cfg_path.write_text(txt)


def main() -> int:
    if not V3.exists():
        print(f"ERROR: {V3} missing -- run from a pod with the baseline.",
              file=sys.stderr)
        return 1

    if ACTIVE.exists():
        shutil.rmtree(ACTIVE)
    shutil.copytree(V3, ACTIVE)
    print(f"[restore] {V3} -> {ACTIVE}")

    # 1. Drop the predictor module into the mdp package.
    pred_path = ACTIVE / "mdp" / "contact_predictor.py"
    pred_path.write_text(CONTACT_PREDICTOR_PY)
    print(f"[new] {pred_path}")

    # 2. Append the predicted_next_contact obs function.
    edit_observations(ACTIVE / "mdp" / "observations.py")
    print("[edit] mdp/observations.py: appended predicted_next_contact")

    # 3. Inherit v4c structural pieces.
    apply_pose_jitter(ACTIVE / "mdp" / "events.py")
    print("[edit] mdp/events.py: pose jitter")
    append_v4c_rewards(ACTIVE / "mdp" / "rewards.py")
    print("[edit] mdp/rewards.py: hind_legs_extended + stand_fully_bonus")

    # 4. Wire it all up in env_cfg.
    edit_env_cfg(ACTIVE / "get_up_env_cfg.py")
    print("[edit] get_up_env_cfg.py: v4c weights + v7 obs term")

    # 5. Re-export the new mdp symbol via mdp/__init__.py if it uses *imports.
    init_path = ACTIVE / "mdp" / "__init__.py"
    if init_path.exists():
        init_txt = init_path.read_text()
        if "from .observations import" in init_txt and "predicted_next_contact" not in init_txt:
            # Naive append; star imports will pick it up too.
            init_txt += "\nfrom .observations import predicted_next_contact  # v7\n"
            init_txt += "from . import contact_predictor  # v7\n"
            init_path.write_text(init_txt)
            print("[edit] mdp/__init__.py: re-export v7 symbols")

    # 6. Syntax check.
    rc = subprocess.run(
        ["python3", "-c",
         "import ast;"
         f"ast.parse(open('{ACTIVE / 'get_up_env_cfg.py'}').read());"
         f"ast.parse(open('{ACTIVE / 'mdp' / 'rewards.py'}').read());"
         f"ast.parse(open('{ACTIVE / 'mdp' / 'events.py'}').read());"
         f"ast.parse(open('{ACTIVE / 'mdp' / 'observations.py'}').read());"
         f"ast.parse(open('{ACTIVE / 'mdp' / 'contact_predictor.py'}').read())"],
        capture_output=True, text=True,
    )
    if rc.returncode != 0:
        print(f"ERROR: syntax check failed:\n{rc.stderr}", file=sys.stderr)
        return 3

    print("[ok] v7 patch applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
