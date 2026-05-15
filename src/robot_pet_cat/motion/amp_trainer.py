"""AMP trainer for cat-style locomotion on Unitree Go2.

Follows the Adversarial Motion Priors recipe (Peng et al. 2021, building on the
earlier Peng et al. 2020 quadruped-from-animal-mocap work):

  1. Take a small set of retargeted cat motion clips (state trajectories in the
     Go2 joint space). Quality bar is low — we want style, not exact tracking.
  2. Train a *style discriminator* D(s, s') that scores transitions as
     "cat-like" vs "policy-generated".
  3. Train the PPO policy with reward = task_reward + lambda * log D(s, s').
     The task reward is just stay-upright + velocity-tracking. The style
     reward is what teaches cat-style gait.

The skills/ module then trains *subgoal-conditioned* policies on top of this
foundation, inheriting the cat motion style via the same discriminator.

Filled in during Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class AMPConfig:
    env_name: str = "Go2Joystick"
    motion_clips_dir: Path = Path("data/motion_clips")  # retargeted .npz files
    num_envs: int = 4096
    total_timesteps: int = 50_000_000
    style_reward_weight: float = 2.0      # weight on log D(s, s')
    task_reward_weight: float = 1.0
    discriminator_lr: float = 1.0e-4
    policy_lr: float = 3.0e-4
    seed: int = 0
    log_to_wandb: bool = True


def train(cfg: AMPConfig) -> None:
    raise NotImplementedError(
        "Phase 2: wire AMP discriminator + PPO. References:\n"
        "  - Peng et al. 2021 'AMP: Adversarial Motion Priors'\n"
        "  - Peng et al. 2020 'Learning Agile Robotic Locomotion Skills by Imitating Animals'\n"
        "  - mujoco_playground.locomotion.go2_joystick for the env baseline"
    )
