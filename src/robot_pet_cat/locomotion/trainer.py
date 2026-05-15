"""PPO trainer for low-level Go2 locomotion.

Phase 2 plan: bootstrap from MuJoCo Playground's `quadruped/joystick` task rather
than building the env from scratch — it already has reasonable reward shaping and
domain randomization. This module is the *entry point* that loads our YAML config,
spins up the env, and dispatches to the underlying trainer (Brax/PPO).

Filled in during Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainerConfig:
    env_name: str = "Go2Joystick"
    num_envs: int = 4096
    total_timesteps: int = 50_000_000
    learning_rate: float = 3e-4
    seed: int = 0
    log_to_wandb: bool = True


def train(cfg: TrainerConfig) -> None:
    raise NotImplementedError(
        "Phase 2: wire up to mujoco_playground.locomotion.go2_joystick + Brax PPO."
    )
