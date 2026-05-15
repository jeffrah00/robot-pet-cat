"""Runs a fine-tuned VLA against the simulated cat-robot at inference time.

Reads observations from the MuJoCo env, queries the VLA for high-level
behavior commands, hands those commands to the locomotion controller. Filled
in during Phase 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class HighLevelCommand:
    """What the VLA emits per step — the locomotion controller's input."""

    target_velocity: tuple[float, float, float]  # vx, vy, yaw_rate
    gait: str = "trot"  # trot | walk | stand | crouch | leap
    head_pitch: float = 0.0
    tail_pose: float = 0.0


@dataclass
class RunnerConfig:
    vla_checkpoint: str = "your-hf-username/smolvla-cat"
    goal_token: str = "explore"  # explore | rest | play | follow | hunt
    inference_hz: int = 5  # VLA queried at low Hz; locomotion runs at 50 Hz


class CatBehaviorRunner:
    def __init__(self, cfg: RunnerConfig) -> None:
        self.cfg = cfg
        self._model: Any | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        raise NotImplementedError(
            "Phase 4: load SmolVLA from Hugging Face checkpoint and warm it up."
        )

    def step(self, observation: dict[str, Any]) -> HighLevelCommand:
        self._ensure_loaded()
        raise NotImplementedError("Phase 4: VLA forward pass → HighLevelCommand.")
