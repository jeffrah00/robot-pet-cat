from __future__ import annotations
from typing import Any
from .skill_policy import LocomotionCommand, Skill


class Trot(Skill):
    """Fast diagonal trot (vx = 1.0 m/s)."""

    name: str = "trot"

    def step(self, obs: dict[str, Any], goal: Any) -> LocomotionCommand:
        return LocomotionCommand(vx=1.0, vy=0.0, yaw_rate=0.0, gait="trot")

    def is_done(self, obs: dict[str, Any], goal: Any) -> bool:
        return False
