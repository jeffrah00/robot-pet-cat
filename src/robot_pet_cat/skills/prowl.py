from __future__ import annotations
from typing import Any
from .skill_policy import LocomotionCommand, Skill


class Prowl(Skill):
    """Slow, low-body stalking creep (vx = 0.2 m/s, gait="crouch")."""

    name: str = "prowl"

    def step(self, obs: dict[str, Any], goal: Any) -> LocomotionCommand:
        return LocomotionCommand(vx=0.2, vy=0.0, yaw_rate=0.0, gait="crouch")

    def is_done(self, obs: dict[str, Any], goal: Any) -> bool:
        return False
