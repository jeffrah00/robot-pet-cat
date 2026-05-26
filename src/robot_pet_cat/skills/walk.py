"""walk -- forward locomotion via mjlab Go1 walker (normal)."""
from __future__ import annotations
from typing import Any
from .skill_policy import LocomotionCommand, Skill


class Walk(Skill):
    """Forward walking; dispatched via gait='walk'."""

    name: str = "walk"

    def step(self, obs: dict[str, Any], goal: Any) -> LocomotionCommand:
        return LocomotionCommand(vx=0.5, vy=0.0, yaw_rate=0.0, gait="walk")

    def is_done(self, obs: dict[str, Any], goal: Any) -> bool:
        return False
