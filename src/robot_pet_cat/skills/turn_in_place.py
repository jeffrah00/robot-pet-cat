from __future__ import annotations
from typing import Any
from .skill_policy import LocomotionCommand, Skill


class TurnInPlace(Skill):
    """Rotate in place (default yaw_rate = 1.0 rad/s, CCW positive).

    Goal can be a float to override the yaw_rate sign/magnitude,
    or None to use the default.
    """

    name: str = "turn_in_place"

    def __init__(self, yaw_rate: float = 1.0) -> None:
        self.yaw_rate = yaw_rate

    def step(self, obs: dict[str, Any], goal: Any) -> LocomotionCommand:
        rate = float(goal) if goal is not None else self.yaw_rate
        return LocomotionCommand(vx=0.0, vy=0.0, yaw_rate=rate, gait="trot")

    def is_done(self, obs: dict[str, Any], goal: Any) -> bool:
        return False
