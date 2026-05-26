"""prowl -- slow stalking creep via mjlab Go1 walker_slow."""
from __future__ import annotations
from typing import Any
from .skill_policy import LocomotionCommand, Skill


class Prowl(Skill):
    """Slow forward walk; dispatched via gait='walk_slow'."""

    name: str = "prowl"

    def step(self, obs: dict[str, Any], goal: Any) -> LocomotionCommand:
        return LocomotionCommand(vx=0.2, vy=0.0, yaw_rate=0.0, gait="walk_slow")

    def is_done(self, obs: dict[str, Any], goal: Any) -> bool:
        return False
