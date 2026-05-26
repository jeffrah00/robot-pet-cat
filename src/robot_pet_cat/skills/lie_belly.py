"""lie_belly -- belly-down pose via mjlab Go1 lie_belly policy.

Backed by Mjlab-LieBelly-Flat-Unitree-Go1.
"""
from __future__ import annotations
from typing import Any
from .skill_policy import LocomotionCommand, Skill


class LieBelly(Skill):
    """Belly-down sphinx/loaf pose; dispatched via gait='lie_belly'."""

    name: str = "lie_belly"

    def step(self, obs: dict[str, Any], goal: Any) -> LocomotionCommand:
        return LocomotionCommand(vx=0.0, vy=0.0, yaw_rate=0.0, gait="lie_belly")

    def is_done(self, obs: dict[str, Any], goal: Any) -> bool:
        return False
