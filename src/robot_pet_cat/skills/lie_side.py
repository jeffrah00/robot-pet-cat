"""lie_side -- side-lying pose via mjlab Go1 lie_side policy.

Backed by Mjlab-LieSide-Flat-Unitree-Go1.
"""
from __future__ import annotations
from typing import Any
from .skill_policy import LocomotionCommand, Skill


class LieSide(Skill):
    """Side-lying rest pose; dispatched via gait='lie_side'."""

    name: str = "lie_side"

    def step(self, obs: dict[str, Any], goal: Any) -> LocomotionCommand:
        return LocomotionCommand(vx=0.0, vy=0.0, yaw_rate=0.0, gait="lie_side")

    def is_done(self, obs: dict[str, Any], goal: Any) -> bool:
        return False
