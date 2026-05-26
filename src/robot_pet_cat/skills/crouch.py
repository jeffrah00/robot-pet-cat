"""crouch -- pre-pounce low pose via mjlab Go1 crouch policy.

Backed by Mjlab-Crouch-Flat-Unitree-Go1 (getup fork with a lower torso target).
"""
from __future__ import annotations
from typing import Any
from .skill_policy import LocomotionCommand, Skill


class Crouch(Skill):
    """Drop into a pre-pounce crouch; dispatched via gait='crouch'."""

    name: str = "crouch"

    def step(self, obs: dict[str, Any], goal: Any) -> LocomotionCommand:
        return LocomotionCommand(vx=0.0, vy=0.0, yaw_rate=0.0, gait="crouch")

    def is_done(self, obs: dict[str, Any], goal: Any) -> bool:
        return False
