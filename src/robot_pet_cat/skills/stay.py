"""stay -- freeze the current joint configuration.

On activation PhysicsCat snapshots `qpos` of all 12 leg joints and writes
those exact targets to the actuators every tick thereafter, so the cat
holds whatever pose it happened to be in when this skill became active.
No policy is called.

Unlike `gait="stand"` (which routes to the walker with a zeroed velocity
command and stands at the walker's *default* standing pose), `gait="stay"`
holds the *current* pose -- post-get_up, post-crouch, mid-stretch, whatever.
"""
from __future__ import annotations
from typing import Any
from .skill_policy import LocomotionCommand, Skill


class Stay(Skill):
    """Freeze the cat in whatever pose it's currently in."""

    name: str = "stay"

    def step(self, obs: dict[str, Any], goal: Any) -> LocomotionCommand:
        return LocomotionCommand(vx=0.0, vy=0.0, yaw_rate=0.0, gait="stay")

    def is_done(self, obs: dict[str, Any], goal: Any) -> bool:
        return False
