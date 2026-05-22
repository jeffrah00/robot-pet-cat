from __future__ import annotations
from typing import Any
from .skill_policy import LocomotionCommand, Skill


class Stand(Skill):
    """Hold the cat in a static upright standing pose.

    Emits zero velocity with gait="stand" so the locomotion policy
    maintains an upright stance without stepping.
    """

    name: str = "stand"

    def step(self, obs: dict[str, Any], goal: Any) -> LocomotionCommand:
        return LocomotionCommand(vx=0.0, vy=0.0, yaw_rate=0.0, gait="stand")

    def is_done(self, obs: dict[str, Any], goal: Any) -> bool:
        # Hold skill  brain decides when to stop.
        return False
