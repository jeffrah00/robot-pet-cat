"""get_up skill -- hand off to the get_up_v4c policy (gait='get_up')."""
from __future__ import annotations
from typing import Any

from .skill_policy import Skill, LocomotionCommand

GET_UP_TRANSITION_STEPS = 60


class GetUp(Skill):
    """Hand off to the get_up_v4c.pt policy until the brain switches skills."""

    name: str = "get_up"

    def __init__(self) -> None:
        self._step_count = 0

    def reset(self) -> None:
        """Called by the brain when transitioning INTO get_up from a
        different skill. Zeroes the step counter so the transition window
        starts fresh."""
        self._step_count = 0

    @property
    def step_count(self) -> int:
        """How many decision ticks the cat has been in get_up."""
        return self._step_count

    @property
    def is_holding(self) -> bool:
        """True once past the transition window."""
        return self._step_count >= GET_UP_TRANSITION_STEPS

    def step(self, obs: dict[str, Any], goal: Any) -> LocomotionCommand:
        """Emit gait='get_up'. PhysicsCat routes substeps through the get_up policy."""
        self._step_count += 1
        return LocomotionCommand(
            vx=0.0,
            vy=0.0,
            yaw_rate=0.0,
            gait="get_up",
        )

    def is_done(self, obs: dict[str, Any], goal: Any) -> bool:
        return False
