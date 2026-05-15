"""Generic skill-policy interface.

Every skill exposes the same shape so the brain can invoke them uniformly.
A skill takes a structured *goal* (target position, target object, etc.) plus
the robot observation, and outputs the locomotion-controller command (velocity,
gait, head pose). Skills are trained with task-specific reward + the shared AMP
style reward inherited from motion/.

The training entry-point for new skills lives in this file too. Phase 3 work.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class LocomotionCommand:
    """Output of every skill — consumed by the AMP-trained low-level policy."""

    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0
    gait: str = "trot"          # trot | walk | stand | crouch | leap
    head_pitch: float = 0.0
    head_yaw: float = 0.0
    body_height: float = 0.30   # nominal Go2 standing height in meters


class Skill(ABC):
    """Base class for a mid-level skill."""

    name: str

    @abstractmethod
    def step(self, obs: dict[str, Any], goal: Any) -> LocomotionCommand: ...

    @abstractmethod
    def is_done(self, obs: dict[str, Any], goal: Any) -> bool: ...


# Each concrete skill becomes a subclass with its own learned (or partly
# learned) policy network. Phase 3 fills these in:
#
#   class WalkTo(Skill): ...
#   class SitStill(Skill): ...
#   class SwatObject(Skill): ...
#   class JumpToSurface(Skill): ...
#
# For Phase 3 we start by training WalkTo and SitStill, then add JumpToSurface
# (using the curriculum-jumping recipe — Atanassov et al. 2024), then SwatObject.
