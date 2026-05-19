"""swat(object_xyz) — approach a small object and paw-strike it.

Per the 2026-05-19 design conversation (faking is enough): the realistic version
of swat on a Go2 is a *two-phase* sequence with no RL training:

  1. walk_to(near_object) — already implemented in walk_to.py
  2. paw-lift gesture — a brief front-leg extension toward the object, executed
     as a keyframe with PD interpolation

This scaffold delegates phase 1 to WalkTo and stubs phase 2 as a held
"front legs forward + body crouched" command for a fixed duration. The real
paw-lift trajectory is a per-joint keyframe sequence that LocomotionCommand
doesn't currently express; future revision should add a per-joint-targets
field or introduce a richer command object.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .skill_policy import LocomotionCommand, Skill
from .walk_to import WalkTo, WalkToGoal


@dataclass(frozen=True)
class SwatGoal:
    """World-frame point to swat. z is ignored for approach but kept for
    completeness so the brain doesn't have to remember which dim to drop."""

    point_xyz: tuple[float, float, float]


# How close we need to be before the strike phase begins (paw range).
APPROACH_RADIUS_M = 0.25

# Duration of the strike phase in sim seconds.
STRIKE_DURATION_S = 0.6


class Swat(Skill):
    """Approach a target XY, then briefly emit a paw-strike pose."""

    name: str = "swat"

    def __init__(self) -> None:
        self._walker = WalkTo(goal_tolerance=APPROACH_RADIUS_M)
        self._strike_start_t: float | None = None

    def reset(self) -> None:
        self._strike_start_t = None

    def step(self, obs: dict[str, Any], goal: Any) -> LocomotionCommand:
        target = self._unpack_goal(goal)
        target_xy = (float(target[0]), float(target[1]))

        # Phase 1: walk into range.
        if not self._walker.is_done(obs, WalkToGoal(target_xy=target_xy)):
            return self._walker.step(obs, WalkToGoal(target_xy=target_xy))

        # Phase 2: strike. Hold a crouched-forward pose for STRIKE_DURATION_S.
        if self._strike_start_t is None:
            self._strike_start_t = self._now(obs)

        # Face the target while striking (head_yaw points at target relative to body).
        root_xy = np.asarray(obs["root_xy"], dtype=np.float32).reshape(2)
        root_yaw = float(obs["root_yaw"])
        dy = float(target[1]) - float(root_xy[1])
        dx = float(target[0]) - float(root_xy[0])
        head_yaw = math.atan2(dy, dx) - root_yaw

        return LocomotionCommand(
            vx=0.0,
            vy=0.0,
            yaw_rate=0.0,
            gait="stand",
            body_height=0.16,  # crouched slightly
            head_yaw=head_yaw,
        )

    def is_done(self, obs: dict[str, Any], goal: Any) -> bool:
        if self._strike_start_t is None:
            return False
        return (self._now(obs) - self._strike_start_t) >= STRIKE_DURATION_S

    @staticmethod
    def _unpack_goal(goal: Any) -> np.ndarray:
        if isinstance(goal, SwatGoal):
            xyz = goal.point_xyz
        else:
            xyz = goal
        arr = np.asarray(xyz, dtype=np.float32).reshape(-1)
        if arr.size != 3:
            raise ValueError(f"swat goal must be 3-D (x, y, z), got shape {arr.shape}")
        return arr

    @staticmethod
    def _now(obs: dict[str, Any]) -> float:
        t = obs.get("t_sim")
        return float(t) if t is not None else time.monotonic()
