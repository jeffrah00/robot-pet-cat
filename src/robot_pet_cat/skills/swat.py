"""swat -- front-leg raise gesture (paw-tap toward object).

PHYSICAL REALITY CHECK (2026-05-20):
The Go2 is a quadruped with NO arm or manipulator. It cannot swat, bat, or
strike objects the way a cat with retractable claws can. The only vaguely
analogous motion is briefly lifting one front leg -- the closest the Go2 can
get to a "paw" gesture.

This skill has been redesigned from "approach + paw-strike fantasy" to:

  Phase 1: walk toward the object (WalkTo, same as before)
  Phase 2: raise the front-left leg (FL lifted, balance on 3 legs) for
           PAW_RAISE_DURATION_S seconds, then lower back to stand.

The raised-leg pose:
  FL: hip=0.0, thigh=-0.5, calf=-0.6  -- leg lifted and angled forward
  FR: hip=+0.1, thigh=0.9, calf=-1.8  -- supporting, near-default
  RL: hip=-0.1, thigh=1.0, calf=-2.0  -- slightly deeper to shift CG back
  RR: hip=+0.1, thigh=1.0, calf=-2.0  -- same

This is the physically honest Go2 implementation of the swat slot. The brain
can still dispatch swat; what it produces is a paw-raise acknowledgment, not
an actual cat-style swat.

NOTE: SKILL_NAMES order is intentionally preserved (swat at index 6) so that
brain checkpoints trained on brain_v4/v5 continue to load without re-mapping.
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
    """World-frame point to gesture toward. z is ignored for approach."""

    point_xyz: tuple[float, float, float]


# How close we need to be before the paw-raise phase begins.
APPROACH_RADIUS_M = 0.30

# Duration of the paw-raise gesture in sim seconds.
PAW_RAISE_DURATION_S = 0.8

# Joint order: FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
#              RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf

# Phase 2a: FL leg raised (brief gesture toward object)
PAW_RAISE_TARGETS = np.array(
    [
         0.00, -0.50, -0.60,   # FL: hip fwd, thigh raised (negative = up/back)
        +0.10,  0.90, -1.80,   # FR: supporting, near-default
        -0.10,  1.00, -2.00,   # RL: slightly deeper for CG shift
        +0.10,  1.00, -2.00,   # RR: slightly deeper for CG shift
    ],
    dtype=np.float32,
)

# Phase 2b: return to normal stand (no joint_targets -- walker policy)
STAND_CMD = LocomotionCommand(vx=0.0, vy=0.0, yaw_rate=0.0, gait="stand")


class Swat(Skill):
    """Approach a target XY, then perform a brief front-leg-raise gesture."""

    name: str = "swat"

    def __init__(self) -> None:
        self._walker = WalkTo(goal_tolerance=APPROACH_RADIUS_M)
        self._raise_start_t: float | None = None

    def reset(self) -> None:
        self._raise_start_t = None

    def step(self, obs: dict[str, Any], goal: Any) -> LocomotionCommand:
        target = self._unpack_goal(goal)
        target_xy = (float(target[0]), float(target[1]))

        # Phase 1: walk into range.
        if not self._walker.is_done(obs, WalkToGoal(target_xy=target_xy)):
            return self._walker.step(obs, WalkToGoal(target_xy=target_xy))

        # Phase 2: paw raise. Hold FL-up pose for PAW_RAISE_DURATION_S.
        if self._raise_start_t is None:
            self._raise_start_t = self._now(obs)

        elapsed = self._now(obs) - self._raise_start_t
        if elapsed < PAW_RAISE_DURATION_S:
            return LocomotionCommand(
                vx=0.0,
                vy=0.0,
                yaw_rate=0.0,
                gait="stand",
                body_height=0.28,
                joint_targets=PAW_RAISE_TARGETS,
            )
        # Gesture done -- return to stand (no joint_targets, walker takes over).
        return STAND_CMD

    def is_done(self, obs: dict[str, Any], goal: Any) -> bool:
        if self._raise_start_t is None:
            return False
        return (self._now(obs) - self._raise_start_t) >= PAW_RAISE_DURATION_S

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
