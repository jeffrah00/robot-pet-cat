"""stretch -- front-legs-extended wake-up stretch, transient.

The front legs sweep forward and straighten while the rear legs hold their
standing position, keeping the haunches elevated. Produces the feline
"downward dog" stretch where the chest is low and the rump stays up.

Unlike sit/lie_down/crouch, stretch has a natural duration: a cat stretches
for ~2s then relaxes. The skill signals is_done() after the duration.

Joint targets (FL/FR/RL/RR, hip/thigh/calf per leg):

  Front legs: swept moderately forward, partially straightened.
    hip   = +-0.10  (near-default splay)
    thigh =  1.50   (leg swept forward, was 1.80 -- reduced to prevent forward tip)
    calf  = -1.40   (moderate bend, was -0.80 -- more bent = less forward pitch)

  Rear legs: standard standing -- haunches stay up.
    hip   = +-0.10
    thigh =  0.85   (near default 0.90)
    calf  = -1.80   (default)

Reduced from v1 to keep all 4 feet planted without pitching forward.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .skill_policy import LocomotionCommand, Skill


# Joint order: FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
#              RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf
STRETCH_JOINT_TARGETS = np.array(
    [
        -0.10,  1.50, -1.40,   # FL: swept forward, moderately straight (was 1.80, -0.80)
        +0.10,  1.50, -1.40,   # FR: swept forward, moderately straight
        -0.10,  0.85, -1.80,   # RL: near-default, haunches up
        +0.10,  0.85, -1.80,   # RR: near-default, haunches up
    ],
    dtype=np.float32,
)


@dataclass
class StretchGoal:
    """How long to hold the stretch, in sim seconds."""

    duration_s: float = 2.0


class Stretch(Skill):
    """Hold an extended front-leg stretch for a fixed duration, then signal done."""

    name: str = "stretch"

    def __init__(self) -> None:
        self._start_t: float | None = None

    def reset(self) -> None:
        self._start_t = None

    def step(self, obs: dict[str, Any], goal: Any) -> LocomotionCommand:
        if self._start_t is None:
            self._start_t = self._now(obs)
        return LocomotionCommand(
            vx=0.0,
            vy=0.0,
            yaw_rate=0.0,
            gait="stand",
            body_height=0.24,
            joint_targets=STRETCH_JOINT_TARGETS,
        )

    def is_done(self, obs: dict[str, Any], goal: Any) -> bool:
        if self._start_t is None:
            return False
        duration = self._unpack_goal(goal)
        return (self._now(obs) - self._start_t) >= duration

    @staticmethod
    def _unpack_goal(goal: Any) -> float:
        if isinstance(goal, StretchGoal):
            return float(goal.duration_s)
        if goal is None:
            return 2.0
        return float(goal)

    @staticmethod
    def _now(obs: dict[str, Any]) -> float:
        t = obs.get("t_sim")
        return float(t) if t is not None else time.monotonic()
