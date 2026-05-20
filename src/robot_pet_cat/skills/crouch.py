"""crouch -- pre-pounce low-and-still pose.

All four legs bend deeper than standing, lowering the center of mass.
Weight is evenly distributed -- front and rear folds are matched to keep
the robot planted without tipping forward.

Joint targets (FL/FR/RL/RR, hip/thigh/calf per leg):

  Front legs: moderate bend, even weight.
    hip   = +-0.10  (near-default splay)
    thigh =  1.00   (moderate forward bend, was 1.20 -- reduced to prevent forward tip)
    calf  = -2.00   (moderate bend, was -2.30)

  Rear legs: slightly deeper, haunches lowered.
    hip   = +-0.10
    thigh =  1.20   (rear haunches pulled under, was 1.45)
    calf  = -2.30   (moderately deep, was -2.55)

Reduced from v1 to prevent forward CoM tip. All 4 feet stay planted.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .skill_policy import LocomotionCommand, Skill


# Joint order: FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
#              RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf
CROUCH_JOINT_TARGETS = np.array(
    [
        -0.10,  1.00, -2.00,   # FL: moderate front bend (was 1.20, -2.30)
        +0.10,  1.00, -2.00,   # FR
        -0.10,  1.20, -2.30,   # RL: rear haunches lower (was 1.45, -2.55)
        +0.10,  1.20, -2.30,   # RR
    ],
    dtype=np.float32,
)


class Crouch(Skill):
    """Drop into a pre-pounce crouch and hold. Brain decides exit."""

    name: str = "crouch"

    def step(self, obs: dict[str, Any], goal: Any) -> LocomotionCommand:
        return LocomotionCommand(
            vx=0.0,
            vy=0.0,
            yaw_rate=0.0,
            gait="stand",
            body_height=0.20,
            joint_targets=CROUCH_JOINT_TARGETS,
        )

    def is_done(self, obs: dict[str, Any], goal: Any) -> bool:
        return False
