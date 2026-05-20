"""sit -- static sitting pose, held until the brain switches skills.

The Go2 approximates a cat sit: front legs remain roughly vertical (upright)
while the rear legs fold partially under the haunches, dropping the rear of
the body toward the floor.

Joint targets (FL/FR/RL/RR, hip/thigh/calf per leg):

  Front legs (FL=0-2, FR=3-5): slightly more forward than stand.
    hip   = +-0.10  (symmetric, slight abduction)
    thigh =  0.55   (gentle forward angle, more vertical than stand's 0.90)
    calf  = -1.20   (moderate bend -- front legs act as pillars)

  Rear legs (RL=6-8, RR=9-11): partial haunches fold, weight balanced.
    hip   = +-0.15  (slight outward splay for stability)
    thigh =  1.20   (partial fold -- conservative, was 1.55)
    calf  = -2.20   (moderately bent -- was -2.65)

Reduced from v1 to prevent rear tipping. Robot stays balanced with weight
distributed across all four contact points.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .skill_policy import LocomotionCommand, Skill


# Joint order: FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
#              RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf
SIT_JOINT_TARGETS = np.array(
    [
        -0.10,  0.55, -1.20,   # FL: semi-upright, slight forward lean
        +0.10,  0.55, -1.20,   # FR: semi-upright, slight forward lean
        -0.15,  1.20, -2.20,   # RL: partial haunch fold (conservative)
        +0.15,  1.20, -2.20,   # RR: partial haunch fold (conservative)
    ],
    dtype=np.float32,
)


class Sit(Skill):
    """Hold a sitting pose. No goal needed (the goal is just 'be sitting')."""

    name: str = "sit"

    def step(self, obs: dict[str, Any], goal: Any) -> LocomotionCommand:
        return LocomotionCommand(
            vx=0.0,
            vy=0.0,
            yaw_rate=0.0,
            gait="stand",
            body_height=0.25,
            joint_targets=SIT_JOINT_TARGETS,
        )

    def is_done(self, obs: dict[str, Any], goal: Any) -> bool:
        return False
