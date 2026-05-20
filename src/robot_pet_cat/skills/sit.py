"""sit -- static sitting pose, held until the brain switches skills.

The Go2 approximates a cat sit: front legs remain roughly vertical (upright)
while the rear legs fold deeply under the haunches, dropping the rear of the
body toward the floor. This is the closest the Go2's joint configuration can
get to a feline sitting posture.

Joint targets (FL/FR/RL/RR, hip/thigh/calf per leg):

  Front legs (FL=0-2, FR=3-5): semi-upright, thigh slightly forward.
    hip  = +-0.10  (symmetric, slight abduction)
    thigh = 0.45   (less forward than stand's 0.90 -- more vertical)
    calf  = -0.90  (less bent -- straighter front legs)

  Rear legs (RL=6-8, RR=9-11): haunches folded under the body.
    hip  = +-0.20  (slight outward splay for stability)
    thigh = 1.55   (deeply forward -- haunches pulled under)
    calf  = -2.65  (fully bent)

This emits joint_targets directly; PhysicsCat bypasses the walker policy
and uses PD control to hold the pose.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .skill_policy import LocomotionCommand, Skill


# Joint order: FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
#              RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf
SIT_JOINT_TARGETS = np.array(
    [
        -0.10,  0.45, -0.90,   # FL: semi-upright front leg
        +0.10,  0.45, -0.90,   # FR: semi-upright front leg
        -0.20,  1.55, -2.65,   # RL: haunches folded under
        +0.20,  1.55, -2.65,   # RR: haunches folded under
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
            body_height=0.22,
            joint_targets=SIT_JOINT_TARGETS,
        )

    def is_done(self, obs: dict[str, Any], goal: Any) -> bool:
        # Sitting is a hold pose -- the brain decides when to exit. Never
        # spontaneously "done" from the skill's own perspective.
        return False
