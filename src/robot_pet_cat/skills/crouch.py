"""crouch -- pre-pounce low-and-still pose.

All four legs bend deeper than standing, lowering the center of mass. The
weight shifts slightly forward (front legs less bent than rear, so the
front stays lower relative to the haunches). This matches feline crouching
behavior where the cat compresses before launching.

The stillness IS the skill: a crouching cat is almost motionless, sometimes
for many seconds, with only minor head/eye motion.

Joint targets (FL/FR/RL/RR, hip/thigh/calf per leg):

  Front legs: deep bend, weight forward.
    hip   = +-0.10  (near-default splay)
    thigh =  1.20   (forward + down, deeper than stand's 0.90)
    calf  = -2.30   (more bent)

  Rear legs: even deeper, haunches lowered.
    hip   = +-0.10
    thigh =  1.45   (rear haunches pulled further under)
    calf  = -2.55

This emits joint_targets directly; PhysicsCat bypasses the walker policy.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .skill_policy import LocomotionCommand, Skill


# Joint order: FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
#              RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf
CROUCH_JOINT_TARGETS = np.array(
    [
        -0.10,  1.20, -2.30,   # FL: deep front bend, weight forward
        +0.10,  1.20, -2.30,   # FR
        -0.10,  1.45, -2.55,   # RL: rear haunches lower
        +0.10,  1.45, -2.55,   # RR
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
            body_height=0.17,
            joint_targets=CROUCH_JOINT_TARGETS,
        )

    def is_done(self, obs: dict[str, Any], goal: Any) -> bool:
        return False
