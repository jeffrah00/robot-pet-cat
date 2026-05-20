"""lie_down -- belly-on-floor sphinx pose, held until the brain switches skills.

All four legs fold under the body. The Go2 drops into a sphinx/prone posture:
hips splayed slightly outward, thighs pulled forward, calves folded back under
the body. This matches the Unitree SDK's built-in damp/fold trajectory.

Joint targets (FL/FR/RL/RR, hip/thigh/calf per leg):

  All legs fold symmetrically:
    hip   = +-0.30  (wider splay than standing -- lets legs fold under)
    thigh =  1.50   (deeply forward -- leg folds in against body)
    calf  = -2.65   (near maximum bend)

This emits joint_targets directly; PhysicsCat bypasses the walker policy
and uses PD control to hold the pose. The trunk rests approximately 0.08 m
above the floor once settled.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .skill_policy import LocomotionCommand, Skill


# Joint order: FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
#              RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf
LIE_DOWN_JOINT_TARGETS = np.array(
    [
        # Loaf pose: all four legs folded symmetrically under the body so the
        # cat sinks onto its own folded legs. No leg extension -> no tipping.
        -0.10,  1.50, -2.50,   # FL: hip neutral, thigh forward, calf tight
        +0.10,  1.50, -2.50,   # FR
        -0.10,  1.50, -2.50,   # RL
        +0.10,  1.50, -2.50,   # RR
    ],
    dtype=np.float32,
)


class LieDown(Skill):
    """Belly-down sphinx pose. Held until brain switches."""

    name: str = "lie_down"

    def step(self, obs: dict[str, Any], goal: Any) -> LocomotionCommand:
        return LocomotionCommand(
            vx=0.0,
            vy=0.0,
            yaw_rate=0.0,
            gait="stand",
            body_height=0.08,
            joint_targets=LIE_DOWN_JOINT_TARGETS,
        )

    def is_done(self, obs: dict[str, Any], goal: Any) -> bool:
        return False
