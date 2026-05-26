"""lie_belly -- belly-on-floor sphinx/loaf pose via manual PD keyframes.

Memory `lie_down_manual_v1` + `lie_belly_v1_findings`: the keyframe-PD
approach was the chosen implementation for this pose; the trained
Mjlab-LieBelly-Flat-Unitree-Go1 policy is loaded by PhysicsCat for future
A/B but is not the live dispatch path here.

Implementation: linearly ramp joint targets from standing defaults toward
a tucked-under loaf pose over RAMP_STEPS ticks, then hold. PhysicsCat sees
LocomotionCommand.joint_targets != None and bypasses any policy, writing
the targets straight to actuators.

Joint order matches GO2_JOINT_NAMES (PhysicsCat lookup order):
  FL_hip, FL_thigh, FL_calf,
  FR_hip, FR_thigh, FR_calf,
  RL_hip, RL_thigh, RL_calf,
  RR_hip, RR_thigh, RR_calf
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .skill_policy import LocomotionCommand, Skill

# Go1 standing defaults (matches GO1_DEFAULT_JOINT_POS in go1_policy.py).
STAND_JOINT_POS = np.array(
    [
        -0.10, 0.90, -1.80,  # FL
        +0.10, 0.90, -1.80,  # FR
        -0.10, 0.90, -1.80,  # RL
        +0.10, 0.90, -1.80,  # RR
    ],
    dtype=np.float32,
)

# Tucked-under loaf targets (belly-down sphinx/prone pose).
# Hips splayed outward so legs fold under without collision; thigh forward,
# calf bent near limit.
LOAF_JOINT_TARGETS = np.array(
    [
        +0.50, 1.60, -2.70,  # FL
        -0.50, 1.60, -2.70,  # FR
        +0.50, 1.60, -2.70,  # RL
        -0.50, 1.60, -2.70,  # RR
    ],
    dtype=np.float32,
)

# Decision ticks to ramp from stand to loaf (~2.4 s at 20 ms/tick).
RAMP_STEPS = 120


class LieBelly(Skill):
    """Belly-down sphinx/loaf pose held until the brain switches skills."""

    name: str = "lie_belly"

    def __init__(self) -> None:
        self._step_count: int = 0

    def reset(self) -> None:
        """Called by the brain on transition INTO lie_belly."""
        self._step_count = 0

    def step(self, obs: dict[str, Any], goal: Any) -> LocomotionCommand:
        """Ramp joint targets from stand toward loaf, then hold."""
        self._step_count += 1
        alpha = min(1.0, self._step_count / RAMP_STEPS)
        targets = (
            (1.0 - alpha) * STAND_JOINT_POS + alpha * LOAF_JOINT_TARGETS
        ).astype(np.float32)
        return LocomotionCommand(
            vx=0.0,
            vy=0.0,
            yaw_rate=0.0,
            gait="stand",
            body_height=0.08,
            joint_targets=targets,
        )

    def is_done(self, obs: dict[str, Any], goal: Any) -> bool:
        return False
