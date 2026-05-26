"""lie_side -- side-lying rest pose (stub).

The robot rolls onto its right side, left legs extending upward and right
legs folding under the body. This is a distinct skill from lie_belly: the
body orientation is ~90 deg rolled rather than prone.

Currently a stub: joint targets approximate a rolled position but have not
been verified against Go1 physics. Render and tune targets before use.

Joint order (MJCF / Go1 policy order):
  FL_hip, FL_thigh, FL_calf,   (left fore -- extends upward when rolled right)
  FR_hip, FR_thigh, FR_calf,   (right fore -- folds under)
  RL_hip, RL_thigh, RL_calf,   (left rear  -- extends upward)
  RR_hip, RR_thigh, RR_calf    (right rear -- folds under)
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .skill_policy import LocomotionCommand, Skill

# Go1 standing defaults.
STAND_JOINT_POS = np.array(
    [
        -0.10, 0.90, -1.80,  # FL
        +0.10, 0.90, -1.80,  # FR
        -0.10, 0.90, -1.80,  # RL
        +0.10, 0.90, -1.80,  # RR
    ],
    dtype=np.float32,
)

# Placeholder side-lying targets (right side down).
# Left legs (FL, RL) extend upward; right legs (FR, RR) fold under body.
# These are UNTUNED -- render and adjust before shipping.
SIDE_JOINT_TARGETS = np.array(
    [
        +0.00, 0.50, -1.20,  # FL: left fore extends upward
        -0.00, 1.50, -2.60,  # FR: right fore folds under
        +0.00, 0.50, -1.20,  # RL: left rear extends upward
        -0.00, 1.50, -2.60,  # RR: right rear folds under
    ],
    dtype=np.float32,
)

# Slightly longer ramp than lie_belly; rolling takes more time.
RAMP_STEPS = 150


class LieSide(Skill):
    """Side-lying rest pose (right side down). Stub -- targets untuned."""

    name: str = "lie_side"

    def __init__(self) -> None:
        self._step_count: int = 0

    def reset(self) -> None:
        """Called by the brain when transitioning INTO lie_side."""
        self._step_count = 0

    def step(self, obs: dict[str, Any], goal: Any) -> LocomotionCommand:
        """Linearly ramp joint_targets from stand toward side-lying, then hold."""
        self._step_count += 1
        alpha = min(1.0, self._step_count / RAMP_STEPS)
        targets = (
            (1.0 - alpha) * STAND_JOINT_POS + alpha * SIDE_JOINT_TARGETS
        ).astype(np.float32)
        return LocomotionCommand(
            vx=0.0,
            vy=0.0,
            yaw_rate=0.0,
            gait="stand",
            body_height=0.05,
            joint_targets=targets,
        )

    def is_done(self, obs: dict[str, Any], goal: Any) -> bool:
        return False
