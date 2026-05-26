"""lie_down -- belly-on-floor sphinx pose, held until the brain switches skills.

All four legs fold under the body. The Go2 drops into a sphinx/prone posture:
hips splayed slightly outward, thighs pulled forward, calves folded back under
the body.

Implementation: manual PD interpolation. No RL training. Each call to step()
linearly ramps joint targets from the Go2 standing defaults toward the tucked-
under loaf targets over RAMP_STEPS decision ticks, then holds. PhysicsCat's
joint_targets bypass drives actuators directly; the walker policy is not used.

Joint order (MJCF / Go2 policy order):
  FL_hip, FL_thigh, FL_calf,
  FR_hip, FR_thigh, FR_calf,
  RL_hip, RL_thigh, RL_calf,
  RR_hip, RR_thigh, RR_calf
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .skill_policy import LocomotionCommand, Skill

# Go2 standing defaults (matches GO2_DEFAULT_JOINT_POS in go2_policy.py)
STAND_JOINT_POS = np.array(
    [
        -0.10, 0.90, -1.80,  # FL
        +0.10, 0.90, -1.80,  # FR
        -0.10, 0.90, -1.80,  # RL
        +0.10, 0.90, -1.80,  # RR
    ],
    dtype=np.float32,
)

# Tucked-under loaf targets.
# Hip splayed outward so legs can fold under without collision.
# Thigh pushed deeply forward; calf bent near maximum range.
LOAF_JOINT_TARGETS = np.array(
    [
        +0.50, 1.60, -2.70,  # FL
        -0.50, 1.60, -2.70,  # FR
        +0.50, 1.60, -2.70,  # RL
        -0.50, 1.60, -2.70,  # RR
    ],
    dtype=np.float32,
)

# Number of decision ticks to ramp from stand to loaf.
# At GO2_POLICY_DT=0.02s this is ~2.4 seconds of sim time.
RAMP_STEPS = 120


class LieDown(Skill):
    """Belly-down sphinx pose. Held until brain switches skills."""

    name: str = "lie_down"

    def __init__(self) -> None:
        self._step_count: int = 0

    def reset(self) -> None:
        """Called by the brain when transitioning INTO lie_down."""
        self._step_count = 0

    def step(self, obs: dict[str, Any], goal: Any) -> LocomotionCommand:
        """Emit a linearly-ramped joint_targets command toward loaf pose."""
        self._step_count += 1
        alpha = min(1.0, self._step_count / RAMP_STEPS)
        targets = ((1.0 - alpha) * STAND_JOINT_POS
                   + alpha * LOAF_JOINT_TARGETS).astype(np.float32)
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
