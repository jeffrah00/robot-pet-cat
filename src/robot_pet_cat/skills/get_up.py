"""get_up -- reverse-ramp from a prone pose back to standing.

Emits joint_targets = GO2_DEFAULT_JOINT_POS, which makes PhysicsCat's keyframe-PD
ramp drive the joints from current pose to the standing default over _RAMP
substeps (~7.5 s with the slow linear ramp). When the brain switches to a
walker-based skill next, joint_targets becomes None and the walker takes over
from (approximately) the standing pose.
"""
from __future__ import annotations

import numpy as np
from typing import Any

from .skill_policy import LocomotionCommand, Skill
from ..brain.go2_policy import GO2_DEFAULT_JOINT_POS

GET_UP_JOINT_TARGETS = GO2_DEFAULT_JOINT_POS.astype(np.float32)


class GetUp(Skill):
    name: str = "get_up"

    def step(self, obs: dict[str, Any], goal: Any) -> LocomotionCommand:
        return LocomotionCommand(
            vx=0.0,
            vy=0.0,
            yaw_rate=0.0,
            gait="stand",
            joint_targets=GET_UP_JOINT_TARGETS,
        )

    def is_done(self, obs: dict[str, Any], goal: Any) -> bool:
        # Time-based: brain holds get_up for ~8 s before switching.
        return float(obs.get("time_in_skill", 0.0)) >= 8.0
