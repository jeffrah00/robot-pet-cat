"""look_at(point_xyz) — orient head toward a world-frame point.

Per the brain design v2 discussion (gaze-first architecture): the head turns
first, body responds later (or not at all). This skill drives head_yaw and
head_pitch in the LocomotionCommand vocabulary; vx/vy/yaw_rate stay zero, so
the body holds whatever pose the previous skill left it in.

For the Go2 specifically the "head" articulation is limited (the trunk-mounted
camera mostly turns via body yaw), so head_yaw/head_pitch may end up driving
body yaw at integration time. The skill interface is forward-compatible with
better head hardware later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .skill_policy import LocomotionCommand, Skill
from .walk_to import _wrap_angle


@dataclass(frozen=True)
class LookAtGoal:
    """World-frame point the head should orient toward."""

    point_xyz: tuple[float, float, float]


# Vertical center the cat's "eye line" sits at when standing. Used to compute
# the pitch component of looking up vs down.
EYE_HEIGHT_M = 0.28


class LookAt(Skill):
    """Orient head (and, on hardware without a head joint, body yaw) toward a point."""

    name: str = "look_at"

    def __init__(
        self,
        *,
        max_head_pitch: float = math.radians(30.0),
        max_head_yaw: float = math.radians(60.0),
        eye_height_m: float = EYE_HEIGHT_M,
    ) -> None:
        self.max_head_pitch = float(max_head_pitch)
        self.max_head_yaw = float(max_head_yaw)
        self.eye_height_m = float(eye_height_m)

    def step(self, obs: dict[str, Any], goal: Any) -> LocomotionCommand:
        target = self._unpack_goal(goal)
        root_xy = np.asarray(obs["root_xy"], dtype=np.float32).reshape(2)
        root_yaw = float(obs["root_yaw"])

        dx = float(target[0] - root_xy[0])
        dy = float(target[1] - root_xy[1])
        dz = float(target[2] - self.eye_height_m)

        # Desired world-frame yaw to face the target.
        desired_yaw = math.atan2(dy, dx)
        yaw_err = _wrap_angle(desired_yaw - root_yaw)
        head_yaw_cmd = max(-self.max_head_yaw, min(self.max_head_yaw, yaw_err))

        # Pitch from horizontal distance + height delta.
        horiz = math.hypot(dx, dy)
        desired_pitch = math.atan2(dz, max(horiz, 1e-3))
        head_pitch_cmd = max(
            -self.max_head_pitch, min(self.max_head_pitch, desired_pitch)
        )

        return LocomotionCommand(
            vx=0.0,
            vy=0.0,
            yaw_rate=0.0,
            gait="stand",
            head_yaw=head_yaw_cmd,
            head_pitch=head_pitch_cmd,
        )

    def is_done(self, obs: dict[str, Any], goal: Any) -> bool:
        # Looking is a hold action; the brain decides when to stop looking.
        return False

    @staticmethod
    def _unpack_goal(goal: Any) -> np.ndarray:
        if isinstance(goal, LookAtGoal):
            xyz = goal.point_xyz
        else:
            xyz = goal
        arr = np.asarray(xyz, dtype=np.float32).reshape(-1)
        if arr.size != 3:
            raise ValueError(f"look_at goal must be 3-D (x, y, z), got shape {arr.shape}")
        return arr
