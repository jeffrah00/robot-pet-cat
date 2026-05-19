"""walk_to(target_xy) — Tier 2 skill that drives the cat to a target XY position.

Per ROADMAP Phase 3, this is the canonical first skill: "velocity-command wrapper,
mostly the AMP policy." The skill itself is a thin analytical controller — it
reads the cat's current root position + heading, computes a body-frame velocity
command pointing toward the target, and returns it as a `LocomotionCommand`.
The frozen AMP motion policy (Tier 1) then translates that command into joint
targets with cat-style gait.

No learned weights here. Character is supplied by the AMP style prior at Tier 1;
goal-directedness is supplied by this skill; *what goal to pick* is supplied by
the Tier 3 brain. Per project memory, the cat is autonomous — no human ever
calls this skill directly; the brain calls it.

Obs contract
------------
The skill expects the obs dict to carry the cat's privileged world-frame state:

    obs["root_xy"]   : ndarray[2]  — world-frame (x, y) of the trunk
    obs["root_yaw"]  : float       — world-frame heading (radians, +z up)

These are extracted by whatever env wrapper sits between the raw MuJoCo state
and the brain — they correspond to `state.data.qpos[:2]` and the yaw component
of `state.data.qpos[3:7]`. Pushing them through the obs dict keeps this skill
decoupled from the MuJoCo backend, so the same code works under brax/mjx, raw
MuJoCo, or a unit-test fixture.

Goal type
---------
`WalkToGoal(target_xy)` — a single 2-D world-frame point. Z is implied by floor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .skill_policy import LocomotionCommand, Skill


@dataclass(frozen=True)
class WalkToGoal:
    """Where to walk to, in world frame."""

    target_xy: tuple[float, float]


def _wrap_angle(a: float) -> float:
    """Wrap an angle to the half-open interval [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class WalkTo(Skill):
    """Drive the AMP locomotion policy toward a target XY in world frame.

    The controller is a two-channel cascade:

      1. Heading channel — rotate to face the goal. yaw_rate is a P-controller
         on the wrapped yaw error, saturated at ``max_yaw_rate``.

      2. Translation channel — once roughly aligned, push forward at a speed
         that ramps from ``max_speed`` (far from goal) to 0 (at goal), with a
         linear slow-down inside ``slow_radius``. Velocity is expressed in the
         BODY frame, so the AMP policy receives the same kind of command it was
         trained on.

    A "facing" weight gates translation by how aligned we are with the target:
    we don't drive forward at full speed while still spinning to face the goal.
    """

    name: str = "walk_to"

    def __init__(
        self,
        *,
        max_speed: float = 0.6,        # m/s — matches render_imit_policy.py's --command 0.6,0,0
        slow_radius: float = 0.5,      # m — start ramping speed down within this
        goal_tolerance: float = 0.15,  # m — done when closer than this
        yaw_kp: float = 2.0,           # rad/s per rad of heading error
        max_yaw_rate: float = 1.5,     # rad/s — cap to keep motion gentle
        facing_threshold: float = math.radians(45.0),
        # Above this much heading error, kill forward velocity until aligned.
    ) -> None:
        if max_speed <= 0:
            raise ValueError(f"max_speed must be positive, got {max_speed}")
        if slow_radius <= 0:
            raise ValueError(f"slow_radius must be positive, got {slow_radius}")
        if goal_tolerance <= 0:
            raise ValueError(f"goal_tolerance must be positive, got {goal_tolerance}")
        self.max_speed = float(max_speed)
        self.slow_radius = float(slow_radius)
        self.goal_tolerance = float(goal_tolerance)
        self.yaw_kp = float(yaw_kp)
        self.max_yaw_rate = float(max_yaw_rate)
        self.facing_threshold = float(facing_threshold)

    # ------------------------------------------------------------------ #
    # Skill interface
    # ------------------------------------------------------------------ #

    def step(self, obs: dict[str, Any], goal: Any) -> LocomotionCommand:
        target = self._unpack_goal(goal)
        root_xy = np.asarray(obs["root_xy"], dtype=np.float32).reshape(2)
        root_yaw = float(obs["root_yaw"])

        # World-frame displacement to target.
        dxy = target - root_xy
        distance = float(np.linalg.norm(dxy))

        if distance < self.goal_tolerance:
            # Already there — hand back a "stand still" command. The brain's
            # is_done check will switch skills on the next tick.
            return LocomotionCommand(vx=0.0, vy=0.0, yaw_rate=0.0, gait="stand")

        # Heading control: turn to face the goal.
        desired_yaw = math.atan2(float(dxy[1]), float(dxy[0]))
        yaw_err = _wrap_angle(desired_yaw - root_yaw)
        yaw_rate = max(-self.max_yaw_rate, min(self.max_yaw_rate, self.yaw_kp * yaw_err))

        # Translation control: forward speed ramps with distance, gated by how
        # well we're facing the goal.
        speed = self.max_speed * min(1.0, distance / self.slow_radius)
        if abs(yaw_err) >= self.facing_threshold:
            # Too far off heading to commit forward motion — rotate in place.
            speed = 0.0

        # Express speed in body frame. After yaw_rate brings us into alignment,
        # the velocity points straight along body +x. Until then we still emit a
        # tiny lateral component so the AMP policy can side-step if its trained
        # behavior allows it.
        vx_body = speed * math.cos(yaw_err)
        vy_body = speed * math.sin(yaw_err)

        return LocomotionCommand(
            vx=vx_body,
            vy=vy_body,
            yaw_rate=yaw_rate,
            gait="trot",
        )

    def is_done(self, obs: dict[str, Any], goal: Any) -> bool:
        target = self._unpack_goal(goal)
        root_xy = np.asarray(obs["root_xy"], dtype=np.float32).reshape(2)
        return bool(np.linalg.norm(target - root_xy) < self.goal_tolerance)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _unpack_goal(goal: Any) -> np.ndarray:
        """Accept a WalkToGoal, a 2-tuple, or any 2-element array."""
        if isinstance(goal, WalkToGoal):
            xy = goal.target_xy
        else:
            xy = goal
        arr = np.asarray(xy, dtype=np.float32).reshape(-1)
        if arr.size != 2:
            raise ValueError(f"walk_to goal must be 2-D (x, y), got shape {arr.shape}")
        return arr
