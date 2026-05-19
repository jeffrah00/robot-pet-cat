"""jump_to(surface_xyz) -- multi-phase scripted jump onto an elevated surface.

State machine:
  APPROACH  -- walk toward the platform approach point
  ALIGN     -- rotate in place to face the platform
  CROUCH    -- lower body for CROUCH_DURATION_S (charge phase)
  LAUNCH    -- one step: BrainEnv injects ballistic impulse
  AIRBORNE  -- stand still while physics handles the arc (~1s)
  PERCH     -- landed; skill is done

Platform geometry is hardcoded from data/go2_scenes/living_room_with_go2.xml:
  <body name="cat_tree" pos="-1.5 -1.5 0.0">
    <geom name="cattree_lvl1" pos=" 0.18 0 0.30" size="0.20 0.20 0.025" .../>
    <geom name="cattree_lvl2" pos="-0.18 0 0.65" size="0.20 0.20 0.025" .../>
    <geom name="cattree_lvl3" pos=" 0.00 0 1.00" size="0.22 0.22 0.025" .../>

BrainEnv checks JumpTo.launch_requested each step (parallel to the swat-impulse
check). When True it calls PhysicsCat.apply_launch_impulse(mj_data, target_xyz),
then calls JumpTo.advance_to_airborne() to flip the skill into the AIRBORNE phase.

This skill uses keyframe driving for phases 1-3 and a physics impulse for the
actual jump arc (Tier 2 mixed approach). Full RL training (Atanassov 2024
curriculum) is the Phase 3 goal; this scripted version gives us a working
visual + obs signal in the meantime.

References:
- Atanassov et al. 2024, "Curriculum-Based Reinforcement Learning for
  Quadrupedal Jumping" -- https://arxiv.org/abs/2401.16337
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .skill_policy import LocomotionCommand, Skill


# ---------------------------------------------------------------------------
# Cat-tree platform geometry (world frame).
# Body at (-1.5, -1.5, 0.0). Top-surface z = geom_pos_z + half_height.
# ---------------------------------------------------------------------------

CAT_TREE_BODY_XY: tuple[float, float] = (-1.5, -1.5)

# Each platform entry: world_xyz of its top surface, plus approach geometry.
PLATFORMS: list[dict] = [
    {
        "name": "lvl1",
        "world_xyz": (-1.5 + 0.18, -1.5, 0.30 + 0.025),
        "approach_xy": (-0.72, -1.5),
        "approach_yaw": math.pi,
    },
    {
        "name": "lvl2",
        "world_xyz": (-1.5 - 0.18, -1.5, 0.65 + 0.025),
        "approach_xy": (-0.72, -1.5),
        "approach_yaw": math.pi,
    },
    {
        "name": "lvl3",
        "world_xyz": (-1.5, -1.5, 1.00 + 0.025),
        "approach_xy": (-0.72, -1.5),
        "approach_yaw": math.pi,
    },
]

DEFAULT_PLATFORM_IDX = 0

# Phase timing knobs.
APPROACH_GOAL_TOL: float = 0.15   # m -- approach done within this radius
ALIGN_GOAL_TOL: float = 0.15      # rad -- aligned within this heading error
CROUCH_DURATION_S: float = 0.8    # s -- crouch-charge duration
AIRBORNE_DURATION_S: float = 1.2  # s -- wait for physics arc to complete


def _wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class JumpPhase(enum.IntEnum):
    APPROACH = 0
    ALIGN = 1
    CROUCH = 2
    LAUNCH = 3
    AIRBORNE = 4
    PERCH = 5


@dataclass(frozen=True)
class JumpToGoal:
    """World-frame target surface point to land on."""

    surface_xyz: tuple[float, float, float]


class JumpTo(Skill):
    """Multi-phase scripted jump onto a cat-tree platform.

    Phases: APPROACH -> ALIGN -> CROUCH -> LAUNCH -> AIRBORNE -> PERCH.

    BrainEnv watches `skill.launch_requested` and calls
    `PhysicsCat.apply_launch_impulse(mj_data, target_xyz)` on the one
    LAUNCH step, then calls `skill.advance_to_airborne()` to continue.
    """

    name: str = "jump_to"

    def __init__(self) -> None:
        self._phase: JumpPhase = JumpPhase.APPROACH
        self._phase_timer: float = 0.0
        self._target: dict = PLATFORMS[DEFAULT_PLATFORM_IDX]

    # ------------------------------------------------------------------ #
    # Properties checked by BrainEnv
    # ------------------------------------------------------------------ #

    @property
    def launch_requested(self) -> bool:
        """True on the one LAUNCH step -- BrainEnv injects the impulse."""
        return self._phase == JumpPhase.LAUNCH

    @property
    def target_xyz(self) -> tuple[float, float, float]:
        return tuple(self._target["world_xyz"])  # type: ignore[return-value]

    def advance_to_airborne(self) -> None:
        """Called by BrainEnv immediately after injecting the impulse."""
        self._phase = JumpPhase.AIRBORNE
        self._phase_timer = 0.0

    # ------------------------------------------------------------------ #
    # Skill interface
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        self._phase = JumpPhase.APPROACH
        self._phase_timer = 0.0

    def step(self, obs: dict[str, Any], goal: Any) -> LocomotionCommand:
        dt = float(obs.get("dt", 0.05))

        root_xy = np.asarray(obs["root_xy"], dtype=np.float32)
        root_yaw = float(obs["root_yaw"])

        # Resolve which platform to target from the goal.
        self._target = self._pick_target(goal)

        if self._phase == JumpPhase.APPROACH:
            return self._step_approach(root_xy, root_yaw)
        if self._phase == JumpPhase.ALIGN:
            return self._step_align(root_yaw)
        if self._phase == JumpPhase.CROUCH:
            return self._step_crouch(dt)
        if self._phase == JumpPhase.LAUNCH:
            # BrainEnv will intercept and inject the impulse this tick.
            # Lean forward as a visual cue.
            return LocomotionCommand(vx=0.3, gait="leap")
        if self._phase == JumpPhase.AIRBORNE:
            self._phase_timer += dt
            if self._phase_timer >= AIRBORNE_DURATION_S:
                self._phase = JumpPhase.PERCH
            return LocomotionCommand(gait="stand")
        # PERCH
        return LocomotionCommand(gait="stand")

    def is_done(self, obs: dict[str, Any], goal: Any) -> bool:
        return self._phase == JumpPhase.PERCH

    # ------------------------------------------------------------------ #
    # Per-phase step helpers
    # ------------------------------------------------------------------ #

    def _step_approach(self, root_xy: np.ndarray, root_yaw: float) -> LocomotionCommand:
        approach_xy = np.asarray(self._target["approach_xy"], dtype=np.float32)
        dxy = approach_xy - root_xy
        distance = float(np.linalg.norm(dxy))

        if distance < APPROACH_GOAL_TOL:
            self._phase = JumpPhase.ALIGN
            self._phase_timer = 0.0
            return LocomotionCommand(gait="stand")

        desired_yaw = math.atan2(float(dxy[1]), float(dxy[0]))
        yaw_err = _wrap_angle(desired_yaw - root_yaw)
        yaw_rate = max(-1.5, min(1.5, 2.0 * yaw_err))
        speed = 0.5 * min(1.0, distance / 0.5)
        if abs(yaw_err) > math.radians(45.0):
            speed = 0.0
        vx_body = speed * math.cos(yaw_err)
        vy_body = speed * math.sin(yaw_err)
        return LocomotionCommand(vx=vx_body, vy=vy_body, yaw_rate=yaw_rate, gait="trot")

    def _step_align(self, root_yaw: float) -> LocomotionCommand:
        desired_yaw = float(self._target["approach_yaw"])
        yaw_err = _wrap_angle(desired_yaw - root_yaw)
        if abs(yaw_err) < ALIGN_GOAL_TOL:
            self._phase = JumpPhase.CROUCH
            self._phase_timer = 0.0
            return LocomotionCommand(gait="stand")
        yaw_rate = max(-1.5, min(1.5, 2.5 * yaw_err))
        return LocomotionCommand(yaw_rate=yaw_rate, gait="trot")

    def _step_crouch(self, dt: float) -> LocomotionCommand:
        self._phase_timer += dt
        if self._phase_timer >= CROUCH_DURATION_S:
            self._phase = JumpPhase.LAUNCH
            self._phase_timer = 0.0
            return LocomotionCommand(gait="leap")
        frac = self._phase_timer / CROUCH_DURATION_S
        body_height = 0.30 - frac * 0.12  # drop 0.30 -> 0.18 m during charge
        return LocomotionCommand(body_height=body_height, gait="stand")

    # ------------------------------------------------------------------ #
    # Goal helper
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pick_target(goal: Any) -> dict:
        """Pick nearest hardcoded platform to the given JumpToGoal, or default."""
        if isinstance(goal, JumpToGoal):
            gx, gy, gz = goal.surface_xyz
            best = PLATFORMS[DEFAULT_PLATFORM_IDX]
            best_d = float("inf")
            for p in PLATFORMS:
                px, py, pz = p["world_xyz"]
                d = math.hypot(px - gx, py - gy) + abs(pz - gz)
                if d < best_d:
                    best_d = d
                    best = p
            return best
        return PLATFORMS[DEFAULT_PLATFORM_IDX]
