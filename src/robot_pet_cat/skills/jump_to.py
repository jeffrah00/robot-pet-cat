"""jump_to(surface_xyz) -- multi-phase scripted jump onto an elevated surface.

State machine:
  APPROACH  -- walk toward the platform approach point
  ALIGN     -- rotate in place to face the platform
  CROUCH    -- lower body for CROUCH_DURATION_S (charge phase)
  LAUNCH    -- joint trajectory: legs extend (push-off pose, ~0.3s)
  AIRBORNE  -- joint trajectory: legs tucked (mid-arc pose, ~1.2s)
  LAND      -- joint trajectory: legs bent deep (shock absorb, ~0.5s)
  PERCH     -- landed; skill is done

JOINT-TRAJECTORY IMPLEMENTATION (no physics impulse):
  Phases LAUNCH/AIRBORNE/LAND use joint_targets rather than a ballistic
  impulse. The robot stays on the ground but goes through the visual
  motions of a jump -- stable for showcase purposes. Full RL training
  (Atanassov 2024 curriculum) is the Phase 3 goal.

Platform geometry is hardcoded from data/go2_scenes/living_room_with_go2.xml:
  <body name="cat_tree" pos="-1.5 -1.5 0.0">
    <geom name="cattree_lvl1" pos=" 0.18 0 0.30" .../>
    <geom name="cattree_lvl2" pos="-0.18 0 0.65" .../>
    <geom name="cattree_lvl3" pos=" 0.00 0 1.00" .../>
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
# ---------------------------------------------------------------------------

CAT_TREE_BODY_XY: tuple[float, float] = (-1.5, -1.5)

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
APPROACH_GOAL_TOL: float = 0.15   # m
ALIGN_GOAL_TOL: float = 0.15      # rad
CROUCH_DURATION_S: float = 0.8    # s -- charge phase
LAUNCH_DURATION_S: float = 0.30   # s -- push-off pose hold
AIRBORNE_DURATION_S: float = 1.2  # s -- mid-arc tuck hold
LAND_DURATION_S: float = 0.50     # s -- landing absorb hold

# ---------------------------------------------------------------------------
# Joint trajectory poses for the fake-jump sequence.
# Joint order: FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
#              RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf
# ---------------------------------------------------------------------------

# Push-off: all legs extended (lower than stand, pushing into ground).
LAUNCH_TARGETS = np.array(
    [
        -0.10,  0.50, -0.80,   # FL: extended, push-off
        +0.10,  0.50, -0.80,   # FR: extended
        -0.10,  0.50, -0.80,   # RL: extended, push-off
        +0.10,  0.50, -0.80,   # RR: extended
    ],
    dtype=np.float32,
)

# Mid-arc: legs tucked under the body.
AIRBORNE_TARGETS = np.array(
    [
        -0.15,  0.80, -1.80,   # FL: front tucked
        +0.15,  0.80, -1.80,   # FR: front tucked
        -0.10,  1.20, -2.20,   # RL: rear pulled up
        +0.10,  1.20, -2.20,   # RR: rear pulled up
    ],
    dtype=np.float32,
)

# Landing absorb: all legs bent deep to absorb impact.
LAND_TARGETS = np.array(
    [
        -0.10,  1.00, -2.00,   # FL: shock absorb
        +0.10,  1.00, -2.00,   # FR: shock absorb
        -0.10,  1.10, -2.10,   # RL: shock absorb
        +0.10,  1.10, -2.10,   # RR: shock absorb
    ],
    dtype=np.float32,
)


def _wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class JumpPhase(enum.IntEnum):
    APPROACH = 0
    ALIGN = 1
    CROUCH = 2
    LAUNCH = 3
    AIRBORNE = 4
    LAND = 5
    PERCH = 6


@dataclass(frozen=True)
class JumpToGoal:
    """World-frame target surface point to land on."""

    surface_xyz: tuple[float, float, float]


class JumpTo(Skill):
    """Multi-phase scripted jump sequence using joint trajectories (no impulse).

    Phases: APPROACH -> ALIGN -> CROUCH -> LAUNCH -> AIRBORNE -> LAND -> PERCH.

    No physics impulse is applied. The robot goes through the visual motions
    of a jump while staying stable on the ground.
    """

    name: str = "jump_to"

    def __init__(self) -> None:
        self._phase: JumpPhase = JumpPhase.APPROACH
        self._phase_timer: float = 0.0
        self._target: dict = PLATFORMS[DEFAULT_PLATFORM_IDX]

    # ------------------------------------------------------------------ #
    # Properties kept for backward compatibility with BrainEnv.
    # launch_requested always returns False -- no impulse will be applied.
    # ------------------------------------------------------------------ #

    @property
    def launch_requested(self) -> bool:
        """Always False -- impulse-based jump replaced by joint trajectory."""
        return False

    @property
    def target_xyz(self) -> tuple[float, float, float]:
        return tuple(self._target["world_xyz"])  # type: ignore[return-value]

    def advance_to_airborne(self) -> None:
        """No-op -- auto-advance handled internally by phase timer."""
        pass

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

        self._target = self._pick_target(goal)

        if self._phase == JumpPhase.APPROACH:
            return self._step_approach(root_xy, root_yaw)
        if self._phase == JumpPhase.ALIGN:
            return self._step_align(root_yaw)
        if self._phase == JumpPhase.CROUCH:
            return self._step_crouch(dt)
        if self._phase == JumpPhase.LAUNCH:
            self._phase_timer += dt
            if self._phase_timer >= LAUNCH_DURATION_S:
                self._phase = JumpPhase.AIRBORNE
                self._phase_timer = 0.0
            return LocomotionCommand(
                vx=0.0, vy=0.0, yaw_rate=0.0, gait="stand",
                body_height=0.28,
                joint_targets=LAUNCH_TARGETS,
            )
        if self._phase == JumpPhase.AIRBORNE:
            self._phase_timer += dt
            if self._phase_timer >= AIRBORNE_DURATION_S:
                self._phase = JumpPhase.LAND
                self._phase_timer = 0.0
            return LocomotionCommand(
                vx=0.0, vy=0.0, yaw_rate=0.0, gait="stand",
                joint_targets=AIRBORNE_TARGETS,
            )
        if self._phase == JumpPhase.LAND:
            self._phase_timer += dt
            if self._phase_timer >= LAND_DURATION_S:
                self._phase = JumpPhase.PERCH
            return LocomotionCommand(
                vx=0.0, vy=0.0, yaw_rate=0.0, gait="stand",
                joint_targets=LAND_TARGETS,
            )
        # PERCH -- stand normally
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
            return LocomotionCommand(gait="stand")
        frac = self._phase_timer / CROUCH_DURATION_S
        body_height = 0.30 - frac * 0.10
        return LocomotionCommand(body_height=body_height, gait="stand")

    @staticmethod
    def _pick_target(goal: Any) -> dict:
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
