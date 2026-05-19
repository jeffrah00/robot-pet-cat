"""stretch — long forward leg extension, transient (cat-like wake-up stretch).

Unlike sit/lie_down, stretch has a *natural duration*: a cat stretches for ~2 s
then relaxes back to whatever pose was before. The skill emits a transient
extended pose for `duration_s`, then reports done so the brain switches away.

Implementation note: the actual joint trajectory (front legs extended forward,
shoulders dropped, hindquarters up) requires per-joint targets that the
LocomotionCommand vocabulary doesn't currently express. For now we fake it with
a slight body-height bump and let the brain switch out after duration. A future
revision should add a per-skill joint-target field to LocomotionCommand or
introduce a richer command object.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .skill_policy import LocomotionCommand, Skill


# Empirical placeholder. Real value comes from observing the cat stretch clip
# (`stretch_table.npz`) at integration time.
STRETCH_BODY_HEIGHT_M = 0.32  # slightly above standing


@dataclass
class StretchGoal:
    """How long to hold the stretch, in sim seconds."""

    duration_s: float = 2.0


class Stretch(Skill):
    """Hold an extended pose for a fixed duration, then signal done."""

    name: str = "stretch"

    def __init__(self) -> None:
        # `_start_t` is set on the first step() call after the skill is selected.
        # The brain (or a wrapping dispatcher) should call .reset() when handing
        # control to this skill to clear the timer; for now we lazy-init on first
        # step and reset whenever is_done() returns True.
        self._start_t: float | None = None

    def reset(self) -> None:
        self._start_t = None

    def step(self, obs: dict[str, Any], goal: Any) -> LocomotionCommand:
        if self._start_t is None:
            self._start_t = self._now(obs)
        return LocomotionCommand(
            vx=0.0,
            vy=0.0,
            yaw_rate=0.0,
            gait="stand",
            body_height=STRETCH_BODY_HEIGHT_M,
        )

    def is_done(self, obs: dict[str, Any], goal: Any) -> bool:
        if self._start_t is None:
            return False
        duration = self._unpack_goal(goal)
        return (self._now(obs) - self._start_t) >= duration

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _unpack_goal(goal: Any) -> float:
        if isinstance(goal, StretchGoal):
            return float(goal.duration_s)
        if goal is None:
            return 2.0
        return float(goal)

    @staticmethod
    def _now(obs: dict[str, Any]) -> float:
        """Sim time if available in obs, else wall-clock. The brain populates
        obs['t_sim'] when running inside the env; tests pass None."""
        t = obs.get("t_sim")
        return float(t) if t is not None else time.monotonic()
