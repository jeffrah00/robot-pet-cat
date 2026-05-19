"""jump_to(surface_xyz) — ballistic jump onto an elevated surface.

This is the one Tier-2 skill we expect to actually train end-to-end with RL
(per ROADMAP and the 2026-05-19 design conversation). The Atanassov et al.
2024 curriculum-based jumping recipe trains this in ~1 day on a single A100.

Keyframe faking is not a real option here: a jump-to is a dynamic skill with
strict timing on push-off, mid-air orientation, and landing absorption, and
the target height varies per call. Hand-tuning a keyframe sequence for every
possible target would be miserable.

This file is the *interface* the brain dispatches against. The actual learned
policy is loaded at runtime from a checkpoint produced by a separate trainer
(not in scope tonight). For now `step` raises NotImplementedError so any
accidental use during brain prototyping fails loudly rather than silently
hanging.

References:
- Atanassov et al. 2024, "Curriculum-Based Reinforcement Learning for
  Quadrupedal Jumping" — https://arxiv.org/abs/2401.16337
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .skill_policy import LocomotionCommand, Skill


@dataclass(frozen=True)
class JumpToGoal:
    """World-frame target surface point to land on."""

    surface_xyz: tuple[float, float, float]


class JumpTo(Skill):
    """Ballistic jump onto an elevated target. RL-trained, not scaffolded yet."""

    name: str = "jump_to"

    def step(self, obs: dict[str, Any], goal: Any) -> LocomotionCommand:
        raise NotImplementedError(
            "JumpTo needs a learned policy. Train one via scripts/train_jump_to.py "
            "(not written) using the Atanassov 2024 curriculum, then load the "
            "checkpoint in __init__."
        )

    def is_done(self, obs: dict[str, Any], goal: Any) -> bool:
        raise NotImplementedError("JumpTo needs a learned policy (see step).")
