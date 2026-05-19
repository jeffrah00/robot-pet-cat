"""crouch — pre-pounce low-and-still pose.

Body drops, weight shifts forward, ready to launch. Held until the brain
switches (usually to a swat or jump_to immediately after). Distinct from sit
(higher and more relaxed) and lie_down (lower and resting).

The cat-behavior signature here is *stillness*: a crouching cat is almost
motionless, sometimes for many seconds, with only minor head/eye motion. The
stillness IS the skill.
"""

from __future__ import annotations

from typing import Any

from .skill_policy import LocomotionCommand, Skill


# Empirical placeholder. Below standing (0.30), above lying (0.08), and below
# sit (0.18). Visually distinguishable from sit by being more horizontal.
CROUCH_BODY_HEIGHT_M = 0.14


class Crouch(Skill):
    """Drop into a pre-pounce crouch and hold. Brain decides exit."""

    name: str = "crouch"

    def step(self, obs: dict[str, Any], goal: Any) -> LocomotionCommand:
        return LocomotionCommand(
            vx=0.0,
            vy=0.0,
            yaw_rate=0.0,
            gait="stand",
            body_height=CROUCH_BODY_HEIGHT_M,
        )

    def is_done(self, obs: dict[str, Any], goal: Any) -> bool:
        return False
