"""lie_down — belly-on-floor rest pose, held until the brain switches skills.

Same pattern as sit but lower. The Go2 firmware has a built-in lie-down command
(joint trajectory baked into the controller), so this stub mostly exists so the
brain has a slot to dispatch into. The actual joint targets at integration time
should match the firmware's choice, not invent a new one.
"""

from __future__ import annotations

from typing import Any

from .skill_policy import LocomotionCommand, Skill


# Empirical placeholder. The Go2 lying pose puts the trunk ~8 cm above the
# floor. Tune at integration time.
LIE_DOWN_BODY_HEIGHT_M = 0.08


class LieDown(Skill):
    """Belly-down rest pose. Held until brain switches."""

    name: str = "lie_down"

    def step(self, obs: dict[str, Any], goal: Any) -> LocomotionCommand:
        return LocomotionCommand(
            vx=0.0,
            vy=0.0,
            yaw_rate=0.0,
            gait="stand",
            body_height=LIE_DOWN_BODY_HEIGHT_M,
        )

    def is_done(self, obs: dict[str, Any], goal: Any) -> bool:
        return False
