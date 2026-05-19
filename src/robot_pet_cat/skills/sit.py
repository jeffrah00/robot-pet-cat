"""sit — static low-haunches pose, held until the brain switches skills.

Per ROADMAP Phase 3 and 2026-05-19 design conversation: most static-pose skills
are NOT trained policies. They're keyframe targets executed by PD control on
the motor cortex. The skill's job is to emit the right LocomotionCommand
continuously until the brain commands a switch.

The actual posture (haunches tucked, front legs propped, body height ~22 cm)
will be tuned against MuJoCo visual inspection. The numbers here are starting
guesses — read them as placeholders, not as final values.
"""

from __future__ import annotations

from typing import Any

from .skill_policy import LocomotionCommand, Skill


# Empirical placeholders. Tune against MuJoCo render until it visually reads
# as a sitting cat.
SIT_BODY_HEIGHT_M = 0.18  # below standing (~0.30) but above lying (~0.08)


class Sit(Skill):
    """Hold a sitting pose. No goal needed (the goal is just 'be sitting')."""

    name: str = "sit"

    def step(self, obs: dict[str, Any], goal: Any) -> LocomotionCommand:
        return LocomotionCommand(
            vx=0.0,
            vy=0.0,
            yaw_rate=0.0,
            gait="stand",
            body_height=SIT_BODY_HEIGHT_M,
        )

    def is_done(self, obs: dict[str, Any], goal: Any) -> bool:
        # Sitting is a hold pose — the brain decides when to exit. Never
        # spontaneously "done" from the skill's own perspective.
        return False
