"""hind_sit -- haunches-down sitting transition driven by the get_up v4b
model_13000.pt checkpoint (packaged as models/hind_sit_v1.pt).

Unlike the static-pose skills (sit, lie_down, stretch) which emit a constant
LocomotionCommand for the walker policy to interpret, hind_sit hands control
off to its own trained policy. The mechanism: the skill emits a
LocomotionCommand with gait="hind_sit", and PhysicsCat detects that gait and
swaps the walker policy for the hind_sit policy for those substeps.

The trained policy was the get_up v4b model at iter 13000 -- Jeff observed
that the policy reliably settles into a "front legs propped, hind legs
folded under" sit at that checkpoint, which is the desired behavior.
The checkpoint lives at /workspace/robot-pet-cat/models/hind_sit_v1.pt.

Duration:
    The brain decides when to exit hind_sit (is_done always returns False).
    The skill internally counts steps so callers can introspect how long
    the cat has been transitioning into / holding the sit.
"""

from __future__ import annotations

from typing import Any

from .skill_policy import LocomotionCommand, Skill


# How many step() calls before we consider the transition complete.
# At brain dt = 0.05 s, 60 steps = 3 s -- the policy settles into its
# terminal sit pose well within that window.
HIND_SIT_TRANSITION_STEPS = 60


class HindSit(Skill):
    """Hand off to the hind_sit_v1.pt policy until the brain switches skills."""

    name: str = "hind_sit"

    def __init__(self) -> None:
        self._step_count = 0

    def reset(self) -> None:
        """Called by the brain when transitioning INTO hind_sit from a
        different skill. Zeroes the step counter so the transition window
        starts fresh."""
        self._step_count = 0

    @property
    def step_count(self) -> int:
        """How many decision ticks the cat has been in hind_sit. Read-only,
        for introspection / debugging."""
        return self._step_count

    @property
    def is_holding(self) -> bool:
        """True once we're past the transition window -- the policy has had
        enough time to converge into its terminal sit pose."""
        return self._step_count >= HIND_SIT_TRANSITION_STEPS

    def step(self, obs: dict[str, Any], goal: Any) -> LocomotionCommand:
        """Emit gait='hind_sit'. PhysicsCat sees this and routes substeps
        through the hind_sit policy. After the transition window the
        policy is settled, so emitting the same gait keeps the cat in
        the sit (the policy is by then producing near-zero corrections).
        """
        self._step_count += 1
        return LocomotionCommand(
            vx=0.0,
            vy=0.0,
            yaw_rate=0.0,
            gait="hind_sit",
        )

    def is_done(self, obs: dict[str, Any], goal: Any) -> bool:
        # Like sit / lie_down, hind_sit is a hold posture. The brain decides
        # when to exit by selecting a different skill.
        return False
