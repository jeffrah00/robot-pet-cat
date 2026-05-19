"""Registry of mid-level skills the brain can dispatch over.

The brain emits `(skill_id, goal)` actions. `SkillRegistry.dispatch(skill_id)`
returns the matching Skill instance; the brain then calls `skill.step(obs, goal)`
to translate into a LocomotionCommand.

Adding a new skill is two lines: import the class, add an entry to `SKILLS`.
Skill IDs are stable strings so the brain's categorical head can be saved and
reloaded without an integer-mapping drift bug. Order in `SKILLS` is also stable
so `SKILL_INDEX[name]` produces a fixed integer for the policy head.

Skills are stateful (Stretch and Swat carry a phase timer); the registry
returns the SAME instance across calls so state is preserved across decision
ticks. Callers should invoke `skill.reset()` (if defined) when the brain
transitions INTO that skill from a different one.
"""

from __future__ import annotations

from typing import Any

from .crouch import Crouch
from .lie_down import LieDown
from .look_at import LookAt
from .sit import Sit
from .skill_policy import Skill
from .stretch import Stretch
from .swat import Swat
from .walk_to import WalkTo

# jump_to is included for the brain's categorical head but raises
# NotImplementedError at step time until a trained policy is loaded.
from .jump_to import JumpTo


def _build_skills() -> dict[str, Skill]:
    """Instantiate the canonical skill set. Stateful skills get exactly one
    instance shared across decision ticks; the brain is responsible for
    `.reset()` on transitions."""
    return {
        "walk_to": WalkTo(),
        "sit": Sit(),
        "lie_down": LieDown(),
        "stretch": Stretch(),
        "crouch": Crouch(),
        "look_at": LookAt(),
        "swat": Swat(),
        "jump_to": JumpTo(),
    }


# Stable insertion order — DON'T reorder; brain checkpoints encode integer
# indices that map back through SKILL_NAMES.
SKILL_NAMES: tuple[str, ...] = (
    "walk_to",
    "sit",
    "lie_down",
    "stretch",
    "crouch",
    "look_at",
    "swat",
    "jump_to",
)

SKILL_INDEX: dict[str, int] = {name: i for i, name in enumerate(SKILL_NAMES)}


class SkillRegistry:
    """Holds one shared instance of each skill plus name <-> index mappings."""

    def __init__(self) -> None:
        self._skills = _build_skills()
        missing = set(SKILL_NAMES) - set(self._skills)
        if missing:
            raise RuntimeError(f"skill registry missing entries: {sorted(missing)}")
        extra = set(self._skills) - set(SKILL_NAMES)
        if extra:
            raise RuntimeError(
                f"skill registry has entries not in SKILL_NAMES: {sorted(extra)}. "
                "Append the new names to SKILL_NAMES at the bottom (never reorder)."
            )

    @property
    def names(self) -> tuple[str, ...]:
        return SKILL_NAMES

    def __len__(self) -> int:
        return len(SKILL_NAMES)

    def __contains__(self, name: str) -> bool:
        return name in self._skills

    def get(self, name: str) -> Skill:
        if name not in self._skills:
            raise KeyError(f"unknown skill {name!r}. known: {sorted(self._skills)}")
        return self._skills[name]

    def get_by_index(self, idx: int) -> Skill:
        if not 0 <= idx < len(SKILL_NAMES):
            raise IndexError(f"skill index {idx} out of range [0, {len(SKILL_NAMES)})")
        return self._skills[SKILL_NAMES[idx]]

    def name_of(self, idx: int) -> str:
        if not 0 <= idx < len(SKILL_NAMES):
            raise IndexError(f"skill index {idx} out of range [0, {len(SKILL_NAMES)})")
        return SKILL_NAMES[idx]

    def index_of(self, name: str) -> int:
        if name not in SKILL_INDEX:
            raise KeyError(f"unknown skill {name!r}")
        return SKILL_INDEX[name]

    def reset(self, name: str) -> None:
        """Reset a skill's internal state. Called by the brain when transitioning
        INTO this skill from a different one. No-op if the skill has no reset()."""
        skill = self.get(name)
        reset = getattr(skill, "reset", None)
        if callable(reset):
            reset()

    def step(self, name: str, obs: dict[str, Any], goal: Any):
        """Convenience: dispatch a step call by skill name."""
        return self.get(name).step(obs, goal)

    def is_done(self, name: str, obs: dict[str, Any], goal: Any) -> bool:
        return self.get(name).is_done(obs, goal)
