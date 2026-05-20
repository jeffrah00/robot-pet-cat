"""Registry of mid-level skills the brain can dispatch over.

The brain emits `(skill_id, goal)` actions. `SkillRegistry.dispatch(skill_id)`
returns the matching Skill instance; the brain then calls `skill.step(obs, goal)`
to translate into a LocomotionCommand.

Adding a new skill is two lines: import the class, add an entry to `SKILLS`.
Skill IDs are stable strings so the brain's categorical head can be saved and
reloaded without an integer-mapping drift bug. Order in `SKILLS` is also stable
so `SKILL_INDEX[name]` produces a fixed integer for the policy head.

Skills are stateful (Stretch carries a phase timer); the registry returns the
SAME instance across calls so state is preserved across decision ticks.
Callers should invoke `skill.reset()` (if defined) when the brain transitions
INTO that skill from a different one.

NOTE: swat was removed 2026-05-20. The action space is now Discrete(8)
(HOLD + 7 skills). Brain checkpoints trained with swat (v4/v5) need retraining.
"""

from __future__ import annotations

from typing import Any

from .crouch import Crouch
from .lie_down import LieDown
from .look_at import LookAt
from .sit import Sit
from .skill_policy import Skill
from .stretch import Stretch
from .walk_to import WalkTo
from .jump_to import JumpTo
from .get_up import GetUp


def _build_skills() -> dict[str, Skill]:
    """Instantiate the canonical skill set."""
    return {
        "walk_to": WalkTo(),
        "sit": Sit(),
        "lie_down": LieDown(),
        "stretch": Stretch(),
        "crouch": Crouch(),
        "look_at": LookAt(),
        "jump_to": JumpTo(),
        "get_up": GetUp(),
    }


# Stable insertion order -- DON'T reorder; brain checkpoints encode integer
# indices that map back through SKILL_NAMES.
# swat removed 2026-05-20: Discrete(9) -> Discrete(8). Needs brain retrain.
SKILL_NAMES: tuple[str, ...] = (
    "walk_to",
    "sit",
    "lie_down",
    "stretch",
    "crouch",
    "look_at",
    "jump_to",
    "get_up",
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
        skill = self.get(name)
        reset = getattr(skill, "reset", None)
        if callable(reset):
            reset()

    def step(self, name: str, obs: dict[str, Any], goal: Any):
        return self.get(name).step(obs, goal)

    def is_done(self, name: str, obs: dict[str, Any], goal: Any) -> bool:
        return self.get(name).is_done(obs, goal)
