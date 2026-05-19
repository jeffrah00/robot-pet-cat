"""Behavioral attractors — v2 architectural layer between mood and skills.

Per docs/brain_design_v2.md: cats don't pick a skill every two seconds. They
enter persistent *modes* (resting, observing, stalking, playing, grooming,
exploring) that last tens of seconds to minutes, and within a mode they pick
among a small, mode-coherent subset of skills.

This module exposes:

  - `Attractor` enum: the six behavioral modes
  - `MODE_SKILLS` table: mode -> tuple of permissible skill names
  - `ModePolicy`: a low-frequency mode-transition policy

`ModePolicy.tick(dt, mood)` is called every step but only considers a
transition every `decision_period_s` AND only after the current mode has been
held for at least `min_dwell_s`. Transitions are sampled from a softmax over
mood-biased per-mode scores. This keeps mode-level behavior coherent
(no thrashing) while still being mood-responsive.

The action_mask helper lets BrainEnv filter the body-policy's action space
down to the active mode's permissible skills (plus the always-allowed hold).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .mood import Mood


class Attractor(str, Enum):
    """The six high-level behavioral modes the brain switches between.

    The values are stable strings so we can checkpoint a current mode without
    worrying about integer mappings drifting.
    """

    RESTING = "resting"
    OBSERVING = "observing"
    STALKING = "stalking"
    PLAYING = "playing"
    GROOMING = "grooming"
    EXPLORING = "exploring"


# Per-mode permissible skill subsets. Within a mode, the body policy picks
# from this subset; transitioning between modes is the mode policy's job.
#
# Mapping rationale:
#   resting:   the cat is settled. Sitting, lying, sometimes idle gazing.
#   observing: alert but not pursuing. Looking around, maybe wandering.
#   stalking:  pre-pounce. Crouched, possibly creeping, locked on something.
#   playing:   active engagement with a play_target. Swatting, chasing, leaping.
#   grooming:  static/quiet. Same skill set as resting until real grooming
#              gestures land; mood is what distinguishes it from resting.
#   exploring: locomotion-dominant, low-engagement. Walking around.
#
# `walk_to` and `look_at` appear in many modes because they're general-purpose
# competence skills, not character-defining.
MODE_SKILLS: dict[Attractor, tuple[str, ...]] = {
    Attractor.RESTING: ("sit", "lie_down", "look_at"),
    Attractor.OBSERVING: ("look_at", "walk_to"),
    Attractor.STALKING: ("crouch", "walk_to", "look_at"),
    Attractor.PLAYING: ("swat", "walk_to", "jump_to"),
    Attractor.GROOMING: ("sit", "lie_down", "look_at"),
    Attractor.EXPLORING: ("walk_to", "look_at"),
}


# --------------------------------------------------------------------------- #
# Mode -> mood bias mapping
# --------------------------------------------------------------------------- #


def _mode_bias_from_mood(curiosity_w: float, comfort_w: float, play_w: float) -> dict[Attractor, float]:
    """Each mode's transition logit is a weighted blend of the three mood weights.

    Coefficients chosen so:
      - sleepy + comfortable mood => RESTING / GROOMING
      - alert curious mood        => OBSERVING / EXPLORING
      - playful mood              => PLAYING
      - mixed alert+playful       => STALKING

    These are starting values; calibrate against observed mode dwell-time
    statistics on real cats once we have telemetry.
    """
    return {
        Attractor.RESTING: 1.0 * comfort_w,
        Attractor.OBSERVING: 0.5 * curiosity_w + 0.2 * play_w,
        Attractor.STALKING: 0.5 * curiosity_w + 0.6 * play_w,
        Attractor.PLAYING: 1.0 * play_w,
        Attractor.GROOMING: 0.3 * comfort_w,
        Attractor.EXPLORING: 1.0 * curiosity_w,
    }


# --------------------------------------------------------------------------- #
# ModePolicy
# --------------------------------------------------------------------------- #


@dataclass
class ModePolicyConfig:
    """Tuning knobs for the mode-switching policy."""

    # Minimum dwell time in any mode before a transition is considered.
    # Below ~5 s the cat looks twitchy; above ~30 s it looks stuck.
    min_dwell_s: float = 8.0
    # How often (in sim seconds) to *evaluate* whether to transition.
    # Fast evaluation + min_dwell gating gives quick reactions when the cat
    # has been in a mode long enough, without busy-polling.
    decision_period_s: float = 4.0
    # Softmax temperature on the mood-bias scores. Higher => more uniform
    # (less mood-driven); lower => sharper, more deterministic.
    transition_temperature: float = 0.8
    initial_mode: Attractor = Attractor.RESTING
    rng_seed: int = 0


@dataclass
class ModePolicy:
    """Low-frequency mode switcher biased by mood.

    The brain instantiates one of these and calls `tick(dt, mood)` every
    BrainEnv step. The policy returns the active mode (possibly unchanged).
    Use `allowed_skills()` to read the current mode's permissible skill names,
    or `allowed_action_mask()` for the boolean mask BrainEnv consumes.
    """

    cfg: ModePolicyConfig = field(default_factory=ModePolicyConfig)
    current_mode: Attractor = field(init=False)
    time_in_mode: float = field(init=False, default=0.0)
    time_since_decision: float = field(init=False, default=0.0)
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self.current_mode = self.cfg.initial_mode
        self.time_in_mode = 0.0
        self.time_since_decision = 0.0
        self.rng = np.random.default_rng(self.cfg.rng_seed)

    def tick(self, dt: float, mood: "Mood") -> Attractor:
        self.time_in_mode += dt
        self.time_since_decision += dt

        # Two gates: evaluation cadence + minimum dwell time.
        if self.time_since_decision < self.cfg.decision_period_s:
            return self.current_mode
        self.time_since_decision = 0.0
        if self.time_in_mode < self.cfg.min_dwell_s:
            return self.current_mode

        cur_w, comf_w, play_w = mood.weights()
        biases = _mode_bias_from_mood(cur_w, comf_w, play_w)
        modes = list(biases.keys())
        scores = np.array([biases[m] for m in modes], dtype=np.float64)
        scaled = scores / max(self.cfg.transition_temperature, 1e-6)
        probs = np.exp(scaled - scaled.max())
        probs /= probs.sum()

        new_idx = int(self.rng.choice(len(modes), p=probs))
        new_mode = modes[new_idx]
        if new_mode != self.current_mode:
            self.current_mode = new_mode
            self.time_in_mode = 0.0
        return self.current_mode

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def allowed_skills(self) -> tuple[str, ...]:
        return MODE_SKILLS[self.current_mode]

    def allowed_action_mask(
        self, n_actions: int, skill_names: tuple[str, ...]
    ) -> np.ndarray:
        """Boolean mask of shape (n_actions,).

        Index 0 (HOLD) is always True — pause-as-default is allowed in every
        mode. Indices 1..N correspond to skill_names[0..N-1] and are True iff
        the skill is in the active mode's MODE_SKILLS entry.

        BrainEnv uses this to clip incoming actions: if a brain emits an
        action outside the mask, the env falls back to HOLD.
        """
        if n_actions < 1 + len(skill_names):
            raise ValueError(
                f"n_actions={n_actions} too small for {len(skill_names)} skills + hold"
            )
        mask = np.zeros(n_actions, dtype=bool)
        mask[0] = True  # hold always allowed
        allowed = set(self.allowed_skills())
        for i, name in enumerate(skill_names):
            if name in allowed:
                mask[1 + i] = True
        return mask

    def force_mode(self, mode: Attractor) -> None:
        """Manually set the mode. Used by tests + by the brain when entering a
        scripted scenario. Resets time_in_mode so dwell gating works."""
        if not isinstance(mode, Attractor):
            mode = Attractor(mode)
        self.current_mode = mode
        self.time_in_mode = 0.0
        self.time_since_decision = 0.0
