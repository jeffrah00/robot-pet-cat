"""Tests for behavioral-attractor module (src/robot_pet_cat/brain/attractor.py)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from robot_pet_cat.brain.attractor import (
    MODE_SKILLS,
    Attractor,
    ModePolicy,
    ModePolicyConfig,
    _mode_bias_from_mood,
)


# --------------------------------------------------------------------------- #
# A tiny stub Mood so attractor tests don't depend on brain.mood internals
# --------------------------------------------------------------------------- #


@dataclass
class _StubMood:
    cur: float = 0.5
    com: float = 0.5
    play: float = 0.5

    def weights(self):
        return self.cur, self.com, self.play


# --------------------------------------------------------------------------- #
# Static structure
# --------------------------------------------------------------------------- #


class TestEnumAndTable:
    def test_six_modes_defined(self) -> None:
        assert len(list(Attractor)) == 6
        names = {m.value for m in Attractor}
        assert names == {
            "resting", "observing", "stalking", "playing", "grooming", "exploring",
        }

    def test_mode_skills_covers_every_mode(self) -> None:
        for mode in Attractor:
            assert mode in MODE_SKILLS, f"{mode} missing from MODE_SKILLS"
            assert len(MODE_SKILLS[mode]) > 0, f"{mode} has empty skill set"

    def test_every_mode_skill_is_a_real_skill(self) -> None:
        from robot_pet_cat.skills.skill_registry import SKILL_NAMES

        valid = set(SKILL_NAMES)
        for mode, skills in MODE_SKILLS.items():
            for s in skills:
                assert s in valid, f"mode {mode} references unknown skill {s!r}"


# --------------------------------------------------------------------------- #
# Mood-bias mapping
# --------------------------------------------------------------------------- #


class TestModeBias:
    def test_sleepy_comfort_mood_favors_resting(self) -> None:
        biases = _mode_bias_from_mood(curiosity_w=0.1, comfort_w=1.0, play_w=0.0)
        # RESTING should be highest given full comfort, no play, low curiosity.
        top = max(biases, key=biases.get)
        assert top == Attractor.RESTING

    def test_playful_mood_favors_playing(self) -> None:
        biases = _mode_bias_from_mood(curiosity_w=0.1, comfort_w=0.1, play_w=1.0)
        top = max(biases, key=biases.get)
        assert top == Attractor.PLAYING

    def test_curious_mood_favors_exploring(self) -> None:
        biases = _mode_bias_from_mood(curiosity_w=1.0, comfort_w=0.1, play_w=0.0)
        top = max(biases, key=biases.get)
        assert top == Attractor.EXPLORING


# --------------------------------------------------------------------------- #
# ModePolicy lifecycle
# --------------------------------------------------------------------------- #


class TestModePolicyDwell:
    def test_initial_mode_returned_until_dwell_elapses(self) -> None:
        cfg = ModePolicyConfig(min_dwell_s=2.0, decision_period_s=0.5,
                               initial_mode=Attractor.STALKING)
        mp = ModePolicy(cfg)
        mood = _StubMood(cur=0.0, com=1.0, play=0.0)  # would prefer RESTING
        for _ in range(3):  # 3 * 0.5s = 1.5s, still inside dwell
            assert mp.tick(0.5, mood) == Attractor.STALKING
        # After 4 ticks (2s) dwell is satisfied; transition becomes possible.
        mp.tick(0.5, mood)
        # The next tick beyond dwell + decision boundary may switch.
        for _ in range(8):
            mp.tick(0.5, mood)
        # With strong comfort bias and many ticks, we expect a transition out.
        assert mp.current_mode != Attractor.STALKING

    def test_decision_period_gate(self) -> None:
        cfg = ModePolicyConfig(min_dwell_s=0.0, decision_period_s=5.0,
                               initial_mode=Attractor.RESTING)
        mp = ModePolicy(cfg)
        mood = _StubMood()
        # Lots of small ticks below decision_period -> never reconsiders.
        for _ in range(20):
            assert mp.tick(0.1, mood) == Attractor.RESTING

    def test_force_mode_resets_timers(self) -> None:
        cfg = ModePolicyConfig(min_dwell_s=10.0)
        mp = ModePolicy(cfg)
        mp.tick(5.0, _StubMood())
        assert mp.time_in_mode == 5.0
        mp.force_mode(Attractor.PLAYING)
        assert mp.current_mode == Attractor.PLAYING
        assert mp.time_in_mode == 0.0
        assert mp.time_since_decision == 0.0


# --------------------------------------------------------------------------- #
# Action mask
# --------------------------------------------------------------------------- #


class TestAllowedActionMask:
    def test_hold_always_allowed(self) -> None:
        from robot_pet_cat.skills.skill_registry import SKILL_NAMES

        for mode in Attractor:
            mp = ModePolicy(ModePolicyConfig(initial_mode=mode))
            mask = mp.allowed_action_mask(1 + len(SKILL_NAMES), SKILL_NAMES)
            assert mask[0] is np.True_ or bool(mask[0]) is True

    def test_mask_matches_mode_skills(self) -> None:
        from robot_pet_cat.skills.skill_registry import SKILL_NAMES

        for mode in Attractor:
            mp = ModePolicy(ModePolicyConfig(initial_mode=mode))
            mask = mp.allowed_action_mask(1 + len(SKILL_NAMES), SKILL_NAMES)
            allowed_set = set(MODE_SKILLS[mode])
            for i, name in enumerate(SKILL_NAMES):
                expected = name in allowed_set
                assert bool(mask[1 + i]) == expected, (
                    f"mode {mode} skill {name}: mask[{1+i}]={bool(mask[1+i])} expected {expected}"
                )

    def test_rejects_undersized_mask(self) -> None:
        from robot_pet_cat.skills.skill_registry import SKILL_NAMES

        mp = ModePolicy()
        with pytest.raises(ValueError):
            mp.allowed_action_mask(2, SKILL_NAMES)  # too small for hold + 8 skills
