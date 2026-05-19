"""Unit tests for the Phase 3 skill stubs and the skill registry."""

from __future__ import annotations

import math

import numpy as np
import pytest

from robot_pet_cat.skills.crouch import Crouch
from robot_pet_cat.skills.jump_to import JumpTo, JumpToGoal
from robot_pet_cat.skills.lie_down import LieDown
from robot_pet_cat.skills.look_at import LookAt, LookAtGoal
from robot_pet_cat.skills.sit import Sit
from robot_pet_cat.skills.skill_policy import LocomotionCommand
from robot_pet_cat.skills.skill_registry import (
    SKILL_INDEX,
    SKILL_NAMES,
    SkillRegistry,
)
from robot_pet_cat.skills.stretch import Stretch, StretchGoal
from robot_pet_cat.skills.swat import Swat, SwatGoal
from robot_pet_cat.skills.walk_to import WalkTo


def _obs(x: float = 0.0, y: float = 0.0, yaw: float = 0.0, t: float | None = 0.0) -> dict:
    o = {"root_xy": np.asarray([x, y], dtype=np.float32), "root_yaw": float(yaw)}
    if t is not None:
        o["t_sim"] = t
    return o


# --------------------------------------------------------------------------- #
# Static-pose skills
# --------------------------------------------------------------------------- #


class TestStaticPoseSkills:
    """Sit / LieDown / Crouch are hold-pose skills: zero velocity, distinct
    body heights, never done from their own perspective."""

    @pytest.mark.parametrize(
        "skill_cls,expected_height_order",
        [
            (LieDown, "lowest"),
            (Crouch, "mid_low"),
            (Sit, "mid_high"),
        ],
    )
    def test_static_pose_zero_velocity(self, skill_cls, expected_height_order) -> None:
        out = skill_cls().step(_obs(), None)
        assert isinstance(out, LocomotionCommand)
        assert out.vx == 0.0 and out.vy == 0.0 and out.yaw_rate == 0.0

    def test_height_ordering_lie_crouch_sit(self) -> None:
        """Cat anatomy: lying < crouching < sitting < standing."""
        lie = LieDown().step(_obs(), None).body_height
        crouch = Crouch().step(_obs(), None).body_height
        sit = Sit().step(_obs(), None).body_height
        assert lie < crouch < sit < 0.30  # 0.30 is the LocomotionCommand default standing

    def test_is_done_returns_false(self) -> None:
        """Static pose skills never spontaneously exit — brain decides."""
        for skill in (Sit(), LieDown(), Crouch()):
            assert skill.is_done(_obs(), None) is False


# --------------------------------------------------------------------------- #
# Stretch
# --------------------------------------------------------------------------- #


class TestStretch:
    def test_holds_then_completes(self) -> None:
        skill = Stretch()
        # First call seeds the timer at t=0.
        cmd0 = skill.step(_obs(t=0.0), StretchGoal(duration_s=2.0))
        assert isinstance(cmd0, LocomotionCommand)
        assert skill.is_done(_obs(t=0.5), StretchGoal(duration_s=2.0)) is False
        # After the duration, is_done flips.
        assert skill.is_done(_obs(t=2.0), StretchGoal(duration_s=2.0)) is True
        assert skill.is_done(_obs(t=3.0), StretchGoal(duration_s=2.0)) is True

    def test_reset_clears_timer(self) -> None:
        skill = Stretch()
        skill.step(_obs(t=0.0), StretchGoal(duration_s=2.0))
        assert skill.is_done(_obs(t=3.0), StretchGoal(duration_s=2.0)) is True
        skill.reset()
        # After reset the timer is unset → is_done is False until next step seeds it.
        assert skill.is_done(_obs(t=3.0), StretchGoal(duration_s=2.0)) is False

    def test_accepts_plain_float_goal(self) -> None:
        skill = Stretch()
        skill.step(_obs(t=0.0), 1.0)
        assert skill.is_done(_obs(t=2.0), 1.0) is True


# --------------------------------------------------------------------------- #
# LookAt
# --------------------------------------------------------------------------- #


class TestLookAt:
    def test_target_in_front_zero_head_yaw(self) -> None:
        """Cat facing +x, target straight ahead at eye height: head_yaw≈0."""
        out = LookAt().step(_obs(yaw=0.0), LookAtGoal(point_xyz=(2.0, 0.0, 0.28)))
        assert out.head_yaw == pytest.approx(0.0, abs=1e-5)
        assert out.head_pitch == pytest.approx(0.0, abs=1e-5)

    def test_target_to_left_positive_head_yaw(self) -> None:
        out = LookAt().step(_obs(yaw=0.0), LookAtGoal(point_xyz=(0.0, 2.0, 0.28)))
        assert out.head_yaw > 0  # turn head left
        # Capped at max_head_yaw (default 60 deg) since +y target needs 90 deg turn.
        assert out.head_yaw == pytest.approx(math.radians(60.0), abs=1e-5)

    def test_target_above_positive_pitch(self) -> None:
        """Target above eye height should produce positive pitch."""
        out = LookAt().step(_obs(yaw=0.0), LookAtGoal(point_xyz=(2.0, 0.0, 1.0)))
        assert out.head_pitch > 0

    def test_target_below_negative_pitch(self) -> None:
        out = LookAt().step(_obs(yaw=0.0), LookAtGoal(point_xyz=(2.0, 0.0, 0.0)))
        assert out.head_pitch < 0

    def test_rejects_non_3d_goal(self) -> None:
        with pytest.raises(ValueError, match="3-D"):
            LookAt().step(_obs(), (1.0, 2.0))


# --------------------------------------------------------------------------- #
# Swat
# --------------------------------------------------------------------------- #


class TestSwat:
    def test_far_from_target_walks(self) -> None:
        """When far from the target, swat delegates to walk_to (nonzero vx)."""
        skill = Swat()
        out = skill.step(_obs(yaw=0.0), SwatGoal(point_xyz=(2.0, 0.0, 0.1)))
        # walk_to facing forward at full speed: vx > 0.
        assert out.vx > 0

    def test_close_to_target_strikes(self) -> None:
        """When within paw range, emits the strike pose (zero velocity, crouched)."""
        skill = Swat()
        # Position the cat 0.1 m from the target (inside default APPROACH_RADIUS_M=0.25).
        out = skill.step(_obs(x=0.0, y=0.0, t=0.0), SwatGoal(point_xyz=(0.1, 0.0, 0.1)))
        assert out.vx == 0.0 and out.vy == 0.0
        assert out.body_height < 0.20  # crouched

    def test_strike_completes_after_duration(self) -> None:
        skill = Swat()
        # Place at target.
        skill.step(_obs(t=0.0), SwatGoal(point_xyz=(0.1, 0.0, 0.1)))
        assert skill.is_done(_obs(t=0.3), SwatGoal(point_xyz=(0.1, 0.0, 0.1))) is False
        assert skill.is_done(_obs(t=1.0), SwatGoal(point_xyz=(0.1, 0.0, 0.1))) is True


# --------------------------------------------------------------------------- #
# JumpTo (still a stub raising NotImplementedError)
# --------------------------------------------------------------------------- #


class TestJumpTo:
    def test_step_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError, match="learned policy"):
            JumpTo().step(_obs(), JumpToGoal(surface_xyz=(2.0, 0.0, 0.4)))

    def test_is_done_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            JumpTo().is_done(_obs(), JumpToGoal(surface_xyz=(2.0, 0.0, 0.4)))


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


class TestSkillRegistry:
    def test_all_names_present(self) -> None:
        reg = SkillRegistry()
        assert len(reg) == 8
        expected = {
            "walk_to",
            "sit",
            "lie_down",
            "stretch",
            "crouch",
            "look_at",
            "swat",
            "jump_to",
        }
        assert set(reg.names) == expected

    def test_name_index_bijection(self) -> None:
        reg = SkillRegistry()
        for name in reg.names:
            assert reg.name_of(reg.index_of(name)) == name
        for i in range(len(reg)):
            assert reg.index_of(reg.name_of(i)) == i

    def test_get_returns_skill_instance(self) -> None:
        reg = SkillRegistry()
        assert isinstance(reg.get("sit"), Sit)
        assert isinstance(reg.get("walk_to"), WalkTo)

    def test_unknown_skill_raises(self) -> None:
        reg = SkillRegistry()
        with pytest.raises(KeyError, match="unknown skill"):
            reg.get("purr")

    def test_index_out_of_range(self) -> None:
        reg = SkillRegistry()
        with pytest.raises(IndexError):
            reg.get_by_index(99)

    def test_reset_is_safe_for_stateless_skill(self) -> None:
        """Static-pose skills don't define reset(); registry.reset() should
        no-op rather than crash."""
        reg = SkillRegistry()
        reg.reset("sit")  # no exception

    def test_reset_clears_state_for_stateful_skill(self) -> None:
        reg = SkillRegistry()
        stretch = reg.get("stretch")
        assert isinstance(stretch, Stretch)
        stretch.step(_obs(t=0.0), StretchGoal(duration_s=2.0))
        assert stretch.is_done(_obs(t=3.0), StretchGoal(duration_s=2.0)) is True
        reg.reset("stretch")
        assert stretch.is_done(_obs(t=3.0), StretchGoal(duration_s=2.0)) is False

    def test_stable_skill_index_walk_to_is_zero(self) -> None:
        """SKILL_NAMES order is part of the public contract — checkpoints encode
        these integer indices. If you change this, you break loadability."""
        assert SKILL_INDEX["walk_to"] == 0
        # And the full ordering is fixed:
        assert SKILL_NAMES == (
            "walk_to",
            "sit",
            "lie_down",
            "stretch",
            "crouch",
            "look_at",
            "swat",
            "jump_to",
        )
