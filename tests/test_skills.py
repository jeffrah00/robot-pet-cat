"""Phase 3 — skills tests.

Unit-level: each skill's analytical/control logic should be exercised without
spinning up MuJoCo. The skill receives a pre-cooked obs dict and goal; we
assert on the LocomotionCommand it returns. End-to-end env tests live in
test_skills_integration.py (not yet written) and require the AMP checkpoint.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from robot_pet_cat.skills.skill_policy import LocomotionCommand
from robot_pet_cat.skills.walk_to import WalkTo, WalkToGoal, _wrap_angle


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _obs(x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> dict:
    """Minimal privileged-state obs the skill consumes."""
    return {"root_xy": np.asarray([x, y], dtype=np.float32), "root_yaw": float(yaw)}


# --------------------------------------------------------------------------- #
# WalkTo
# --------------------------------------------------------------------------- #


class TestWalkToCommand:
    """The body-frame velocity command should always point toward the goal."""

    def test_returns_locomotion_command(self) -> None:
        skill = WalkTo()
        out = skill.step(_obs(), WalkToGoal(target_xy=(2.0, 0.0)))
        assert isinstance(out, LocomotionCommand)
        assert out.gait == "trot"
        assert out.body_height == pytest.approx(0.30)

    def test_facing_target_drives_straight_forward(self) -> None:
        """Cat at origin facing +x, goal at (2, 0): vx > 0, vy ≈ 0, yaw_rate ≈ 0."""
        skill = WalkTo(max_speed=0.6)
        out = skill.step(_obs(yaw=0.0), WalkToGoal((2.0, 0.0)))
        assert out.vx == pytest.approx(0.6, abs=1e-6)
        assert out.vy == pytest.approx(0.0, abs=1e-6)
        assert out.yaw_rate == pytest.approx(0.0, abs=1e-6)

    def test_goal_to_side_triggers_yaw_and_kills_translation(self) -> None:
        """Cat at origin facing +x, goal at (0, 2): heading error = +pi/2, which
        is past facing_threshold, so forward speed should drop to zero while
        yaw_rate spins us toward the target."""
        skill = WalkTo()
        out = skill.step(_obs(yaw=0.0), WalkToGoal((0.0, 2.0)))
        assert out.vx == pytest.approx(0.0, abs=1e-6)
        assert out.vy == pytest.approx(0.0, abs=1e-6)
        assert out.yaw_rate > 0  # turn left (counterclockwise) toward +y
        assert out.yaw_rate <= skill.max_yaw_rate

    def test_slight_offset_keeps_translating(self) -> None:
        """30 deg offset is inside facing_threshold (45 deg). The cat should
        still move forward (body +x) but with a small lateral component and a
        nonzero yaw_rate trying to align."""
        skill = WalkTo()
        offset = math.radians(30.0)
        out = skill.step(
            _obs(yaw=0.0),
            WalkToGoal((math.cos(offset) * 2.0, math.sin(offset) * 2.0)),
        )
        assert out.vx > 0
        assert out.vy > 0
        # Speed is preserved on the body-frame velocity vector.
        speed = math.hypot(out.vx, out.vy)
        assert speed == pytest.approx(skill.max_speed, abs=1e-5)
        assert out.yaw_rate > 0

    def test_speed_ramps_down_inside_slow_radius(self) -> None:
        """At slow_radius / 2 from goal, speed should be half of max."""
        skill = WalkTo(max_speed=0.6, slow_radius=0.5)
        # Place target slightly more than goal_tolerance away so we're inside
        # the slow ramp but not yet "done". 0.25 m is half of slow_radius.
        out = skill.step(_obs(yaw=0.0), WalkToGoal((0.25, 0.0)))
        speed = math.hypot(out.vx, out.vy)
        assert speed == pytest.approx(0.3, abs=1e-5)

    def test_yaw_rate_is_clipped(self) -> None:
        """Goal directly behind: heading error magnitude ≈ pi, P-controller
        wants ≫ max_yaw_rate, output should be exactly max_yaw_rate."""
        skill = WalkTo(yaw_kp=10.0, max_yaw_rate=1.5)
        out = skill.step(_obs(yaw=0.0), WalkToGoal((-2.0, 0.0)))
        assert abs(out.yaw_rate) == pytest.approx(skill.max_yaw_rate, abs=1e-6)

    def test_zero_speed_at_goal(self) -> None:
        """Inside goal_tolerance: stand still."""
        skill = WalkTo(goal_tolerance=0.15)
        out = skill.step(_obs(0.0, 0.05, yaw=0.0), WalkToGoal((0.0, 0.0)))
        assert out.vx == 0.0
        assert out.vy == 0.0
        assert out.yaw_rate == 0.0
        assert out.gait == "stand"


class TestWalkToIsDone:
    def test_done_when_inside_tolerance(self) -> None:
        skill = WalkTo(goal_tolerance=0.15)
        assert skill.is_done(_obs(0.1, 0.0), WalkToGoal((0.0, 0.0))) is True

    def test_not_done_when_outside_tolerance(self) -> None:
        skill = WalkTo(goal_tolerance=0.15)
        assert skill.is_done(_obs(1.0, 0.0), WalkToGoal((0.0, 0.0))) is False

    def test_accepts_raw_tuple_goal(self) -> None:
        """is_done and step should both accept a plain (x, y) tuple, not just WalkToGoal."""
        skill = WalkTo(goal_tolerance=0.15)
        assert skill.is_done(_obs(0.0, 0.0), (1.0, 0.0)) is False
        # step too
        skill.step(_obs(), (1.0, 0.0))


class TestWalkToValidation:
    def test_rejects_non_2d_goal(self) -> None:
        skill = WalkTo()
        with pytest.raises(ValueError, match="2-D"):
            skill.step(_obs(), (1.0, 2.0, 3.0))

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_speed": 0.0},
            {"max_speed": -1.0},
            {"slow_radius": 0.0},
            {"goal_tolerance": 0.0},
        ],
    )
    def test_rejects_nonpositive_lengths(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            WalkTo(**kwargs)


# --------------------------------------------------------------------------- #
# Angle wrap (used internally; tested directly because the closed-form
# behavior is easy to assert on and it's exercised in every step call).
# --------------------------------------------------------------------------- #


class TestWrapAngle:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            (0.0, 0.0),
            (math.pi / 2, math.pi / 2),
            (-math.pi / 2, -math.pi / 2),
            (math.pi, -math.pi),       # boundary collapses to -pi (half-open interval)
            (math.pi + 0.1, -math.pi + 0.1),
            (-math.pi - 0.1, math.pi - 0.1),
            (3 * math.pi, -math.pi),
        ],
    )
    def test_wraps_to_half_open_interval(self, raw: float, expected: float) -> None:
        assert _wrap_angle(raw) == pytest.approx(expected, abs=1e-6)
