"""Tests for brain.rewards (ComfortReward, PlayReward, HoldBonusReward, composite).

Full ICM behavior is tested in test_brain_curiosity.py; this file only
checks the API guard on CuriosityReward.compute() and that the composite
reward function plumbs `curiosity_value` correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_pet_cat.brain.rewards import (
    CatState,
    ComfortReward,
    CompositeRewardConfig,
    CuriosityReward,
    HoldBonusReward,
    PlayReward,
    compute_composite_reward,
)
from robot_pet_cat.scene import SceneEntity, SceneState, SemanticFlags


def _cat(x=0.0, y=0.0, body_height=0.30, speed=0.0, active_skill=None) -> CatState:
    return CatState(
        xy=np.asarray([x, y], dtype=np.float32),
        yaw=0.0,
        body_height=body_height,
        speed=speed,
        active_skill=active_skill,
        time_in_skill=0.0,
    )


def _scene(*entities: SceneEntity) -> SceneState:
    return SceneState(entities=list(entities))


def _couch(x: float = 2.0, y: float = 0.0) -> SceneEntity:
    return SceneEntity(
        name="couch",
        kind="body",
        pos_xyz=np.asarray([x, y, 0.2], dtype=np.float32),
        velocity_xyz=None,
        flags=SemanticFlags(elevated=True, soft=True),
    )


def _window(x: float = -2.0, y: float = 0.0) -> SceneEntity:
    return SceneEntity(
        name="window",
        kind="site",
        pos_xyz=np.asarray([x, y, 0.01], dtype=np.float32),
        velocity_xyz=None,
        flags=SemanticFlags(warm=True),
    )


def _ball(x: float = 0.0, y: float = 0.0, vx: float = 0.0, vy: float = 0.0) -> SceneEntity:
    return SceneEntity(
        name="ball",
        kind="body",
        pos_xyz=np.asarray([x, y, 0.1], dtype=np.float32),
        velocity_xyz=np.asarray([vx, vy, 0.0], dtype=np.float32),
        flags=SemanticFlags(play_target=True),
    )


# --------------------------------------------------------------------------- #
# ComfortReward
# --------------------------------------------------------------------------- #


class TestComfortReward:
    def test_zero_when_alone_no_warm_no_couch(self) -> None:
        scene = _scene()
        r = ComfortReward().compute(scene, _cat(x=0.0, y=0.0))
        assert r == 0.0

    def test_on_couch_gets_elevated_plus_soft(self) -> None:
        scene = _scene(_couch(x=2.0, y=0.0))
        cr = ComfortReward()
        r_on = cr.compute(scene, _cat(x=2.0, y=0.0))
        r_off = cr.compute(scene, _cat(x=4.0, y=4.0))
        assert r_on == pytest.approx(cr.elevated_weight + cr.soft_weight, abs=1e-6)
        assert r_off == 0.0

    def test_warmth_decays_with_distance(self) -> None:
        cr = ComfortReward()
        scene = _scene(_window(x=0.0, y=0.0))
        r_near = cr.compute(scene, _cat(x=0.0, y=0.0))
        r_mid = cr.compute(scene, _cat(x=1.0, y=0.0))
        r_far = cr.compute(scene, _cat(x=5.0, y=0.0))
        assert r_near == pytest.approx(cr.warm_weight, abs=1e-6)
        assert r_near > r_mid > r_far
        assert r_far == pytest.approx(0.0, abs=1e-3)

    def test_lying_down_kills_on_couch_credit(self) -> None:
        scene = _scene(_couch(2.0, 0.0))
        cr = ComfortReward()
        r = cr.compute(scene, _cat(x=2.0, y=0.0, body_height=0.08))
        assert r == 0.0


# --------------------------------------------------------------------------- #
# PlayReward
# --------------------------------------------------------------------------- #


class TestPlayReward:
    def test_zero_when_no_play_targets(self) -> None:
        assert PlayReward().compute(_scene(), _cat()) == 0.0

    def test_zero_when_ball_far_away(self) -> None:
        scene = _scene(_ball(x=3.0, y=0.0))
        r = PlayReward(paw_range_m=0.4).compute(scene, _cat(x=0.0, y=0.0))
        assert r == 0.0

    def test_moving_ball_in_range_scores_speed(self) -> None:
        scene = _scene(_ball(x=0.1, y=0.0, vx=2.0, vy=0.0))
        pr = PlayReward(paw_range_m=0.4, play_speed_weight=1.0)
        r = pr.compute(scene, _cat(x=0.0, y=0.0))
        assert r == pytest.approx(2.0, abs=1e-5)

    def test_swat_active_adds_causal_bonus(self) -> None:
        scene = _scene(_ball(x=0.1, y=0.0, vx=0.0, vy=0.0))
        pr = PlayReward(paw_range_m=0.4, causal_bonus=0.5, causal_skills=("swat",))
        r_idle = pr.compute(scene, _cat(x=0.0, y=0.0, active_skill="sit"))
        r_swat = pr.compute(scene, _cat(x=0.0, y=0.0, active_skill="swat"))
        assert r_idle == 0.0
        assert r_swat == pytest.approx(0.5, abs=1e-6)


# --------------------------------------------------------------------------- #
# HoldBonusReward (pause-as-default)
# --------------------------------------------------------------------------- #


class TestHoldBonusReward:
    def test_pays_when_cat_still_and_no_play_target_near(self) -> None:
        scene = _scene()
        r = HoldBonusReward(bonus=0.01).compute(scene, _cat(x=0.0, y=0.0))
        assert r == pytest.approx(0.01, abs=1e-9)

    def test_zero_when_cat_moving(self) -> None:
        scene = _scene()
        r = HoldBonusReward().compute(scene, _cat(speed=0.2))
        assert r == 0.0

    def test_zero_when_play_target_nearby(self) -> None:
        scene = _scene(_ball(x=0.3, y=0.0))
        hb = HoldBonusReward(saliency_range_m=0.6)
        r = hb.compute(scene, _cat(x=0.0, y=0.0))
        assert r == 0.0

    def test_zero_when_active_skill_is_non_hold(self) -> None:
        scene = _scene()
        hb = HoldBonusReward(hold_skills=("sit", "lie_down"))
        r_sit = hb.compute(scene, _cat(active_skill="sit"))
        r_walk = hb.compute(scene, _cat(active_skill="walk_to"))
        assert r_sit > 0
        assert r_walk == 0.0

    def test_allows_none_skill(self) -> None:
        scene = _scene()
        r = HoldBonusReward().compute(scene, _cat(active_skill=None))
        assert r > 0


# --------------------------------------------------------------------------- #
# Composite
# --------------------------------------------------------------------------- #


class _StubMood:
    """Tiny mood stub matching the Mood.weights() protocol."""

    def __init__(self, cur=0.5, com=0.5, play=0.5):
        self._w = (cur, com, play)

    def weights(self):
        return self._w


class TestCompositeReward:
    def test_returns_breakdown_dict(self) -> None:
        scene = _scene(_couch())
        cat = _cat(x=2.0, y=0.0)
        out = compute_composite_reward(scene, cat, _StubMood())
        for k in ("total", "comfort", "play", "hold", "curiosity"):
            assert k in out
        assert out["curiosity"] == 0.0  # default when no value supplied
        assert out["total"] != 0.0  # on couch => non-zero comfort

    def test_curiosity_value_passed_through_and_weighted(self) -> None:
        """The supplied curiosity_value flows into the breakdown and is
        scaled by cfg.curiosity_w * mood_w_curiosity."""
        scene = _scene()
        cat = _cat(speed=0.2)  # disables hold bonus
        mood = _StubMood(cur=0.5, com=0.0, play=0.0)
        cfg = CompositeRewardConfig(
            curiosity_w=2.0, comfort_w=0.0, play_w=0.0, hold_w=0.0
        )
        out = compute_composite_reward(
            scene, cat, mood, cfg=cfg, curiosity_value=0.3
        )
        # total = curiosity_w * mood_w_curiosity * value = 2.0 * 0.5 * 0.3 = 0.3
        assert out["curiosity"] == pytest.approx(0.3, abs=1e-9)
        assert out["total"] == pytest.approx(0.3, abs=1e-9)

    def test_mood_weight_modulates_comfort_subterm(self) -> None:
        scene = _scene(_couch())
        cat = _cat(x=2.0, y=0.0)
        out_high = compute_composite_reward(scene, cat, _StubMood(com=1.0))
        out_low = compute_composite_reward(scene, cat, _StubMood(com=0.0))
        assert out_high["total"] > out_low["total"]

    def test_hold_term_not_mood_modulated(self) -> None:
        """Pause-default applies regardless of mood (curious cat in a quiet
        room is still still)."""
        scene = _scene()
        cat = _cat()
        out_high = compute_composite_reward(
            scene, cat, _StubMood(com=1.0, play=1.0, cur=1.0)
        )
        out_low = compute_composite_reward(
            scene, cat, _StubMood(com=0.0, play=0.0, cur=0.0)
        )
        assert out_high["hold"] == out_low["hold"]
        assert out_high["hold"] > 0


# --------------------------------------------------------------------------- #
# CuriosityReward API guard (full ICM behavior is in test_brain_curiosity.py)
# --------------------------------------------------------------------------- #


class TestCuriosityComputeGuard:
    def test_compute_raises_typeerror(self) -> None:
        """The (scene, cat, mood) entry point is intentionally a hard error.

        ICM needs (feat_t, action, feat_tp1); using .compute(...) would mean
        the env stopped passing curiosity_value and silently got a zero.
        """
        pytest.importorskip("torch")
        c = CuriosityReward(obs_dim=4, n_actions=3, hidden=8, feature_dim=4, seed=0)
        with pytest.raises(TypeError):
            c.compute(None, None, None)
