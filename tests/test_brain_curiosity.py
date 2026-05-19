"""Unit tests for the ICM CuriosityReward in brain.rewards.

These tests exercise the trainable forward+inverse model end-to-end on tiny
synthetic data. Everything runs CPU-only and finishes in well under a second.

Skipped automatically if torch isn't installed so the broader brain test
suite stays green in environments without the optional `brain` extras.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from robot_pet_cat.brain.rewards import CuriosityReward, Transition  # noqa: E402


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------- #
# Shape and contract tests
# --------------------------------------------------------------------------- #


class TestICMContract:
    def test_intrinsic_reward_is_finite_nonnegative_scalar(self) -> None:
        rng = _rng(0)
        c = CuriosityReward(obs_dim=8, n_actions=4, hidden=16, feature_dim=8, seed=0)
        feat_t = rng.standard_normal(8).astype(np.float32)
        feat_tp1 = rng.standard_normal(8).astype(np.float32)
        r = c.intrinsic_reward(feat_t, action=2, feat_tp1=feat_tp1)
        assert isinstance(r, float)
        assert np.isfinite(r)
        assert r >= 0.0

    def test_intrinsic_reward_rejects_wrong_shape(self) -> None:
        c = CuriosityReward(obs_dim=4, n_actions=3, hidden=8, feature_dim=4, seed=0)
        bad = np.zeros(5, dtype=np.float32)
        good = np.zeros(4, dtype=np.float32)
        with pytest.raises(ValueError, match="feature vector shape"):
            c.intrinsic_reward(bad, 0, good)
        with pytest.raises(ValueError, match="feature vector shape"):
            c.intrinsic_reward(good, 0, bad)

    def test_intrinsic_reward_rejects_out_of_range_action(self) -> None:
        c = CuriosityReward(obs_dim=4, n_actions=3, hidden=8, feature_dim=4, seed=0)
        feat = np.zeros(4, dtype=np.float32)
        with pytest.raises(ValueError, match="out of range"):
            c.intrinsic_reward(feat, action=3, feat_tp1=feat)
        with pytest.raises(ValueError, match="out of range"):
            c.intrinsic_reward(feat, action=-1, feat_tp1=feat)

    def test_update_empty_batch_is_noop(self) -> None:
        c = CuriosityReward(obs_dim=4, n_actions=3, hidden=8, feature_dim=4, seed=0)
        out = c.update([])
        assert out == {"loss": 0.0, "forward": 0.0, "inverse": 0.0}

    def test_update_returns_finite_scalars(self) -> None:
        rng = _rng(2)
        c = CuriosityReward(obs_dim=4, n_actions=3, hidden=8, feature_dim=4, seed=2)
        batch = [
            Transition(
                feat_t=rng.standard_normal(4).astype(np.float32),
                action=int(rng.integers(0, 3)),
                feat_tp1=rng.standard_normal(4).astype(np.float32),
            )
            for _ in range(16)
        ]
        out = c.update(batch)
        for k in ("loss", "forward", "inverse"):
            assert k in out
            assert isinstance(out[k], float)
            assert np.isfinite(out[k])
            assert out[k] >= 0.0


# --------------------------------------------------------------------------- #
# Learning behavior
# --------------------------------------------------------------------------- #


def _toy_dynamics(rng: np.random.Generator, obs_dim: int, n_actions: int):
    """Build a fixed, learnable deterministic dynamics:

      s_{t+1} = s_t + A[a_t]
    """
    A = rng.standard_normal((n_actions, obs_dim)).astype(np.float32) * 0.5
    return A


def _make_batch(rng, A, obs_dim, n_actions, n=64):
    batch = []
    for _ in range(n):
        s = rng.standard_normal(obs_dim).astype(np.float32)
        a = int(rng.integers(0, n_actions))
        s2 = s + A[a]
        batch.append(Transition(feat_t=s, action=a, feat_tp1=s2))
    return batch


class TestICMLearning:
    def test_intrinsic_reward_decreases_after_repeated_updates(self) -> None:
        """The defining test: ICM should *get bored* of a transition it has
        been trained on repeatedly. We construct a fixed (s, a, s') triple
        and show that average reward drops after enough .update() steps."""
        rng = _rng(3)
        obs_dim, n_actions = 4, 3
        A = _toy_dynamics(rng, obs_dim, n_actions)
        c = CuriosityReward(
            obs_dim=obs_dim,
            n_actions=n_actions,
            hidden=32,
            feature_dim=8,
            beta=0.2,
            reward_scale=0.5,
            lr=3e-3,
            seed=3,
        )

        s = rng.standard_normal(obs_dim).astype(np.float32)
        a = 1
        s2 = s + A[a]

        r_before = c.intrinsic_reward(s, a, s2)

        # Train on the same transition many times. ICM should learn it.
        for _ in range(200):
            c.update([Transition(feat_t=s, action=a, feat_tp1=s2)])

        r_after = c.intrinsic_reward(s, a, s2)
        assert r_after < r_before, (
            f"expected familiarity to decrease curiosity: before={r_before:.4f} "
            f"after={r_after:.4f}"
        )
        # Should drop by at least 50% on this trivial single-point task.
        assert r_after < 0.5 * r_before + 1e-6

    def test_inverse_loss_decreases_on_consistent_dynamics(self) -> None:
        """Inverse model should learn to recover actions from feature diffs."""
        rng = _rng(5)
        obs_dim, n_actions = 4, 3
        A = _toy_dynamics(rng, obs_dim, n_actions)
        c = CuriosityReward(
            obs_dim=obs_dim, n_actions=n_actions, hidden=32, feature_dim=8,
            beta=0.5,  # weight inverse heavier so it actually learns fast
            lr=3e-3,
            seed=5,
        )
        eval_batch = _make_batch(rng, A, obs_dim, n_actions, n=128)

        out0 = c.update(eval_batch)
        loss_early = out0["inverse"]
        for _ in range(80):
            c.update(_make_batch(rng, A, obs_dim, n_actions, n=64))
        out1 = c.update(eval_batch)
        loss_late = out1["inverse"]

        assert loss_late < loss_early, (
            f"inverse loss did not decrease: early={loss_early:.4f} "
            f"late={loss_late:.4f}"
        )


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #


class TestICMCheckpointing:
    def test_state_dict_round_trip(self) -> None:
        rng = _rng(6)
        c1 = CuriosityReward(obs_dim=4, n_actions=3, hidden=8, feature_dim=4, seed=6)
        # Train a few steps so weights are non-trivial
        for _ in range(10):
            batch = [
                Transition(
                    feat_t=rng.standard_normal(4).astype(np.float32),
                    action=int(rng.integers(0, 3)),
                    feat_tp1=rng.standard_normal(4).astype(np.float32),
                )
                for _ in range(8)
            ]
            c1.update(batch)

        sd = c1.state_dict()

        # Fresh instance with a different seed, then load
        c2 = CuriosityReward(obs_dim=4, n_actions=3, hidden=8, feature_dim=4, seed=99)
        c2.load_state_dict(sd)

        s = rng.standard_normal(4).astype(np.float32)
        s2 = rng.standard_normal(4).astype(np.float32)
        r1 = c1.intrinsic_reward(s, 0, s2)
        r2 = c2.intrinsic_reward(s, 0, s2)
        assert r1 == pytest.approx(r2, abs=1e-6)


# --------------------------------------------------------------------------- #
# Integration with compute_composite_reward
# --------------------------------------------------------------------------- #


class TestICMIntegrationWithComposite:
    def test_full_step_through_composite_reward(self) -> None:
        """End-to-end: compute ICM reward, feed it through
        compute_composite_reward, verify the breakdown reflects it."""
        from robot_pet_cat.brain.rewards import (
            CatState,
            CompositeRewardConfig,
            compute_composite_reward,
        )
        from robot_pet_cat.scene import SceneState

        rng = _rng(7)
        c = CuriosityReward(obs_dim=4, n_actions=3, hidden=8, feature_dim=4, seed=7)
        feat_t = rng.standard_normal(4).astype(np.float32)
        feat_tp1 = rng.standard_normal(4).astype(np.float32)
        r_icm = c.intrinsic_reward(feat_t, 1, feat_tp1)
        assert r_icm > 0

        class _M:
            def weights(self):
                return (1.0, 0.0, 0.0)  # only curiosity weight is on

        cat = CatState(
            xy=np.zeros(2, dtype=np.float32),
            yaw=0.0, body_height=0.30, speed=0.0,
            active_skill=None, time_in_skill=0.0,
        )
        cfg = CompositeRewardConfig(curiosity_w=1.0, comfort_w=0.0, play_w=0.0, hold_w=0.0)
        out = compute_composite_reward(
            SceneState(entities=[]), cat, _M(), cfg=cfg, curiosity_value=r_icm,
        )
        assert out["curiosity"] == pytest.approx(r_icm, abs=1e-9)
        assert out["total"] == pytest.approx(r_icm, abs=1e-9)
