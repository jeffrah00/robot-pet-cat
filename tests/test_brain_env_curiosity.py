"""Tests for the ICM-curiosity wiring added to BrainEnv.

These tests exist separately from test_brain_env.py because:
  - They require torch (curiosity-enabled construction instantiates ICM).
  - They cover behavior that is OPT-IN; the base env tests verify the
    default (curiosity_enabled=False) path is unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # CuriosityReward needs torch

from robot_pet_cat.brain.curiosity_feat import (  # noqa: E402
    CURIOSITY_OBS_DIM_V0,
    obs_dict_to_curiosity_vec,
)
from robot_pet_cat.brain.env import BrainEnv, BrainEnvConfig  # noqa: E402
from robot_pet_cat.brain.rewards import Transition  # noqa: E402


# --------------------------------------------------------------------------- #
# Feature vector helper
# --------------------------------------------------------------------------- #


class TestCuriosityFeatureVector:
    def test_shape_and_dtype(self) -> None:
        env = BrainEnv(BrainEnvConfig(curiosity_enabled=False))
        obs = env.reset()
        vec = obs_dict_to_curiosity_vec(obs)
        assert vec.shape == (CURIOSITY_OBS_DIM_V0,)
        assert vec.dtype == np.float32
        assert np.all(np.isfinite(vec))

    def test_yaw_encoded_as_cos_sin(self) -> None:
        """Two cats differing only in yaw should differ only in the
        cos/sin slots of the feature vector."""
        env_a = BrainEnv(BrainEnvConfig(initial_cat_yaw=0.0))
        env_b = BrainEnv(BrainEnvConfig(initial_cat_yaw=1.5))
        a = obs_dict_to_curiosity_vec(env_a.reset())
        b = obs_dict_to_curiosity_vec(env_b.reset())
        # Slots: mood(6) + cat_xy(2) + cos,sin(2) + ...
        # cat_xy unchanged; mood is freshly initialized in each env, so
        # we only check the yaw slots are the ones we'd expect.
        assert a[8] == pytest.approx(np.cos(0.0), abs=1e-6)
        assert a[9] == pytest.approx(np.sin(0.0), abs=1e-6)
        assert b[8] == pytest.approx(np.cos(1.5), abs=1e-6)
        assert b[9] == pytest.approx(np.sin(1.5), abs=1e-6)


# --------------------------------------------------------------------------- #
# BrainEnv with curiosity disabled (default)
# --------------------------------------------------------------------------- #


class TestBrainEnvCuriosityDisabled:
    def test_curiosity_is_none(self) -> None:
        env = BrainEnv(BrainEnvConfig(curiosity_enabled=False))
        assert env.curiosity is None
        env.reset()
        _, _, _, _, info = env.step(0)
        assert "curiosity_transition" not in info
        assert info["reward_breakdown"]["curiosity"] == 0.0


# --------------------------------------------------------------------------- #
# BrainEnv with curiosity enabled
# --------------------------------------------------------------------------- #


class TestBrainEnvCuriosityEnabled:
    def _make(self, **kw) -> BrainEnv:
        cfg = BrainEnvConfig(
            curiosity_enabled=True,
            curiosity_seed=0,
            curiosity_feature_dim=8,
            curiosity_hidden=16,
            **kw,
        )
        return BrainEnv(cfg)

    def test_curiosity_instantiated(self) -> None:
        env = self._make()
        assert env.curiosity is not None
        assert env.curiosity.obs_dim == CURIOSITY_OBS_DIM_V0
        assert env.curiosity.n_actions == env.action_space_n

    def test_step_emits_curiosity_transition(self) -> None:
        env = self._make()
        env.reset()
        _, _, _, _, info = env.step(0)
        assert "curiosity_transition" in info
        tr = info["curiosity_transition"]
        assert isinstance(tr, Transition)
        assert tr.feat_t.shape == (CURIOSITY_OBS_DIM_V0,)
        assert tr.feat_tp1.shape == (CURIOSITY_OBS_DIM_V0,)
        assert tr.action == 0
        assert np.all(np.isfinite(tr.feat_t))
        assert np.all(np.isfinite(tr.feat_tp1))

    def test_breakdown_carries_curiosity_value(self) -> None:
        """With curiosity wired but composite weight 0, the breakdown
        records the raw curiosity scalar but it doesn't affect total."""
        env = self._make()
        env.reset()
        _, _, _, _, info = env.step(1)  # any non-hold action
        brk = info["reward_breakdown"]
        # ICM-from-random-init produces non-zero curiosity error
        assert brk["curiosity"] > 0.0
        # Default curiosity_w=0 means it doesn't enter the total.
        weighted = (
            env.r_cfg.curiosity_w
            * brk["_mood_w_curiosity"]
            * brk["curiosity"]
        )
        assert brk["total"] == pytest.approx(
            weighted + env.r_cfg.comfort_w * brk["_mood_w_comfort"] * brk["comfort"]
            + env.r_cfg.play_w * brk["_mood_w_play"] * brk["play"]
            + env.r_cfg.hold_w * brk["hold"],
            abs=1e-9,
        )

    def test_can_train_curiosity_from_emitted_transitions(self) -> None:
        """End-to-end: roll out a few steps, collect transitions, run
        .update() on them, and confirm losses are finite scalars."""
        env = self._make()
        env.reset()
        batch = []
        for a in [0, 1, 2, 0, 0, 3, 0]:
            _, _, _, _, info = env.step(a)
            batch.append(info["curiosity_transition"])
        out = env.curiosity.update(batch)
        for k in ("loss", "forward", "inverse"):
            assert k in out
            assert np.isfinite(out[k])

    def test_reset_seeds_prev_feature_vector(self) -> None:
        """After reset, the first step must have a valid prev-feature buffer
        (i.e., emits a transition immediately, not on step 2)."""
        env = self._make()
        env.reset()
        _, _, _, _, info = env.step(0)
        assert "curiosity_transition" in info


# --------------------------------------------------------------------------- #
# Gym shim
# --------------------------------------------------------------------------- #


class TestBrainGymEnv:
    def test_spaces(self) -> None:
        gym = pytest.importorskip("gymnasium")
        from robot_pet_cat.brain.gym_env import make_brain_gym_env

        env = make_brain_gym_env()
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == env.brain.action_space_n
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert env.observation_space.shape == (CURIOSITY_OBS_DIM_V0,)
        assert env.observation_space.dtype == np.float32

    def test_reset_returns_obs_and_info(self) -> None:
        pytest.importorskip("gymnasium")
        from robot_pet_cat.brain.gym_env import make_brain_gym_env

        env = make_brain_gym_env()
        obs, info = env.reset(seed=42)
        assert obs.shape == (CURIOSITY_OBS_DIM_V0,)
        assert isinstance(info, dict)

    def test_step_returns_5_tuple(self) -> None:
        pytest.importorskip("gymnasium")
        from robot_pet_cat.brain.gym_env import make_brain_gym_env

        env = make_brain_gym_env()
        env.reset()
        out = env.step(0)
        assert len(out) == 5
        obs, r, term, trunc, info = out
        assert obs.shape == (CURIOSITY_OBS_DIM_V0,)
        assert isinstance(r, float)
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert "reward_breakdown" in info

    def test_full_episode_runs(self) -> None:
        pytest.importorskip("gymnasium")
        from robot_pet_cat.brain.gym_env import make_brain_gym_env

        cfg = BrainEnvConfig(max_steps_per_episode=20)
        env = make_brain_gym_env(cfg)
        env.reset()
        for i in range(20):
            obs, r, term, trunc, info = env.step(i % env.action_space.n)
            assert np.all(np.isfinite(obs))
            if trunc:
                break
        assert trunc, "episode should have truncated at max_steps_per_episode"

    def test_curiosity_enabled_passes_through_info(self) -> None:
        pytest.importorskip("gymnasium")
        from robot_pet_cat.brain.gym_env import make_brain_gym_env

        cfg = BrainEnvConfig(
            curiosity_enabled=True,
            curiosity_seed=1,
            curiosity_feature_dim=8,
            curiosity_hidden=16,
        )
        env = make_brain_gym_env(cfg)
        env.reset()
        _, _, _, _, info = env.step(0)
        assert "curiosity_transition" in info
        assert isinstance(info["curiosity_transition"], Transition)
