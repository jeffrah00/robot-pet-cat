"""Smoke tests for the brain PPO trainer.

These tests construct a small env, train PPO for ~256 timesteps total,
and verify the plumbing all the way through: rollout, callbacks fire,
ICM gets a non-empty batch and runs an update, the model returns.

Each test runs in under a couple of seconds on CPU.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")
pytest.importorskip("gymnasium")

from robot_pet_cat.brain.env import BrainEnvConfig  # noqa: E402
from robot_pet_cat.brain.policy import BrainConfig, train  # noqa: E402
from robot_pet_cat.brain.rewards import CompositeRewardConfig  # noqa: E402
from robot_pet_cat.brain.runner import BrainTrainConfig, train_brain  # noqa: E402


def _tiny_cfg(curiosity: bool = False, total: int = 128) -> BrainTrainConfig:
    return BrainTrainConfig(
        env_cfg=BrainEnvConfig(
            curiosity_enabled=curiosity,
            curiosity_seed=0,
            curiosity_feature_dim=8,
            curiosity_hidden=16,
            max_steps_per_episode=64,
        ),
        composite_cfg=CompositeRewardConfig(curiosity_w=0.0),
        total_timesteps=total,
        n_steps=64,
        batch_size=64,
        n_epochs=1,
        learning_rate=3.0e-4,
        policy_kwargs=dict(net_arch=[32, 32]),
        device="cpu",
        seed=0,
    )


class TestPPOSmoke:
    def test_runs_without_curiosity(self) -> None:
        cfg = _tiny_cfg(curiosity=False, total=128)
        model = train_brain(cfg)
        # SB3 stores how many env steps we trained against
        assert model.num_timesteps >= cfg.total_timesteps

    def test_runs_with_curiosity_and_fires_update(self) -> None:
        cfg = _tiny_cfg(curiosity=True, total=128)
        model = train_brain(cfg)
        # After training, ICM should have run at least one update.
        # We can probe this by checking that the env's curiosity module
        # exists and its optimizer has stepped (Adam tracks step count).
        # SB3 wraps our env in Monitor; walk to the real _BrainGymEnv.
        gym_env = model.get_env().envs[0].unwrapped
        brain = gym_env.brain
        assert brain.curiosity is not None
        first_param = next(iter(brain.curiosity.encoder.parameters()))
        # Adam stores its step count in state[param]["step"]
        opt_state = brain.curiosity.optimizer.state.get(first_param, {})
        assert opt_state.get("step", 0) > 0, "ICM optimizer should have stepped"

    def test_legacy_brainconfig_entrypoint(self) -> None:
        """The legacy `train(BrainConfig)` shim should still work."""
        cfg = BrainConfig(total_timesteps=64)
        # This will use defaults; we just want it not to crash.
        # Note: BrainConfig translates total_timesteps but n_steps stays
        # at the runner default (64), so 64 steps == 1 rollout.
        model = train(cfg)
        assert model.num_timesteps >= 64


class TestTrainerInfra:
    def test_callbacks_dont_break_on_empty_batch(self) -> None:
        """If curiosity is disabled, the CuriosityUpdateCallback should
        be a no-op (and not crash on empty batches)."""
        cfg = _tiny_cfg(curiosity=False, total=128)
        model = train_brain(cfg)
        # Survived the rollout without raising. Sanity-check the env
        # is still callable post-training.
        obs, _ = model.get_env().envs[0].unwrapped.reset()
        assert obs.shape == (15,)

    def test_can_save_and_smoke(self, tmp_path) -> None:
        save_path = tmp_path / "smoke_brain.zip"
        cfg = _tiny_cfg(curiosity=False, total=128)
        cfg.save_model_to = save_path
        train_brain(cfg)
        assert save_path.exists(), "save_model_to should write the zip"
