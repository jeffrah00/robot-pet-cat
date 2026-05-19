"""Gym(nasium)-compatible adapter around BrainEnv.

Wraps the dict-obs/dispatch-int-action BrainEnv into the standard gymnasium
Env interface so it can be fed into stable_baselines3, rsl_rl, sb3-contrib,
or hand-rolled PPO. Two adapters:

  - BrainGymEnv: observation is a flat Box of the v0 curiosity feature
    vector (same projection as ICM uses; see brain/curiosity_feat.py).
    This is the simplest shape for a fresh PPO baseline.
  - The underlying BrainEnv keeps its dict obs for code that wants the
    rich form -- accessible via env.unwrapped.brain.

We deliberately do NOT subclass gymnasium.Env at import time so this
module imports cleanly even where gymnasium isn't installed -- gymnasium
is a lazy import in the factory. The returned object IS a gymnasium.Env
instance.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .curiosity_feat import CURIOSITY_OBS_DIM_V0, obs_dict_to_curiosity_vec
from .env import BrainEnv, BrainEnvConfig


def make_brain_gym_env(cfg: Optional[BrainEnvConfig] = None):
    """Construct a gymnasium.Env wrapping a fresh BrainEnv.

    Action space: Discrete(action_space_n) where index 0 is HOLD.
    Observation space: Box of shape (CURIOSITY_OBS_DIM_V0,) float32.

    Reward, terminated, truncated, info come straight from BrainEnv.step.
    info["reward_breakdown"] and info["curiosity_transition"] (when
    cfg.curiosity_enabled) pass through unmodified.
    """
    import gymnasium as gym

    class _BrainGymEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self) -> None:
            super().__init__()
            self.brain = BrainEnv(cfg)
            self.action_space = gym.spaces.Discrete(self.brain.action_space_n)
            # Observation bounds are loose -- mood is in [-1,1] but xy is
            # unbounded; we say [-inf, inf] and let downstream normalize.
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(CURIOSITY_OBS_DIM_V0,),
                dtype=np.float32,
            )

        def reset(self, *, seed=None, options=None):
            if seed is not None:
                # Re-seed the mood OU process via the underlying numpy RNG.
                # The brain mood doesn't expose seeding yet; future work.
                np.random.seed(int(seed))
            obs_dict = self.brain.reset()
            return obs_dict_to_curiosity_vec(obs_dict), {}

        def step(self, action):
            obs_dict, reward, terminated, truncated, info = self.brain.step(int(action))
            return (
                obs_dict_to_curiosity_vec(obs_dict),
                float(reward),
                bool(terminated),
                bool(truncated),
                info,
            )

        def render(self):
            return None

        def close(self):
            pass

    return _BrainGymEnv()
