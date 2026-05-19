"""Gym(nasium)-compatible adapter around BrainEnv.

Wraps the dict-obs/dispatch-int-action BrainEnv into the standard gymnasium
Env interface so it can be fed into stable_baselines3 or hand-rolled PPO.
Two observation versions:

  - obs_version=0 (default): flat 15-dim v0 curiosity vector. Backward-
    compatible with all existing checkpoints.
  - obs_version=1: 271-dim v1 vector = v0 (15) + cat_eye grayscale pixels
    (256). Requires cfg.use_cat_eye_obs=True and cfg.use_physics_cat=True.

The underlying BrainEnv stays accessible via env.unwrapped.brain for code
that wants the rich dict obs.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .curiosity_feat import (
    CURIOSITY_OBS_DIM_V0,
    CURIOSITY_OBS_DIM_V1,
    obs_dict_to_curiosity_vec,
    obs_dict_to_curiosity_vec_v1,
)
from .env import BrainEnv, BrainEnvConfig


def make_brain_gym_env(
    cfg: Optional[BrainEnvConfig] = None,
    obs_version: int = 0,
):
    """Construct a gymnasium.Env wrapping a fresh BrainEnv.

    Args:
      cfg: BrainEnvConfig. Set cfg.use_cat_eye_obs=True together with
           obs_version=1 to get visual observations.
      obs_version: 0 = 15-dim v0 vector; 1 = 271-dim v1 vector with
                   cat_eye grayscale pixels.

    Action space: Discrete(action_space_n) where index 0 is HOLD.
    Observation space: Box of shape (obs_dim,) float32.
    """
    import gymnasium as gym

    if obs_version == 1:
        obs_dim = CURIOSITY_OBS_DIM_V1
        _vec_fn = obs_dict_to_curiosity_vec_v1
    else:
        obs_dim = CURIOSITY_OBS_DIM_V0
        _vec_fn = obs_dict_to_curiosity_vec

    class _BrainGymEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self) -> None:
            super().__init__()
            self.brain = BrainEnv(cfg)
            self.action_space = gym.spaces.Discrete(self.brain.action_space_n)
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32,
            )

        def reset(self, *, seed=None, options=None):
            if seed is not None:
                np.random.seed(int(seed))
            obs_dict = self.brain.reset()
            return _vec_fn(obs_dict), {}

        def step(self, action):
            obs_dict, reward, terminated, truncated, info = self.brain.step(int(action))
            return (
                _vec_fn(obs_dict),
                float(reward),
                bool(terminated),
                bool(truncated),
                info,
            )

        def render(self):
            return None

        def close(self):
            if self.brain._cat_eye_renderer is not None:
                self.brain._cat_eye_renderer.close()

    return _BrainGymEnv()
