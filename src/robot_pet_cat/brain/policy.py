"""High-level RL policy that chooses skills.

Inputs per decision (every dt_s of sim time):
  - flat feature vector (mood + proprioception + nearest-play-target dx/dy)
  - implicit recurrent state via PPO's policy network (no LSTM yet)

Output:
  - discrete skill_id (0=HOLD, 1..N=skill[i-1])

Training uses PPO over the composite reward (curiosity + comfort + play
+ hold), with each component weighted by `CompositeRewardConfig` and (for
curiosity/comfort/play) mood-modulated by `brain.mood.Mood.weights()`.

The actual training loop is in `brain/runner.py`; this module exposes
the legacy `train(cfg)` entrypoint by delegating to it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .runner import BrainTrainConfig, train_brain


@dataclass
class BrainConfig:
    """Legacy summary config kept for backwards compat.

    Real training knobs now live on `BrainTrainConfig` (in runner.py),
    which wraps `BrainEnvConfig` and `CompositeRewardConfig`. This shim
    exists so older docstrings / scripts that reference `BrainConfig`
    keep importing cleanly.
    """

    decision_period_s: float = 2.0
    num_skills: int = 9
    mood_dim: int = 6
    sampling_temperature: float = 0.8
    curiosity_lr: float = 1.0e-4
    policy_lr: float = 3.0e-4
    total_timesteps: int = 20_000_000


def train(cfg: BrainConfig | BrainTrainConfig | None = None):
    """Train the brain policy. Returns the trained PPO model.

    Accepts either the legacy `BrainConfig` (a few summary knobs) or the
    real `BrainTrainConfig`. Passing `None` uses all defaults from
    `BrainTrainConfig` -- useful for smoke runs.
    """
    if cfg is None:
        train_cfg = BrainTrainConfig()
    elif isinstance(cfg, BrainTrainConfig):
        train_cfg = cfg
    elif isinstance(cfg, BrainConfig):
        train_cfg = BrainTrainConfig(
            learning_rate=cfg.policy_lr,
            total_timesteps=cfg.total_timesteps,
        )
        train_cfg.env_cfg.curiosity_lr = cfg.curiosity_lr
    else:
        raise TypeError(f"unsupported cfg type: {type(cfg).__name__}")
    return train_brain(train_cfg)
