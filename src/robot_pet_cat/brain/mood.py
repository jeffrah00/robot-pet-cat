"""Mood latent — slow-drifting state vector that biases the brain.

Six dimensions, each with a circadian-ish drift schedule and a small amount of
Ornstein-Uhlenbeck noise so two identical scenes don't produce identical
behavior across sessions:

  energy        — high = playful; low = sleepy.
  alertness     — high = scans scene; low = stares blankly.
  sociability   — high = approach human; low = retreat.
  curiosity_w   — weight on the curiosity reward stream.
  comfort_w     — weight on the comfort reward stream.
  play_w        — weight on the play reward stream.

`update(dt)` is called every sim step; `weights()` returns the (curiosity,
comfort, play) tuple consumed by brain.policy. The exact mapping from internal
state → weights is intentionally smooth and continuous so behavior shifts feel
gradual.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MoodConfig:
    decay_per_minute: float = 0.05
    noise_sigma: float = 0.02
    rng_seed: int = 0


@dataclass
class Mood:
    cfg: MoodConfig = field(default_factory=MoodConfig)
    state: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.cfg.rng_seed)
        # Start with a mild positive prior on energy + curiosity so the cat
        # doesn't immediately go to sleep.
        self.state[:] = self.rng.uniform(-0.1, 0.4, size=6).astype(np.float32)

    def update(self, dt_s: float) -> None:
        # OU drift toward zero plus noise.
        k = self.cfg.decay_per_minute * (dt_s / 60.0)
        self.state -= k * self.state
        self.state += self.cfg.noise_sigma * np.sqrt(dt_s) * \
            self.rng.standard_normal(6).astype(np.float32)
        self.state = np.clip(self.state, -1.0, 1.0)

    def weights(self) -> tuple[float, float, float]:
        """Returns (curiosity, comfort, play) reward weights in [0, 1]."""

        def sig(x: float) -> float:
            return float(1.0 / (1.0 + np.exp(-2.0 * x)))

        return sig(self.state[3]), sig(self.state[4]), sig(self.state[5])
