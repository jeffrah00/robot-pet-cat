"""The three intrinsic reward streams the brain optimizes against.

Each reward is computed online from the simulator state. The brain.policy
composes them with mood-modulated weights and runs PPO over the sum.

Phase 4 fills in the real implementations; the dataclasses here pin down the
shape so the brain training loop can be built against the interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class CuriosityReward:
    """ICM-style intrinsic curiosity: prediction error of a forward dynamics model.

    Reference: Pathak et al. 2017. Side effect: novel objects + novel locations
    become attractive without us hand-coding what "novel" means.
    """

    embed_dim: int = 64

    def compute(self, obs_t: np.ndarray, obs_tp1: np.ndarray, action: int) -> float:
        raise NotImplementedError("Phase 4")


@dataclass
class ComfortReward:
    """Dense reward for being in a 'good resting spot'.

    Combines several MuJoCo scene-state signals:
      + elevated (body z > floor by some threshold)
      + on a surface tagged `soft` (couch, bed, cat tree)
      + near a surface tagged `warm` (window sill in sun, heater)
      + low body acceleration (resting, not moving)
    """

    elevated_weight: float = 0.4
    soft_weight: float = 0.4
    warm_weight: float = 0.2

    def compute(self, scene_state: dict[str, Any]) -> float:
        raise NotImplementedError("Phase 4")


@dataclass
class PlayReward:
    """Reward triggered by interaction with movable objects.

    + proportional to ball velocity when ball is within paw range
    + bonus when the cat caused the velocity change (causal credit, simple)
    + small ongoing reward for looking at moving objects
    """

    def compute(self, scene_state: dict[str, Any]) -> float:
        raise NotImplementedError("Phase 4")
