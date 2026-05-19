"""ICM (Intrinsic Curiosity Module) neural networks.

Internal helper for `CuriosityReward` in `brain/rewards.py`. Kept in its own
module so the torch dependency is isolated and the reward dataclasses in
rewards.py stay focused on the composition layer.

Reference: Pathak et al. 2017, "Curiosity-driven Exploration by Self-supervised
Prediction" (https://arxiv.org/abs/1705.05363).

Three small MLPs:
  - FeatureEncoder: obs -> feature vector (phi).
  - ForwardModel:   (phi(s_t), a_t)  -> phi_hat(s_{t+1}).
  - InverseModel:   (phi(s_t), phi(s_{t+1})) -> a_hat.

The intrinsic reward is the L2 prediction error of the forward model; the
inverse model exists only to shape phi so that features are sensitive to the
parts of the observation the agent can actually influence (and insensitive
to white-noise / TV-static distractors). Both losses share the encoder.

Actions are one-hot encoded before being fed to the forward model; the
inverse model emits logits over the discrete action set.
"""

from __future__ import annotations

import torch
from torch import nn


class FeatureEncoder(nn.Module):
    """obs (R^obs_dim) -> feature (R^feature_dim). Simple MLP."""

    def __init__(self, obs_dim: int, feature_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, feature_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class ForwardModel(nn.Module):
    """(phi, a_onehot) -> phi_hat_next."""

    def __init__(self, feature_dim: int, n_actions: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + n_actions, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, feature_dim),
        )

    def forward(self, phi: torch.Tensor, a_onehot: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([phi, a_onehot], dim=-1))


class InverseModel(nn.Module):
    """(phi_t, phi_t+1) -> action logits."""

    def __init__(self, feature_dim: int, n_actions: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * feature_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, phi_t: torch.Tensor, phi_tp1: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([phi_t, phi_tp1], dim=-1))


def action_one_hot(action: torch.Tensor, n_actions: int) -> torch.Tensor:
    """Convert a long-tensor of action indices to a float one-hot matrix."""
    if action.dim() == 0:
        action = action.unsqueeze(0)
    return torch.nn.functional.one_hot(action.long(), num_classes=n_actions).float()
