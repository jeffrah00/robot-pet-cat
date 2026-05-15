"""High-level RL policy that chooses skills.

Inputs per decision (every ~1-3 sec of sim time):
  - first-person camera frame (or a privileged scene-state vector during training)
  - last-skill identity + time-since-skill-start
  - mood latent (from brain.mood)

Output:
  - skill_id (categorical over the skills registry)
  - target (continuous - depends on skill semantics)
  - temperature for sampling

Training uses PPO over the composite reward (curiosity + comfort + play),
with each component weighted by the mood-modulated weighting in brain.mood.
Phase 4 work.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BrainConfig:
    decision_period_s: float = 2.0
    num_skills: int = 9
    mood_dim: int = 6
    sampling_temperature: float = 0.8
    curiosity_lr: float = 1.0e-4
    policy_lr: float = 3.0e-4
    total_timesteps: int = 20_000_000


def train(cfg: BrainConfig) -> None:
    raise NotImplementedError(
        "Phase 4: PPO over composite reward, with ICM curiosity bonus. "
        "References:\n"
        "  - Pathak et al. 2017 'Curiosity-Driven Exploration by Self-Supervised Prediction'\n"
        "  - Burda et al. 2018 'Large-Scale Study of Curiosity-Driven Learning'"
    )
