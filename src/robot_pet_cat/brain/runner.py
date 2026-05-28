"""PPO training loop for the Tier 3 brain.

Wires `make_brain_gym_env(...)` to stable_baselines3 `PPO` with two
brain-specific hooks:

  - `CuriosityUpdateCallback`: after each rollout, gather all
    `info["curiosity_transition"]` records from the rollout buffer and
    call `env.unwrapped.brain.curiosity.update(transitions)`. This is
    what makes the ICM forward+inverse losses actually decrease over
    training. No-op when curiosity is disabled.
  - `RewardBreakdownCallback`: each rollout, summarize the per-term
    reward breakdown (comfort/play/hold/curiosity) and log it to wandb
    if a run is active, otherwise just stash on `self.locals` for
    smoke-test introspection.

The trainer is designed to be smoke-runnable CPU-only with a 256-step
total budget. Real runs swap in a GPU device, a longer
`total_timesteps`, and a wandb run via the config.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .env import BrainEnvConfig
from .rewards import CompositeRewardConfig, Transition


@dataclass
class BrainTrainConfig:
    """Knobs for one PPO training run on BrainEnv.

    `env_cfg` controls the environment (scene, dt, curiosity wiring).
    `composite_cfg` controls reward weights (curiosity_w, etc).
    PPO hyperparams kept at SB3 defaults except for batch_size /
    n_steps so the smoke test runs in seconds.
    """

    # Environment
    env_cfg: BrainEnvConfig = field(default_factory=BrainEnvConfig)
    composite_cfg: CompositeRewardConfig = field(default_factory=CompositeRewardConfig)

    # PPO
    total_timesteps: int = 200
    n_steps: int = 64
    batch_size: int = 64
    n_epochs: int = 2
    learning_rate: float = 3.0e-4
    # Entropy coefficient. SB3's default is 0.0; that's *wrong for cats* --
    # the policy collapses to a single optimal behavior. We want sustained
    # behavioral variability. 0.05-0.1 is the cat-friendly range; do not
    # decay this. See memory: cats-are-stochastic.
    ent_coef: float = 0.05
    # PPO clipping range. SB3 default is 0.2; 0.1 slows policy updates and
    # helps prevent collapse when the reward signal is sparse / skewed.
    clip_range: float = 0.2
    policy_kwargs: Optional[dict] = None  # e.g. dict(net_arch=[64,64])
    device: str = "cpu"
    seed: Optional[int] = None

    # Logging / persistence
    log_dir: Optional[Path] = None
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    save_model_to: Optional[Path] = None


def _import_sb3():
    """Lazy import so test collection doesn't need SB3."""
    import stable_baselines3 as sb3
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList

    return sb3, PPO, BaseCallback, CallbackList


def train_brain(cfg: BrainTrainConfig) -> Any:
    """Run one PPO training pass; return the trained model.

    The env is constructed fresh inside this function. The composite
    reward weights from `cfg.composite_cfg` are pushed onto the env's
    `r_cfg` after construction so they take effect for all rollouts.
    """
    from .gym_env import make_brain_gym_env

    sb3, PPO, BaseCallback, CallbackList = _import_sb3()

    env = make_brain_gym_env(cfg.env_cfg)
    env.brain.r_cfg = cfg.composite_cfg

    class CuriosityUpdateCallback(BaseCallback):
        """At each rollout end, batch transitions and update ICM."""

        def __init__(self) -> None:
            super().__init__()
            self._batch: list[Transition] = []

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                tr = info.get("curiosity_transition") if isinstance(info, dict) else None
                if tr is not None:
                    self._batch.append(tr)
            return True

        def _on_rollout_end(self) -> None:
            curiosity = env.unwrapped.curiosity
            if curiosity is None or not self._batch:
                self._batch = []
                return
            losses = curiosity.update(self._batch)
            if self.logger is not None:
                for k, v in losses.items():
                    self.logger.record(f"curiosity/{k}", v)
                self.logger.record("curiosity/batch_size", len(self._batch))
            # SB3's internal logger doesn't auto-sync to wandb; log explicitly.
            try:
                import wandb
                if wandb.run is not None:
                    payload = {f"curiosity/{k}": v for k, v in losses.items()}
                    payload["curiosity/batch_size"] = len(self._batch)
                    wandb.log(payload, step=self.num_timesteps)
            except Exception:
                pass
            self._batch = []

    class RewardBreakdownCallback(BaseCallback):
        """Track per-term reward means across a rollout."""

        def __init__(self) -> None:
            super().__init__()
            self._sums: dict[str, float] = {}
            self._n: int = 0

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                if not isinstance(info, dict):
                    continue
                brk = info.get("reward_breakdown")
                if brk is None:
                    continue
                for k, v in brk.items():
                    if k.startswith("_"):
                        continue
                    self._sums[k] = self._sums.get(k, 0.0) + float(v)
                self._n += 1
            return True

        def _on_rollout_end(self) -> None:
            if self._n == 0:
                return
            means = {k: v / self._n for k, v in self._sums.items()}
            if self.logger is not None:
                for k, v in means.items():
                    self.logger.record(f"reward/{k}_mean", v)
                self.logger.record("reward/rollout_steps", self._n)
            try:
                import wandb
                if wandb.run is not None:
                    payload = {f"reward/{k}_mean": v for k, v in means.items()}
                    payload["reward/rollout_steps"] = self._n
                    wandb.log(payload, step=self.num_timesteps)
            except Exception:
                pass
            self._sums = {}
            self._n = 0

    callbacks = CallbackList([CuriosityUpdateCallback(), RewardBreakdownCallback()])

    # Load .env from repo root so WANDB_API_KEY / WANDB_ENTITY are available
    # even on fresh pods where `wandb login` hasn't been called.
    _env_file = Path(__file__).parents[3] / ".env"
    if _env_file.exists():
        try:
            from dotenv import load_dotenv as _load_dotenv
            _load_dotenv(_env_file, override=False)
        except ImportError:
            # python-dotenv not installed -- parse manually (no extra deps)
            with open(_env_file) as _f:
                for _line in _f:
                    _line = _line.strip()
                    if not _line or _line.startswith("#") or "=" not in _line:
                        continue
                    _k, _, _v = _line.partition("=")
                    os.environ.setdefault(_k.strip(), _v.strip())

    # Optional wandb
    if cfg.wandb_project:
        try:
            import wandb

            wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                name=cfg.wandb_run_name,
                config={
                    "total_timesteps": cfg.total_timesteps,
                    "n_steps": cfg.n_steps,
                    "batch_size": cfg.batch_size,
                    "clip_range": cfg.clip_range,
                    "curiosity_enabled": cfg.env_cfg.curiosity_enabled,
                    "curiosity_w": cfg.composite_cfg.curiosity_w,
                    "comfort_w": cfg.composite_cfg.comfort_w,
                    "play_w": cfg.composite_cfg.play_w,
                    "hold_w": cfg.composite_cfg.hold_w,
                },
                reinit=True,
            )
        except Exception as e:  # don't kill a training run for a logging hiccup
            print(f"[runner] wandb init failed: {e}")

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        learning_rate=cfg.learning_rate,
        ent_coef=cfg.ent_coef,
        clip_range=cfg.clip_range,
        policy_kwargs=cfg.policy_kwargs,
        device=cfg.device,
        seed=cfg.seed,
        verbose=1,
        tensorboard_log=str(cfg.log_dir) if cfg.log_dir else None,
    )

    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=callbacks,
        progress_bar=False,
    )

    if cfg.save_model_to is not None:
        cfg.save_model_to.parent.mkdir(parents=True, exist_ok=True)
        model.save(cfg.save_model_to)
        print(f"[runner] saved -> {cfg.save_model_to}")

    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass

    return model
