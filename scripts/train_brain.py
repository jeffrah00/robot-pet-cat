#!/usr/bin/env python3
"""Launch a brain-policy PPO training run.

Usage:
  python scripts/train_brain.py                       # smoke run (CPU, 200 steps)
  python scripts/train_brain.py --steps 200000        # longer run
  python scripts/train_brain.py --steps 1000000 \\
      --curiosity --curiosity-w 0.05 \\
      --wandb-project robot-pet-cat \\
      --wandb-entity jeffrah89-personal \\
      --wandb-run-name brain_v0_curio \\
      --save checkpoints/brain/brain_v0.zip

Defaults are tuned to be smoke-runnable in a few seconds on CPU. Real
training runs should set --steps, --device cuda, and a wandb run name.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from robot_pet_cat.brain.env import BrainEnvConfig
from robot_pet_cat.brain.rewards import CompositeRewardConfig
from robot_pet_cat.brain.runner import BrainTrainConfig, train_brain


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200, help="total_timesteps for PPO")
    p.add_argument("--n-steps", type=int, default=64, help="PPO rollout length")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=3.0e-4)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--curiosity", action="store_true",
                   help="enable ICM curiosity wiring in the env")
    p.add_argument("--curiosity-w", type=float, default=0.0,
                   help="weight on the curiosity term in the composite reward")
    p.add_argument("--comfort-w", type=float, default=1.0)
    p.add_argument("--play-w", type=float, default=1.0)
    p.add_argument("--hold-w", type=float, default=1.0)
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--log-dir", default=None,
                   help="tensorboard log dir (also used for wandb sync if --wandb-project set)")
    p.add_argument("--save", default=None, help="path to save the trained PPO zip")
    args = p.parse_args()

    env_cfg = BrainEnvConfig(curiosity_enabled=args.curiosity)
    composite_cfg = CompositeRewardConfig(
        curiosity_w=args.curiosity_w,
        comfort_w=args.comfort_w,
        play_w=args.play_w,
        hold_w=args.hold_w,
    )

    cfg = BrainTrainConfig(
        env_cfg=env_cfg,
        composite_cfg=composite_cfg,
        total_timesteps=args.steps,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.lr,
        device=args.device,
        seed=args.seed,
        log_dir=Path(args.log_dir) if args.log_dir else None,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        save_model_to=Path(args.save) if args.save else None,
    )

    print(f"[train_brain] starting run: steps={args.steps}, "
          f"curiosity={args.curiosity}, curiosity_w={args.curiosity_w}, "
          f"device={args.device}")
    model = train_brain(cfg)
    print("[train_brain] done. n_calls=", getattr(model, "num_timesteps", "?"))


if __name__ == "__main__":
    main()
