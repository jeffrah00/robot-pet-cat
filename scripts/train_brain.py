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

from robot_pet_cat.brain.attractor import ModePolicy, ModePolicyConfig
from robot_pet_cat.brain.env import BrainEnvConfig
from robot_pet_cat.brain.rewards import CompositeRewardConfig
from robot_pet_cat.brain.runner import BrainTrainConfig, train_brain


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200,
                   help="total_timesteps for PPO. Smoke test: 200 (CPU, <5s). "
                        "Production brain run: 150_000. The skill-selector policy "
                        "converges by ~150k steps (entropy flat by 32k, EV>0.85 "
                        "by 50k). 500k is wasteful -- see brain_v3_stochastic W&B "
                        "data where entropy was -2.13 from step 32k to 235k.")
    p.add_argument("--n-steps", type=int, default=64, help="PPO rollout length")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=3.0e-4)
    p.add_argument("--ent-coef", type=float, default=0.05,
                   help="PPO entropy coefficient. Default 0.05 keeps the "
                        "policy stochastic; see memory cats-are-stochastic.")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--curiosity", action="store_true",
                   help="enable ICM curiosity wiring in the env")
    p.add_argument("--curiosity-w", type=float, default=0.05,
                   help="weight on the curiosity term in the composite reward")
    p.add_argument("--comfort-w", type=float, default=1.0)
    p.add_argument("--play-w", type=float, default=0.3,
                   help="play reward weight. Default 0.3 (was 1.0); v0 baseline "
                        "showed play term needed to stop dominating 30x.")
    p.add_argument("--hold-w", type=float, default=1.0)
    p.add_argument("--mode-soft-pass", type=float, default=0.3,
                   help="probability of letting an out-of-attractor-mode action "
                        "through instead of forcing HOLD. 0 = hard mask (v0); "
                        "1 = ignore the mask entirely.")
    p.add_argument("--no-attractor", action="store_true",
                   help="disable the ModePolicy attractor layer entirely. "
                        "Default: attractors ON (the cat changes its mind "
                        "every ~5-20s as the active mode switches).")
    p.add_argument("--attractor-seed", type=int, default=0,
                   help="rng seed for the ModePolicy transition sampler")
    p.add_argument("--mode-min-dwell", type=float, default=8.0,
                   help="ModePolicy min_dwell_s. Lower = faster regime changes "
                        "(more visible behavior shifts).")
    p.add_argument("--mode-decision-period", type=float, default=4.0,
                   help="ModePolicy decision_period_s. Lower = more frequent "
                        "transition evaluations.")
    p.add_argument("--mode-temperature", type=float, default=0.8,
                   help="ModePolicy softmax temperature on mood-bias scores. "
                        "Lower = more decisive mode picks.")
    p.add_argument("--initial-cat-x", type=float, default=0.0,
                   help="Spawn x for the cat at episode reset. Default 0.")
    p.add_argument("--initial-cat-y", type=float, default=0.0,
                   help="Spawn y for the cat at episode reset. Default 0.")
    p.add_argument("--skill-min-duration", type=float, default=0.0,
                   help="Minimum seconds a non-HOLD skill stays active before "
                        "the policy can switch to a different non-HOLD skill. "
                        "HOLD always interrupts. Default 0 = per-step switching. "
                        "~1.5 = cat-like commitment.")
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--log-dir", default=None,
                   help="tensorboard log dir (also used for wandb sync if --wandb-project set)")
    p.add_argument("--save", default=None, help="path to save the trained PPO zip")
    args = p.parse_args()

    # Build the env config. Attractors on by default so the cat changes
    # its mind every ~5-20s; pass --no-attractor to opt out.
    mode_policy = None
    if not args.no_attractor:
        mode_policy = ModePolicy(
            cfg=ModePolicyConfig(
                rng_seed=args.attractor_seed,
                min_dwell_s=args.mode_min_dwell,
                decision_period_s=args.mode_decision_period,
                transition_temperature=args.mode_temperature,
            )
        )
    env_cfg = BrainEnvConfig(
        curiosity_enabled=args.curiosity,
        mode_policy=mode_policy,
        initial_cat_xy=(args.initial_cat_x, args.initial_cat_y),
        min_skill_duration_s=args.skill_min_duration,
    )
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
        ent_coef=args.ent_coef,
        device=args.device,
        seed=args.seed,
        log_dir=Path(args.log_dir) if args.log_dir else None,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        save_model_to=Path(args.save) if args.save else None,
    )
    # mode_soft_pass flows into the env config; default 0.3 lets ~30% of
    # out-of-mode actions through instead of forcing HOLD.
    env_cfg.mode_soft_pass = args.mode_soft_pass

    print(f"[train_brain] starting run: steps={args.steps}, "
          f"curiosity={args.curiosity}, curiosity_w={args.curiosity_w}, "
          f"ent_coef={args.ent_coef}, play_w={args.play_w}, "
          f"attractor={'ON' if mode_policy else 'OFF'}, "
          f"min_dwell={args.mode_min_dwell}, "
          f"mode_soft_pass={args.mode_soft_pass}, "
          f"initial_xy=({args.initial_cat_x},{args.initial_cat_y}), "
          f"device={args.device}")
    model = train_brain(cfg)
    print("[train_brain] done. n_calls=", getattr(model, "num_timesteps", "?"))


if __name__ == "__main__":
    main()
