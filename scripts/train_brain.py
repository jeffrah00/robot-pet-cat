#!/usr/bin/env python3
"""Launch a brain-policy PPO training run.

Usage:
  python scripts/train_brain.py                       # smoke run (CPU, 200 steps)
  python scripts/train_brain.py --steps 200000        # longer run
  python scripts/train_brain.py --steps 500000 \
      --curiosity --curiosity-w 0.05 \
      --decision-period-min-steps 40 \
      --decision-period-max-steps 300 \
      --wandb-project robot-pet-cat \
      --wandb-entity jeffrah89-personal \
      --wandb-run-name brain_v3_stochastic \
      --save checkpoints/brain/brain_v3_stochastic.zip

Defaults are tuned to be smoke-runnable in a few seconds on CPU. Real
training runs should set --steps, --device cpu (MlpPolicy is faster on
CPU than GPU for this obs size), and a wandb run name.

Key v3 addition: --decision-period-min-steps / --max-steps wire the
stochastic action-repeat into training (not just rendering). The policy
needs to experience the slow decision cadence during training to learn
the right macro-action distribution -- see memory stochastic-decision-period.
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
    p.add_argument("--steps", type=int, default=200, help="total_timesteps for PPO")
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
    p.add_argument("--decision-period-min-steps", type=int, default=1,
                   help="Min env-steps per skill decision (stochastic action-repeat). "
                        "1 = per-step policy query every tick (v0-v2 default). "
                        "40 = 2 s cat-like dwell at dt=0.05. "
                        "IMPORTANT: train and render with the same value -- "
                        "a policy trained at 20Hz doesn't learn the slower macro-action "
                        "rhythm (see memory stochastic-decision-period).")
    p.add_argument("--decision-period-max-steps", type=int, default=1,
                   help="Max env-steps per skill decision. 300 = 15 s max dwell. "
                        "v3 canonical: min=40, max=300 gives uniform [2-15 s] dwell.")
    p.add_argument("--use-physics-cat", action="store_true",
                   help="Use PhysicsCat (real Go2 MuJoCo) instead of KinematicCat stub. "
                        "~1000x slower per step; recommended for eval/render rather than "
                        "routine training. Requires --walker-policy-path.")
    p.add_argument("--walker-policy-path", default=None,
                   help="Path to trained Go2 velocity-walker policy (.onnx preferred, "
                        "or .pt RSL-RL checkpoint, or directory). "
                        "Required when --use-physics-cat.")
    p.add_argument("--use-cat-eye-obs", action="store_true",
                   help="Include 16x16 grayscale cat-eye camera frame in obs "
                        "(obs_version=1, 271-dim). Requires --use-physics-cat.")
    p.add_argument("--initial-cat-x", type=float, default=0.0,
                   help="Spawn x for the cat at episode reset. Default 0.")
    p.add_argument("--initial-cat-y", type=float, default=0.0,
                   help="Spawn y for the cat at episode reset. Default 0.")
    p.add_argument("--skill-min-duration", type=float, default=0.0,
                   help="Minimum seconds a non-HOLD skill stays active before "
                        "the policy can switch to a different non-HOLD skill. "
                        "HOLD always interrupts. Default 0 = per-step switching. "
                        "Deprecated: prefer --decision-period-* instead.")
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
        decision_period_min_steps=args.decision_period_min_steps,
        decision_period_max_steps=args.decision_period_max_steps,
        use_physics_cat=args.use_physics_cat,
        walker_policy_path=Path(args.walker_policy_path) if args.walker_policy_path else None,
        use_cat_eye_obs=args.use_cat_eye_obs,
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
          f"decision_period=[{args.decision_period_min_steps},{args.decision_period_max_steps}], "
          f"physics={'ON' if args.use_physics_cat else 'OFF'}, "
          f"cat_eye={'ON' if args.use_cat_eye_obs else 'OFF'}, "
          f"initial_xy=({args.initial_cat_x},{args.initial_cat_y}), "
          f"device={args.device}")
    model = train_brain(cfg)
    print("[train_brain] done. n_calls=", getattr(model, "num_timesteps", "?"))


if __name__ == "__main__":
    main()
