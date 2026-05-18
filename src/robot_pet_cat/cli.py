"""Top-level CLI. Wires subcommands as the project grows.

Usage:
    rpc sim --render
    rpc train-motion    --config configs/motion/cat_amp.yaml       # v1
    rpc train-motion-v2 --config configs/motion/cat_amp_v2.yaml    # true AMP
    rpc train-skill     --skill walk_to
    rpc train-brain     --config configs/brain/cat_brain.yaml
    rpc retarget        --clips data/motion_clips_raw
    rpc demo
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rpc", description="robot-pet-cat CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sim = sub.add_parser("sim", help="Load the Unitree Go2 in MuJoCo and step it.")
    sim.add_argument("--render", action="store_true", help="Open a viewer window.")
    sim.add_argument("--steps", type=int, default=1000)

    tmotion = sub.add_parser(
        "train-motion",
        help="Tier 1 v1: AMP with frozen D, brax stock PPO.",
    )
    tmotion.add_argument("--config", required=True)

    tmotion_v2 = sub.add_parser(
        "train-motion-v2",
        help="Tier 1 v2: true AMP with live D-vs-rollouts, custom PPO loop.",
    )
    tmotion_v2.add_argument("--config", required=True)

    tmotion_imit = sub.add_parser(
        "train-motion-imitation",
        help="Tier 1 Phase 1: kinematic imitation + RSI (no discriminator).",
    )
    tmotion_imit.add_argument("--config", required=True)

    tskill = sub.add_parser(
        "train-skill", help="Tier 2: a single skill (walk_to, sit, jump_to, swat, ...).",
    )
    tskill.add_argument("--skill", required=True)
    tskill.add_argument("--config", default=None)

    tbrain = sub.add_parser(
        "train-brain",
        help="Tier 3: high-level RL with curiosity + comfort + play + mood.",
    )
    tbrain.add_argument("--config", required=True)

    retarget = sub.add_parser(
        "retarget",
        help="Retarget cat keypoint clips to Go2 reference trajectories.",
    )
    retarget.add_argument(
        "--clips", required=True,
        help="Directory of keypoint .json files.",
    )
    retarget.add_argument(
        "--out", default="data/motion_clips",
        help="Output directory for .npz files (default: data/motion_clips).",
    )

    sub.add_parser("demo", help="Run the cat in the household scene for N seconds.")

    args = parser.parse_args(argv)

    if args.cmd == "sim":
        from robot_pet_cat.sim.mujoco_env import smoke_test
        return smoke_test(render=args.render, steps=args.steps)

    if args.cmd == "train-motion":
        from pathlib import Path
        import yaml
        from robot_pet_cat.motion.amp_trainer import (
            AMPTrainConfig, PPOConfig, train,
        )
        with open(args.config) as f:
            raw = yaml.safe_load(f)
        ppo_kwargs = raw.pop("ppo", {})
        for k in ("motion_clips_dir", "checkpoint_dir"):
            if k in raw:
                raw[k] = Path(raw[k])
        cfg = AMPTrainConfig(ppo=PPOConfig(**ppo_kwargs), **raw)
        train(cfg)
        return 0

    if args.cmd == "train-motion-v2":
        from pathlib import Path
        import yaml
        from robot_pet_cat.motion.amp_trainer_v2 import (
            AMPTrainConfigV2,
            PPOConfig as PPOConfigV2,
            train as train_v2,
        )
        with open(args.config) as f:
            raw = yaml.safe_load(f)
        ppo_kwargs = raw.pop("ppo", {})
        for k in ("motion_clips_dir", "checkpoint_dir", "init_policy_ckpt"):
            if k in raw and raw[k] is not None:
                raw[k] = Path(raw[k])
        cfg = AMPTrainConfigV2(ppo=PPOConfigV2(**ppo_kwargs), **raw)
        train_v2(cfg)
        return 0

    if args.cmd == "train-motion-imitation":
        from pathlib import Path
        import yaml
        from robot_pet_cat.motion.imitation_trainer import (
            ImitationTrainConfig, PPOConfig as PPOConfigImit, train as train_imit,
        )
        with open(args.config) as f:
            raw = yaml.safe_load(f)
        ppo_kwargs = raw.pop("ppo", {})
        for k in ("motion_clips_dir", "checkpoint_dir"):
            if k in raw and raw[k] is not None:
                raw[k] = Path(raw[k])
        cfg = ImitationTrainConfig(ppo=PPOConfigImit(**ppo_kwargs), **raw)
        train_imit(cfg)
