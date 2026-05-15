"""Top-level CLI. Wires subcommands as the project grows.

Usage:
    rpc sim --render
    rpc train-motion --config configs/motion/cat_amp.yaml
    rpc train-skill  --skill walk_to
    rpc train-brain  --config configs/brain/cat_brain.yaml
    rpc retarget     --clips data/motion_clips_raw
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
        "train-motion", help="Tier 1: AMP-trained cat-style locomotion."
    )
    tmotion.add_argument("--config", required=True)

    tskill = sub.add_parser(
        "train-skill", help="Tier 2: a single skill (walk_to, sit, jump_to, swat, ...)."
    )
    tskill.add_argument("--skill", required=True)
    tskill.add_argument("--config", default=None)

    tbrain = sub.add_parser(
        "train-brain",
        help="Tier 3: high-level RL with curiosity + comfort + play + mood.",
    )
    tbrain.add_argument("--config", required=True)

    retarget = sub.add_parser(
        "retarget", help="Extract pose + retarget cat clips to Go2 reference trajectories."
    )
    retarget.add_argument("--clips", required=True, help="Directory of raw .mp4 clips.")

    sub.add_parser("demo", help="Run the cat in the household scene for N seconds.")

    args = parser.parse_args(argv)

    if args.cmd == "sim":
        from robot_pet_cat.sim.mujoco_env import smoke_test

        return smoke_test(render=args.render, steps=args.steps)
    if args.cmd == "train-motion":
        print(f"[stub] would AMP-train motion with {args.config}")
        return 0
    if args.cmd == "train-skill":
        print(f"[stub] would train skill {args.skill} (config={args.config})")
        return 0
    if args.cmd == "train-brain":
        print(f"[stub] would train brain with {args.config}")
        return 0
    if args.cmd == "retarget":
        print(f"[stub] would retarget clips in {args.clips}")
        return 0
    if args.cmd == "demo":
        print("[stub] would spawn cat in living room and run for 60 seconds")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
