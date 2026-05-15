"""Top-level CLI. Wires subcommands as the project grows.

Usage:
    rpc sim --render
    rpc train-locomotion --config configs/locomotion/unitree_go2_ppo.yaml
    rpc collect-data --source pexels
    rpc finetune-vla --config configs/vla/smolvla_cat.yaml
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

    train = sub.add_parser("train-locomotion", help="PPO training for low-level walking.")
    train.add_argument("--config", required=True)

    data = sub.add_parser("collect-data", help="Scrape and process cat videos.")
    data.add_argument("--source", choices=["pexels", "youtube", "manifest"], required=True)

    ft = sub.add_parser("finetune-vla", help="Fine-tune SmolVLA on cat behaviors.")
    ft.add_argument("--config", required=True)

    args = parser.parse_args(argv)

    if args.cmd == "sim":
        from robot_pet_cat.sim.mujoco_env import smoke_test

        return smoke_test(render=args.render, steps=args.steps)
    if args.cmd == "train-locomotion":
        print(f"[stub] would train locomotion with {args.config}")
        return 0
    if args.cmd == "collect-data":
        print(f"[stub] would collect data from {args.source}")
        return 0
    if args.cmd == "finetune-vla":
        print(f"[stub] would fine-tune VLA with {args.config}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
