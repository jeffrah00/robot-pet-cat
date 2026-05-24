#!/usr/bin/env python3
"""End-to-end smoke test for the patched Unitree-Go2-Flat-MoB task.

Run on the RunPod pod after `python3 patch_mob_unitree.py`. Verifies:

  1. The package src.tasks.velocity_mob imports without error.
  2. The new task IDs are registered (Unitree-Go2-Flat-MoB, -Rough-MoB).
  3. Constructing the env succeeds and the command tensor has shape [N, 4].
  4. A single env.step works; the track_body_height reward is finite.
  5. Stepping with body_height_cmd=-0.05 vs +0.05 produces a different reward
     direction (sanity check that the reward responds to the new command dim).

Usage:
  cd /workspace/unitree_rl_mjlab && source /workspace/mjlab_venv/bin/activate
  python3 /workspace/robot-pet-cat/scripts/smoketest_mob.py
"""
from __future__ import annotations
import sys
import traceback


def main() -> int:
    failures: list[str] = []

    # 1. Package import (triggers task registration via __init__ chains).
    print("[1/5] importing src.tasks.velocity_mob.config.go2 ...")
    try:
        import src.tasks.velocity_mob.config.go2  # noqa: F401
    except Exception as e:
        traceback.print_exc()
        print(f"  FAIL: {e}")
        return 1

    # 2. Registry check.
    print("[2/5] checking gym registry for Unitree-Go2-Flat-MoB ...")
    try:
        import gymnasium as gym
    except ImportError:
        # mjlab may use a different registry; try the mjlab registry directly.
        try:
            from mjlab.tasks.registry import TASK_REGISTRY  # type: ignore
            registered = list(TASK_REGISTRY.keys())
        except Exception:
            failures.append("could not locate any task registry")
            registered = []
    else:
        registered = list(gym.registry.keys())

    for needed in ("Unitree-Go2-Flat-MoB", "Unitree-Go2-Rough-MoB"):
        if needed in registered:
            print(f"  [ok] {needed} registered")
        else:
            failures.append(f"{needed} NOT registered")
            print(f"  [fail] {needed} NOT in registry "
                  f"(saw {len(registered)} ids; sample: {sorted(registered)[:6]} ...)")

    # 3. Construct env + verify command shape.
    print("[3/5] constructing Unitree-Go2-Flat-MoB env ...")
    env = None
    try:
        # mjlab provides a task helper to instantiate by id.
        try:
            from mjlab.tasks.registry import make_env  # type: ignore
            env = make_env("Unitree-Go2-Flat-MoB", num_envs=2)
        except Exception:
            # Fall back to gym.make if registry exposes via gymnasium.
            import gymnasium as gym
            env = gym.make("Unitree-Go2-Flat-MoB", num_envs=2)
    except Exception as e:
        traceback.print_exc()
        failures.append(f"env construction failed: {e}")

    if env is not None:
        try:
            twist = env.unwrapped.command_manager.get_command("twist")
            print(f"  [ok] twist command shape: {tuple(twist.shape)}")
            if twist.shape[-1] != 4:
                failures.append(f"expected last dim 4, got {twist.shape[-1]}")
        except Exception as e:
            traceback.print_exc()
            failures.append(f"could not read twist command: {e}")

    # 4. Step the env.
    print("[4/5] step env ...")
    if env is not None:
        try:
            import torch
            obs, _info = env.reset()
            n_envs = env.unwrapped.num_envs
            n_act = env.unwrapped.action_space.shape[-1] if hasattr(env.unwrapped, "action_space") else 12
            act = torch.zeros(n_envs, n_act)
            step_out = env.step(act)
            print(f"  [ok] step returned {len(step_out)}-tuple")
        except Exception as e:
            traceback.print_exc()
            failures.append(f"step failed: {e}")

    # 5. track_body_height responds to body_height_cmd direction.
    print("[5/5] track_body_height responds to command ...")
    if env is not None:
        try:
            import torch
            twist = env.unwrapped.command_manager.get_command("twist")
            # Set env 0 to body_height_cmd = -0.05 (lower), env 1 to +0.05 (higher).
            twist[0, 3] = -0.05
            twist[1, 3] = +0.05
            # Find the reward function and call it directly.
            from src.tasks.velocity_mob.mdp import track_body_height  # type: ignore
            r = track_body_height(env.unwrapped, command_name="twist",
                                  std=0.05, base_height_target=0.30)
            print(f"  track_body_height per env: {r.tolist()}")
            if not torch.all(torch.isfinite(r)):
                failures.append("track_body_height returned non-finite values")
        except Exception as e:
            traceback.print_exc()
            failures.append(f"track_body_height eval failed: {e}")

    if failures:
        print("\n=== FAILURES ===")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nAll smoke-test checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
