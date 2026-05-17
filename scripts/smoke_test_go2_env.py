#!/usr/bin/env python3
"""Smoke-test the Go2 env wiring end to end.

Run this on a machine with the [motion] extras installed (mujoco_playground
+ brax + jax + mujoco). It verifies:

  1. robot_pet_cat.motion.go2_env imports and registers "Go2JoystickFlatTerrain"
     into mujoco_playground.registry.
  2. registry.load("Go2JoystickFlatTerrain") succeeds (model compiles).
  3. The compiled model has the bodies/sites/geoms we expect (base, FL_foot,
     FR_foot, RL_foot, RR_foot, and the FL/FR/RL/RR foot geoms).
  4. A single env.step() runs without error.

Usage:
  python scripts/smoke_test_go2_env.py
"""

from __future__ import annotations

import sys
import traceback


def main() -> int:
    failures: list[str] = []

    # 1. Registration
    print("[1/4] importing go2_env (triggers registration)...")
    try:
        from robot_pet_cat.motion import go2_env  # noqa: F401
        from mujoco_playground import registry
    except Exception as e:
        traceback.print_exc()
        print(f"  FAIL: {e}")
        return 1

    avail = sorted(registry.ALL_ENVS) if hasattr(registry, "ALL_ENVS") else []
    print(f"  registry.ALL_ENVS contains {len(avail)} envs")
    if "Go2JoystickFlatTerrain" not in avail:
        # Not fatal -- our legacy-API patch route doesn't always advertise via
        # ALL_ENVS but still resolves via registry.load.
        print("  (Go2JoystickFlatTerrain not in ALL_ENVS; will rely on patched load)")

    # 2. Load
    print("[2/4] registry.load('Go2JoystickFlatTerrain')...")
    try:
        env = registry.load("Go2JoystickFlatTerrain")
    except Exception as e:
        traceback.print_exc()
        print(f"  FAIL: {e}")
        return 1
    print(f"  loaded: {type(env).__name__}")

    # 3. Model inspection
    print("[3/4] inspecting compiled model...")
    mj_model = getattr(env, "mj_model", None) or getattr(env, "_mj_model", None)
    if mj_model is None:
        failures.append("env has no mj_model / _mj_model attribute")
    else:
        import mujoco
        expect_bodies = ["base"]
        expect_sites = ["imu", "FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        expect_geoms = ["floor", "FL", "FR", "RL", "RR"]
        for b in expect_bodies:
            try:
                _ = mj_model.body(b)
                print(f"  body  {b:<10} OK")
            except (KeyError, Exception) as e:
                failures.append(f"missing body {b}: {e}")
        for s in expect_sites:
            try:
                _ = mj_model.site(s)
                print(f"  site  {s:<10} OK")
            except (KeyError, Exception) as e:
                failures.append(f"missing site {s}: {e}")
        for g in expect_geoms:
            try:
                _ = mj_model.geom(g)
                print(f"  geom  {g:<10} OK")
            except (KeyError, Exception) as e:
                failures.append(f"missing geom {g}: {e}")
        print(f"  nq={mj_model.nq}  nv={mj_model.nv}  nu={mj_model.nu}")

    # 4. Step
    print("[4/4] reset + single step...")
    try:
        import jax
        import jax.numpy as jnp
        rng = jax.random.PRNGKey(0)
        state = env.reset(rng)
        act = jnp.zeros((mj_model.nu if mj_model is not None else env.action_size,))
        state2 = env.step(state, act)
        print(f"  obs shape: {state2.obs.shape if hasattr(state2.obs, 'shape') else type(state2.obs).__name__}")
        print(f"  reward: {float(state2.reward):.4f}")
    except Exception as e:
        traceback.print_exc()
        failures.append(f"step failed: {e}")

    if failures:
        print("\n=== FAILURES ===")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nAll checks passed. Go2 env is ready for AMP training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
