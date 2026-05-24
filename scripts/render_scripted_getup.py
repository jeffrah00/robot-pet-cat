#!/usr/bin/env python3
"""Slim render for the scripted get_up demo.

Tighter memory budget than render_brain_3d.py: avoids onnxruntime by
substituting a stub walker that emits zero actions. Pod cgroup is
~489 MB; loading the trained walker via onnxruntime pushes us over
that limit before any rendering can happen. We don't need a real
walker for this demo -- the only thing we want to see is:

  1. Cat spawned standing (PhysicsCat default pose).
  2. Roll/yaw impulse at t=1.0s knocks it over.
  3. At t=2.5s gait switches to "get_up" and ScriptedGetUpPolicy
     drives joint targets through tuck -> push -> stand keyframes.

A stub walker (always-zero action) means the cat just holds its
default standing pose under PD until we knock it over. That's fine.

Usage (pod):
  MUJOCO_GL=osmesa python render_scripted_getup.py \\
      --out renders/get_up_scripted_v1.mp4 --steps 200
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np


def _log(msg: str) -> None:
    print(f"[scripted_getup] {msg}", flush=True)


class _StubWalker:
    """Walker substitute that emits zero actions (keeps cat in default pose).

    Conforms to the WalkerPolicy callable protocol. obs_dim=47 matches the
    real velocity walker. We don't use it -- we knock the cat over and
    immediately switch to get_up gait -- but PhysicsCat is built around
    having SOME walker, and this avoids loading onnxruntime.
    """
    obs_dim: int = 47

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        n = obs.shape[0] if obs.ndim >= 2 else 1
        return np.zeros((n, 12), dtype=np.float32)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="renders/get_up_scripted_v1.mp4")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=320)
    p.add_argument("--knock-at", type=float, default=1.0,
                   help="Sim time at which to apply the knock-down impulse.")
    p.add_argument("--get-up-at", type=float, default=2.5,
                   help="Sim time at which to switch to gait='get_up'.")
    p.add_argument("--start-prone", action="store_true",
                   help="Initialize cat belly-down (chassis at z=0.06, legs "
                        "tucked PRONE) and skip the knock-down. The scripted "
                        "get_up policy then ramps it back to standing.")
    args = p.parse_args()

    # Lazy imports
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    sys.path.insert(0, "/workspace/robot-pet-cat/src")

    import mujoco
    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio  # type: ignore

    from robot_pet_cat.brain.env import BrainEnv, BrainEnvConfig
    from robot_pet_cat.brain.scripted_get_up import ScriptedGetUpPolicy
    from robot_pet_cat.skills.skill_policy import LocomotionCommand

    _log("building BrainEnv with PhysicsCat + stub walker + ScriptedGetUpPolicy")
    cfg = BrainEnvConfig(
        use_physics_cat=True,
        walker_policy_path=_StubWalker(),  # type: ignore[arg-type]
        initial_cat_xy=(0.0, 0.0),
        max_steps_per_episode=args.steps + 10,
        get_up_policy_path=ScriptedGetUpPolicy(),  # type: ignore[arg-type]
    )
    t0 = time.perf_counter()
    env = BrainEnv(cfg)
    env.reset()
    _log(f"  built in {time.perf_counter()-t0:.1f}s, xy={env.cat.xy} z={env.cat.body_height:.3f}")

    if args.start_prone:
        # Override the post-reset pose to belly-down. Chassis just above the
        # floor; joints in PRONE (hips splayed 0.30, thighs forward 1.50,
        # calves fully folded -2.60) so feet sit at chassis level beside hips.
        PRONE_JOINT_POS = np.array([
            -0.30, 1.50, -2.60,   # FL
            +0.30, 1.50, -2.60,   # FR
            -0.30, 1.50, -2.60,   # RL
            +0.30, 1.50, -2.60,   # RR
        ], dtype=np.float32)
        cat = env.cat
        mj_data = env.mj_data
        mj_data.qpos[cat._base_qpos_adr + 2] = 0.06
        for i in range(12):
            mj_data.qpos[cat._joint_qpos_adr[i]] = float(PRONE_JOINT_POS[i])
        mj_data.qvel[:] = 0.0
        # Pre-load actuators with the prone targets so PD doesn't fight at t=0.
        for i in range(12):
            mj_data.ctrl[cat._actuator_ids[i]] = float(PRONE_JOINT_POS[i])
        mujoco.mj_forward(env.mj_model, mj_data)
        cat._refresh_pose_cache(mj_data)
        _log(f"  start-prone: cat at z={cat.body_height:.3f}, knock-down disabled")

    _log(f"creating renderer {args.width}x{args.height}")
    renderer = mujoco.Renderer(env.mj_model, width=args.width, height=args.height)

    # Chase camera tracking the cat
    base_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = base_id
    cam.distance = 2.0
    cam.azimuth = 135.0
    cam.elevation = -25.0
    cam.lookat[:] = [0.0, 0.0, 0.18]

    # Output writer
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=args.fps, codec="libx264")

    # Rollout loop -- we bypass env.step() entirely and drive PhysicsCat
    # directly with explicit LocomotionCommands. This way we don't pay
    # for the skill registry / mood / reward bookkeeping.
    _log(f"rolling out {args.steps} steps -> {out_path}")
    cat = env.cat
    mj_data = env.mj_data
    dt = env.cfg.dt_s

    knocked = False
    # --start-prone fires get_up immediately and skips the knock-down.
    switched = bool(args.start_prone)
    if switched:
        _log(f"  starting in gait='get_up' (start-prone mode)")
    t_loop = time.perf_counter()
    for step in range(args.steps):
        t_sim = step * dt

        # Knock-down impulse (disabled in --start-prone mode)
        if (not args.start_prone) and (not knocked) and t_sim >= args.knock_at:
            vadr = cat._base_qvel_adr
            mj_data.qvel[vadr + 0] = 0.5
            mj_data.qvel[vadr + 2] = 1.2
            mj_data.qvel[vadr + 3] = 12.0  # roll about body +x
            knocked = True
            _log(f"  knocked at t={t_sim:.2f}s")

        # Switch gait
        if not switched and t_sim >= args.get_up_at:
            switched = True
            _log(f"  switching to gait='get_up' at t={t_sim:.2f}s")

        gait = "get_up" if switched else "stand"
        cmd = LocomotionCommand(vx=0.0, vy=0.0, yaw_rate=0.0, gait=gait)
        cat.step(mj_data, cmd, dt)

        # Render frame
        renderer.update_scene(mj_data, camera=cam)
        img = renderer.render()
        writer.append_data(img)

        if (step + 1) % 25 == 0:
            _log(f"  step {step+1:4d}/{args.steps}  "
                 f"t={t_sim:5.2f}s  z={cat.body_height:.3f}  yaw={cat.yaw:+.2f}")

    writer.close()
    _log(f"done in {time.perf_counter()-t_loop:.1f}s -> {out_path}  "
         f"({out_path.stat().st_size/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
