"""Standalone headless render of the lie_down skill in isolation.

Usage (from /workspace/robot-pet-cat):
    source /workspace/mjlab_venv/bin/activate
    MUJOCO_GL=osmesa python scripts/render_lie_down.py \\
        --policy /workspace/mjlab_playground/logs/rsl_rl/go1_velocity/2026-05-25_05-54-47/2026-05-25_05-54-47.onnx \\
        --out renders/lie_down_manual_v1.mp4

The script:
    1. Builds a PhysicsCat-backed BrainEnv (loads walker policy + Go1 model).
    2. Runs 30 stand ticks to settle (0.6 s sim).
    3. Runs LieDown.step() for 400 ticks (8 s sim, covers full ramp + hold).
    4. Renders one frame per tick at 20 fps -> ~14 s wall video.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

# ---- repo on path ----
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from robot_pet_cat.brain.env import BrainEnv, BrainEnvConfig
from robot_pet_cat.skills.lie_down import LieDown
from robot_pet_cat.brain.physics_cat import LocomotionCommand


def _log(msg: str) -> None:
    print(f"[render_lie_down]  {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="Walker policy path (.onnx or .pt)")
    ap.add_argument("--out", default="renders/lie_down_manual_v1.mp4")
    ap.add_argument("--stand-ticks", type=int, default=30,
                    help="Settle ticks in stand before triggering lie_down")
    ap.add_argument("--lie-ticks", type=int, default=400,
                    help="Ticks to run lie_down (covers 120-tick ramp + hold)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Build env
    # ------------------------------------------------------------------ #
    _log("building BrainEnv with PhysicsCat...")
    t0 = time.perf_counter()
    cfg = BrainEnvConfig(
        use_physics_cat=True,
        walker_policy_path=Path(args.policy),
        max_steps_per_episode=args.stand_ticks + args.lie_ticks + 10,
    )
    env = BrainEnv(cfg)
    _log(f"  env built in {time.perf_counter()-t0:.1f}s")

    obs = env.reset()
    mj_model = env.mj_model
    mj_data  = env.mj_data
    cat      = env.cat
    dt       = cfg.dt_s

    # ------------------------------------------------------------------ #
    # Renderer + tracking camera
    # ------------------------------------------------------------------ #
    renderer = mujoco.Renderer(mj_model, width=args.width, height=args.height)

    # Chase camera: track base_link from behind-and-above (same as render_brain_3d)
    base_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    chase = mujoco.MjvCamera()
    chase.type        = mujoco.mjtCamera.mjCAMERA_TRACKING
    chase.trackbodyid = base_id
    chase.distance    = 2.5
    chase.azimuth     = 120.0
    chase.elevation   = -20.0
    chase.lookat[:]   = [0.0, 0.0, 0.20]
    _log(f"tracking base_link body_id={base_id}, distance={chase.distance}")

    frames: list[np.ndarray] = []
    render_every = max(1, round(1.0 / (args.fps * dt)))
    _log(f"render_every={render_every} ticks (dt={dt}s, fps={args.fps})")

    def capture() -> None:
        renderer.update_scene(mj_data, camera=chase)
        frames.append(renderer.render().copy())

    # ------------------------------------------------------------------ #
    # Settle in stand
    # ------------------------------------------------------------------ #
    _log(f"settling for {args.stand_ticks} stand ticks...")
    stand_cmd = LocomotionCommand(vx=0.0, vy=0.0, yaw_rate=0.0, gait="stand")
    for tick in range(args.stand_ticks):
        cat.step(mj_data, stand_cmd, dt)
        if tick % render_every == 0:
            capture()

    # ------------------------------------------------------------------ #
    # Run lie_down skill
    # ------------------------------------------------------------------ #
    _log(f"running lie_down for {args.lie_ticks} ticks (ramp={120} + hold)...")
    skill = LieDown()
    skill.reset()
    for tick in range(args.lie_ticks):
        cmd = skill.step({}, None)
        cat.step(mj_data, cmd, dt)
        if tick % render_every == 0:
            capture()
        if tick % 50 == 0:
            alpha = min(1.0, (tick + 1) / 120)
            _log(f"  tick {tick:4d}/{args.lie_ticks}  alpha={alpha:.2f}  frames={len(frames)}")

    # ------------------------------------------------------------------ #
    # Write video
    # ------------------------------------------------------------------ #
    _log(f"writing {len(frames)} frames to {out_path} ...")
    # Sanity-check first frame
    if frames:
        f0 = frames[0]
        _log(f"  frame[0] shape={f0.shape} max={f0.max()} mean={f0.mean():.1f}")

    try:
        import mediapy
        mediapy.write_video(str(out_path), frames, fps=args.fps)
    except ImportError:
        # Fallback: use imageio
        import imageio.v2 as iio
        writer = iio.get_writer(str(out_path), fps=args.fps, quality=8)
        for f in frames:
            writer.append_data(f)
        writer.close()

    _log(f"done -> {out_path}  ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
