"""Headless render: get_up -> lie_belly skill chain.

Usage (from /workspace/robot-pet-cat):
    source /workspace/robot-pet-cat/.env
    source /workspace/mjlab_venv/bin/activate
    MUJOCO_GL=osmesa python scripts/render_lie_belly.py \\
        --policy /workspace/mjlab_playground/logs/rsl_rl/go1_velocity/2026-05-25_05-54-47/2026-05-25_05-54-47.onnx \\
        --out renders/lie_belly_go1_v1.mp4

Pipeline:
    Phase 1 (get_up, recorded): run gait="get_up" for up to --getup-ticks.
        Stops early once cat.body_height > STAND_HEIGHT_THRESH (robot is
        upright). This forces a known standing pose before the belly transition.
    Phase 2 (lie_belly, recorded): run LieBelly.step() for --lie-ticks.
        120-tick linear ramp thigh->1.60 / calf->-2.70, then hold.
        PhysicsCat's joint_targets bypass drives actuators directly.

Rendering rules:
    - MUJOCO_GL=osmesa  (EGL throws eglQueryString error on this pod)
    - /workspace/mjlab_venv  (never system Python)
    - source /workspace/robot-pet-cat/.env first (WANDB_API_KEY etc.)
    - No Xvfb needed with osmesa
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from robot_pet_cat.brain.env import BrainEnv, BrainEnvConfig
from robot_pet_cat.skills.get_up import GetUp
from robot_pet_cat.skills.lie_belly import LieBelly
from robot_pet_cat.brain.physics_cat import LocomotionCommand
from robot_pet_cat.brain.go2_policy import load_get_up_policy

# Body height threshold above which the robot is considered to be standing.
STAND_HEIGHT_THRESH = 0.25  # metres


def _log(msg: str) -> None:
    print(f"[render_lie_belly]  {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="models/go1_walker_v0.pt",
                    help="Walker policy path")
    ap.add_argument("--getup-policy", default="models/get_up_go1_v1.pt",
                    help="Mjlab go1_getup checkpoint")
    ap.add_argument("--out", default="renders/lie_belly_go1_v1.mp4")
    ap.add_argument("--getup-ticks", type=int, default=300,
                    help="Max ticks for get_up phase (stops early when standing)")
    ap.add_argument("--lie-ticks", type=int, default=400,
                    help="Ticks to run lie_belly (covers 120-tick ramp + hold)")
    ap.add_argument("--width",  type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps",    type=int, default=20)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Build env
    # ------------------------------------------------------------------ #
    _log("building BrainEnv with PhysicsCat...")
    t0 = time.perf_counter()
    total_ticks = args.getup_ticks + args.lie_ticks + 10
    cfg = BrainEnvConfig(
        use_physics_cat=True,
        walker_policy_path=Path(args.policy),
        max_steps_per_episode=total_ticks,
    )
    env = BrainEnv(cfg)
    _log(f"  env built in {time.perf_counter()-t0:.1f}s")

    obs      = env.reset()
    mj_model = env.mj_model
    mj_data  = env.mj_data
    cat      = env.cat
    dt       = cfg.dt_s

    # Inject mjlab get_up policy
    cat.get_up_policy = load_get_up_policy(args.getup_policy)
    cat._last_active_policy = "walker"
    print(f"[render_lie_belly]  get_up policy obs_dim={cat.get_up_policy.obs_dim}", flush=True)
    # Spawn prone
    import mujoco as _mj
    _adr = cat._base_qpos_adr; _vadr = cat._base_qvel_adr
    _pj = [+0.30,1.60,-2.70, -0.30,1.60,-2.70, +0.30,1.60,-2.70, -0.30,1.60,-2.70]
    mj_data.qpos[_adr:_adr+7] = [0.,0.,0.06, 1.,0.,0.,0.]
    mj_data.qvel[_vadr:_vadr+6] = 0.
    for _i in range(12):
        mj_data.qpos[cat._joint_qpos_adr[_i]] = _pj[_i]
        mj_data.qvel[cat._joint_qvel_adr[_i]] = 0.
        mj_data.ctrl[cat._actuator_ids[_i]] = _pj[_i]
    _mj.mj_forward(mj_model, mj_data); cat._refresh_pose_cache(mj_data)
    print(f"[render_lie_belly]  spawned prone z={cat.body_height:.3f}m", flush=True)

    # ------------------------------------------------------------------ #
    # Renderer + tracking camera
    # ------------------------------------------------------------------ #
    renderer = mujoco.Renderer(mj_model, width=args.width, height=args.height)

    # Chase camera: track trunk (Go1) or base_link (Go2) from behind-and-above.
    _base_name = (
        "trunk"
        if mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "trunk") >= 0
        else "base_link"
    )
    base_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, _base_name)
    chase = mujoco.MjvCamera()
    chase.type        = mujoco.mjtCamera.mjCAMERA_TRACKING
    chase.trackbodyid = base_id
    chase.distance    = 2.5
    chase.azimuth     = 120.0
    chase.elevation   = -20.0
    chase.lookat[:]   = [0.0, 0.0, 0.20]
    _log(f"tracking body '{_base_name}' id={base_id}")

    frames: list[np.ndarray] = []
    render_every = max(1, round(1.0 / (args.fps * dt)))
    _log(f"render_every={render_every} ticks  (dt={dt}s, fps={args.fps})")

    def capture() -> None:
        renderer.update_scene(mj_data, camera=chase)
        frames.append(renderer.render().copy())

    # ------------------------------------------------------------------ #
    # Phase 1: get_up skill -- recorded
    # Run gait="get_up" until the body height exceeds STAND_HEIGHT_THRESH
    # or we exhaust --getup-ticks. This forces a known upright stance before
    # the lie_belly transition (avoids belly-to-belly with no visible getup).
    # ------------------------------------------------------------------ #
    _log(f"Phase 1: get_up (max {args.getup_ticks} ticks, "
         f"early-stop at height>{STAND_HEIGHT_THRESH}m)...")
    get_up_skill = GetUp()
    get_up_skill.reset()
    get_up_cmd = LocomotionCommand(vx=0.0, vy=0.0, yaw_rate=0.0, gait="get_up")

    ticks_used = 0
    for tick in range(args.getup_ticks):
        get_up_skill.step({}, None)   # advance internal step counter
        cat.step(mj_data, get_up_cmd, dt)
        ticks_used += 1
        if tick % render_every == 0:
            capture()
        if tick % 50 == 0:
            h = cat.body_height
            _log(f"  get_up tick {tick:4d}  height={h:.3f}m  frames={len(frames)}")
        # Early stop: robot is upright
        if cat.body_height > STAND_HEIGHT_THRESH and tick > 3:
            _log(f"  standing at tick {tick} (height={cat.body_height:.3f}m), "
                 "moving to lie_belly")
            break

    _log(f"  get_up phase done: {ticks_used} ticks, {len(frames)} frames, "
         f"final height={cat.body_height:.3f}m")

    # ------------------------------------------------------------------ #
    # Phase 2: lie_belly skill -- recorded
    # LieBelly.step() linearly ramps joint_targets from STAND_JOINT_POS to
    # LOAF_JOINT_TARGETS (thigh=1.60, calf=-2.70) over 120 ticks, then holds.
    # ------------------------------------------------------------------ #
    _log(f"Phase 2: lie_belly for {args.lie_ticks} ticks (ramp=120 + hold)...")
    skill = LieBelly()
    skill.reset()
    for tick in range(args.lie_ticks):
        cmd = skill.step({}, None)
        cat.step(mj_data, cmd, dt)
        if tick % render_every == 0:
            capture()
        if tick % 50 == 0:
            alpha = min(1.0, (tick + 1) / 120)
            _log(f"  lie_belly tick {tick:4d}/{args.lie_ticks}  "
                 f"alpha={alpha:.2f}  frames={len(frames)}")

    # ------------------------------------------------------------------ #
    # Write video
    # ------------------------------------------------------------------ #
    _log(f"writing {len(frames)} frames to {out_path} ...")
    if frames:
        f0 = frames[0]
        _log(f"  frame[0] shape={f0.shape} max={f0.max()} mean={f0.mean():.1f}")

    try:
        import mediapy
        mediapy.write_video(str(out_path), frames, fps=args.fps)
    except ImportError:
        import imageio.v2 as iio
        writer = iio.get_writer(str(out_path), fps=args.fps, quality=8)
        for f in frames:
            writer.append_data(f)
        writer.close()

    size_kb = out_path.stat().st_size // 1024
    _log(f"done -> {out_path}  ({size_kb} KB)")
    if size_kb < 50:
        _log("WARNING: output is very small -- may be grey/blank frames. "
             "Check MUJOCO_GL=osmesa is set and osmesa/ffmpeg are installed.")


if __name__ == "__main__":
    main()
