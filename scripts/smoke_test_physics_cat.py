#!/usr/bin/env python3
"""Smoke test PhysicsCat: load the Go2 walker, drive it in the living room.

Designed to run on the RunPod pod (where model_6400.pt + unitree_rl_mjlab live).

Usage:
  python scripts/smoke_test_physics_cat.py \\
      --policy /workspace/unitree_rl_mjlab/logs/.../policy.onnx \\
      --steps 200 \\
      --command 0.3 0.0 0.0 \\
      --render /workspace/robot-pet-cat/renders/physics_cat_smoke.mp4

What it checks (in order):
  1. The merged scene compiles with the menagerie Go2 assets.
  2. The walker policy file loads (.onnx or .pt) and produces 12-d output.
  3. PhysicsCat resets to a clean standing pose at z=0.32.
  4. Under a constant forward command, the robot stays up and moves > 0.2 m
     in 10 sim seconds.
  5. (Optional) An mp4 is rendered showing the quadruped in the living room.

Exit code 0 on success, 1 on failure. Designed for stdout-only diagnostics
so it works fine in non-interactive pod sessions.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np


def _log(msg: str) -> None:
    print(f"[smoke] {msg}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--policy",
        required=True,
        help="Path to walker policy: .onnx, .pt, or directory containing either.",
    )
    p.add_argument(
        "--scene",
        default=None,
        help="Path to merged living-room+Go2 XML (defaults to "
        "data/go2_scenes/living_room_with_go2.xml).",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Number of brain-env steps to roll out (dt=0.05 -> 10 sim seconds at 200).",
    )
    p.add_argument(
        "--command",
        nargs=3,
        type=float,
        default=[0.3, 0.0, 0.0],
        metavar=("vx", "vy", "yaw_rate"),
        help="Constant locomotion command throughout the rollout.",
    )
    p.add_argument(
        "--render",
        default=None,
        help="If set, write an mp4 of the rollout from a free MuJoCo camera.",
    )
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument(
        "--initial-xy",
        nargs=2,
        type=float,
        default=[0.0, 0.0],
        metavar=("x", "y"),
    )
    p.add_argument("--initial-yaw", type=float, default=0.0)
    args = p.parse_args()

    # --- 1) Compile the merged scene ----------------------------------- #
    import mujoco
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from robot_pet_cat.brain.physics_cat import PhysicsCat, PhysicsCatConfig
    from robot_pet_cat.brain.go2_policy import GO2_POLICY_DT, load_go2_walker_policy
    from robot_pet_cat.motion.go2_base import get_assets
    from robot_pet_cat.skills.skill_policy import LocomotionCommand

    scene_path = (
        Path(args.scene)
        if args.scene
        else Path(__file__).resolve().parents[1]
        / "data"
        / "go2_scenes"
        / "living_room_with_go2.xml"
    )
    _log(f"Compiling scene: {scene_path}")
    xml_str = scene_path.read_text()
    assets = get_assets()
    t0 = time.perf_counter()
    model = mujoco.MjModel.from_xml_string(xml_str, assets=assets)
    data = mujoco.MjData(model)
    _log(
        f"  -> compiled in {time.perf_counter()-t0:.2f} s "
        f"(nbody={model.nbody}, njnt={model.njnt}, nu={model.nu})"
    )

    # --- 2) Load the walker policy -------------------------------------- #
    policy_path = Path(args.policy)
    _log(f"Loading walker policy: {policy_path}")
    t0 = time.perf_counter()
    policy = load_go2_walker_policy(policy_path)
    _log(f"  -> loaded in {time.perf_counter()-t0:.2f} s ({type(policy).__name__})")

    # --- 3) Instantiate PhysicsCat ------------------------------------- #
    cat = PhysicsCat(model, policy, PhysicsCatConfig())
    cat.reset(
        data,
        xy=tuple(args.initial_xy),
        yaw=float(args.initial_yaw),
        body_height=0.32,
    )
    _log(f"PhysicsCat reset; cat.xy={cat.xy}, yaw={cat.yaw:.3f}, z={cat.body_height:.3f}")

    # Validate observation shape against the policy's expected input.
    obs0 = cat._build_observation(data, np.zeros(3, dtype=np.float32))
    _log(f"  observation shape={obs0.shape} (expected 47 for flat env)")

    # Run policy once just to validate the round trip.
    try:
        act = policy(obs0[None, :])
    except Exception as e:
        _log(f"FATAL: policy(obs) failed: {e!r}")
        return 1
    if act.shape != (1, 12):
        _log(f"FATAL: policy returned shape {act.shape}, expected (1, 12)")
        return 1
    _log(f"  policy(obs) OK; action range=[{act.min():+.3f}, {act.max():+.3f}]")

    # --- 4) Roll out --------------------------------------------------- #
    vx, vy, wz = args.command
    cmd = LocomotionCommand(vx=vx, vy=vy, yaw_rate=wz, gait="trot")
    _log(
        f"Rolling out {args.steps} steps "
        f"@ dt=0.05 (= {args.steps * 0.05:.1f} sim s) "
        f"with command vx={vx}, vy={vy}, yaw_rate={wz}"
    )

    # Set up renderer if requested.
    renderer = None
    writer = None
    if args.render is not None:
        try:
            import imageio.v2 as imageio
        except ImportError:
            import imageio  # type: ignore
        renderer = mujoco.Renderer(model, width=args.width, height=args.height)
        Path(args.render).parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(args.render, fps=args.fps, codec="libx264")

    pose_history = []
    fell_over = False
    t_loop = time.perf_counter()
    for step in range(args.steps):
        cat.step(data, cmd, dt=0.05)
        pose_history.append((cat.xy[0], cat.xy[1], cat.body_height, cat.yaw))

        # Crude fall detection: base height collapses below ~0.12 m.
        if cat.body_height < 0.12:
            fell_over = True
            _log(f"  !! fell over at step {step} (z={cat.body_height:.3f})")
            break

        if renderer is not None and writer is not None:
            renderer.update_scene(data, camera=-1)  # use free camera default
            frame = renderer.render()
            writer.append_data(frame)

        if step % 50 == 0:
            _log(
                f"  step {step:4d}: xy=({cat.xy[0]:+.3f}, {cat.xy[1]:+.3f}) "
                f"z={cat.body_height:.3f} yaw={cat.yaw:+.3f} speed={cat.last_speed:.2f}"
            )

    elapsed = time.perf_counter() - t_loop
    n_done = len(pose_history)
    _log(
        f"  rollout: {n_done} steps in {elapsed:.2f} s "
        f"(real-time factor = {n_done * 0.05 / max(elapsed, 1e-6):.2f}x)"
    )

    if writer is not None:
        writer.close()
        _log(f"  rendered -> {args.render}")

    # --- 5) Pass/fail -------------------------------------------------- #
    final_xy = pose_history[-1]
    dist = math.hypot(final_xy[0] - args.initial_xy[0], final_xy[1] - args.initial_xy[1])
    _log(f"Final pose: xy=({final_xy[0]:+.3f}, {final_xy[1]:+.3f}) z={final_xy[2]:.3f}")
    _log(f"Distance traveled: {dist:.3f} m")

    cmd_norm = math.hypot(vx, vy)
    moving = cmd_norm >= 0.1 or abs(wz) >= 0.1
    moved_enough = dist > 0.20 if moving else dist < 0.10

    if fell_over:
        _log("RESULT: FAIL (robot fell over)")
        return 1
    if not moved_enough:
        verb = "moved too little" if moving else "drifted too much while stationary"
        _log(f"RESULT: FAIL (robot {verb}: {dist:.3f} m)")
        return 1
    _log("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
