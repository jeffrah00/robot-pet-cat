#!/usr/bin/env python3
"""Skill showcase render using render_brain_3d.py as the base.

Same dual-camera layout + PIL HUD as render_brain_3d.py, but the action
stream is a scripted skill sequence rather than a PPO checkpoint.

Output panel layout:
  +------------------------------+----------------+
  |   Third-person (chase cam)   |   cat_eye      |
  |   + HUD overlay              |   (POV inset)  |
  +------------------------------+----------------+

Skills in showcase (swat removed, jump_to uses joint-trajectory):
  stand -> sit -> stand -> lie_down -> stand -> crouch -> stand
        -> stretch -> stand -> walk_to -> stand -> jump_to -> stand

Usage (pod):
  MUJOCO_GL=osmesa python scripts/render_skill_showcase_v3.py \\
      --policy /workspace/unitree_rl_mjlab/logs/Go2JoystickFlatEnv/*/policy.onnx \\
      --out renders/skill_showcase_v3.mp4
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Showcase schedule: (hold_seconds, skill_name, hud_label)
# stand entries use skill_name="" -- the env holds (action=0).
# ---------------------------------------------------------------------------

SHOWCASE: list[tuple[float, str, str]] = [
    (2.0, "",         "STAND"),
    (5.0, "sit",      "SIT"),
    (2.0, "",         "STAND"),
    (5.0, "lie_down", "LIE DOWN"),
    (2.0, "",         "STAND"),
    (4.0, "crouch",   "CROUCH"),
    (2.0, "",         "STAND"),
    (4.0, "stretch",  "STRETCH"),
    (2.0, "",         "STAND"),
    (6.0, "walk_to",  "WALK TO"),
    (2.0, "",         "STAND"),
    (6.0, "jump_to",  "JUMP (trajectory)"),
    (3.0, "",         "STAND"),
]

TOTAL_S = sum(d for d, _, _ in SHOWCASE)


def _showcase_action(showcase_t: float, env) -> tuple[int, str]:
    """Return (action_int, label) for the given showcase clock time."""
    acc = 0.0
    for duration, skill_name, label in SHOWCASE:
        acc += duration
        if showcase_t < acc:
            if not skill_name:
                return 0, label  # HOLD
            skills = list(env.skill_names)
            if skill_name in skills:
                return 1 + skills.index(skill_name), label
            return 0, label
    return 0, "STAND"


# ---------------------------------------------------------------------------
# PIL HUD (same as render_brain_3d.py)
# ---------------------------------------------------------------------------

def _load_font(size: int = 18):
    from PIL import ImageFont
    for cand in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ):
        if os.path.exists(cand):
            try:
                return ImageFont.truetype(cand, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _draw_hud(img_array: np.ndarray, lines: list[str], font) -> np.ndarray:
    from PIL import Image, ImageDraw
    im = Image.fromarray(img_array)
    draw = ImageDraw.Draw(im, "RGBA")
    line_h = font.size + 6
    panel_w = max((font.getlength(line) for line in lines), default=0) + 24
    panel_h = line_h * len(lines) + 12
    draw.rectangle([(6, 6), (6 + panel_w, 6 + panel_h)], fill=(0, 0, 0, 170))
    y = 12
    for ln in lines:
        draw.text((14, y), ln, fill=(240, 240, 240, 255), font=font)
        y += line_h
    return np.array(im)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _log(s: str) -> None:
    print(f"[showcase_v3] {s}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True,
                   help="Go2 walker policy (.onnx, .pt, or run dir).")
    p.add_argument("--steps", type=int, default=0,
                   help="Override total steps (0 = auto from schedule).")
    p.add_argument("--out", default="renders/skill_showcase_v3.mp4")
    p.add_argument("--width-main",  type=int, default=960)
    p.add_argument("--height-main", type=int, default=640)
    p.add_argument("--width-eye",   type=int, default=320)
    p.add_argument("--height-eye",  type=int, default=320)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--initial-xy", nargs=2, type=float, default=[0.5, 0.5])
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    import mujoco
    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio  # type: ignore

    from robot_pet_cat.brain.env import BrainEnv, BrainEnvConfig

    brain_dt = 0.05
    total_s = TOTAL_S
    total_steps = args.steps if args.steps > 0 else int(total_s / brain_dt) + 20

    cfg = BrainEnvConfig(
        use_physics_cat=True,
        walker_policy_path=Path(args.policy),
        initial_cat_xy=tuple(args.initial_xy),
        max_steps_per_episode=total_steps + 100,
        mode_policy=None,
        decision_period_min_steps=1,
        decision_period_max_steps=1,
    )
    _log("building BrainEnv with PhysicsCat...")
    t0 = time.perf_counter()
    env = BrainEnv(cfg)
    _log(f"  env built in {time.perf_counter()-t0:.1f}s  "
         f"skills={list(env.skill_names)}")

    obs = env.reset()
    _log(f"reset OK; cat at xy={obs['cat_xy']}")

    _log(f"renderers: main={args.width_main}x{args.height_main}, "
         f"eye={args.width_eye}x{args.height_eye}")
    renderer_main = mujoco.Renderer(env.mj_model,
                                    width=args.width_main,
                                    height=args.height_main)
    renderer_eye  = mujoco.Renderer(env.mj_model,
                                    width=args.width_eye,
                                    height=args.height_eye)

    base_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    chase = mujoco.MjvCamera()
    chase.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    chase.trackbodyid = base_id
    chase.distance = 2.8
    chase.azimuth = 135.0
    chase.elevation = -25.0
    chase.lookat[:] = [0.0, 0.0, 0.25]

    font = _load_font(size=18)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out_path, fps=args.fps, codec="libx264")

    _log(f"rolling out {total_steps} steps -> {out_path}")
    _log(f"showcase total: {total_s:.1f}s  ({len(SHOWCASE)} segments)")

    showcase_t = 0.0

    for step in range(total_steps):
        action, skill_label = _showcase_action(showcase_t, env)
        showcase_t = min(showcase_t + brain_dt, total_s)

        obs, reward, terminated, truncated, info = env.step(action)
        if truncated:
            break

        # Third-person render
        renderer_main.update_scene(env.mj_data, camera=chase)
        img_main = renderer_main.render()

        # Cat-eye render
        eye_cam_id = mujoco.mj_name2id(env.mj_model,
                                        mujoco.mjtObj.mjOBJ_CAMERA, "cat_eye")
        renderer_eye.update_scene(env.mj_data, camera=eye_cam_id)
        img_eye = renderer_eye.render()

        # HUD
        cmd_arr = getattr(env.cat, "_last_cmd_arr", np.zeros(3))
        hud = [
            f"t = {env.t_sim:6.2f}s   step {step:4d}",
            f"skill : {skill_label}",
            f"cmd   : vx={cmd_arr[0]:+.2f}  vy={cmd_arr[1]:+.2f}  wz={cmd_arr[2]:+.2f}",
            f"xy    : ({env.cat.xy[0]:+.2f}, {env.cat.xy[1]:+.2f})   z={env.cat.body_height:.2f}",
            f"yaw   : {env.cat.yaw:+.2f} rad   speed={env.cat.last_speed:.2f} m/s",
        ]
        img_main = _draw_hud(img_main, hud, font)

        # Composite: main | eye panel
        h_main, w_main = img_main.shape[:2]
        h_eye,  w_eye  = img_eye.shape[:2]
        canvas = np.zeros((h_main, w_main + w_eye, 3), dtype=np.uint8)
        canvas[:, :w_main] = img_main
        y_off = (h_main - h_eye) // 2
        canvas[y_off:y_off + h_eye, w_main:w_main + w_eye] = img_eye

        # Eye panel border + label
        from PIL import Image, ImageDraw
        im = Image.fromarray(canvas)
        draw = ImageDraw.Draw(im, "RGBA")
        draw.rectangle(
            [(w_main, y_off - 1), (w_main + w_eye - 1, y_off + h_eye)],
            outline=(255, 255, 255, 230), width=2,
        )
        draw.text((w_main + 8, y_off + 6), "cat_eye (POV)",
                  fill=(255, 255, 255, 240), font=font)
        canvas = np.array(im)

        writer.append_data(canvas)

    writer.close()
    size_mb = out_path.stat().st_size / 1e6 if out_path.exists() else 0
    _log(f"done: {out_path} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
