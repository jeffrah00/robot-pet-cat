#!/usr/bin/env python3
"""Render a skill showcase video -- each skill held for a fixed duration.

Cycles through: stand -> sit -> lie_down -> stand -> crouch -> stretch ->
                stand -> walk_to (short walk) -> stand -> swat (paw-raise) ->
                stand -> jump_to

Each static pose is held for POSE_HOLD_S sim-seconds so it's clearly visible.
The robot starts in standing position and returns to stand between poses.

Usage (pod):
  MUJOCO_GL=osmesa python scripts/render_skill_showcase.py \\
      --policy /workspace/unitree_rl_mjlab/logs/rsl_rl/go2_velocity/\\
2026-05-18_22-14-29/policy.onnx \\
      --out renders/skill_showcase.mp4

Output: side-by-side (third-person + cat_eye), HUD showing current skill.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np


MOOD_NAMES = [
    "calm_w",
    "playful_w",
    "hungry_w",
    "tired_w",
    "curious_w",
]
MOOD_FORMAT = "mood [{:.2f} {:.2f} {:.2f} {:.2f} {:.2f}]"


# ---------------------------------------------------------------------------
# Showcase schedule: (duration_s, skill_name_or_None, label)
# None skill_name = HOLD (stand still with walker policy).
# ---------------------------------------------------------------------------
SHOWCASE = [
    (2.0,  None,       "STAND (baseline)"),
    (3.5,  "sit",      "SIT -- haunches folded, front upright"),
    (2.0,  None,       "STAND (recovery)"),
    (3.5,  "lie_down", "LIE DOWN -- sphinx pose, belly low"),
    (2.0,  None,       "STAND (recovery)"),
    (3.5,  "crouch",   "CROUCH -- pre-pounce, weight forward"),
    (2.0,  None,       "STAND (recovery)"),
    (3.0,  "stretch",  "STRETCH -- front legs extended, chest down"),
    (2.0,  None,       "STAND (recovery)"),
    (4.0,  "walk_to",  "WALK_TO -- velocity-cmd locomotion"),
    (2.0,  None,       "STAND (recovery)"),
    (3.0,  "swat",     "SWAT -- paw-raise (Go2 limit: no arm)"),
    (2.0,  None,       "STAND (recovery)"),
    (5.0,  "jump_to",  "JUMP_TO -- ballistic arc to cat tree"),
    (2.0,  None,       "STAND (final)"),
]


def _scripted_action(t_sim: float, showcase: list, env) -> tuple[int, str]:
    """Return (action_idx, label) for current sim time."""
    acc = 0.0
    for dur, skill_name, label in showcase:
        acc += dur
        if t_sim < acc:
            if skill_name is None:
                return 0, label
            skills = env.skill_names
            if skill_name not in skills:
                return 0, label + " [NOT IN REGISTRY]"
            return 1 + skills.index(skill_name), label
    return 0, "STAND (end)"


def _hud_lines(obs: dict, env, label: str) -> list[str]:
    """Build HUD text lines."""
    lines = []
    lines.append(f"t={env.t_sim:6.2f}s  step={env.episode_step:4d}")
    lines.append(f"SKILL: {label}")
    active = env.active_skill_name or "hold"
    lines.append(f"active: {active}  in_skill: {env.time_in_skill:.1f}s")
    xy = obs.get("root_xy", [0, 0])
    z = obs.get("cat_body_height", 0)
    yaw = obs.get("root_yaw", 0)
    speed = obs.get("last_speed", 0)
    lines.append(f"xy=({xy[0]:.2f},{xy[1]:.2f}) z={z:.3f} yaw={math.degrees(yaw):.0f}deg")
    lines.append(f"speed={speed:.2f}m/s")
    mood = obs.get("mood", np.zeros(5))
    lines.append(MOOD_FORMAT.format(*mood))
    rd = getattr(env, "last_reward_breakdown", {})
    if rd:
        total = rd.get("total", 0)
        lines.append(f"reward: {total:.3f}")
    return lines


def _draw_hud(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Draw HUD text into the top-left of frame."""
    try:
        import cv2

        y = 18
        for line in lines:
            cv2.putText(
                frame, line, (8, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (0, 0, 0), 2, cv2.LINE_AA,
            )
            cv2.putText(
                frame, line, (8, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (220, 220, 50), 1, cv2.LINE_AA,
            )
            y += 18
    except ImportError:
        pass
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Skill showcase render")
    parser.add_argument("--policy", required=True,
                        help="Path to Go2 ONNX walker policy")
    parser.add_argument("--out", default="renders/skill_showcase.mp4",
                        help="Output mp4 path")
    parser.add_argument("--fps", type=int, default=30,
                        help="Output video FPS")
    parser.add_argument("--width", type=int, default=640,
                        help="Third-person camera width")
    parser.add_argument("--height", type=int, default=480,
                        help="Third-person camera height")
    args = parser.parse_args()

    policy_path = Path(args.policy)
    if not policy_path.exists():
        print(f"[ERROR] Policy not found: {policy_path}", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_s = sum(d for d, _, _ in SHOWCASE)
    # steps per brain tick (BrainEnvConfig.dt_s = 0.05s by default)
    brain_dt = 0.05
    total_steps = int(math.ceil(total_s / brain_dt)) + 20

    print(f"[showcase] total sim time: {total_s:.1f}s  steps: {total_steps}")
    print(f"[showcase] output: {out_path}")

    # ------------------------------------------------------------------ #
    # Build env
    # ------------------------------------------------------------------ #
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    from robot_pet_cat.brain.env import BrainEnv, BrainEnvConfig
    from robot_pet_cat.brain.attractor import AttractorConfig

    cfg = BrainEnvConfig(
        policy_path=str(policy_path),
        use_physics_cat=True,
        max_steps_per_episode=total_steps + 50,
        attractor=AttractorConfig(enabled=False),
        initial_attractor_mode="PLAYING",
    )
    env = BrainEnv(cfg)
    obs, _ = env.reset()

    # ------------------------------------------------------------------ #
    # Set up renderer
    # ------------------------------------------------------------------ #
    import mujoco
    import mujoco.renderer

    W, W_eye = args.width, args.width // 3
    H = args.height
    renderer = mujoco.renderer.Renderer(env.mj_model, height=H, width=W)
    renderer_eye = mujoco.renderer.Renderer(env.mj_model, height=H // 2, width=W_eye)

    # Camera: chase cam behind the robot
    def _render_main(mj_data, yaw: float) -> np.ndarray:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        xy = mj_data.qpos[:2] if hasattr(mj_data, "qpos") else np.zeros(2)
        # Use base body position from qpos (offset by scene bodies)
        try:
            base_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_BODY, "base")
            xpos = mj_data.xpos[base_id]
            cam.lookat[:] = xpos[:3]
        except Exception:
            cam.lookat[:] = [0, 0, 0.3]
        cam.distance = 2.8
        cam.elevation = -18.0
        cam.azimuth = math.degrees(yaw) + 160.0
        renderer.update_scene(mj_data, camera=cam)
        return renderer.render().copy()

    def _render_eye(mj_data) -> np.ndarray:
        renderer_eye.update_scene(mj_data, camera="cat_eye")
        return renderer_eye.render().copy()

    # ------------------------------------------------------------------ #
    # Video writer
    # ------------------------------------------------------------------ #
    import imageio

    # Composite frame dimensions
    eye_h = H // 2
    eye_w = W_eye
    composite_w = W + eye_w
    composite_h = H

    writer = imageio.get_writer(str(out_path), fps=args.fps, codec="libx264",
                                quality=7, macro_block_size=None)

    # Render one frame every render_every brain steps
    steps_per_frame = max(1, int(round((1.0 / args.fps) / brain_dt)))

    print(f"[showcase] steps_per_frame={steps_per_frame}, brain_dt={brain_dt}s")

    current_label = "STAND (baseline)"
    step = 0
    while step < total_steps:
        action, label = _scripted_action(env.t_sim, SHOWCASE, env)
        current_label = label

        obs, _, terminated, truncated, info = env.step(action)
        step += 1

        if step % steps_per_frame == 0 or step == 1:
            yaw = float(obs.get("root_yaw", 0.0))
            frame_main = _render_main(env.mj_data, yaw)
            frame_eye = _render_eye(env.mj_data)

            hud = _hud_lines(obs, env, current_label)
            frame_main = _draw_hud(frame_main, hud)

            # Composite: main | eye (padded to H)
            eye_padded = np.zeros((H, eye_w, 3), dtype=np.uint8)
            eye_padded[:eye_h, :, :] = frame_eye[:eye_h, :eye_w, :]
            composite = np.concatenate([frame_main, eye_padded], axis=1)
            writer.append_data(composite)

        if terminated or truncated:
            break

    writer.close()
    print(f"[showcase] done. wrote {step} steps -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
