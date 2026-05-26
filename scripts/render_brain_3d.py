#!/usr/bin/env python3
"""Render a Go2 brain rollout in 3D with a HUD + cat's-eye inset.

Output panel layout:

  +------------------------------+----------------+
  |                              |                |
  |   Third-person (chase cam)   |   cat_eye      |
  |   + HUD overlay              |   (POV inset)  |
  |                              |                |
  +------------------------------+----------------+

The HUD shows:
  * mode (attractor mode name from ModePolicy, if enabled)
  * skill (active_skill_name from BrainEnv)
  * the LocomotionCommand that PhysicsCat is currently tracking
  * mood vector
  * cat pose (xy/z/yaw/speed)
  * sim time + step

How the action stream is driven:
  * --checkpoint PATH : load a SB3 PPO model from PATH and use it (so mode
    + skill change for real).
  * --scripted        : cycle a hand-coded sequence of skills (hold ->
    walk_to -> swat -> lie_down -> stretch -> hold) every ~3 sim seconds.
    No checkpoint needed; useful for sanity-checking the scene visuals.
  * default            : if --checkpoint exists try it; else fall back to
    scripted.

Render backends:
  Needs a working MuJoCo GL backend. On a headless pod, set
    MUJOCO_GL=osmesa
  before running (apt-get install libosmesa6 libgl1 if missing).

Usage (pod):
  python scripts/render_brain_3d.py \\
      --policy /workspace/unitree_rl_mjlab/logs/.../policy.onnx \\
      --steps 600 --out renders/brain_v0_3d.mp4 \\
      [--checkpoint checkpoints/brain/brain_v0_baseline.zip] \\
      [--scripted]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Mood label conventions (matches src/robot_pet_cat/brain/mood.py).
# --------------------------------------------------------------------------- #

MOOD_NAMES = [
    "energy",
    "alertness",
    "sociability",
    "curiosity_w",
    "comfort_w",
    "play_w",
]


# --------------------------------------------------------------------------- #
# Scripted action stream -- for the no-checkpoint path.
# --------------------------------------------------------------------------- #


SCRIPTED_SCHEDULE_S = [
    # 2026-05-25 post-Go1 pivot: cycle through the canonical 6-skill set.
    # Skill name appears in HUD; underlying locomotion is the Go2 walker.
    (0.0,  "walk_normal"),   # settle into active walk (knock-down impulse at t=1.0)
    (3.0,  "get_up"),         # recovery
    (6.0,  "walk_slow"),      # gentle locomotion
    (9.0,  "crouch"),          # alert low posture
    (12.0, "lie_belly"),       # rest on belly
    (15.0, "lie_side"),        # rest on side
    (18.0, "walk_normal"),     # back to active
]


def _scripted_action(t_sim: float, env) -> int:
    """Pick a skill index based on absolute sim time."""
    name = SCRIPTED_SCHEDULE_S[0][1]
    for boundary_t, n in SCRIPTED_SCHEDULE_S:
        if t_sim >= boundary_t:
            name = n
    skills = env.skill_names
    if name == "hold" or name not in skills:
        return 0
    return 1 + skills.index(name)


# --------------------------------------------------------------------------- #
# HUD drawing via PIL.
# --------------------------------------------------------------------------- #


def _load_font(size: int = 16):
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
    """Overlay text top-left of `img_array` (RGB H,W,3 uint8). Returns new array."""
    from PIL import Image, ImageDraw

    im = Image.fromarray(img_array)
    draw = ImageDraw.Draw(im, "RGBA")
    # Black translucent background panel so text reads on any scene color.
    line_h = font.size + 6
    panel_w = max((font.getlength(line) for line in lines), default=0) + 24
    panel_h = line_h * len(lines) + 12
    draw.rectangle([(6, 6), (6 + panel_w, 6 + panel_h)], fill=(0, 0, 0, 170))
    y = 12
    for ln in lines:
        draw.text((14, y), ln, fill=(240, 240, 240, 255), font=font)
        y += line_h
    return np.array(im)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def _log(s: str) -> None:
    print(f"[render3d] {s}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--policy",
        required=True,
        help="Path to the Go2 walker policy (.onnx, .pt, or run dir).",
    )
    p.add_argument(
        "--checkpoint",
        default="checkpoints/brain/brain_v0_baseline.zip",
        help="SB3 PPO checkpoint for the brain policy. Used when --scripted "
        "is not set and the file exists.",
    )
    p.add_argument(
        "--scripted",
        action="store_true",
        help="Use the hand-coded action schedule instead of a PPO model.",
    )
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--out", default="renders/brain_3d.mp4")
    p.add_argument("--width-main", type=int, default=960)
    p.add_argument("--height-main", type=int, default=640)
    p.add_argument("--width-eye", type=int, default=320)
    p.add_argument("--height-eye", type=int, default=320)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument(
        "--initial-xy", nargs=2, type=float, default=[0.5, 0.5],
        help="Spawn xy of the cat in the room.",
    )
    p.add_argument(
        "--mode-init", default="EXPLORING",
        help="Initial Attractor mode for the brain. Pass NONE to disable mode_policy.",
    )
    p.add_argument(
        "--no-mode-policy", action="store_true",
        help="Disable mode policy entirely.",
    )
    p.add_argument(
        "--scripted-get-up", action="store_true",
        help="Replace the trained get_up .pt policy with the keyframe-based "
        "ScriptedGetUpPolicy (deterministic joint-angle ramp).",
    )
    p.add_argument(
        "--get-up-checkpoint",
        default="models/get_up_v7_4300.pt",
        help="Path to the trained get_up policy. Ignored when "
        "--scripted-get-up is set.",
    )
    p.add_argument(
        "--knock-down", action="store_true",
        help="At sim t=1.0s, apply an angular impulse to roll the cat onto "
        "its side so the scripted get_up has something to recover from.",
    )
    args = p.parse_args()

    # Lazy imports so --help doesn't pull mujoco.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    # --- Load PPO checkpoint BEFORE onnxruntime/BrainEnv ------------------ #
    # On pods with old NVIDIA drivers onnxruntime grabs the GPU context when
    # it loads the walker ONNX.  If torch (via SB3) runs after that it
    # segfaults inside torch._C._cuda_getDeviceCount().  Loading the PPO
    # weights here -- before BrainEnv is constructed -- avoids that race.
    ppo_model = None
    if not args.scripted and Path(args.checkpoint).is_file():
        try:
            from stable_baselines3 import PPO
            _log(f"pre-loading PPO checkpoint {args.checkpoint} (before BrainEnv)...")
            ppo_model = PPO.load(args.checkpoint, device="cpu")
            _log(f"  PPO loaded OK ({ppo_model.action_space})")
        except Exception as e:
            _log(f"  could not load checkpoint ({type(e).__name__}: {e}); "
                 "falling back to scripted")
            ppo_model = None
    if ppo_model is None:
        _log("using scripted action stream")

    import mujoco
    from PIL import Image  # noqa
    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio  # type: ignore

    from robot_pet_cat.brain.env import BrainEnv, BrainEnvConfig
    from robot_pet_cat.brain.attractor import Attractor, ModePolicy, ModePolicyConfig
    from robot_pet_cat.skills.skill_policy import LocomotionCommand
    from robot_pet_cat.brain.gym_env import obs_dict_to_curiosity_vec

    # --- Build the env ------------------------------------------------- #
    mode_policy = None
    if not args.no_mode_policy:
        try:
            init_mode = Attractor[args.mode_init.upper()]
        except KeyError:
            _log(f"unknown attractor {args.mode_init}, falling back to EXPLORING")
            init_mode = Attractor.EXPLORING
        mode_policy = ModePolicy(ModePolicyConfig(initial_mode=init_mode))

    # Resolve get_up policy: scripted callable or trained checkpoint.
    if args.scripted_get_up:
        from robot_pet_cat.brain.scripted_get_up import ScriptedGetUpPolicy
        get_up_obj = ScriptedGetUpPolicy()
        _log("using ScriptedGetUpPolicy (keyframe joint-angle sequence)")
    else:
        get_up_obj = Path(args.get_up_checkpoint)
        _log(f"using trained get_up policy: {get_up_obj}")

    cfg = BrainEnvConfig(
        use_physics_cat=True,
        walker_policy_path=Path(args.policy),
        initial_cat_xy=tuple(args.initial_xy),
        max_steps_per_episode=args.steps + 1,
        mode_policy=mode_policy,
        decision_period_min_steps=40,
        decision_period_max_steps=120,
        hind_sit_policy_path=Path("models/hind_sit_v1.pt"),
        get_up_policy_path=get_up_obj,
    )
    _log("building BrainEnv with PhysicsCat (this loads the walker)...")
    t0 = time.perf_counter()
    env = BrainEnv(cfg)
    _log(f"  env built in {time.perf_counter()-t0:.1f}s")

    obs = env.reset()
    _log(f"reset OK; cat at xy={obs['cat_xy']} yaw={obs['cat_yaw']:.3f}")

    # --- Renderers ----------------------------------------------------- #
    _log(f"creating renderers: main={args.width_main}x{args.height_main}, "
         f"eye={args.width_eye}x{args.height_eye}")
    renderer_main = mujoco.Renderer(env.mj_model, width=args.width_main, height=args.height_main)
    renderer_eye = mujoco.Renderer(env.mj_model, width=args.width_eye, height=args.height_eye)

    # Chase camera: track the cat from behind-and-above.
    # Try Go1 trunk first, fall back to Go2 base_link
    _base_name = "trunk" if mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_BODY, "trunk") >= 0 else "base_link"
    base_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_BODY, _base_name)
    chase = mujoco.MjvCamera()
    chase.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    chase.trackbodyid = base_id
    chase.distance = 2.8
    chase.azimuth = 135.0
    chase.elevation = -25.0
    chase.lookat[:] = [0.0, 0.0, 0.25]

    font = _load_font(size=18)

    # --- Output writer ------------------------------------------------- #
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out_path, fps=args.fps, codec="libx264")

    # --- Roll out + render --------------------------------------------- #
    _log(f"rolling out {args.steps} steps -> {out_path}")
    knock_done = False
    t_loop = time.perf_counter()
    for step in range(args.steps):
        # Optional knock-down: apply an angular impulse near sim t=1.0s so
        # the scripted get_up has something to recover from. The scripted
        # action schedule picks up "get_up" at t=2.5s and fires GetUp ->
        # gait="get_up" -> ScriptedGetUpPolicy in PhysicsCat.
        if args.knock_down and not knock_done and env.t_sim >= 1.0:
            vadr = env.cat._base_qvel_adr
            env.mj_data.qvel[vadr + 0] = 0.5   # small forward push
            env.mj_data.qvel[vadr + 2] = 1.2   # small upward kick
            env.mj_data.qvel[vadr + 3] = 12.0  # roll rate about body +x
            knock_done = True
            _log(f"  knocked cat down at t={env.t_sim:.2f}s")
        if ppo_model is not None:
            flat_obs = obs_dict_to_curiosity_vec(obs)
            action, _ = ppo_model.predict(flat_obs, deterministic=False)
            action = int(action)
        else:
            action = _scripted_action(env.t_sim, env)

        obs, reward, terminated, truncated, info = env.step(action)

        # Render third-person + cat eye.
        renderer_main.update_scene(env.mj_data, camera=chase)
        img_main = renderer_main.render()

        eye_cam_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "cat_eye")
        renderer_eye.update_scene(env.mj_data, camera=eye_cam_id)
        img_eye = renderer_eye.render()

        # HUD lines.
        skill = env.active_skill_name or "(hold)"
        mode = obs.get("mode", "n/a")
        cmd_arr = getattr(env.cat, "_last_cmd_arr", np.zeros(3))
        mood_state = obs.get("mood", np.zeros(6))
        hud = [
            f"t = {env.t_sim:6.2f}s   step {step:4d}",
            f"mode  : {mode}",
            f"skill : {skill}",
            f"cmd   : vx={cmd_arr[0]:+.2f}  vy={cmd_arr[1]:+.2f}  wz={cmd_arr[2]:+.2f}",
            f"xy    : ({env.cat.xy[0]:+.2f}, {env.cat.xy[1]:+.2f})   z={env.cat.body_height:.2f}",
            f"yaw   : {env.cat.yaw:+.2f} rad   speed={env.cat.last_speed:.2f} m/s",
            "mood  : " + " ".join(
                f"{n[:4]}={float(mood_state[i]):+.2f}" for i, n in enumerate(MOOD_NAMES[:3])
            ),
            "        " + " ".join(
                f"{n[:4]}={float(mood_state[i + 3]):+.2f}" for i, n in enumerate(MOOD_NAMES[3:])
            ),
        ]
        img_main = _draw_hud(img_main, hud, font)

        # Composite side-by-side. Pad the eye view height to main height.
        h_main, w_main = img_main.shape[:2]
        h_eye, w_eye = img_eye.shape[:2]
        canvas = np.zeros((h_main, w_main + w_eye, 3), dtype=np.uint8)
        canvas[:, :w_main] = img_main
        # Center the eye vertically, then label it.
        y_off = (h_main - h_eye) // 2
        canvas[y_off:y_off + h_eye, w_main:w_main + w_eye] = img_eye

        # Mini label on the cat eye panel.
        from PIL import Image, ImageDraw
        im = Image.fromarray(canvas)
        draw = ImageDraw.Draw(im, "RGBA")
        # Border around eye view
        draw.rectangle(
            [(w_main, y_off - 1), (w_main + w_eye - 1, y_off + h_eye)],
            outline=(255, 255, 255, 230),
            width=2,
        )
        draw.text((w_main + 8, y_off + 6), "cat_eye (POV)",
                  fill=(255, 255, 255, 240), font=font)
        canvas = np.array(im)

        writer.append_data(canvas)

    writer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
