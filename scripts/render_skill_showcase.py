#!/usr/bin/env python3
"""Skill showcase render v2 -- slow deliberate motions with HUD overlay.

Fixes vs v1:
  1. PIL-based HUD (cv2 not available on pod).
  2. Independent showcase clock -- physics resets don't restart the schedule.
  3. Falls tolerated for 3s before physics reset; showcase continues.
  4. All pose skills take 5-7s (3s interpolation + hold time).

Usage:
  MUJOCO_GL=osmesa python scripts/render_skill_showcase.py \
      --policy /workspace/unitree_rl_mjlab/logs/rsl_rl/go2_velocity/\
2026-05-18_22-14-29/policy.onnx \
      --out renders/skill_showcase_v2.mp4
"""
from __future__ import annotations
import argparse, math, sys
from pathlib import Path
import numpy as np

MOOD_FORMAT = "mood [{:.2f} {:.2f} {:.2f} {:.2f} {:.2f}]"

# ---------------------------------------------------------------------------
# Showcase schedule  (longer durations -- 3s interp + hold time per pose)
# ---------------------------------------------------------------------------
SHOWCASE = [
    (3.0,  None,        "STAND (baseline)"),
    (6.0,  "sit",       "SIT -- haunches folded, front upright"),
    (3.0,  None,        "STAND (recovery)"),
    (6.0,  "lie_down",  "LIE DOWN -- sphinx pose, belly low"),
    (3.0,  None,        "STAND (recovery)"),
    (5.0,  "crouch",    "CROUCH -- pre-pounce, weight forward"),
    (3.0,  None,        "STAND (recovery)"),
    (5.0,  "stretch",   "STRETCH -- front legs extended, chest down"),
    (3.0,  None,        "STAND (recovery)"),
    (6.0,  "walk_to",   "WALK_TO -- casual saunter"),
    (3.0,  None,        "STAND (recovery)"),
    (5.0,  "swat",      "SWAT -- paw-raise"),
    (3.0,  None,        "STAND (recovery)"),
    (7.0,  "jump_to",   "JUMP_TO -- ballistic arc to cat tree"),
    (3.0,  None,        "STAND (final)"),
]

MAX_FALLEN_STEPS = 60   # 3 s at brain_dt=0.05
FALLEN_Z_THRESH  = 0.12 # m


def _scripted_action(showcase_t: float, showcase: list, env) -> tuple:
    """Map showcase_t to (action_idx, label)."""
    acc = 0.0
    for dur, skill, label in showcase:
        acc += dur
        if showcase_t < acc:
            if skill is None:
                return 0, label
            skills = env.skill_names
            if skill not in skills:
                return 0, label + " [MISSING]"
            return 1 + skills.index(skill), label
    return 0, "STAND (end)"


def _hud_lines(obs: dict, env, label: str) -> list:
    t  = getattr(env, "t_sim", 0.0)
    st = getattr(env, "episode_step", 0)
    active = getattr(env, "active_skill_name", None) or "hold"
    tin    = getattr(env, "time_in_skill", 0.0)
    xy  = obs.get("root_xy",        [0.0, 0.0])
    z   = float(obs.get("cat_body_height", 0.0))
    yaw = float(obs.get("root_yaw",       0.0))
    spd = float(obs.get("last_speed",     0.0))
    mood = obs.get("mood", np.zeros(5))
    return [
        f"t={t:6.2f}s  step={st:4d}",
        f">>> {label} <<<",
        f"active={active}  in_skill={tin:.1f}s",
        f"xy=({float(xy[0]):.2f},{float(xy[1]):.2f})  z={z:.3f}  yaw={math.degrees(yaw):.0f}deg",
        f"speed={spd:.2f} m/s",
        MOOD_FORMAT.format(*mood),
    ]


def _draw_hud(frame: np.ndarray, lines: list) -> np.ndarray:
    """PIL-based HUD; cv2 fallback."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        img  = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        font = bold = None
        for fp in ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                   "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"]:
            try:
                font = ImageFont.truetype(fp, 13)
                bold = ImageFont.truetype(fp, 14)
                break
            except Exception:
                pass
        if font is None:
            font = bold = ImageFont.load_default()
        y = 6
        for i, line in enumerate(lines):
            f = bold if i == 1 else font
            col = (255, 255, 80) if i == 1 else (220, 220, 220)
            draw.text((9, y+1), line, fill=(0,0,0), font=f)
            draw.text((8, y),   line, fill=col,     font=f)
            y += 17
        return np.array(img)
    except Exception:
        pass
    try:
        import cv2
        y = 18
        for i, line in enumerate(lines):
            col = (80,255,255) if i==1 else (220,220,50)
            cv2.putText(frame, line, (8,y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,0,0),  2, cv2.LINE_AA)
            cv2.putText(frame, line, (8,y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col,      1, cv2.LINE_AA)
            y += 18
    except Exception:
        pass
    return frame


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--out",   default="renders/skill_showcase_v2.mp4")
    p.add_argument("--fps",   type=int, default=30)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height",type=int, default=480)
    args = p.parse_args()

    policy_path = Path(args.policy)
    if not policy_path.exists():
        print(f"[ERROR] policy not found: {policy_path}", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_s    = sum(d for d,_,_ in SHOWCASE)
    brain_dt   = 0.05
    total_steps = int(total_s / brain_dt) + 40
    print(f"[showcase-v2] {total_s:.1f}s sim  {total_steps} steps  -> {out_path}")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from robot_pet_cat.brain.env import BrainEnv, BrainEnvConfig

    env = BrainEnv(BrainEnvConfig(
        walker_policy_path=policy_path,
        use_physics_cat=True,
        max_steps_per_episode=total_steps + 100,
    ))
    _r  = env.reset()
    obs = _r[0] if isinstance(_r, tuple) else _r

    import mujoco, mujoco.renderer
    W     = args.width
    W_eye = (args.width // 3) & ~1
    H     = args.height
    rend  = mujoco.renderer.Renderer(env.mj_model, height=H, width=W)
    reye  = mujoco.renderer.Renderer(env.mj_model, height=H//2, width=W_eye)

    def _frame_main(yaw):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        try:
            bid = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_BODY, "base")
            cam.lookat[:] = env.mj_data.xpos[bid][:3]
        except Exception:
            cam.lookat[:] = [0, 0, 0.3]
        cam.distance  = 2.8
        cam.elevation = -18.0
        cam.azimuth   = math.degrees(yaw) + 160.0
        rend.update_scene(env.mj_data, camera=cam)
        return rend.render().copy()

    def _frame_eye():
        reye.update_scene(env.mj_data, camera="cat_eye")
        return reye.render().copy()

    import imageio
    writer = imageio.get_writer(str(out_path), fps=args.fps,
                                codec="libx264", quality=7, macro_block_size=None)
    spf   = max(1, int(round((1.0/args.fps) / brain_dt)))  # steps per frame

    showcase_t   = 0.0
    step         = 0
    fallen_steps = 0
    cur_label    = "STAND (baseline)"

    while step < total_steps:
        showcase_t += brain_dt
        action, cur_label = _scripted_action(showcase_t, SHOWCASE, env)

        obs, _, terminated, truncated, _ = env.step(action)
        step += 1

        # Track fall duration
        z = float(obs.get("cat_body_height", 1.0))
        fallen_steps = (fallen_steps + 1) if z < FALLEN_Z_THRESH else 0

        # Reset physics after sustained fall (showcase_t keeps advancing)
        if fallen_steps > MAX_FALLEN_STEPS:
            print(f"[showcase-v2] sustained fall at showcase_t={showcase_t:.1f}s -- reset")
            _r  = env.reset()
            obs = _r[0] if isinstance(_r, tuple) else _r
            fallen_steps = 0

        if truncated:   # only hard-stop on max-steps
            break

        if step % spf == 0 or step == 1:
            yaw  = float(obs.get("root_yaw", 0.0))
            fm   = _frame_main(yaw)
            fe   = _frame_eye()
            fm   = _draw_hud(fm, _hud_lines(obs, env, cur_label))
            pad  = np.zeros((H, W_eye, 3), dtype=np.uint8)
            pad[:H//2, :, :] = fe[:H//2, :W_eye, :]
            writer.append_data(np.concatenate([fm, pad], axis=1))

    writer.close()
    print(f"[showcase-v2] done. {step} steps -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
