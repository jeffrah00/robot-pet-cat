#!/usr/bin/env python3
"""Render a trained brain policy's rollout to mp4 (top-down 2D view).

The cat is a KinematicCat in BrainEnv -- there's no robot MJCF yet -- so a
top-down view is the natural representation. We draw the living room scene
(floor + couch + ball + window region) from `assets/scenes/living_room.xml`
and overlay the cat as a triangle (yaw-aware), with a trajectory trail and
a per-frame info block (t, active skill, reward breakdown).

Usage:
  python scripts/render_brain.py \\
      --checkpoint checkpoints/brain/brain_v0_baseline.zip \\
      --steps 600 \\
      --out renders/brain_v0_baseline.mp4

Notes:
  - The action distribution is sampled (`deterministic=False`) so the
    video shows the *trained policy as it acts*, including residual
    stochasticity. Pass --deterministic for the argmax-policy view.
  - The env is reconstructed with curiosity_enabled=True so the
    transitions are emitted and printed in the trace -- but the env
    DOES NOT call .update() during rendering, so the policy weights
    don't shift.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Rectangle, Circle, Polygon
from stable_baselines3 import PPO

from robot_pet_cat.brain.env import BrainEnv, BrainEnvConfig
from robot_pet_cat.brain.gym_env import make_brain_gym_env


# Scene geometry (must match assets/scenes/living_room.xml)
COUCH_CENTER = (2.0, 0.0)
COUCH_HALF = (0.6, 0.3)
WINDOW_CENTER = (-2.0, 0.0)
WINDOW_HALF = (1.0, 1.0)
BALL_RADIUS = 0.06

# Cat marker geometry (top-down)
CAT_BODY_LEN = 0.30   # rough Go2 trunk length
CAT_BODY_WIDTH = 0.18


def _cat_triangle(xy, yaw, length=CAT_BODY_LEN, width=CAT_BODY_WIDTH):
    """Return three vertices for an isoceles triangle pointing along yaw."""
    half_w = width / 2
    nose = (length, 0.0)
    rear_l = (-length * 0.5, +half_w)
    rear_r = (-length * 0.5, -half_w)
    pts = np.array([nose, rear_l, rear_r])  # (3,2)
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T + np.asarray(xy)


def _ball_xy_from_brain(brain) -> tuple:
    """Pull the ball's current xy from MuJoCo state."""
    scene_state = brain._extract_scene_state()
    for ent in scene_state.filter(play_target=True):
        return (float(ent.pos_xyz[0]), float(ent.pos_xyz[1]))
    return (0.0, 1.0)  # default spawn


def collect_trajectory(model, gym_env, n_steps: int, deterministic: bool) -> list:
    """Roll out, record per-step state. Returns a list of dicts."""
    history = []
    obs, _ = gym_env.reset()
    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        action = int(action)
        obs, reward, terminated, truncated, info = gym_env.step(action)
        brain = gym_env.brain
        history.append(
            {
                "t": float(brain.t_sim),
                "cat_xy": tuple(map(float, brain.cat.xy)),
                "cat_yaw": float(brain.cat.yaw),
                "ball_xy": _ball_xy_from_brain(brain),
                "skill": brain.active_skill_name or "hold",
                "reward": dict(info.get("reward_breakdown", {})),
                "action": action,
            }
        )
        if terminated or truncated:
            obs, _ = gym_env.reset()
    return history


def render_to_mp4(history: list, out_path: Path, fps: int = 20, trail_len: int = 40) -> None:
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)
    ax.set_aspect("equal")
    ax.set_facecolor("#f0eee6")  # warm floor tone
    ax.grid(True, alpha=0.15)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    # Static scene
    couch = Rectangle(
        (COUCH_CENTER[0] - COUCH_HALF[0], COUCH_CENTER[1] - COUCH_HALF[1]),
        2 * COUCH_HALF[0],
        2 * COUCH_HALF[1],
        color="#735540",
        alpha=0.9,
        label="couch (elevated, soft)",
    )
    ax.add_patch(couch)
    window = Rectangle(
        (WINDOW_CENTER[0] - WINDOW_HALF[0], WINDOW_CENTER[1] - WINDOW_HALF[1]),
        2 * WINDOW_HALF[0],
        2 * WINDOW_HALF[1],
        color="#f4d03f",
        alpha=0.30,
        label="window (warm)",
    )
    ax.add_patch(window)

    # Dynamic markers
    ball = Circle(history[0]["ball_xy"], 0.10, color="#c0392b", zorder=4, label="ball")
    ax.add_patch(ball)

    cat_patch = Polygon(
        _cat_triangle(history[0]["cat_xy"], history[0]["cat_yaw"]),
        color="#222222",
        zorder=5,
        label="cat",
    )
    ax.add_patch(cat_patch)

    (trail_line,) = ax.plot([], [], "-", color="#222", alpha=0.30, lw=1.2, zorder=3)

    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)

    # Info block (top-left)
    info_text = ax.text(
        -3.1,
        3.05,
        "",
        fontsize=11,
        fontfamily="monospace",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#bbb", "boxstyle": "round,pad=0.4"},
        zorder=10,
    )

    def update(frame: int):
        h = history[frame]
        cat_patch.set_xy(_cat_triangle(h["cat_xy"], h["cat_yaw"]))
        ball.center = h["ball_xy"]

        lo = max(0, frame - trail_len)
        xs = [history[i]["cat_xy"][0] for i in range(lo, frame + 1)]
        ys = [history[i]["cat_xy"][1] for i in range(lo, frame + 1)]
        trail_line.set_data(xs, ys)

        rew = h["reward"]
        info_text.set_text(
            f"t = {h['t']:6.2f} s   step = {frame:4d}\n"
            f"skill        = {h['skill']:>10s}\n"
            f"action idx   = {h['action']}\n"
            f"reward terms (per step):\n"
            f"  total      = {rew.get('total', 0):+.3f}\n"
            f"  play       = {rew.get('play', 0):+.3f}\n"
            f"  comfort    = {rew.get('comfort', 0):+.3f}\n"
            f"  hold       = {rew.get('hold', 0):+.3f}\n"
            f"  curiosity  = {rew.get('curiosity', 0):+.3f}"
        )
        return [cat_patch, ball, trail_line, info_text]

    anim = FuncAnimation(fig, update, frames=len(history), interval=1000 // fps, blit=False)
    writer = FFMpegWriter(
        fps=fps,
        metadata={"title": "robot-pet-cat brain rollout", "artist": "robot-pet-cat"},
        bitrate=1800,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=writer)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/brain/brain_v0_baseline.zip")
    p.add_argument("--steps", type=int, default=600, help="env steps to render (dt=0.05 -> 30s at 600)")
    p.add_argument("--out", default="renders/brain_v0_baseline.mp4")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--trail", type=int, default=40)
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="argmax-policy view (no sampling); default sampled",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = BrainEnvConfig(
        curiosity_enabled=True,
        curiosity_seed=args.seed,
        curiosity_feature_dim=8,
        curiosity_hidden=16,
        max_steps_per_episode=args.steps + 1,  # don't truncate mid-render
    )
    env = make_brain_gym_env(cfg)
    print(f"[render_brain] loading {args.checkpoint}")
    model = PPO.load(args.checkpoint, env=env)

    print(f"[render_brain] rolling out {args.steps} steps (deterministic={args.deterministic})")
    history = collect_trajectory(model, env, n_steps=args.steps, deterministic=args.deterministic)

    print(f"[render_brain] encoding to {args.out}")
    render_to_mp4(history, Path(args.out), fps=args.fps, trail_len=args.trail)
    print(f"[render_brain] done -> {args.out}")


if __name__ == "__main__":
    main()
