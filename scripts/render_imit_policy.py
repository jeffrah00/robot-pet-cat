#!/usr/bin/env python3
"""Render an MP4 of a trained imitation-conditioned policy in the Go2 env.

Mirrors scripts/render_motion_policy.py (the AMP renderer), but matches the
observation shape and network architecture that imitation_trainer.py expects:

  - obs['state'] is augmented with a 2-D phase clock [sin(phase), cos(phase)]
    that gets concatenated onto the env's raw 48-D state -> 50-D total.
  - policy MLP hidden sizes are (512, 256, 128) (default in ImitationTrainConfig).

At training time, `phase` is derived from the frame index within each reference
clip. At inference time we don't have a specific clip, so we cycle a synthetic
phase with a user-chosen period (`--period-steps`). Tune that knob to match the
natural gait frequency the policy learned.

Usage:
    .venv/bin/python -u scripts/render_imit_policy.py \\
        --ckpt checkpoints/motion_imitation_v1/imit_v1_latest.pkl \\
        --out  checkpoints/motion_imitation_v1/render_imit_v1_latest.mp4 \\
        --command 0.6,0,0 --duration-s 10 --period-steps 100
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

# Match scripts/render_motion_policy.py: pick headless GL backends *before*
# mujoco is imported, so a pod with no DISPLAY still gets an OpenGL context.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import jax
import jax.numpy as jnp
import mujoco
import imageio.v2 as imageio

from brax.training.acme import running_statistics
from brax.training.agents.ppo.networks import make_inference_fn, make_ppo_networks

# Reuse the project's env builder so wrapping matches training exactly.
from robot_pet_cat.motion.amp_env import AMPEnvConfig, make_env


# Match ImitationTrainConfig defaults in src/robot_pet_cat/motion/imitation_trainer.py.
POLICY_HIDDEN = (512, 256, 128)
VALUE_HIDDEN = (512, 256, 128)


def _to_spec(size):
    return jax.tree_util.tree_map(
        lambda n: jax.ShapeDtypeStruct((int(n),), jnp.float32),
        size,
    )


def _state_to_mj_data(state, mj_model):
    """Sync brax/mjx env state.data into a mujoco MjData for rendering."""
    d = mujoco.MjData(mj_model)
    qpos = np.asarray(state.data.qpos).reshape(-1)
    qvel = np.asarray(state.data.qvel).reshape(-1)
    d.qpos[: qpos.shape[0]] = qpos
    d.qvel[: qvel.shape[0]] = qvel
    mujoco.mj_forward(mj_model, d)
    return d


def _augment_obs_with_phase(obs, phase_sin_cos):
    """Append [sin(phase), cos(phase)] to obs['state'] (or to root obs)."""
    if isinstance(obs, dict):
        obs2 = dict(obs)
        obs2["state"] = jnp.concatenate([obs2["state"], phase_sin_cos], axis=-1)
        return obs2
    return jnp.concatenate([obs, phase_sin_cos], axis=-1)


def render_policy(
    ckpt_path: Path,
    out_mp4: Path,
    duration_s: float,
    fps: int,
    seed: int,
    command,
    deterministic: bool,
    width: int,
    height: int,
    camera,
    period_steps: int,
):
    print(f"[render-imit] loading params from {ckpt_path}", flush=True)
    payload = pickle.loads(ckpt_path.read_bytes())
    if isinstance(payload, tuple):
        normalizer_params, policy_params = payload[0], payload[1]
    elif isinstance(payload, dict):
        normalizer_params = payload.get("normalizer") or payload.get("normalizer_params")
        policy_params = payload.get("policy") or payload.get("policy_params")
    else:
        raise RuntimeError(f"unknown ckpt payload type: {type(payload)}")

    print(f"[render-imit] normalizer present: {normalizer_params is not None}", flush=True)

    # Build the env the same way the imitation trainer does, but DO NOT wrap
    # with mjx_wrapper.wrap_for_brax_training -- that vmaps over a batch dim
    # and breaks single-env rollout. We want the raw, unbatched env here.
    env_cfg = AMPEnvConfig(
        base_env_name="Go2JoystickFlatTerrain",
    )
    raw_env = make_env(env_cfg)
    env = raw_env  # render unbatched
    print(f"[render-imit] env loaded (unbatched): {type(env).__name__}", flush=True)

    # Compute obs_size including the +2 phase clock dims, matching training.
    obs_size_base = env.observation_size
    if isinstance(obs_size_base, dict):
        obs_size_aug = {
            k: ((v[0] + 2,) + v[1:] if k == "state" and isinstance(v, tuple) else v)
            for k, v in obs_size_base.items()
        }
    else:
        obs_size_aug = obs_size_base + 2

    obs_spec = (
        {k: _to_spec(v) for k, v in obs_size_aug.items()}
        if isinstance(obs_size_aug, dict)
        else _to_spec(obs_size_aug)
    )
    if normalizer_params is None:
        normalizer_params = running_statistics.init_state(obs_spec)

    ppo_nets = make_ppo_networks(
        observation_size=obs_size_aug,
        action_size=env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=POLICY_HIDDEN,
        value_hidden_layer_sizes=VALUE_HIDDEN,
    )
    inference_fn = make_inference_fn(ppo_nets)(
        (normalizer_params, policy_params), deterministic=deterministic
    )
    inference_fn = jax.jit(inference_fn)

    mj_model = getattr(env, "mj_model", None) or getattr(env, "_mj_model", None)
    if mj_model is None:
        raise RuntimeError("env has no mj_model/_mj_model attribute")

    print("[render-imit] jit-compiling reset/step", flush=True)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(seed)
    rng, rk = jax.random.split(rng)
    state = jit_reset(rk)

    # Apply velocity command override.
    if command is not None and "command" in state.info:
        cmd = jnp.asarray(command, dtype=jnp.float32)
        new_info = {**state.info, "command": cmd}
        state = state.replace(info=new_info)
        print(
            f"[render-imit] override command -> {tuple(float(c) for c in command)}",
            flush=True,
        )

    n_steps = int(duration_s * fps)
    print(
        f"[render-imit] rolling out {duration_s:.1f}s = {n_steps} frames at {fps} fps "
        f"(phase period = {period_steps} steps)",
        flush=True,
    )

    renderer = mujoco.Renderer(mj_model, width=width, height=height)
    frames = []
    t0 = time.time()
    rewards = []
    for i in range(n_steps):
        phase = 2.0 * np.pi * (i % period_steps) / float(period_steps)
        phase_sc = jnp.asarray(
            [np.sin(phase), np.cos(phase)], dtype=jnp.float32
        )
        obs_aug = _augment_obs_with_phase(state.obs, phase_sc)

        rng, rk = jax.random.split(rng)
        act, _ = inference_fn(obs_aug, rk)
        state = jit_step(state, act)
        rewards.append(float(state.reward))

        mj_data = _state_to_mj_data(state, mj_model)
        if camera is not None:
            renderer.update_scene(mj_data, camera=camera)
        else:
            renderer.update_scene(mj_data)
        frames.append(renderer.render())

        if (i + 1) % 30 == 0:
            print(
                f"[render-imit]   frame {i+1}/{n_steps}  "
                f"wall {time.time()-t0:.1f}s  reward(last)={rewards[-1]:+.3f}",
                flush=True,
            )

    print(
        f"[render-imit] encoding {out_mp4} ({len(frames)} frames @ {fps} fps)",
        flush=True,
    )
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_mp4, frames, fps=fps, codec="libx264")
    print(f"[render-imit] done -> {out_mp4}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--duration-s", type=float, default=10.0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--command",
        type=str,
        default=None,
        help="vx,vy,wz override (m/s, m/s, rad/s). Default: env-provided.",
    )
    p.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of taking the policy mean.",
    )
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument(
        "--camera",
        type=str,
        default=None,
        help="MuJoCo camera name; default uses the model's default cam.",
    )
    p.add_argument(
        "--period-steps",
        type=int,
        default=100,
        help="Phase-clock period in ctrl steps. Try 60 / 100 / 200 to find the "
        "natural gait frequency the policy learned. Default 100.",
    )
    args = p.parse_args()

    if not args.ckpt.is_file():
        print(f"[render-imit] no checkpoint at {args.ckpt}", file=sys.stderr)
        return 1

    command = None
    if args.command:
        try:
            command = tuple(float(x) for x in args.command.split(","))
            if len(command) != 3:
                raise ValueError
        except ValueError:
            print(
                f"[render-imit] --command must be 3 floats vx,vy,wz; "
                f"got {args.command!r}",
                file=sys.stderr,
            )
            return 1

    render_policy(
        ckpt_path=args.ckpt,
        out_mp4=args.out,
        duration_s=args.duration_s,
        fps=args.fps,
        seed=args.seed,
        command=command,
        deterministic=not args.stochastic,
        width=args.width,
        height=args.height,
        camera=args.camera,
        period_steps=args.period_steps,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
