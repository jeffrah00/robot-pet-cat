#!/usr/bin/env python3
"""Render an MP4 of a trained AMP policy walking around in the Go2 env.

What it does, in order:
  1. Builds the same Go2 Joystick env we trained against (same wrapper, same
     constants patch). Single env, no vmap.
  2. Reconstructs the brax PPO inference function from the network factory
     used in amp_trainer + the pickled params at checkpoints/motion/ppo_params.pkl.
  3. Steps the env for --duration_s seconds in eager mode. Per step:
       - inference_fn(obs) -> action
       - env.step(state, action)
       - sync mj_data with state.data.qpos / qvel
       - mujoco.Renderer.render() -> RGB frame
  4. Encodes the frames into an MP4 via imageio.

Usage:
    python scripts/render_motion_policy.py
    python scripts/render_motion_policy.py --ckpt checkpoints/motion/ppo_params.pkl \
        --out checkpoints/motion/render.mp4 --duration-s 8 --fps 30
    python scripts/render_motion_policy.py --command 1.0,0,0   # forward 1 m/s
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

# MUST come before any `import mujoco`. On a headless box (RunPod, CI, any
# server without X11) MuJoCo defaults to GLFW and fails with
#   "an OpenGL platform library has not been loaded into this process".
# EGL uses the NVIDIA driver's offscreen path directly -- no display needed.
# OSMesa is the CPU fallback if no GPU is visible.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np


def _load_params(ckpt_path: Path):
    """Load a v1 or v2 checkpoint and return (normalizer_params, policy_params).

    v1 (amp_trainer.py): pickled as either a single object, a (policy, value)
       tuple, or a (normalizer, policy, value) triple depending on brax version.
    v2 (amp_trainer_v2.py): pickled as a dict
       {"policy", "value", "disc", "normalizer", "iteration"}.
    """
    with ckpt_path.open("rb") as f:
        raw = pickle.load(f)

    if isinstance(raw, dict) and "policy" in raw:
        # v2 dict format
        return raw.get("normalizer"), raw["policy"]
    if isinstance(raw, tuple) and len(raw) == 3:
        normalizer, policy, _value = raw
        return normalizer, policy
    if isinstance(raw, tuple) and len(raw) == 2:
        # (policy, value) -- no normalizer saved. Return None and let the
        # render caller build a default one.
        return None, raw[0]
    # Bare params (no normalizer). Caller handles.
    return None, raw


def _build_env_and_inference(seed: int):
    """Mirror the env + network construction from amp_trainer."""
    import jax
    from brax.training.agents.ppo import networks as ppo_networks

    from robot_pet_cat.motion.amp_env import AMPEnvConfig, make_env

    env_cfg = AMPEnvConfig(
        base_env_name="Go2JoystickFlatTerrain",
        num_envs=1,
        episode_length_s=20.0,
        seed=seed,
    )
    env = make_env(env_cfg)

    # NOTE: for rendering we do NOT wrap with wrap_for_brax_training. We want
    # eager single-env stepping; that wrapper vmaps and produces brax-shaped
    # State which we'd then have to unwrap.

    # Reconstruct the inference network. Hidden sizes must match what we
    # trained with (cat_amp.yaml: 512,256,128 / 512,256,128).
    obs_size = env.observation_size
    act_size = env.action_size
    ppo_nets = ppo_networks.make_ppo_networks(
        observation_size=obs_size,
        action_size=act_size,
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
    )
    make_inference_fn = ppo_networks.make_inference_fn(ppo_nets)
    return env, make_inference_fn


def _state_to_mj_data(state, mj_model):
    """Copy qpos/qvel out of an mjx state into a fresh MjData for rendering."""
    import mujoco

    mj_data = mujoco.MjData(mj_model)
    # mujoco_playground stores mjx data on state.data (or state.pipeline_state
    # if wrapped). Try both.
    data = getattr(state, "data", None) or getattr(state, "pipeline_state", None)
    if data is None:
        raise RuntimeError(f"state has neither .data nor .pipeline_state: {state}")
    mj_data.qpos[:] = np.asarray(data.qpos)
    mj_data.qvel[:] = np.asarray(data.qvel)
    if hasattr(data, "ctrl") and data.ctrl is not None:
        mj_data.ctrl[:] = np.asarray(data.ctrl)
    mujoco.mj_forward(mj_model, mj_data)
    return mj_data


def render_policy(
    ckpt_path: Path,
    out_mp4: Path,
    duration_s: float,
    fps: int,
    seed: int,
    command: tuple[float, float, float] | None,
    deterministic: bool,
    width: int,
    height: int,
    camera: str | None,
) -> None:
    import jax
    import jax.numpy as jnp
    import mujoco
    import imageio.v3 as iio

    print(f"[render] loading params from {ckpt_path}")
    normalizer_params, policy_params = _load_params(ckpt_path)
    print(f"[render]   normalizer present: {normalizer_params is not None}")

    print("[render] building env + inference network")
    env, make_inference_fn = _build_env_and_inference(seed=seed)

    # make_inference_fn expects a (normalizer_params, policy_params) tuple
    # in the brax convention. If we didn't recover a normalizer, build a
    # neutral one (mean=0/std=1) so the policy at least runs -- it'll
    # produce out-of-scale actions but better than crashing.
    if normalizer_params is None:
        from brax.training.acme import running_statistics, specs
        obs_size = env.observation_size
        def _to_spec(size):
            shape = (size,) if isinstance(size, int) else tuple(size)
            return specs.Array(shape, jnp.float32)
        if isinstance(obs_size, dict):
            obs_spec = {k: _to_spec(v) for k, v in obs_size.items()}
        else:
            obs_spec = _to_spec(obs_size)
        normalizer_params = running_statistics.init_state(obs_spec)

    inference_fn = make_inference_fn(
        (normalizer_params, policy_params), deterministic=deterministic,
    )
    inference_fn = jax.jit(inference_fn)

    mj_model = getattr(env, "mj_model", None) or getattr(env, "_mj_model", None)
    if mj_model is None:
        raise RuntimeError("env has no mj_model / _mj_model attribute to render against")

    print("[render] jit-compiling reset/step")
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(seed)
    rng, rk = jax.random.split(rng)
    state = jit_reset(rk)

    # If a command was provided, write it into state.info so the joystick
    # reward/observation uses it. Go1JoystickFlatTerrain stores the command
    # in state.info under "command" (3D: vx, vy, wz).
    if command is not None and "command" in state.info:
        cmd = jnp.asarray(command, dtype=jnp.float32)
        new_info = {**state.info, "command": cmd}
        state = state.replace(info=new_info)
        print(f"[render] override command -> {command}")

    n_steps = int(duration_s * fps)
    ctrl_dt = 1.0 / fps
    print(f"[render] rolling out {duration_s:.1f}s = {n_steps} frames at {fps} fps")

    renderer = mujoco.Renderer(mj_model, width=width, height=height)
    frames = []
    t0 = time.time()
    for i in range(n_steps):
        rng, rk = jax.random.split(rng)
        act, _ = inference_fn(state.obs, rk)
        state = jit_step(state, act)

        mj_data = _state_to_mj_data(state, mj_model)
        if camera is not None:
            renderer.update_scene(mj_data, camera=camera)
        else:
            renderer.update_scene(mj_data)
        frames.append(renderer.render())

        if (i + 1) % 30 == 0 or (i + 1) == n_steps:
            wall = time.time() - t0
            print(f"[render]   frame {i+1}/{n_steps}  wall {wall:.1f}s  "
                  f"reward(last)={float(state.reward):+.3f}")

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    print(f"[render] encoding {out_mp4} ({len(frames)} frames @ {fps} fps)")
    iio.imwrite(out_mp4, np.stack(frames), fps=fps)
    print(f"[render] done -> {out_mp4}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path,
                   default=Path("checkpoints/motion/ppo_params.pkl"),
                   help="Pickled PPO params from amp_trainer.")
    p.add_argument("--out", type=Path,
                   default=Path("checkpoints/motion/render.mp4"))
    p.add_argument("--duration-s", type=float, default=10.0, dest="duration_s")
    p.add_argument("--fps", type=int, default=30,
                   help="Output video fps; also the control rate during rollout.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--command", type=str, default=None,
                   help="vx,vy,wz override (m/s, m/s, rad/s). Default: env-provided.")
    p.add_argument("--stochastic", action="store_true",
                   help="Sample actions from the policy distribution instead of mean.")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--camera", type=str, default=None,
                   help="MuJoCo camera name; default uses the model's default cam.")
    args = p.parse_args()

    if not args.ckpt.is_file():
        print(f"[render] no checkpoint at {args.ckpt}", file=sys.stderr)
        return 1

    command = None
    if args.command:
        try:
            command = tuple(float(x) for x in args.command.split(","))
            if len(command) != 3:
                raise ValueError
        except ValueError:
            print(f"[render] --command must be 3 floats vx,vy,wz; got {args.command!r}",
                  file=sys.stderr)
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
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
