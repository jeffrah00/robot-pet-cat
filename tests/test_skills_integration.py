"""Phase 3 integration test — Tier 2 (skill) talks to Tier 1 (frozen AMP policy).

Loads the same env builder + trained AMP imitation policy that
scripts/render_imit_policy.py uses, then drives it from WalkTo for a few seconds
of sim time and asserts the cat makes meaningful progress toward the goal. This
exercises:

  - root_xy / root_yaw extraction from state.data.qpos
  - obs-dict plumbing from privileged state into the skill
  - LocomotionCommand → state.info["command"] dispatch
  - phase-clock obs augmentation (matches imitation_trainer.py training)
  - JAX-jitted reset + step

The test is gated on:
  1. jax, brax, mujoco being importable
  2. an imit_v1_*.pkl checkpoint being present under checkpoints/motion_imitation_v1/

If either gate fails it skips. That's the right call for a laptop CI run; on
the RunPod pod where the checkpoint and JAX stack both live, it should execute.
"""

from __future__ import annotations

import math
import os
import pickle
from pathlib import Path

import numpy as np
import pytest

# Headless GL — must be set BEFORE mujoco import. See render_imit_policy.py
# header comment for rationale.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


# --------------------------------------------------------------------------- #
# Gating: skip cleanly when deps or checkpoint are missing
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = REPO_ROOT / "checkpoints" / "motion_imitation_v1"


def _find_checkpoint() -> Path | None:
    """Prefer imit_v1_latest.pkl; otherwise the highest-iter saved checkpoint."""
    if not CKPT_DIR.is_dir():
        return None
    latest = CKPT_DIR / "imit_v1_latest.pkl"
    if latest.is_file():
        return latest
    candidates = sorted(CKPT_DIR.glob("imit_v1_it*.pkl"))
    return candidates[-1] if candidates else None


jax = pytest.importorskip("jax", reason="JAX not installed in this env")
brax_ppo = pytest.importorskip(
    "brax.training.agents.ppo.networks",
    reason="brax not installed in this env",
)
mujoco = pytest.importorskip("mujoco", reason="mujoco not installed in this env")
amp_env_mod = pytest.importorskip(
    "robot_pet_cat.motion.amp_env",
    reason="amp_env (and its submodule deps) not importable",
)

CKPT_PATH = _find_checkpoint()
if CKPT_PATH is None:
    pytest.skip(
        f"no AMP imitation checkpoint under {CKPT_DIR}; train one first",
        allow_module_level=True,
    )


import jax.numpy as jnp  # noqa: E402 — guarded by importorskip above

from brax.training.acme import running_statistics  # noqa: E402
from brax.training.agents.ppo.networks import (  # noqa: E402
    make_inference_fn,
    make_ppo_networks,
)

from robot_pet_cat.skills.walk_to import WalkTo, WalkToGoal  # noqa: E402


# Matches imitation_trainer.py defaults (and render_imit_policy.py).
POLICY_HIDDEN = (512, 256, 128)
VALUE_HIDDEN = (512, 256, 128)
PHASE_PERIOD_STEPS = 100


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _yaw_from_quat(qw: float, qx: float, qy: float, qz: float) -> float:
    """Yaw angle (rotation about +z) from a (w, x, y, z) MuJoCo quaternion."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(math.atan2(siny_cosp, cosy_cosp))


def _to_spec(size):
    return jax.tree_util.tree_map(
        lambda n: jax.ShapeDtypeStruct((int(n),), jnp.float32),
        size,
    )


def _augment_obs_with_phase(obs, phase_sin_cos):
    if isinstance(obs, dict):
        obs2 = dict(obs)
        obs2["state"] = jnp.concatenate([obs2["state"], phase_sin_cos], axis=-1)
        return obs2
    return jnp.concatenate([obs, phase_sin_cos], axis=-1)


def _build_inference_fn(ckpt_path: Path, env):
    """Mirror render_imit_policy.py's load+build path."""
    payload = pickle.loads(ckpt_path.read_bytes())
    if isinstance(payload, tuple):
        normalizer_params, policy_params = payload[0], payload[1]
    elif isinstance(payload, dict):
        normalizer_params = payload.get("normalizer") or payload.get("normalizer_params")
        policy_params = payload.get("policy") or payload.get("policy_params")
    else:
        raise RuntimeError(f"unknown ckpt payload type: {type(payload)}")

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
        (normalizer_params, policy_params), deterministic=True
    )
    return jax.jit(inference_fn)


# --------------------------------------------------------------------------- #
# The actual test
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_walk_to_drives_cat_toward_goal() -> None:
    """End-to-end Tier 2 → Tier 1 seam: WalkTo commands an AMP-trained policy
    to a target 1.5 m in front of the cat; over 5 s of sim time the distance to
    target should shrink to at most half the starting distance.

    The bar is intentionally loose. We're validating *plumbing*, not policy
    quality. If WalkTo's command never reaches the policy, distance won't move.
    If it reaches but is corrupted, distance might move the wrong way. Both
    failure modes are caught.
    """
    env_cfg = amp_env_mod.AMPEnvConfig(base_env_name="Go2JoystickFlatTerrain")
    env = amp_env_mod.make_env(env_cfg)

    inference_fn = _build_inference_fn(CKPT_PATH, env)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(0)
    rng, rk = jax.random.split(rng)
    state = jit_reset(rk)

    qpos0 = np.asarray(state.data.qpos).reshape(-1)
    start_xy = qpos0[:2].astype(np.float32)
    # Goal: 1.5 m forward in the world frame +x direction. The reset pose puts
    # the cat near origin facing roughly +x, so this is a natural straight walk
    # with mild heading correction.
    goal = WalkToGoal(target_xy=(float(start_xy[0] + 1.5), float(start_xy[1])))

    skill = WalkTo()

    # 5 s @ 50 Hz ctrl = 250 steps. Long enough for the heading controller to
    # settle and the AMP gait to clock through a few full cycles.
    n_steps = 250
    distances: list[float] = []

    for i in range(n_steps):
        qpos = np.asarray(state.data.qpos).reshape(-1)
        root_xy = qpos[:2].astype(np.float32)
        root_yaw = _yaw_from_quat(qpos[3], qpos[4], qpos[5], qpos[6])

        cmd = skill.step({"root_xy": root_xy, "root_yaw": root_yaw}, goal)

        # Dispatch the velocity command into env.info exactly the same way
        # render_imit_policy.py does — this is the AMP policy's joystick input.
        if "command" in state.info:
            new_cmd = jnp.asarray([cmd.vx, cmd.vy, cmd.yaw_rate], dtype=jnp.float32)
            state = state.replace(info={**state.info, "command": new_cmd})

        # Phase clock, same convention as render_imit_policy.py.
        phase = 2.0 * math.pi * (i % PHASE_PERIOD_STEPS) / float(PHASE_PERIOD_STEPS)
        phase_sc = jnp.asarray([math.sin(phase), math.cos(phase)], dtype=jnp.float32)
        obs_aug = _augment_obs_with_phase(state.obs, phase_sc)

        rng, rk = jax.random.split(rng)
        act, _ = inference_fn(obs_aug, rk)
        state = jit_step(state, act)

        # Track world-frame distance to the goal target.
        d = float(np.linalg.norm(np.asarray(state.data.qpos[:2]) - np.asarray(goal.target_xy)))
        distances.append(d)

    d0 = distances[0]
    d_final = distances[-1]
    d_min = min(distances)

    # Three asserts of increasing strictness:
    # (a) the cat did not flop and stay still
    # (b) some step got the cat at least halfway there
    # (c) the final step is closer than the start (we didn't overshoot wildly)
    assert d_min < d0 - 0.1, (
        f"cat didn't move toward goal: start={d0:.2f}m, min={d_min:.2f}m"
    )
    assert d_min < 0.5 * d0 + 0.01, (
        f"cat didn't get within half of start distance: start={d0:.2f}m, min={d_min:.2f}m"
    )
    assert d_final < d0, (
        f"final distance regressed past start: start={d0:.2f}m, final={d_final:.2f}m"
    )

    print(
        f"\n[walk_to-integration] start={d0:.3f}m  min={d_min:.3f}m  final={d_final:.3f}m"
    )
