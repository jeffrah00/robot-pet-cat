"""Env wrapper for AMP training over Go2 locomotion."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass
class AMPEnvConfig:
    base_env_name: str = "Go2JoystickFlatTerrain"
    num_envs: int = 4096
    episode_length_s: float = 20.0
    action_repeat: int = 1
    use_qvel_in_amp_features: bool = False
    seed: int = 0


def _resolve_joystick_default_config():
    """Get the Joystick default config without instantiating an env.

    Playground exposes default_config as a module-level function in the
    joystick module, not as an instance method on Joystick. The previous
    env.default_config() call worked in older playground versions but
    raises AttributeError on current installs.
    """
    import importlib

    for mod_name in (
        "mujoco_playground._src.locomotion.go1.joystick",
        "mujoco_playground.locomotion.go1.joystick",
        "mujoco_playground._src.locomotion.unitree_go1.joystick",
    ):
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        if hasattr(mod, "default_config") and callable(mod.default_config):
            return mod.default_config()
    return None


def make_env(cfg: AMPEnvConfig):
    """Construct the underlying mujoco_playground env.

    Resolves default config externally rather than via env.default_config()
    (which raises AttributeError on current playground). Casts numeric
    overrides through int() because the default config is an
    ml_collections.ConfigDict and its fields are type-locked:
      - episode_length is an INT number of physics steps (not seconds).
        Go2 joystick env runs at 50 Hz, so multiply seconds by 50.
      - action_repeat is also INT.
    """
    from robot_pet_cat.motion import go2_env  # noqa: F401  # registers Go2

    from mujoco_playground import registry

    available = sorted(registry.ALL_ENVS) if hasattr(registry, "ALL_ENVS") else []

    env_cfg = _resolve_joystick_default_config()
    if env_cfg is not None:
        if hasattr(env_cfg, "episode_length"):
            try:
                env_cfg.episode_length = int(cfg.episode_length_s * 50)
            except (AttributeError, TypeError):
                pass
        if hasattr(env_cfg, "action_repeat"):
            try:
                env_cfg.action_repeat = int(cfg.action_repeat)
            except (AttributeError, TypeError):
                pass
        # Force JAX backend so we dodge the mujoco/warp ABI mismatch in
        # mjx.put_model.
        if hasattr(env_cfg, "impl"):
            try:
                env_cfg.impl = "jax"
            except (AttributeError, TypeError):
                pass

    try:
        if env_cfg is None:
            env = registry.load(cfg.base_env_name)
        else:
            env = registry.load(cfg.base_env_name, env_cfg)
    except (ValueError, KeyError) as e:
        go2 = [n for n in available if "go2" in n.lower()]
        msg_lines = [
            f"Failed to load env {cfg.base_env_name!r}.",
            "Likely a naming-convention mismatch with this mujoco_playground version,",
            "or robot_pet_cat.motion.go2_env failed to register the Go2 env.",
            f"All envs ({len(available)}): {available}",
            f"Go2 envs: {go2}",
            f"Underlying error: {e}",
        ]
        raise RuntimeError("\n".join(msg_lines)) from e
    return env


def extract_amp_features(
    qpos: jnp.ndarray,
    qvel: jnp.ndarray | None,
    use_qvel: bool,
) -> jnp.ndarray:
    """Per-state feature vector D consumes (qpos, optionally + qvel)."""
    if use_qvel:
        if qvel is None:
            raise ValueError("use_qvel=True but qvel was None")
        return jnp.concatenate([qpos, qvel], axis=-1)
    return qpos


def build_transitions_for_amp(
    qpos_traj: jnp.ndarray,
    qvel_traj: jnp.ndarray | None,
    use_qvel: bool,
) -> jnp.ndarray:
    """Build (B, 2 * feat_dim) AMP transition features from a (T, B, dim) rollout."""
    feats = extract_amp_features(qpos_traj, qvel_traj, use_qvel)
    s_t = feats[:-1]
    s_tp1 = feats[1:]
    pairs = jnp.concatenate([s_t, s_tp1], axis=-1)
    T_minus_1, B, two_feat = pairs.shape
    return pairs.reshape(T_minus_1 * B, two_feat)


def split_rng(rng: jax.Array, n: int) -> list[jax.Array]:
    keys = jax.random.split(rng, n)
    return [keys[i] for i in range(n)]
