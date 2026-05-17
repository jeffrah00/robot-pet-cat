"""Env wrapper for AMP training over Go2 locomotion.

We don't augment the reward inside env.step() because the discriminator's
parameters change every training iteration and threading them through the
env's functional state is awkward. Instead, the env returns vanilla
mujoco_playground rollouts; the trainer computes the AMP style reward
afterwards in a separate JAX-jitted function and adds it to the task reward
before the PPO update.

This keeps the env definition clean and decouples the AMP machinery from
the underlying RL infrastructure.
"""

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
    joystick module, not as an instance/class method on Joystick. The
    previous `env.default_config()` call worked in older playground
    versions but raises AttributeError on current installs.
    """
    import importlib  # noqa: PLC0415

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

    Lazy import so this file is parseable on machines without the heavy
    [motion] extras installed. If the named env doesn't exist (the registry's
    naming convention has drifted between versions), the error message lists
    every available env so the user can fix the config without grepping.

    We unconditionally import robot_pet_cat.motion.go2_env first so the Go2
    Joystick env gets registered before any registry lookup. mujoco_playground
    only ships Go1 envs out of the box.
    """
    from robot_pet_cat.motion import go2_env  # noqa: F401, PLC0415  # side-effect: registers Go2

    from mujoco_playground import registry  # noqa: PLC0415

    available = sorted(registry.ALL_ENVS) if hasattr(registry, "ALL_ENVS") else []

    # Resolve default config externally rather than via env.default_config(),
    # which raises AttributeError on current playground versions.
    env_cfg = _resolve_joystick_default_config()
    if env_cfg is not None:
        if hasattr(env_cfg, "episode_length"):
            env_cfg.episode_length = cfg.episode_length_s
        if hasattr(env_cfg, "action_repeat"):
            env_cfg.action_repeat = cfg.action_repeat
        # Force JAX backend here too -- our go2 wrapper does it on the inner
        # Joystick instantiation, but if registry.load(name, config) honors
        # this config without going through our wrapper's path we want jax.
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
    """Return the per-state feature vector D consumes.

    AMP's discriminator works on a feature representation of the state, not
    the full observation. We use the robot's joint positions (and optionally
    velocities), not the policy observation, because that's what AMP cares
    about stylistically.
    """
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
    """Build (B, 2 * feat_dim) AMP transition features from a rollout.

    Inputs are rollouts of shape (T, B, dim). Output is a flat batch of
    (T-1) * B transitions ready for the discriminator.
    """
    feats = extract_amp_features(qpos_traj, qvel_traj, use_qvel)
    s_t = feats[:-1]
    s_tp1 = feats[1:]
    pairs = jnp.concatenate([s_t, s_tp1], axis=-1)
    T_minus_1, B, two_feat = pairs.shape
    return pairs.reshape(T_minus_1 * B, two_feat)


def split_rng(rng: jax.Array, n: int) -> list[jax.Array]:
    keys = jax.random.split(rng, n)
    return [keys[i] for i in range(n)]
