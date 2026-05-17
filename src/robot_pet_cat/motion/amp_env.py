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


def make_env(cfg: AMPEnvConfig):
    """Construct the underlying mujoco_playground env.

    Lazy import so this file is parseable on machines without the heavy
    [motion] extras installed. If the named env doesn't exist (the registry's
    naming convention has drifted between versions), the error message lists
    every available env so the user can fix the config without grepping.
    """
    from mujoco_playground import registry  # noqa: PLC0415

    available = sorted(registry.ALL_ENVS) if hasattr(registry, "ALL_ENVS") else []
    try:
        env = registry.load(cfg.base_env_name)
    except (ValueError, KeyError) as e:
        go2 = [n for n in available if "go2" in n.lower()]
        msg_lines = [
            f"Failed to load env {cfg.base_env_name!r}.",
            "Likely a naming-convention mismatch with this mujoco_playground version.",
            f"All envs ({len(available)}): {available}",
            f"Go2 envs: {go2}",
            f"Underlying error: {e}",
        ]
        raise RuntimeError("\n".join(msg_lines)) from e
    env_cfg = env.default_config()
    if hasattr(env_cfg, "episode_length"):
        env_cfg.episode_length = cfg.episode_length_s
    if hasattr(env_cfg, "action_repeat"):
        env_cfg.action_repeat = cfg.action_repeat
    env = registry.load(cfg.base_env_name, env_cfg)
    return env


def extract_amp_features(
    qpos: jnp.ndarray,
    qvel: jnp.ndarray | None,
    use_qvel: bool,
) -> jnp.ndarray:
    """Return the per-state feature vector D consumes.

    AMP's discriminator works on a feature representation of the state, not
    the full observation. We use the robot's joint positions (and optionally
    velocities) -- not the policy observation -- because that's what AMP cares
    about stylistically.

    Args:
        qpos: (..., qpos_dim) joint positions including root.
        qvel: (..., qvel_dim) joint velocities, or None.
        use_qvel: whether to concatenate qvel into the feature.
    Returns:
        features: (..., qpos_dim) or (..., qpos_dim + qvel_dim).
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

    Inputs are rollouts of shape (T, B, dim) where T is rollout length and B
    is number of parallel envs. Output is a flat batch of (T-1) * B transitions
    suitable for direct use in the discriminator.
    """
    feats = extract_amp_features(qpos_traj, qvel_traj, use_qvel)  # (T, B, feat)
    s_t = feats[:-1]
    s_tp1 = feats[1:]
    pairs = jnp.concatenate([s_t, s_tp1], axis=-1)
    T_minus_1, B, two_feat = pairs.shape
    return pairs.reshape(T_minus_1 * B, two_feat)


def split_rng(rng: jax.Array, n: int) -> list[jax.Array]:
    """Convenience wrapper around jax.random.split."""
    keys = jax.random.split(rng, n)
    return [keys[i] for i in range(n)]
