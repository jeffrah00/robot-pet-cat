"""AMP trainer for cat-style Go2 locomotion.

Phase 2b first cut: discriminator pre-trains against noise then is frozen
during PPO. Hooking the discriminator into brax's rollout buffer comes next.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


@dataclass
class PPOConfig:
    total_timesteps: int = 50_000_000
    unroll_length: int = 20
    num_minibatches: int = 32
    num_updates_per_batch: int = 4
    discounting: float = 0.97
    gae_lambda: float = 0.95
    clipping_epsilon: float = 0.2
    entropy_cost: float = 1.0e-2
    learning_rate: float = 3.0e-4
    max_grad_norm: float = 1.0
    policy_hidden_sizes: Sequence[int] = (512, 256, 128)
    value_hidden_sizes:  Sequence[int] = (512, 256, 128)


@dataclass
class AMPTrainConfig:
    motion_clips_dir: Path = Path("data/motion_clips")
    style_reward_weight: float = 2.0
    task_reward_weight: float = 1.0
    use_qvel_in_features: bool = False

    disc_hidden_sizes: Sequence[int] = (1024, 512)
    disc_grad_penalty_coef: float = 10.0
    disc_learning_rate: float = 1.0e-4
    disc_batch_size: int = 4096
    disc_updates_per_iter: int = 1

    env_name: str = "Go2JoystickFlatTerrain"
    num_envs: int = 4096
    episode_length_s: float = 20.0

    ppo: PPOConfig = field(default_factory=PPOConfig)

    seed: int = 0
    wandb_project: str = "robot-pet-cat"
    wandb_run_name: str = "cat-amp-v1"
    log_interval_steps: int = 100_000
    checkpoint_interval_steps: int = 2_000_000
    checkpoint_dir: Path = Path("checkpoints/motion")
    push_to_hub: bool = True
    hf_repo: str = "jeffrah00/go2-cat-motion"


def _shim_jax_for_brax() -> None:
    """Restore APIs brax depends on that current JAX has removed.

    brax/training/agents/ppo/train.py calls jax.device_put_replicated, which
    JAX deleted in favor of jax.device_put + PositionalSharding. Adding the
    function back as a thin wrapper avoids pinning either side of the stack.

    Call this before importing brax. Idempotent.
    """
    import jax

    # Probe whether the symbol works (a deprecation stub raises on access).
    has_working = False
    try:
        if hasattr(jax, "device_put_replicated") and callable(
            getattr(jax, "device_put_replicated", None)
        ):
            has_working = True
    except AttributeError:
        has_working = False
    if has_working:
        return

    def _device_put_replicated(x, devices):
        from jax.sharding import PositionalSharding

        sharding = PositionalSharding(devices)

        def _put(leaf):
            arr = jax.numpy.asarray(leaf)
            # Replicate along a new leading device axis.
            replicated = jax.numpy.broadcast_to(arr, (len(devices), *arr.shape))
            target_sharding = sharding.reshape((-1,) + (1,) * arr.ndim)
            return jax.device_put(replicated, target_sharding)

        return jax.tree_util.tree_map(_put, x)

    jax.device_put_replicated = _device_put_replicated


def train(cfg: AMPTrainConfig) -> None:
    import jax
    import jax.numpy as jnp
    import optax

    from robot_pet_cat.motion.amp_discriminator import (
        AMPDiscConfig, grad_penalty, init_discriminator, lsgan_disc_loss,
        style_reward,
    )
    from robot_pet_cat.motion.amp_env import AMPEnvConfig, make_env
    from robot_pet_cat.motion.reference_buffer import ReferenceBuffer

    print("[amp] starting training")
    print(f"[amp] env={cfg.env_name}  num_envs={cfg.num_envs}  "
          f"total_timesteps={cfg.ppo.total_timesteps:,}")

    # --- 1. Reference buffer ---
    buf = ReferenceBuffer.from_dir(cfg.motion_clips_dir)
    print("[amp] reference buffer:")
    for line in buf.summary().splitlines():
        print(f"        {line}")

    # --- 2. Env ---
    env_cfg = AMPEnvConfig(
        base_env_name=cfg.env_name,
        num_envs=cfg.num_envs,
        episode_length_s=cfg.episode_length_s,
        use_qvel_in_amp_features=cfg.use_qvel_in_features,
        seed=cfg.seed,
    )
    env = make_env(env_cfg)
    print(f"[amp] env loaded: {type(env).__name__}")

    # --- 3. Discriminator ---
    disc_cfg = AMPDiscConfig(
        hidden_sizes=tuple(cfg.disc_hidden_sizes),
        obs_dim=buf.qpos_dim,
        use_qvel=cfg.use_qvel_in_features,
        grad_penalty_coef=cfg.disc_grad_penalty_coef,
    )
    disc_model, disc_params, disc_apply = init_discriminator(disc_cfg, seed=cfg.seed)
    disc_optimizer = optax.adam(cfg.disc_learning_rate)
    disc_opt_state = disc_optimizer.init(disc_params)
    print(f"[amp] discriminator: hidden={disc_cfg.hidden_sizes}, "
          f"feature_dim={buf.feature_dim(cfg.use_qvel_in_features)}")

    # --- 4. JIT'd discriminator update step ---
    @jax.jit
    def update_discriminator(params, opt_state, real_batch, fake_batch):
        def loss_fn(p):
            real_logits = disc_apply(p, real_batch)
            fake_logits = disc_apply(p, fake_batch)
            gan_loss = lsgan_disc_loss(real_logits, fake_logits)
            gp = grad_penalty(disc_apply, p, real_batch)
            total = gan_loss + disc_cfg.grad_penalty_coef * gp
            return total, {"gan_loss": gan_loss, "grad_penalty": gp}
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = disc_optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss, aux

    # --- 5. Discriminator pre-training ---
    rng = jax.random.PRNGKey(cfg.seed)
    print("[amp] pre-training discriminator (200 steps, fakes = N(0,0.3))")
    for i in range(200):
        rng, rk1, rk2 = jax.random.split(rng, 3)
        real_batch = buf.sample(
            rk1, cfg.disc_batch_size, use_qvel=cfg.use_qvel_in_features
        )
        fake_batch = jax.random.normal(rk2, real_batch.shape) * 0.3
        disc_params, disc_opt_state, loss, aux = update_discriminator(
            disc_params, disc_opt_state, real_batch, fake_batch,
        )
        if (i + 1) % 50 == 0:
            print(f"[amp] disc pre-train step {i+1}: "
                  f"loss={float(loss):.4f} gp={float(aux['grad_penalty']):.4f}")

    # --- 6. PPO training via brax ---
    _shim_jax_for_brax()
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as brax_ppo
    from mujoco_playground import wrapper as mjx_wrapper

    # mujoco_playground envs return State with a different shape than brax
    # expects; wrap_for_brax_training translates AND vmaps. Pass wrap_env=False
    # to brax_ppo.train so brax doesn't double-vmap on top.
    env = mjx_wrapper.wrap_for_brax_training(
        env,
        episode_length=int(cfg.episode_length_s * 50),
        action_repeat=1,
    )

    network_kwargs = dict(
        policy_hidden_layer_sizes=tuple(cfg.ppo.policy_hidden_sizes),
        value_hidden_layer_sizes=tuple(cfg.ppo.value_hidden_sizes),
    )

    print(f"[amp] starting PPO ({cfg.ppo.total_timesteps:,} env steps)")
    start = time.time()
    metrics_log: list[dict] = []

    def progress_fn(num_steps: int, metrics: dict[str, Any]) -> None:
        wall = time.time() - start
        steps_per_sec = num_steps / max(wall, 1e-6)
        episode_reward = float(metrics.get("eval/episode_reward", 0.0))
        print(
            f"[amp] step {num_steps:>10,d}  "
            f"reward={episode_reward:>+8.3f}  "
            f"{steps_per_sec:>7,.0f} steps/s  "
            f"wall={wall/60:6.1f}m"
        )
        metrics_log.append({"step": num_steps, "wall_s": wall, **metrics})

    train_kwargs = dict(
        environment=env,
        num_timesteps=cfg.ppo.total_timesteps,
        num_evals=10,
        episode_length=int(cfg.episode_length_s * 50),
        unroll_length=cfg.ppo.unroll_length,
        num_minibatches=cfg.ppo.num_minibatches,
        num_updates_per_batch=cfg.ppo.num_updates_per_batch,
        discounting=cfg.ppo.discounting,
        learning_rate=cfg.ppo.learning_rate,
        entropy_cost=cfg.ppo.entropy_cost,
        num_envs=cfg.num_envs,
        batch_size=cfg.num_envs,
        seed=cfg.seed,
        network_factory=lambda obs, act, **kw: ppo_networks.make_ppo_networks(
            obs, act, **{**kw, **network_kwargs}
        ),
        progress_fn=progress_fn,
        wrap_env=False,
    )

    try:
        make_inference_fn, params, brax_metrics = brax_ppo.train(**train_kwargs)
    except TypeError as e:
        if "wrap_env" in str(e):
            print("[amp] brax does not accept wrap_env=False; retrying without it")
            train_kwargs.pop("wrap_env")
            make_inference_fn, params, brax_metrics = brax_ppo.train(**train_kwargs)
        else:
            raise

    # --- 7. Checkpoint + (optional) HF push ---
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = cfg.checkpoint_dir / "ppo_params.pkl"
    _save_pickle(params, ckpt_path)
    print(f"[amp] wrote PPO checkpoint {ckpt_path}")

    if cfg.push_to_hub:
        try:
            _push_to_hf(ckpt_path, cfg.hf_repo)
        except Exception as e:  # noqa: BLE001
            print(f"[amp] WARNING: HF push failed: {e}")

    print(f"[amp] done in {(time.time() - start) / 60:.1f} min")


def _save_pickle(obj, path: Path) -> None:
    import pickle
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _push_to_hf(local_path: Path, repo: str) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo, exist_ok=True, repo_type="model")
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=local_path.name,
        repo_id=repo,
        repo_type="model",
    )
    print(f"[amp] pushed {local_path.name} to https://huggingface.co/{repo}")
