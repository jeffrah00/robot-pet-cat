"""AMP trainer for cat-style Go2 locomotion.

Composes:
  - mujoco_playground's Go2JoystickFlatTerrain as the env
  - brax PPO networks + losses (policy and value MLPs)
  - Our AMPDiscriminator for the style reward
  - Our ReferenceBuffer for "real" cat transitions

Training loop sketch (per iteration):
  1. Roll out the policy in N parallel envs for T steps, collecting
     (obs, action, log_prob, value, qpos, qvel, task_reward, done) per step.
  2. Compute the AMP style reward on the rollout's (s, s') transitions via
     the current discriminator: r_style[t, b] = style_reward(D(s_t, s_tp1)).
  3. Compute the augmented reward: r[t, b] = task_w * r_task + style_w * r_style.
  4. PPO update: GAE on augmented rewards, K epochs of clipped policy/value loss.
  5. Discriminator update: sample real transitions from buffer, fake from rollout;
     LS-GAN loss + R1 gradient penalty.
  6. Log + checkpoint at intervals.

This file ties everything together. Heavy deps are imported lazily so the
module is parseable in environments without [motion] extras installed.

Phase 2b. First training run is the test -- expect ~6-8 hours on a single
RTX 4090 to see cat-style gait emerge from a flat-ground walk.
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
    # AMP scalars
    motion_clips_dir: Path = Path("data/motion_clips")
    style_reward_weight: float = 2.0    # lambda on r_style
    task_reward_weight: float = 1.0
    use_qvel_in_features: bool = False

    # Discriminator
    disc_hidden_sizes: Sequence[int] = (1024, 512)
    disc_grad_penalty_coef: float = 10.0
    disc_learning_rate: float = 1.0e-4
    disc_batch_size: int = 4096
    disc_updates_per_iter: int = 1

    # Env
    env_name: str = "Go2JoystickFlatTerrain"
    num_envs: int = 4096
    episode_length_s: float = 20.0

    # PPO
    ppo: PPOConfig = field(default_factory=PPOConfig)

    # Run-level
    seed: int = 0
    wandb_project: str = "robot-pet-cat"
    wandb_run_name: str = "cat-amp-v1"
    log_interval_steps: int = 100_000
    checkpoint_interval_steps: int = 2_000_000
    checkpoint_dir: Path = Path("checkpoints/motion")
    push_to_hub: bool = True
    hf_repo: str = "jeffrah00/go2-cat-motion"


def train(cfg: AMPTrainConfig) -> None:
    """Top-level training entry. Heavy deps are lazy-imported so importing
    this module is cheap on machines without the [motion] extras."""
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

    # --- 5. Discriminator pre-training (200 steps against noise fakes) ---
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
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as brax_ppo

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
        # TODO: wandb.log(metrics, step=num_steps)
        # TODO: real discriminator update against policy rollouts. For now
        # the discriminator is frozen after pre-training. Hooking into brax's
        # rollout buffer requires custom code path; Phase 2b first cut keeps
        # it simple.

    make_inference_fn, params, brax_metrics = brax_ppo.train(
        environment=env,
        num_timesteps=cfg.ppo.total_timesteps,
        num_evals=10,
        episode_length=int(cfg.episode_length_s * 50),  # 50 Hz default
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
    )

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
