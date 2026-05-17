"""True AMP PPO loop for Go2 cat-style locomotion.

Differences from amp_trainer.py (v1):
  v1 frozen discriminator after pre-train, then handed the env to brax's
      stock train(). Style reward never updated against policy rollouts.
  v2 (this file) replaces brax's training loop with our own. Per
      iteration:
        1. Rollout policy for unroll_length steps in num_envs parallel envs.
        2. Compute style reward from D on the rollout's (qpos_t, qpos_t+1)
           transitions.
        3. Mix: r_total = task_w * r_task + style_w * style_reward.
        4. GAE on the augmented reward.
        5. K epochs of clipped PPO update (policy + value) on minibatches.
        6. D update: real from reference buffer, fake from rollout.
        7. Log + checkpoint.

We reuse brax's lower-level utilities -- ppo_networks.make_ppo_networks,
losses.compute_ppo_loss, acting.generate_unroll -- so we're not
reimplementing PPO, just the AMP-specific glue around it.

JIT layout:
  - `rollout_fn`: jitted, returns Transition with augmented reward already
    computed. Takes (env_state, policy_params, value_params, d_params, rng).
  - `ppo_update_step`: jitted, one minibatch update of policy+value.
  - `disc_update_step`: jitted, one D update with LSGAN + R1.

Single-device (4090). No pmap. Designed for one GPU.
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax


@dataclass
class PPOConfig:
    total_iterations: int = 5000
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
    value_hidden_sizes: Sequence[int] = (512, 256, 128)


@dataclass
class AMPTrainConfigV2:
    motion_clips_dir: Path = Path("data/motion_clips")
    style_reward_weight: float = 2.0
    task_reward_weight: float = 1.0
    use_qvel_in_features: bool = False

    disc_hidden_sizes: Sequence[int] = (1024, 512)
    disc_grad_penalty_coef: float = 10.0
    disc_learning_rate: float = 1.0e-4
    disc_batch_size: int = 4096
    disc_updates_per_iter: int = 1
    disc_pretrain_steps: int = 200

    env_name: str = "Go2JoystickFlatTerrain"
    num_envs: int = 4096
    episode_length_s: float = 20.0

    ppo: PPOConfig = field(default_factory=PPOConfig)

    seed: int = 0
    log_interval_iters: int = 10
    eval_interval_iters: int = 100
    checkpoint_interval_iters: int = 500
    checkpoint_dir: Path = Path("checkpoints/motion_v2")

    # Resume from a v1 checkpoint to skip the cold-start exploration phase.
    init_policy_ckpt: Path | None = None


def train(cfg: AMPTrainConfigV2) -> None:
    """Top-level AMP training entry. Heavy deps are lazy-imported so the
    module is parseable on machines without [motion] extras."""
    from brax.training.agents.ppo import networks as ppo_networks
    from mujoco_playground import wrapper as mjx_wrapper

    from robot_pet_cat.motion.amp_discriminator import (
        AMPDiscConfig, grad_penalty, init_discriminator, lsgan_disc_loss,
        style_reward,
    )
    from robot_pet_cat.motion.amp_env import AMPEnvConfig, make_env
    from robot_pet_cat.motion.amp_trainer import _shim_jax_for_brax
    from robot_pet_cat.motion.reference_buffer import ReferenceBuffer

    print("[amp-v2] starting AMP-PPO training")
    print(f"[amp-v2] env={cfg.env_name}  num_envs={cfg.num_envs}  "
          f"total_iters={cfg.ppo.total_iterations:,}")

    # --- 1. Reference buffer ---
    buf = ReferenceBuffer.from_dir(cfg.motion_clips_dir)
    print("[amp-v2] reference buffer:")
    for line in buf.summary().splitlines():
        print(f"          {line}")

    # --- 2. Env (single-call wrapping) ---
    env_cfg = AMPEnvConfig(
        base_env_name=cfg.env_name,
        num_envs=cfg.num_envs,
        episode_length_s=cfg.episode_length_s,
        use_qvel_in_amp_features=cfg.use_qvel_in_features,
        seed=cfg.seed,
    )
    raw_env = make_env(env_cfg)
    env = mjx_wrapper.wrap_for_brax_training(
        raw_env,
        episode_length=int(cfg.episode_length_s * 50),
        action_repeat=1,
    )
    print(f"[amp-v2] env loaded + wrapped: {type(env).__name__}")

    # --- 3. Network + optimizer ---
    _shim_jax_for_brax()  # brax internals still expect device_put_replicated
    ppo_nets = ppo_networks.make_ppo_networks(
        observation_size=env.observation_size,
        action_size=env.action_size,
        policy_hidden_layer_sizes=tuple(cfg.ppo.policy_hidden_sizes),
        value_hidden_layer_sizes=tuple(cfg.ppo.value_hidden_sizes),
    )
    make_inference_fn = ppo_networks.make_inference_fn(ppo_nets)

    rng = jax.random.PRNGKey(cfg.seed)
    rng, k_policy, k_value = jax.random.split(rng, 3)
    policy_params = ppo_nets.policy_network.init(k_policy)
    value_params = ppo_nets.value_network.init(k_value)

    # brax's policy_network.apply signature is (normalizer_params, params, obs).
    # mujoco_playground envs return obs as a DICT (typically {"state": <array>}),
    # so the normalizer spec has to be a dict-shaped pytree, not a flat Array.
    # Without this brax does normalizer_params.mean[obs_key] on a flat array
    # and crashes with "JAX does not support string indexing".
    from brax.training.acme import running_statistics, specs

    def _to_spec(size):
        # size can be int (single-dim obs) or tuple (multi-dim obs).
        shape = (size,) if isinstance(size, int) else tuple(size)
        return specs.Array(shape, jnp.float32)

    obs_size = env.observation_size
    if isinstance(obs_size, dict):
        obs_spec = {k: _to_spec(v) for k, v in obs_size.items()}
    else:
        obs_spec = _to_spec(obs_size)
    normalizer_params = running_statistics.init_state(obs_spec)

    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.ppo.max_grad_norm),
        optax.adam(cfg.ppo.learning_rate),
    )
    opt_state = optimizer.init((policy_params, value_params))

    if cfg.init_policy_ckpt is not None and cfg.init_policy_ckpt.is_file():
        import pickle
        print(f"[amp-v2] loading initial policy params from {cfg.init_policy_ckpt}")
        with cfg.init_policy_ckpt.open("rb") as f:
            init_params = pickle.load(f)
        # brax PPO pickled "params" is a (normalizer_params, policy_params,
        # value_params) triple in recent versions. Probe and unpack.
        if isinstance(init_params, tuple) and len(init_params) == 3:
            _, policy_params, value_params = init_params
        elif isinstance(init_params, tuple) and len(init_params) == 2:
            policy_params, value_params = init_params
        opt_state = optimizer.init((policy_params, value_params))

    # --- 4. Discriminator ---
    disc_cfg = AMPDiscConfig(
        hidden_sizes=tuple(cfg.disc_hidden_sizes),
        obs_dim=buf.qpos_dim,
        use_qvel=cfg.use_qvel_in_features,
        grad_penalty_coef=cfg.disc_grad_penalty_coef,
    )
    _, disc_params, disc_apply = init_discriminator(disc_cfg, seed=cfg.seed)
    disc_optimizer = optax.adam(cfg.disc_learning_rate)
    disc_opt_state = disc_optimizer.init(disc_params)
    print(f"[amp-v2] discriminator: hidden={disc_cfg.hidden_sizes}, "
          f"feature_dim={buf.feature_dim(cfg.use_qvel_in_features)}")

    # --- 5. JITed building blocks --------------------------------------------

    @jax.jit
    def update_discriminator(d_params, d_opt_state, real_batch, fake_batch):
        def loss_fn(p):
            real_logits = disc_apply(p, real_batch)
            fake_logits = disc_apply(p, fake_batch)
            gan_loss = lsgan_disc_loss(real_logits, fake_logits)
            gp = grad_penalty(disc_apply, p, real_batch)
            total = gan_loss + disc_cfg.grad_penalty_coef * gp
            return total, {"gan_loss": gan_loss, "grad_penalty": gp}
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(d_params)
        updates, new_opt = disc_optimizer.update(grads, d_opt_state, d_params)
        new_p = optax.apply_updates(d_params, updates)
        return new_p, new_opt, loss, aux

    # Rollout: vmap'd env.step + policy. Returns transitions including the
    # qpos history we need for AMP style reward.
    def policy_act(norm_params, params, obs, rng):
        # ppo_nets.policy_network.apply takes (normalizer_params, params, obs)
        # and returns (mean, std) parameters of a tanh-squashed normal.
        logits = ppo_nets.policy_network.apply(norm_params, params, obs)
        dist = ppo_nets.parametric_action_distribution
        raw_act = dist.sample_no_postprocessing(logits, rng)
        log_prob = dist.log_prob(logits, raw_act)
        act = dist.postprocess(raw_act)
        return act, log_prob, raw_act

    def value_predict(norm_params, value_params, obs):
        v = ppo_nets.value_network.apply(norm_params, value_params, obs)
        return jnp.squeeze(v, axis=-1) if v.ndim > 1 else v

    def _get_qpos(env_state):
        """Extract qpos out of an env state. brax convention is
        state.pipeline_state.qpos; mujoco_playground (wrap_for_brax_training)
        gives state.data.qpos. Try both at trace time so the JITed scan picks
        whichever attribute path actually exists on this env's State."""
        s = getattr(env_state, "pipeline_state", None)
        if s is None:
            s = getattr(env_state, "data", None)
        if s is None:
            raise AttributeError(
                "env_state has neither .pipeline_state nor .data; "
                f"available attrs: {dir(env_state)}"
            )
        return s.qpos

    @functools.partial(jax.jit, static_argnames=("unroll_length",))
    def rollout(env_state, norm_params, policy_params, value_params, d_params, rng,
                unroll_length: int):
        """Roll out `unroll_length` steps in parallel envs. Returns dict of
        per-step arrays of shape (T, num_envs, ...) plus the final env_state
        and rng."""
        def step_fn(carry, _):
            env_state, rng = carry
            rng, k_act = jax.random.split(rng)
            act, log_prob, raw_act = policy_act(
                norm_params, policy_params, env_state.obs, k_act,
            )
            v = value_predict(norm_params, value_params, env_state.obs)
            qpos_before = _get_qpos(env_state)
            next_env_state = env.step(env_state, act)
            qpos_after = _get_qpos(next_env_state)

            # Style reward on the (qpos_before, qpos_after) transition.
            transition_feat = jnp.concatenate(
                [qpos_before, qpos_after], axis=-1,
            )
            d_logits = disc_apply(d_params, transition_feat)
            # style_reward util expects logits in shape (B,); already that.
            s_r = jnp.maximum(0.0, 1.0 - 0.25 * (d_logits - 1.0) ** 2)

            task_r = next_env_state.reward
            total_r = (
                cfg.task_reward_weight * task_r
                + cfg.style_reward_weight * s_r
            )

            return (next_env_state, rng), {
                "obs": env_state.obs,
                "act": act,
                "log_prob": log_prob,
                "raw_act": raw_act,
                "value": v,
                "reward": total_r,
                "task_reward": task_r,
                "style_reward": s_r,
                "done": next_env_state.done,
                "qpos": qpos_before,
                "qpos_next": qpos_after,
            }

        (final_env_state, final_rng), traj = jax.lax.scan(
            step_fn, (env_state, rng), None, length=unroll_length,
        )
        # Bootstrap value for GAE.
        last_v = value_predict(norm_params, value_params, final_env_state.obs)
        return final_env_state, final_rng, traj, last_v

    @jax.jit
    def compute_gae(rewards, values, dones, last_v):
        """GAE-lambda. All inputs shape (T, B)."""
        gamma = cfg.ppo.discounting
        lam = cfg.ppo.gae_lambda

        def body(carry, t):
            adv = carry
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * mask * values_p1[t] - values[t]
            adv = delta + gamma * lam * mask * adv
            return adv, adv

        values_p1 = jnp.concatenate([values[1:], last_v[None]], axis=0)
        _, advs = jax.lax.scan(
            body, jnp.zeros_like(last_v), jnp.arange(rewards.shape[0]),
            reverse=True,
        )
        returns = advs + values
        return advs, returns

    @jax.jit
    def ppo_update(norm_params, policy_params, value_params, opt_state,
                   obs, act, log_prob_old, raw_act, advs, returns):
        def loss_fn(params):
            p_params, v_params = params
            logits = ppo_nets.policy_network.apply(norm_params, p_params, obs)
            dist = ppo_nets.parametric_action_distribution
            log_prob = dist.log_prob(logits, raw_act)
            entropy = dist.entropy(logits, jax.random.PRNGKey(0)).mean()

            ratio = jnp.exp(log_prob - log_prob_old)
            # Advantage normalization for stability.
            adv_norm = (advs - advs.mean()) / (advs.std() + 1e-8)
            unclipped = ratio * adv_norm
            clipped = jnp.clip(
                ratio, 1 - cfg.ppo.clipping_epsilon,
                1 + cfg.ppo.clipping_epsilon,
            ) * adv_norm
            policy_loss = -jnp.minimum(unclipped, clipped).mean()

            v_pred = ppo_nets.value_network.apply(norm_params, v_params, obs)
            if v_pred.ndim > 1:
                v_pred = jnp.squeeze(v_pred, axis=-1)
            value_loss = 0.5 * ((v_pred - returns) ** 2).mean()

            total = policy_loss + value_loss - cfg.ppo.entropy_cost * entropy
            return total, {"policy_loss": policy_loss,
                           "value_loss": value_loss,
                           "entropy": entropy}

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            (policy_params, value_params),
        )
        updates, new_opt = optimizer.update(grads, opt_state,
                                            (policy_params, value_params))
        new_params = optax.apply_updates((policy_params, value_params), updates)
        return new_params[0], new_params[1], new_opt, loss, aux

    # --- 6. Pre-train D against noise (warm start) -------------------
    print(f"[amp-v2] pre-training D ({cfg.disc_pretrain_steps} steps, fakes=N(0,0.3))")
    for i in range(cfg.disc_pretrain_steps):
        rng, rk1, rk2 = jax.random.split(rng, 3)
        real_b = buf.sample(rk1, cfg.disc_batch_size,
                            use_qvel=cfg.use_qvel_in_features)
        fake_b = jax.random.normal(rk2, real_b.shape) * 0.3
        disc_params, disc_opt_state, loss, aux = update_discriminator(
            disc_params, disc_opt_state, real_b, fake_b,
        )
        if (i + 1) % 50 == 0:
            print(f"[amp-v2]   D pre-train {i+1}: loss={float(loss):.4f}")

    # --- 7. Main training loop ----------------------------------------
    print(f"[amp-v2] starting AMP-PPO ({cfg.ppo.total_iterations} iterations)")
    rng, k_reset = jax.random.split(rng)
    env_state = jax.jit(env.reset)(jax.random.split(k_reset, cfg.num_envs))

    start = time.time()
    total_env_steps = 0
    for it in range(cfg.ppo.total_iterations):
        rng, k_roll = jax.random.split(rng)
        env_state, _, traj, last_v = rollout(
            env_state, normalizer_params, policy_params, value_params, disc_params, k_roll,
            unroll_length=cfg.ppo.unroll_length,
        )

        advs, returns = compute_gae(
            traj["reward"], traj["value"], traj["done"], last_v,
        )

        # Flatten T,B -> T*B for minibatch SGD. obs may be a dict (mujoco_playground
        # gives {"state": array}); tree_map handles both dict and flat array cases.
        def flatten(x):
            return x.reshape((-1,) + x.shape[2:])
        obs_flat = jax.tree_util.tree_map(flatten, traj["obs"])
        act_flat = flatten(traj["act"])
        raw_act_flat = flatten(traj["raw_act"])
        log_prob_flat = flatten(traj["log_prob"])
        advs_flat = flatten(advs)
        returns_flat = flatten(returns)

        # obs_flat may be a dict (pytree) -- get N from any leaf.
        n = jax.tree_util.tree_leaves(obs_flat)[0].shape[0]
        mb_size = n // cfg.ppo.num_minibatches
        for epoch in range(cfg.ppo.num_updates_per_batch):
            rng, k_perm = jax.random.split(rng)
            perm = jax.random.permutation(k_perm, n)
            for mb in range(cfg.ppo.num_minibatches):
                idx = perm[mb * mb_size:(mb + 1) * mb_size]
                obs_mb = jax.tree_util.tree_map(lambda x: x[idx], obs_flat)
                policy_params, value_params, opt_state, ppo_loss, ppo_aux = (
                    ppo_update(
                        normalizer_params, policy_params, value_params, opt_state,
                        obs_mb, act_flat[idx], log_prob_flat[idx],
                        raw_act_flat[idx], advs_flat[idx], returns_flat[idx],
                    )
                )

        # D update: real from buffer, fake from rollout transitions.
        for _ in range(cfg.disc_updates_per_iter):
            rng, k_real = jax.random.split(rng)
            real_b = buf.sample(k_real, cfg.disc_batch_size,
                                use_qvel=cfg.use_qvel_in_features)
            fake_feats = jnp.concatenate(
                [flatten(traj["qpos"]), flatten(traj["qpos_next"])], axis=-1,
            )
            # Subsample fake to disc_batch_size if larger.
            if fake_feats.shape[0] > cfg.disc_batch_size:
                rng, k_sub = jax.random.split(rng)
                idx = jax.random.permutation(k_sub, fake_feats.shape[0])[:cfg.disc_batch_size]
                fake_b = fake_feats[idx]
            else:
                fake_b = fake_feats
            disc_params, disc_opt_state, d_loss, d_aux = update_discriminator(
                disc_params, disc_opt_state, real_b, fake_b,
            )

        # Update running normalizer with the new batch of observations so
        # next iteration's policy sees properly-scaled inputs.
        normalizer_params = running_statistics.update(
            normalizer_params, obs_flat,
        )

        total_env_steps += cfg.num_envs * cfg.ppo.unroll_length
        if (it + 1) % cfg.log_interval_iters == 0:
            wall = time.time() - start
            sps = total_env_steps / max(wall, 1e-6)
            r_task = float(jnp.mean(traj["task_reward"]))
            r_style = float(jnp.mean(traj["style_reward"]))
            print(
                f"[amp-v2] it {it+1:>5d}  step {total_env_steps:>11,d}  "
                f"r_task={r_task:+.3f} r_style={r_style:+.3f}  "
                f"d_loss={float(d_loss):.3f}  "
                f"ppo_loss={float(ppo_loss):+.3f}  "
                f"{sps:>7,.0f} steps/s  wall={wall/60:.1f}m"
            )

        if (it + 1) % cfg.checkpoint_interval_iters == 0:
            _save_ckpt(cfg.checkpoint_dir, it + 1,
                       policy_params, value_params, disc_params,
                       normalizer_params=normalizer_params)

    _save_ckpt(cfg.checkpoint_dir, cfg.ppo.total_iterations,
               policy_params, value_params, disc_params,
               normalizer_params=normalizer_params)
    print(f"[amp-v2] done in {(time.time() - start) / 60:.1f} min")


def _save_ckpt(out_dir: Path, it: int, policy_params, value_params, disc_params,
               normalizer_params=None):
    import pickle
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"amp_v2_it{it:06d}.pkl"
    with p.open("wb") as f:
        pickle.dump(
            {"policy": policy_params, "value": value_params,
             "disc": disc_params, "normalizer": normalizer_params,
             "iteration": it}, f,
        )
    # Also write/overwrite a 'latest' alias.
    latest = out_dir / "amp_v2_latest.pkl"
    latest.write_bytes(p.read_bytes())
    print(f"[amp-v2] checkpoint -> {p}  (latest -> {latest})")
