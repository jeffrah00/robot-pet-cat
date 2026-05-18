"""Phase 1 kinematic imitation trainer for Go2 cat-style locomotion.

Replaces the AMP discriminator with a direct per-step pose-matching reward:

    r_imit = exp(-||q_joints_policy - q_joints_ref||^2 / sigma_sq)

where q_joints are the 12 Go2 joint angles (qpos[7:], skipping the 7-DOF
floating base).  Combined with Reference State Initialization (RSI) -- seeding
a fraction of envs from random reference frames at each rollout start -- this
gives a dense, well-shaped signal that converges where AMP stalled.

Why AMP stalled (train_v3/v4):
  - r_task never exceeded +0.016 after 500 warmup iters (policy stuck at
    near-zero velocity local minimum)
  - AMP style reward (scale ~0.16) dominated task reward (scale ~0.016) 5:1,
    so the policy rationally abandoned velocity tracking to chase style reward
  - Without RSI, the policy only ever sees the default standing pose at reset,
    so it can never experience what mid-gait looks like from the inside

What changes here:
  - No discriminator, no LSGAN, no adversarial instability
  - Imitation reward is dense: random policy gets ~0.01-0.05, good tracking
    gets ~0.7-0.9 -- the reward gradient exists everywhere
  - RSI seeds 80% of envs from reference frames before each rollout, so the
    policy immediately trains on mid-walk / mid-trot states
  - PPO loop, GAE, normalizer update are identical to amp_trainer_v2.py
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax


@dataclass
class PPOConfig:
    total_iterations: int = 3000
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
class ImitationTrainConfig:
    '''Config for Phase 1 kinematic imitation training.'''
    motion_clips_dir: Path = Path("data/motion_clips_loco")

    # Reward weights.  Imitation reward is in [0, 1]; task reward starts
    # ~0.016 and grows as the policy learns to walk.  Equal weights are fine
    # because the imitation gradient guides the policy toward walking, which
    # naturally raises task reward over time.
    task_reward_weight: float = 1.0
    imitation_reward_weight: float = 1.0

    # Sharpness of the pose-matching Gaussian.
    # sigma_sq=0.5 gives r_imit~0.79 at 0.1 rad avg joint error (12 joints),
    # r_imit~0.38 at 0.2 rad, r_imit~0.003 at 0.5 rad.
    imitation_sigma_sq: float = 0.5

    # Fraction of envs to seed from a random reference frame at each rollout
    # start (Reference State Initialization).  0.8 is aggressive but
    # necessary for cold-start training with no prior locomotion policy.
    rsi_prob: float = 0.8

    env_name: str = "Go2JoystickFlatTerrain"
    num_envs: int = 4096
    episode_length_s: float = 20.0

    ppo: PPOConfig = field(default_factory=PPOConfig)

    seed: int = 0
    log_interval_iters: int = 10
    eval_interval_iters: int = 100
    checkpoint_interval_iters: int = 500
    checkpoint_dir: Path = Path("checkpoints/motion_imitation_v1")


def _load_clip_arrays(motion_clips_dir: Path):
    '''Load all reference clips as contiguous numpy arrays for fast indexing.

    Returns
    -------
    all_qpos : (total_frames, nq) float32
    all_qvel : (total_frames, nv) float32  or  None if not present in .npz
    clip_starts  : (n_clips,) int32 -- first frame index of each clip in all_qpos
    clip_lengths : (n_clips,) int32 -- number of frames per clip
    clip_names   : list[str]
    nq           : int
    '''
    paths = sorted(Path(motion_clips_dir).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(
            f"No .npz reference clips found in {motion_clips_dir}. "
            f"Run `rpc retarget` first."
        )

    qpos_list, qvel_list, clip_names = [], [], []
    any_qvel = True

    for p in paths:
        data = np.load(p)
        qpos = np.asarray(data["qpos"], dtype=np.float32)
        if len(qpos) < 2:
            continue
        qpos_list.append(qpos)
        clip_names.append(p.stem)
        if "qvel" in data.files and any_qvel:
            qvel_list.append(np.asarray(data["qvel"], dtype=np.float32))
        else:
            any_qvel = False
            qvel_list.clear()

    if not qpos_list:
        raise ValueError("All clips had fewer than 2 frames.")

    all_qpos = np.concatenate(qpos_list, axis=0)
    all_qvel = np.concatenate(qvel_list, axis=0) if any_qvel and qvel_list else None
    clip_lengths = np.array([len(a) for a in qpos_list], dtype=np.int32)
    clip_starts = np.zeros(len(clip_lengths), dtype=np.int32)
    clip_starts[1:] = np.cumsum(clip_lengths[:-1])

    return all_qpos, all_qvel, clip_starts, clip_lengths, clip_names, all_qpos.shape[1]


def train(cfg: ImitationTrainConfig) -> None:
    '''Top-level imitation training entry.  Heavy deps are lazy-imported.'''
    from brax.training.acme import running_statistics, specs
    from brax.training.agents.ppo import networks as ppo_networks
    from mujoco_playground import wrapper as mjx_wrapper

    from robot_pet_cat.motion.amp_env import AMPEnvConfig, make_env
    from robot_pet_cat.motion.amp_trainer import _shim_jax_for_brax

    print("[imit] starting Phase 1 kinematic imitation training")
    print(f"[imit] env={cfg.env_name}  num_envs={cfg.num_envs}  "
          f"total_iters={cfg.ppo.total_iterations:,}  "
          f"rsi_prob={cfg.rsi_prob}  sigma_sq={cfg.imitation_sigma_sq}")

    # -------------------------------------------------------------------------
    # 1. Reference clip arrays (Python / numpy -- NOT JAX, for fast indexing)
    # -------------------------------------------------------------------------
    all_qpos, all_qvel, clip_starts, clip_lengths, clip_names, nq = \
        _load_clip_arrays(cfg.motion_clips_dir)
    n_clips = len(clip_names)
    print(f"[imit] reference clips: {n_clips} clips, "
          f"{all_qpos.shape[0]} frames, nq={nq}, "
          f"{'qvel present' if all_qvel is not None else 'no qvel in clips (RSI will zero qvel)'}")
    for name, length in zip(clip_names, clip_lengths):
        print(f"[imit]   {name}: {length} frames")

    # -------------------------------------------------------------------------
    # 2. Env
    # -------------------------------------------------------------------------
    env_cfg = AMPEnvConfig(
        base_env_name=cfg.env_name,
        num_envs=cfg.num_envs,
        episode_length_s=cfg.episode_length_s,
        seed=cfg.seed,
    )
    raw_env = make_env(env_cfg)
    env = mjx_wrapper.wrap_for_brax_training(
        raw_env,
        episode_length=int(cfg.episode_length_s * 50),
        action_repeat=1,
    )
    print(f"[imit] env loaded + wrapped: {type(env).__name__}")

    # -------------------------------------------------------------------------
    # 3. Networks + optimizer
    # -------------------------------------------------------------------------
    _shim_jax_for_brax()
    ppo_nets = ppo_networks.make_ppo_networks(
        observation_size=env.observation_size,
        action_size=env.action_size,
        policy_hidden_layer_sizes=tuple(cfg.ppo.policy_hidden_sizes),
        value_hidden_layer_sizes=tuple(cfg.ppo.value_hidden_sizes),
    )

    rng = jax.random.PRNGKey(cfg.seed)
    rng, k_policy, k_value = jax.random.split(rng, 3)
    policy_params = ppo_nets.policy_network.init(k_policy)
    value_params = ppo_nets.value_network.init(k_value)

    def _to_spec(size):
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

    # -------------------------------------------------------------------------
    # 4. Python-level reference frame tracking (numpy, per-env state)
    # -------------------------------------------------------------------------
    rng_np = np.random.default_rng(cfg.seed)
    ref_clip_idx = rng_np.integers(0, n_clips, size=cfg.num_envs).astype(np.int32)
    ref_frame_idx = np.array(
        [rng_np.integers(0, clip_lengths[c]) for c in ref_clip_idx], dtype=np.int32
    )

    def _sample_ref_seq():
        '''Build (num_envs, unroll_length, nq) JAX array of reference poses.'''
        T = cfg.ppo.unroll_length
        lengths_e = clip_lengths[ref_clip_idx]       # (num_envs,)
        starts_e = clip_starts[ref_clip_idx]         # (num_envs,)
        t_off = np.arange(T, dtype=np.int32)         # (T,)
        # frame_indices[e, t] = (ref_frame_idx[e] + t) % lengths_e[e]
        frame_indices = (ref_frame_idx[:, None] + t_off[None, :]) % lengths_e[:, None]
        global_indices = starts_e[:, None] + frame_indices   # (num_envs, T)
        seq = all_qpos[global_indices]               # (num_envs, T, nq)
        return jnp.asarray(seq, dtype=jnp.float32)

    def _advance_ref_frames():
        '''Advance per-env frame pointers by unroll_length (with clip wrap).'''
        T = cfg.ppo.unroll_length
        ref_frame_idx[:] = (ref_frame_idx + T) % clip_lengths[ref_clip_idx]

    # -------------------------------------------------------------------------
    # 5. Reference State Initialization (RSI)
    # -------------------------------------------------------------------------
    @jax.jit
    def _apply_rsi_jit(env_state, ref_qpos_j, ref_qvel_j, rsi_mask_j):
        '''Replace qpos/qvel for envs where rsi_mask=True.
        Works on mjx.Data and brax State objects that expose .replace().
        '''
        _ps_attr = "pipeline_state" if hasattr(env_state, "pipeline_state") else "data"
        ps = getattr(env_state, _ps_attr)
        nq_env = ps.qpos.shape[-1]
        nv_env = ps.qvel.shape[-1]
        # Clip reference arrays to env dims in case of shape mismatch.
        rq = ref_qpos_j[:, :nq_env]
        rv = ref_qvel_j[:, :nv_env]
        new_qpos = jnp.where(rsi_mask_j[:, None], rq, ps.qpos)
        new_qvel = jnp.where(rsi_mask_j[:, None], rv, ps.qvel)
        new_ps = ps.replace(qpos=new_qpos, qvel=new_qvel)
        return env_state.replace(**{_ps_attr: new_ps})

    def _do_rsi(env_state):
        '''Python-level RSI: reseed a random fraction of envs from reference frames.'''
        rsi_mask_np = rng_np.uniform(size=cfg.num_envs) < cfg.rsi_prob
        n_rsi = int(rsi_mask_np.sum())
        if n_rsi > 0:
            new_clips = rng_np.integers(0, n_clips, size=n_rsi).astype(np.int32)
            new_frames = np.array(
                [rng_np.integers(0, clip_lengths[c]) for c in new_clips], dtype=np.int32
            )
            ref_clip_idx[rsi_mask_np] = new_clips
            ref_frame_idx[rsi_mask_np] = new_frames

        # Build ref arrays for ALL envs; JAX where-mask handles RSI vs keep.
        gf = clip_starts[ref_clip_idx] + ref_frame_idx  # (num_envs,)
        ref_qpos_np = all_qpos[gf]                       # (num_envs, nq)

        _ps3 = getattr(env_state, "pipeline_state", None) or env_state.data
        nv_env = _ps3.qvel.shape[-1]
        if all_qvel is not None:
            ref_qvel_np = all_qvel[gf][:, :nv_env]
        else:
            ref_qvel_np = np.zeros((cfg.num_envs, nv_env), dtype=np.float32)

        rsi_mask_j = jnp.asarray(rsi_mask_np)
        ref_qpos_j = jnp.asarray(ref_qpos_np, dtype=jnp.float32)
        ref_qvel_j = jnp.asarray(ref_qvel_np, dtype=jnp.float32)
        return _apply_rsi_jit(env_state, ref_qpos_j, ref_qvel_j, rsi_mask_j)

    # -------------------------------------------------------------------------
    # 6. JIT building blocks
    # -------------------------------------------------------------------------
    def _policy_act(norm_params, params, obs, rng):
        logits = ppo_nets.policy_network.apply(norm_params, params, obs)
        dist = ppo_nets.parametric_action_distribution
        raw_act = dist.sample_no_postprocessing(logits, rng)
        log_prob = dist.log_prob(logits, raw_act)
        act = dist.postprocess(raw_act)
        return act, log_prob, raw_act

    def _value_predict(norm_params, val_params, obs):
        v = ppo_nets.value_network.apply(norm_params, val_params, obs)
        return jnp.squeeze(v, axis=-1) if v.ndim > 1 else v

    def _get_qpos(env_state):
        s = getattr(env_state, "pipeline_state", None)
        if s is None:
            s = getattr(env_state, "data", None)
        if s is None:
            raise AttributeError(
                "env_state has neither .pipeline_state nor .data; "
                f"available: {dir(env_state)}"
            )
        return s.qpos

    # Capture reward scalars and sigma as Python constants so they're
    # baked into the JIT trace (avoids retracing when cfg changes).
    _task_w = cfg.task_reward_weight
    _imit_w = cfg.imitation_reward_weight
    _sigma_sq = cfg.imitation_sigma_sq

    @functools.partial(jax.jit, static_argnames=("unroll_length",))
    def rollout(env_state, ref_qpos_seq, norm_params, policy_params, value_params,
                rng, unroll_length: int):
        '''Roll out unroll_length steps with per-step imitation reward.

        ref_qpos_seq : (num_envs, unroll_length, nq) -- transposed to
                       (unroll_length, num_envs, nq) as scan input.
        '''
        ref_T = jnp.transpose(ref_qpos_seq, (1, 0, 2))   # (T, num_envs, nq)

        def step_fn(carry, ref_qpos_t):
            env_state, rng = carry
            rng, k_act = jax.random.split(rng)
            act, log_prob, raw_act = _policy_act(norm_params, policy_params,
                                                 env_state.obs, k_act)
            v = _value_predict(norm_params, value_params, env_state.obs)

            q_policy = _get_qpos(env_state)              # (num_envs, nq)
            next_state = env.step(env_state, act)

            # Imitation reward: 12 joint angles only (skip 7-DOF floating base)
            q_joints = q_policy[:, 7:]                   # (num_envs, 12)
            q_ref_j  = ref_qpos_t[:, 7:]                # (num_envs, 12)
            r_imit = jnp.exp(
                -jnp.sum((q_joints - q_ref_j) ** 2, axis=-1) / _sigma_sq
            )                                            # (num_envs,)
            r_task = next_state.reward                   # (num_envs,)
            total_r = _task_w * r_task + _imit_w * r_imit

            return (next_state, rng), {
                "obs":            env_state.obs,
                "act":            act,
                "log_prob":       log_prob,
                "raw_act":        raw_act,
                "value":          v,
                "reward":         total_r,
                "task_reward":    r_task,
                "imit_reward":    r_imit,
                "done":           next_state.done,
            }

        (final_state, final_rng), traj = jax.lax.scan(
            step_fn, (env_state, rng), ref_T, length=unroll_length,
        )
        last_v = _value_predict(norm_params, value_params, final_state.obs)
        return final_state, final_rng, traj, last_v

    @jax.jit
    def compute_gae(rewards, values, dones, last_v):
        '''GAE-lambda.  All inputs shape (T, B).'''
        gamma = cfg.ppo.discounting
        lam   = cfg.ppo.gae_lambda

        def body(carry, t):
            adv = carry
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * mask * values_p1[t] - values[t]
            adv = delta + gamma * lam * mask * adv
            return adv, adv

        values_p1 = jnp.concatenate([values[1:], last_v[None]], axis=0)
        _, advs = jax.lax.scan(
            body, jnp.zeros_like(last_v), jnp.arange(rewards.shape[0]), reverse=True,
        )
        return advs, advs + values

    @jax.jit
    def ppo_update(norm_params, policy_params, value_params, opt_state,
                   obs, act, log_prob_old, raw_act, advs, returns):
        def loss_fn(params):
            p_params, v_params = params
            logits = ppo_nets.policy_network.apply(norm_params, p_params, obs)
            dist = ppo_nets.parametric_action_distribution
            log_prob = dist.log_prob(logits, raw_act)
            entropy  = dist.entropy(logits, jax.random.PRNGKey(0)).mean()
            ratio    = jnp.exp(log_prob - log_prob_old)
            adv_norm = (advs - advs.mean()) / (advs.std() + 1e-8)
            unclipped = ratio * adv_norm
            clipped = jnp.clip(ratio,
                                1 - cfg.ppo.clipping_epsilon,
                                1 + cfg.ppo.clipping_epsilon) * adv_norm
            policy_loss = -jnp.minimum(unclipped, clipped).mean()
            v_pred = ppo_nets.value_network.apply(norm_params, v_params, obs)
            if v_pred.ndim > 1:
                v_pred = jnp.squeeze(v_pred, axis=-1)
            value_loss  = 0.5 * ((v_pred - returns) ** 2).mean()
            total = policy_loss + value_loss - cfg.ppo.entropy_cost * entropy
            return total, {"policy_loss": policy_loss, "value_loss": value_loss,
                           "entropy": entropy}

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            (policy_params, value_params)
        )
        updates, new_opt = optimizer.update(grads, opt_state,
                                            (policy_params, value_params))
        new_params = optax.apply_updates((policy_params, value_params), updates)
        return new_params[0], new_params[1], new_opt, loss, aux

    # -------------------------------------------------------------------------
    # 7. Main loop
    # -------------------------------------------------------------------------
    print(f"[imit] starting PPO ({cfg.ppo.total_iterations} iterations)")
    rng, k_reset = jax.random.split(rng)
    env_state = jax.jit(env.reset)(jax.random.split(k_reset, cfg.num_envs))

    # Initial RSI: seed all envs from reference frames before iteration 0.
    print("[imit] applying initial RSI ...")
    env_state = _do_rsi(env_state)
    print("[imit] initial RSI done -- beginning training")

    start = time.time()
    total_env_steps = 0

    for it in range(cfg.ppo.total_iterations):
        # Reference sequence for this rollout (Python -> JAX).
        ref_qpos_seq = _sample_ref_seq()   # (num_envs, T, nq)

        rng, k_roll = jax.random.split(rng)
        env_state, _, traj, last_v = rollout(
            env_state, ref_qpos_seq, normalizer_params,
            policy_params, value_params, k_roll,
            unroll_length=cfg.ppo.unroll_length,
        )

        # Advance reference frame pointers.
        _advance_ref_frames()

        # GAE on augmented reward.
        advs, returns = compute_gae(
            traj["reward"], traj["value"], traj["done"], last_v,
        )

        # Flatten (T, B, ...) -> (T*B, ...) for minibatch SGD.
        def flatten(x):
            return x.reshape((-1,) + x.shape[2:])
        obs_flat      = jax.tree_util.tree_map(flatten, traj["obs"])
        act_flat      = flatten(traj["act"])
        raw_act_flat  = flatten(traj["raw_act"])
        log_prob_flat = flatten(traj["log_prob"])
        advs_flat     = flatten(advs)
        returns_flat  = flatten(returns)

        n = jax.tree_util.tree_leaves(obs_flat)[0].shape[0]
        mb_size = n // cfg.ppo.num_minibatches

        for _epoch in range(cfg.ppo.num_updates_per_batch):
            rng, k_perm = jax.random.split(rng)
            perm = jax.random.permutation(k_perm, n)
            for mb in range(cfg.ppo.num_minibatches):
                idx = perm[mb * mb_size:(mb + 1) * mb_size]
                obs_mb = jax.tree_util.tree_map(lambda x: x[idx], obs_flat)
                policy_params, value_params, opt_state, ppo_loss, ppo_aux = ppo_update(
                    normalizer_params, policy_params, value_params, opt_state,
                    obs_mb, act_flat[idx], log_prob_flat[idx],
                    raw_act_flat[idx], advs_flat[idx], returns_flat[idx],
                )

        # Update running obs normalizer.
        normalizer_params = running_statistics.update(normalizer_params, obs_flat)

        # RSI for next rollout: reseed rsi_prob fraction of envs.
        env_state = _do_rsi(env_state)

        total_env_steps += cfg.num_envs * cfg.ppo.unroll_length
        if (it + 1) % cfg.log_interval_iters == 0:
            wall = time.time() - start
            sps  = total_env_steps / max(wall, 1e-6)
            r_task = float(jnp.mean(traj["task_reward"]))
            r_imit = float(jnp.mean(traj["imit_reward"]))
            print(
                f"[imit] it {it+1:>5d}  step {total_env_steps:>11,d}  "
                f"r_task={r_task:+.3f} r_imit={r_imit:+.3f}  "
                f"ppo_loss={float(ppo_loss):+.3f}  "
                f"{sps:>7,.0f} steps/s  wall={wall/60:.1f}m"
            )

        if (it + 1) % cfg.checkpoint_interval_iters == 0:
            _save_ckpt(cfg.checkpoint_dir, it + 1,
                       policy_params, value_params, normalizer_params)

    _save_ckpt(cfg.checkpoint_dir, cfg.ppo.total_iterations,
               policy_params, value_params, normalizer_params)
    print(f"[imit] done in {(time.time() - start) / 60:.1f} min")


def _save_ckpt(out_dir: Path, it: int, policy_params, value_params,
               normalizer_params=None):
    import pickle
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"imit_v1_it{it:06d}.pkl"
    with p.open("wb") as f:
        pickle.dump(
            {"policy": policy_params, "value": value_params,
             "normalizer": normalizer_params, "iteration": it}, f,
        )
    latest = out_dir / "imit_v1_latest.pkl"
    latest.write_bytes(p.read_bytes())
    print(f"[imit] checkpoint -> {p}  (latest -> {latest})")
