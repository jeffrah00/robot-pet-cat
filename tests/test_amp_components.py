"""Phase 2b unit tests for the AMP components that don't require the heavy
[motion] extras (brax, mujoco_playground) -- just JAX/Flax.

The end-to-end training-loop test lives in tests/test_amp_trainer.py and is
marked as requiring those extras.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

# Skip the entire module if JAX isn't available.
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from robot_pet_cat.motion.amp_discriminator import (
    AMPDiscConfig,
    grad_penalty,
    init_discriminator,
    lsgan_disc_loss,
    make_transition_features,
    style_reward,
)
from robot_pet_cat.motion.reference_buffer import ReferenceBuffer


# -- Discriminator -----------------------------------------------------------


def test_discriminator_init_and_forward():
    cfg = AMPDiscConfig(obs_dim=19, hidden_sizes=(64, 32))
    model, params, apply_fn = init_discriminator(cfg, seed=0)
    transition = jnp.zeros((8, 2 * 19), dtype=jnp.float32)
    logits = apply_fn(params, transition)
    assert logits.shape == (8,)


def test_lsgan_loss_signs():
    """LS-GAN loss: lower for well-separated logits, higher otherwise."""
    real = jnp.array([1.0, 1.0, 1.0])      # at target
    fake = jnp.array([-1.0, -1.0, -1.0])   # at target
    perfect = lsgan_disc_loss(real, fake)
    confused = lsgan_disc_loss(jnp.zeros(3), jnp.zeros(3))
    assert float(perfect) < float(confused)


def test_style_reward_bounded():
    cfg = AMPDiscConfig(obs_dim=19, hidden_sizes=(32, 16))
    _, params, apply_fn = init_discriminator(cfg, seed=1)
    # Crank up D outputs by feeding outliers; reward should still clip.
    transitions = jnp.array(np.random.randn(32, 2 * 19) * 5.0, dtype=jnp.float32)
    r = style_reward(apply_fn, params, transitions, cfg)
    assert r.shape == (32,)
    assert float(jnp.min(r)) >= cfg.style_reward_min - 1e-6
    assert float(jnp.max(r)) <= cfg.style_reward_clip + 1e-6


def test_grad_penalty_finite():
    cfg = AMPDiscConfig(obs_dim=19, hidden_sizes=(32, 16))
    _, params, apply_fn = init_discriminator(cfg, seed=2)
    real = jnp.array(np.random.randn(16, 2 * 19), dtype=jnp.float32)
    gp = grad_penalty(apply_fn, params, real)
    assert jnp.isfinite(gp)


def test_make_transition_features_shape():
    a = jnp.zeros((4, 19))
    b = jnp.ones((4, 19))
    f = make_transition_features(a, b)
    assert f.shape == (4, 38)
    assert float(jnp.sum(f[:, :19])) == 0.0
    assert float(jnp.sum(f[:, 19:])) == 4 * 19


# -- Reference buffer --------------------------------------------------------


def _write_dummy_clip(path: Path, T: int = 20, qpos_dim: int = 19) -> None:
    rng = np.random.default_rng(0)
    qpos = rng.standard_normal((T, qpos_dim)).astype(np.float32)
    qvel = rng.standard_normal((T, qpos_dim - 1)).astype(np.float32)
    np.savez_compressed(
        path,
        qpos=qpos,
        qvel=qvel,
        dt=np.array([1.0 / 30.0]),
        fps=np.array([30.0]),
    )


def test_reference_buffer_loads_and_samples(tmp_path):
    _write_dummy_clip(tmp_path / "clip1.npz", T=30)
    _write_dummy_clip(tmp_path / "clip2.npz", T=15)
    buf = ReferenceBuffer.from_dir(tmp_path)
    assert buf.n_transitions == (30 - 1) + (15 - 1)
    assert buf.qpos_dim == 19

    rng = jax.random.PRNGKey(0)
    batch = buf.sample(rng, batch_size=64, use_qvel=False)
    assert batch.shape == (64, 2 * 19)


def test_reference_buffer_qvel_mode(tmp_path):
    _write_dummy_clip(tmp_path / "clip.npz", T=25)
    buf = ReferenceBuffer.from_dir(tmp_path)
    rng = jax.random.PRNGKey(1)
    batch = buf.sample(rng, batch_size=32, use_qvel=True)
    assert batch.shape == (32, 2 * (19 + 18))


def test_reference_buffer_empty_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        ReferenceBuffer.from_dir(tmp_path)


# -- Integration: discriminator can train on buffer samples -----------------


def test_discriminator_update_step(tmp_path):
    """Wire the buffer + discriminator + a single optax step end-to-end."""
    optax = pytest.importorskip("optax")

    _write_dummy_clip(tmp_path / "clip.npz", T=40)
    buf = ReferenceBuffer.from_dir(tmp_path)

    cfg = AMPDiscConfig(obs_dim=buf.qpos_dim, hidden_sizes=(32, 16))
    _, params, apply_fn = init_discriminator(cfg, seed=3)
    optimizer = optax.adam(1e-4)
    opt_state = optimizer.init(params)

    rng = jax.random.PRNGKey(4)
    real = buf.sample(rng, 64, use_qvel=False)
    fake = jax.random.normal(rng, real.shape) * 0.3

    def loss_fn(p):
        real_l = apply_fn(p, real)
        fake_l = apply_fn(p, fake)
        return lsgan_disc_loss(real_l, fake_l)

    loss_before = float(loss_fn(params))
    grads = jax.grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    loss_after = float(loss_fn(new_params))

    # Loss should at least not blow up; usually decreases on the first step.
    assert np.isfinite(loss_before)
    assert np.isfinite(loss_after)
