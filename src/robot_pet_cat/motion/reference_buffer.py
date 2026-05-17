"""Reference motion clip buffer for AMP training.

Loads retargeted .npz clips from `data/motion_clips/`, extracts (s, s')
transitions, and serves random mini-batches to the discriminator.

The buffer is fully in-memory because reference data for AMP is tiny
(handful of clips, each a few hundred frames at most). For our project even
30 clips at 30 fps for 10 sec = 9000 transitions total -- nothing.

Caller flow during training:
    >>> buf = ReferenceBuffer.from_dir(Path("data/motion_clips"))
    >>> rng, batch_rng = jax.random.split(rng)
    >>> real = buf.sample(batch_rng, batch_size=128, use_qvel=False)
    >>> # real has shape (128, 2 * obs_dim); pass to discriminator
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class ClipMeta:
    name: str
    n_frames: int
    fps: float
    dt: float


class ReferenceBuffer:
    """In-memory buffer of (qpos[t], qpos[t+1]) and optionally (qvel) pairs.

    Stores transitions as a flat numpy array of shape (N_transitions, 2*feat).
    Indexing is uniform across all loaded clips weighted by length.
    """

    def __init__(
        self,
        qpos_transitions: np.ndarray,
        qvel_transitions: np.ndarray | None,
        clips: list[ClipMeta],
    ) -> None:
        if qpos_transitions.ndim != 3 or qpos_transitions.shape[1] != 2:
            raise ValueError(
                f"qpos_transitions should be (N, 2, qpos_dim); got "
                f"{qpos_transitions.shape}"
            )
        self.qpos_transitions = qpos_transitions
        self.qvel_transitions = qvel_transitions
        self.clips = clips
        self._n = qpos_transitions.shape[0]
        self._qpos_dim = qpos_transitions.shape[2]
        self._qvel_dim = (
            qvel_transitions.shape[2] if qvel_transitions is not None else 0
        )

    @property
    def n_transitions(self) -> int:
        return self._n

    @property
    def qpos_dim(self) -> int:
        return self._qpos_dim

    @property
    def qvel_dim(self) -> int:
        return self._qvel_dim

    def feature_dim(self, use_qvel: bool) -> int:
        """Single-state feature dim (so (s, s') concatenation is 2 * this)."""
        return self._qpos_dim + (self._qvel_dim if use_qvel else 0)

    # ---- Construction --------------------------------------------------

    @classmethod
    def from_dir(cls, motion_clips_dir: Path) -> "ReferenceBuffer":
        """Load every *.npz under motion_clips_dir."""
        motion_clips_dir = Path(motion_clips_dir)
        paths = sorted(motion_clips_dir.glob("*.npz"))
        if not paths:
            raise FileNotFoundError(
                f"No .npz reference clips found in {motion_clips_dir}. "
                f"Run `rpc retarget` first."
            )
        return cls.from_paths(paths)

    @classmethod
    def from_paths(cls, paths: list[Path]) -> "ReferenceBuffer":
        qpos_list: list[np.ndarray] = []
        qvel_list: list[np.ndarray] = []
        any_qvel = False
        clips: list[ClipMeta] = []

        for p in paths:
            data = np.load(p)
            qpos = np.asarray(data["qpos"], dtype=np.float32)  # (T, qpos_dim)
            if qpos.shape[0] < 2:
                continue  # need at least one transition
            qpos_pairs = np.stack([qpos[:-1], qpos[1:]], axis=1)  # (T-1, 2, dim)
            qpos_list.append(qpos_pairs)

            if "qvel" in data.files:
                qvel = np.asarray(data["qvel"], dtype=np.float32)
                qvel_pairs = np.stack([qvel[:-1], qvel[1:]], axis=1)
                qvel_list.append(qvel_pairs)
                any_qvel = True

            fps = float(data["fps"][0]) if "fps" in data.files else 30.0
            clips.append(
                ClipMeta(
                    name=p.stem,
                    n_frames=int(qpos.shape[0]),
                    fps=fps,
                    dt=1.0 / fps,
                )
            )

        if not qpos_list:
            raise ValueError(f"All clips in {paths!r} had fewer than 2 frames.")

        qpos_transitions = np.concatenate(qpos_list, axis=0)
        qvel_transitions = (
            np.concatenate(qvel_list, axis=0) if any_qvel and qvel_list else None
        )
        return cls(qpos_transitions, qvel_transitions, clips)

    # ---- Sampling ------------------------------------------------------

    def sample(
        self,
        rng: jax.Array,
        batch_size: int,
        use_qvel: bool = False,
    ) -> jnp.ndarray:
        """Sample `batch_size` transitions as a flat (B, 2 * feat_dim) array.

        Returns a JAX array (already on device). Useful for direct use in the
        discriminator. Caller passes their own rng.
        """
        if use_qvel and self.qvel_transitions is None:
            raise ValueError(
                "use_qvel=True but loaded clips have no 'qvel' field"
            )
        idx = jax.random.randint(rng, (batch_size,), 0, self._n)
        # Move the underlying numpy arrays through jnp once; this gets cached
        # by JAX so repeated sampling is cheap.
        qpos_pairs = jnp.asarray(self.qpos_transitions)[idx]  # (B, 2, qpos_dim)
        feats = qpos_pairs.reshape(batch_size, -1)            # (B, 2 * qpos_dim)
        if use_qvel:
            qvel_pairs = jnp.asarray(self.qvel_transitions)[idx]
            qvel_feats = qvel_pairs.reshape(batch_size, -1)
            # Layout per state: [qpos, qvel]; transition: [s_t (qpos+qvel),
            # s_tp1 (qpos+qvel)]. Reshape to interleave properly.
            qpos_pairs = qpos_pairs.reshape(batch_size, 2, -1)
            qvel_pairs = qvel_pairs.reshape(batch_size, 2, -1)
            states = jnp.concatenate([qpos_pairs, qvel_pairs], axis=-1)
            feats = states.reshape(batch_size, -1)
        return feats

    # ---- Diagnostics ---------------------------------------------------

    def summary(self) -> str:
        lines = [
            f"ReferenceBuffer: {self.n_transitions} transitions across "
            f"{len(self.clips)} clip(s)"
        ]
        for c in self.clips:
            lines.append(f"  {c.name}: {c.n_frames} frames @ {c.fps:.1f} fps")
        lines.append(
            f"  qpos_dim={self.qpos_dim}, "
            f"qvel_dim={self.qvel_dim or 'n/a'}"
        )
        return "\n".join(lines)
