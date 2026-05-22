"""Loader for the unitree_rl_mjlab Go2 velocity walker policy + hind_sit policy.

The walker on the pod is trained by unitree_rl_mjlab (RSL-RL backend) against
the Go2 velocity task. Each training save writes both:

  - logs/.../model_<iter>.pt   -- RSL-RL OnPolicyRunner checkpoint
  - logs/.../policy.onnx       -- exported ONNX (obs-normalizer baked in)

ONNX is the preferred path for external integration. The .pt path is the
fallback when only the runner checkpoint is on disk. It requires rsl_rl
to be installed and rebuilds the actor MLP + normalizer manually before
loading weights.

Walker policy contract (47 dims, flat env):
    base_ang_vel(3) + projected_gravity(3) + command(3) + phase(2)
        + joint_pos(12) + joint_vel(12) + last_action(12)

Hind_sit policy contract (42 dims, from the get_up task):
    base_ang_vel(3) + projected_gravity(3) + joint_pos(12) + joint_vel(12)
        + last_action(12)
The hind_sit policy was the get_up v4b checkpoint at model_13000.pt and is
packaged at models/hind_sit_v1.pt. The actor MLP shape is the same as the
walker (512, 256, 128) -- only the input dim differs.

Joint order (MJCF):
    FL_hip, FL_thigh, FL_calf,
    FR_hip, FR_thigh, FR_calf,
    RL_hip, RL_thigh, RL_calf,
    RR_hip, RR_thigh, RR_calf
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

import numpy as np


# Default joint positions in MJCF compile order (FL, FR, RL, RR).
GO2_DEFAULT_JOINT_POS = np.array(
    [
        -0.10, 0.90, -1.80,   # FL
        +0.10, 0.90, -1.80,   # FR
        -0.10, 0.90, -1.80,   # RL
        +0.10, 0.90, -1.80,   # RR
    ],
    dtype=np.float32,
)

GO2_JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]

GO2_ACTION_SCALE = 0.25
GO2_PHYSICS_DT = 0.005
GO2_POLICY_DECIMATION = 4
GO2_POLICY_DT = GO2_PHYSICS_DT * GO2_POLICY_DECIMATION  # 0.02 s, 50 Hz
GO2_PHASE_PERIOD = 0.6
GO2_PHASE_STAND_THRESHOLD = 0.1


class WalkerPolicy(Protocol):
    """Callable that maps obs[N,*] -> action[N,12]. Obs dim depends on which
    policy: 47 for the walker, 42 for hind_sit."""

    def __call__(self, obs: np.ndarray) -> np.ndarray: ...


# --------------------------------------------------------------------------- #
# ONNX path -- preferred when available
# --------------------------------------------------------------------------- #


class _OnnxWalkerPolicy:
    """Wraps onnxruntime InferenceSession around any Go2 policy export."""

    def __init__(self, onnx_path: Path, providers: Optional[list[str]] = None):
        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_options,
            providers=providers or ["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        self._input_shape = tuple(self._session.get_inputs()[0].shape)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[None, :]
        out = self._session.run(
            [self._output_name],
            {self._input_name: obs.astype(np.float32, copy=False)},
        )[0]
        return np.asarray(out, dtype=np.float32)


# --------------------------------------------------------------------------- #
# .pt path -- requires rsl_rl style checkpoints
# --------------------------------------------------------------------------- #


@dataclass
class TorchWalkerConfig:
    """Generic config for a torch-loaded rsl_rl actor. Use obs_dim=47 for
    the walker, obs_dim=42 for hind_sit. Same hidden dims either way."""
    obs_dim: int = 47
    action_dim: int = 12
    hidden_dims: tuple = (512, 256, 128)
    activation: str = "elu"
    device: str = "cpu"


class _TorchWalkerPolicy:
    """Rebuilds the RSL-RL actor + EmpiricalNormalization and loads weights.

    Keys in model_<iter>.pt look like:
        actor.0.weight, actor.0.bias, actor.2.weight, actor.2.bias, ...
        obs_normalizer.running_mean, obs_normalizer.running_var,
            obs_normalizer.count
    """

    def __init__(self, ckpt_path: Path, cfg: Optional[TorchWalkerConfig] = None):
        import torch
        import torch.nn as nn

        self._torch = torch
        self.cfg = cfg or TorchWalkerConfig()
        self.device = torch.device(self.cfg.device)

        act = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}[self.cfg.activation]
        dims = [self.cfg.obs_dim, *self.cfg.hidden_dims, self.cfg.action_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act())
        self.actor = nn.Sequential(*layers).to(self.device).eval()

        self.normalizer_mean = torch.zeros(self.cfg.obs_dim, device=self.device)
        self.normalizer_var = torch.ones(self.cfg.obs_dim, device=self.device)
        self._has_normalizer = False

        self._load_checkpoint(ckpt_path)

    def _load_checkpoint(self, ckpt_path: Path) -> None:
        """Load weights + obs normalizer from an rsl_rl-style checkpoint.

        Supports two checkpoint layouts in the wild:
          a) Old / vanilla rsl_rl:
             {"model_state_dict": {"actor.0.weight": ..., "actor.0.bias": ...,
              "obs_normalizer.running_mean": ..., "obs_normalizer.running_var": ...}}
          b) mjlab (unitree_rl_mjlab) save format:
             {"actor_state_dict": {"mlp.0.weight": ..., "mlp.0.bias": ...,
              "obs_normalizer._mean": ..., "obs_normalizer._var": ...,
              "distribution.std_param": ..., ...}, "critic_state_dict": ...}
        """
        torch = self._torch
        blob = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)

        # Prefer the mjlab layout if present.
        if isinstance(blob, dict) and "actor_state_dict" in blob:
            asd = blob["actor_state_dict"]
            actor_weights: dict = {}
            for k, v in asd.items():
                if k.startswith("mlp."):
                    # Strip "mlp." -> matches our nn.Sequential layer indices.
                    actor_weights[k[len("mlp."):]] = v
                elif k.startswith("obs_normalizer."):
                    tail = k.split(".", 1)[1]
                    # mjlab uses `_mean` and `_var` (with the underscore);
                    # they are shape [1, obs_dim], so squeeze to [obs_dim].
                    if tail in ("_mean", "mean", "running_mean"):
                        m = v
                        if m.ndim == 2 and m.shape[0] == 1:
                            m = m.squeeze(0)
                        self.normalizer_mean = m.to(self.device)
                        self._has_normalizer = True
                    elif tail in ("_var", "var", "running_var"):
                        vv = v
                        if vv.ndim == 2 and vv.shape[0] == 1:
                            vv = vv.squeeze(0)
                        self.normalizer_var = vv.to(self.device)
                        self._has_normalizer = True
                # `distribution.std_param` and other non-actor fields ignored.
            if not actor_weights:
                raise RuntimeError(
                    "actor_state_dict has no mlp.* weights. Keys: "
                    + str(list(asd.keys())[:20])
                )
        else:
            # Old format.
            sd = blob.get("model_state_dict") or blob.get("state_dict") or blob
            if not isinstance(sd, dict):
                raise RuntimeError(
                    "Unrecognized checkpoint shape: " + type(sd).__name__
                )
            actor_weights = {}
            for k, v in sd.items():
                if k.startswith("actor."):
                    actor_weights[k[len("actor."):]] = v
                elif k.startswith("policy.actor."):
                    actor_weights[k[len("policy.actor."):]] = v
                elif k.startswith("obs_normalizer.") or k.startswith("normalizer."):
                    tail = k.split(".", 1)[1]
                    if tail in ("running_mean", "mean"):
                        self.normalizer_mean = v.to(self.device)
                        self._has_normalizer = True
                    elif tail in ("running_var", "var"):
                        self.normalizer_var = v.to(self.device)
                        self._has_normalizer = True
            if not actor_weights:
                raise RuntimeError(
                    "No actor weights found. Keys in state_dict: "
                    + str(list(sd.keys())[:20])
                )

        missing, _ = self.actor.load_state_dict(actor_weights, strict=False)
        actor_param_keys = set(self.actor.state_dict().keys())
        real_missing = [k for k in missing if k in actor_param_keys]
        if real_missing:
            raise RuntimeError(
                "Actor checkpoint missing keys: " + str(real_missing[:5])
            )

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        torch = self._torch
        obs_arr = np.asarray(obs, dtype=np.float32)
        if obs_arr.ndim == 1:
            obs_arr = obs_arr[None, :]
        with torch.no_grad():
            x = torch.from_numpy(obs_arr).to(self.device)
            if self._has_normalizer:
                x = (x - self.normalizer_mean) / torch.sqrt(self.normalizer_var + 1e-8)
            out = self.actor(x)
        return out.de