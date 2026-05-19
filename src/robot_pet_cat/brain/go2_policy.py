"""Loader for the unitree_rl_mjlab Go2 velocity walker policy.

The walker on the pod is trained by unitree_rl_mjlab (RSL-RL backend) against
the Go2 velocity task. Each training save writes both:

  - logs/.../model_<iter>.pt   -- RSL-RL OnPolicyRunner checkpoint
  - logs/.../policy.onnx       -- exported ONNX (obs-normalizer baked in)

ONNX is the preferred path for external integration because:
  1. The empirical-mean/std observation normalizer is embedded inside the
     ONNX graph -- no rsl_rl dependency required at load time.
  2. The whole pipeline runs in onnxruntime, which is much lighter than
     a torch+rsl_rl stack and works on CPU at ~50 Hz for a single robot.

The .pt path is the fallback when only the runner checkpoint is on disk.
It requires rsl_rl to be installed and rebuilds the actor MLP + normalizer
manually before loading weights.

Policy contract (matches velocity_env_cfg.py + unitree_go2_flat_env_cfg):

  Observation (47 dims for flat env, in this order):
    base_ang_vel        (3)   rad/s, body frame, from gyro
    projected_gravity   (3)   unit gravity in body frame
    command             (3)   [vx, vy, yaw_rate]
    phase               (2)   [sin(2*pi*t/0.6), cos(2*pi*t/0.6)],
                              zeroed when ||command|| < 0.1
    joint_pos           (12)  q - q_default
    joint_vel           (12)  dq
    last_action         (12)  previous raw policy output

  The rough-terrain variant additionally has a `height_scan` term
  (160 dims for the default 1.6x1.0m / 0.1 resolution grid). We do
  not include it here: the Tier 3 brain runs on a flat living-room
  floor, so the scan is degenerate. If model_6400.pt was trained on
  rough terrain, pass `obs_extra_zero_dims=160` to PhysicsCat to pad
  the observation with zeros.

  Action (12 dims):
    raw output a. Joint target = q_default + 0.25 * a.

Joint order (MJCF order in menagerie's go2_mjx.xml):
    FR_hip, FR_thigh, FR_calf,
    FL_hip, FL_thigh, FL_calf,
    RR_hip, RR_thigh, RR_calf,
    RL_hip, RL_thigh, RL_calf
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

import numpy as np


# Default joint positions, MJCF order = FL, FR, RL, RR (verified by
# mj_id2name against the compiled scene). This matches the order
# menagerie's go2_mjx.xml emits actuators/joints in, which is also
# the order mjlab trains the policy against.
GO2_DEFAULT_JOINT_POS = np.array(
    [
        -0.10, 0.90, -1.80,   # FL
        +0.10, 0.90, -1.80,   # FR
        -0.10, 0.90, -1.80,   # RL
        +0.10, 0.90, -1.80,   # RR
    ],
    dtype=np.float32,
)

# Joint name order, FL/FR/RL/RR (matches MJCF compile order).
GO2_JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]

# Per the env config: action delta is scaled by 0.25 before adding to default.
GO2_ACTION_SCALE = 0.25

# Physics dt and decimation expected by the walker.
GO2_PHYSICS_DT = 0.005
GO2_POLICY_DECIMATION = 4
GO2_POLICY_DT = GO2_PHYSICS_DT * GO2_POLICY_DECIMATION  # 0.02 s, 50 Hz

# Phase period (sec) -- gait phase wraps every 0.6 s of "moving" command.
GO2_PHASE_PERIOD = 0.6
# Phase is zeroed when command magnitude below this threshold.
GO2_PHASE_STAND_THRESHOLD = 0.1


class WalkerPolicy(Protocol):
    """Callable that maps obs [N,48] -> action [N,12]."""

    def __call__(self, obs: np.ndarray) -> np.ndarray: ...


# --------------------------------------------------------------------------- #
# ONNX path -- preferred
# --------------------------------------------------------------------------- #


class _OnnxWalkerPolicy:
    """Wraps onnxruntime InferenceSession around the Go2 walker export."""

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
        # Resolve I/O names. The export usually names the input "obs"
        # (or "input") and produces one output; be defensive about both.
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
# .pt path -- requires rsl_rl
# --------------------------------------------------------------------------- #


@dataclass
class TorchWalkerConfig:
    obs_dim: int = 47
    action_dim: int = 12
    hidden_dims: tuple = (512, 256, 128)
    activation: str = "elu"
    device: str = "cpu"


class _TorchWalkerPolicy:
    """Rebuilds the RSL-RL actor + EmpiricalNormalization and loads weights.

    The model_<iter>.pt file from rsl_rl OnPolicyRunner.save has the shape:
        {
          "model_state_dict": {
              "obs_normalizer.running_mean": ...,
              "obs_normalizer.running_var": ...,
              "obs_normalizer.count": ...,
              "actor.0.weight": ..., "actor.0.bias": ...,
              "actor.2.weight": ..., ...,
              "actor.<last>.weight": ..., "actor.<last>.bias": ...,
              ...critic, std...
          },
          "optimizer_state_dict": ...,
          "iter": ...,
          "infos": ...,
        }

    The actor module in rsl_rl 2.x is an `nn.Sequential` so the keys are
    "actor.0", "actor.2", "actor.4", "actor.6" for our four Linear layers.
    """

    def __init__(self, ckpt_path: Path, cfg: Optional[TorchWalkerConfig] = None):
        import torch
        import torch.nn as nn

        self._torch = torch
        self.cfg = cfg or TorchWalkerConfig()
        self.device = torch.device(self.cfg.device)

        # Build MLP: Linear-ELU-Linear-ELU-Linear-ELU-Linear (no output activation).
        act = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}[self.cfg.activation]
        dims = [self.cfg.obs_dim, *self.cfg.hidden_dims, self.cfg.action_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act())
        self.actor = nn.Sequential(*layers).to(self.device).eval()

        # Build empirical normalizer (running mean / var / count).
        self.normalizer_mean = torch.zeros(self.cfg.obs_dim, device=self.device)
        self.normalizer_var = torch.ones(self.cfg.obs_dim, device=self.device)
        self._has_normalizer = False

        self._load_checkpoint(ckpt_path)

    # ------------------------------------------------------------------ #

    def _load_checkpoint(self, ckpt_path: Path) -> None:
        torch = self._torch
        blob = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)

        sd = blob.get("model_state_dict") or blob.get("state_dict") or blob
        if not isinstance(sd, dict):
            raise RuntimeError(
                f"Unrecognized checkpoint shape: {type(sd).__name__}. "
                f"Top-level keys: {list(blob.keys()) if isinstance(blob, dict) else 'n/a'}"
            )

        # Pull out actor and obs_normalizer weights. Strip rsl_rl prefixes.
        actor_weights: dict = {}
        for k, v in sd.items():
            # rsl_rl 2.x: keys look like "actor.0.weight", "actor.2.weight", ...
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
                f"{list(sd.keys())[:20]} (truncated)"
            )

        missing, unexpected = self.actor.load_state_dict(actor_weights, strict=False)
        # Filter to real misses (ignore stds, etc. that aren't actor params).
        actor_param_keys = set(self.actor.state_dict().keys())
        real_missing = [k for k in missing if k in actor_param_keys]
        if real_missing:
            raise RuntimeError(
                f"Actor checkpoint missing {len(real_missing)} keys, e.g. "
                f"{real_missing[:5]}. Architecture mismatch?"
            )

    # ------------------------------------------------------------------ #

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        torch = self._torch
        obs_arr = np.asarray(obs, dtype=np.float32)
        if obs_arr.ndim == 1:
            obs_arr = obs_arr[None, :]
        with torch.no_grad():
            x = torch.from_numpy(obs_arr).to(self.device)
            if self._has_normalizer:
                # x_hat = (x - mean) / sqrt(var + eps)
                x = (x - self.normalizer_mean) / torch.sqrt(self.normalizer_var + 1e-8)
            out = self.actor(x)
        return out.detach().cpu().numpy().astype(np.float32)


# --------------------------------------------------------------------------- #
# Public loader
# --------------------------------------------------------------------------- #


def load_go2_walker_policy(
    policy_path: str | Path,
    device: str = "cpu",
) -> WalkerPolicy:
    """Load a Go2 velocity walker. Path can be .onnx or .pt.

    Selection rules:
      - .onnx extension       -> onnxruntime path (preferred).
      - .pt / .pth extension  -> torch + rsl_rl style path.
      - directory             -> look for policy.onnx, else model_*.pt
                                 (highest iteration number).

    Returns a callable of shape obs[N,48] -> action[N,12].
    """
    path = Path(policy_path)
    if path.is_dir():
        onnx_candidate = path / "policy.onnx"
        if onnx_candidate.is_file():
            path = onnx_candidate
        else:
            pts = sorted(
                path.glob("model_*.pt"),
                key=lambda p: _iter_from_name(p.name),
                reverse=True,
            )
            if not pts:
                raise FileNotFoundError(
                    f"No policy.onnx or model_*.pt found under {path}"
                )
            path = pts[0]

    if not path.is_file():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".onnx":
        return _OnnxWalkerPolicy(path)
    if suffix in (".pt", ".pth"):
        return _TorchWalkerPolicy(path, TorchWalkerConfig(device=device))
    raise ValueError(
        f"Unrecognized walker policy extension {suffix!r}; "
        "expected .onnx, .pt, or .pth"
    )


def _iter_from_name(name: str) -> int:
    """Parse iter index out of 'model_6400.pt'-style filenames; 0 on failure."""
    stem = Path(name).stem
    if "_" not in stem:
        return 0
    tail = stem.rsplit("_", 1)[1]
    try:
        return int(tail)
    except ValueError:
        return 0
