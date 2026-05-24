"""Scripted get_up skill -- deterministic joint-angle keyframes.

Replaces the failed RL get_up track (v3..v10) with a hand-tuned PD-target
sequence. Conforms to the WalkerPolicy callable protocol so PhysicsCat can
dispatch to it the same way it dispatches to a trained .pt policy.

Background
----------
Eight+ single-policy RL variants (v3, v4, v5, v6, v7, v7.1, v7.2, v7.3, v8,
v9) all converged on degenerate poses (sphynx, belly-flat, splayed) instead
of standing recovery. v10b only "succeeded" via a curriculum-at-deployment
bug. Jeff (2026-05-24) called the RL track failed and asked for a scripted
fallback that just forces joint angles through PD control. This is that
fallback.

How it works
------------
PhysicsCat exposes the standard Go2 walker pipeline:
    joint_target = GO2_DEFAULT_JOINT_POS + action_scale * action     (action_scale = 0.25)
    mj_data.ctrl[actuator_i] = joint_target[i]
The compiled menagerie actuators are PD controllers (kp ~ 25, kd ~ 0.5)
that drive joints toward those targets. So if we want joint angle q_i, we
emit action_i = (q_i - q_default_i) / 0.25 .

This callable maintains an internal phase clock (advanced 0.02s per call,
matching the dispatch cadence = decimation * physics_dt = 4 * 0.005) and
linearly interpolates between a small set of keyframes. PhysicsCat zeros
last_action when switching to get_up; we also expose .reset() and rely on
PhysicsCat calling it on activation so the phase restarts at t=0 each time
the skill fires.

Keyframe design (recovery from arbitrary fallen pose):
  t=0.0..0.30s  drive all joints to "tuck": hip abduction 0, thighs forward
                (1.4 rad), calves deeply folded (-2.6 rad). Pulls feet in
                under the body so the COM sits over the contact patch.
  t=0.30..0.80  hold the tuck. PD damping settles the body; if the cat is
                on its side, the curled legs act like a fulcrum.
  t=0.80..1.60  push to standing: hip back to default abduction (+/-0.10),
                thighs back to 0.9, calves to -1.8. The extension torques
                lever the body upright.
  t>=1.60s      hold standing default forever.

Anything past 1.6s is just "stay at default", which equals action=0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .go2_policy import GO2_ACTION_SCALE, GO2_DEFAULT_JOINT_POS


# Keyframes as 12-d joint-angle vectors, in MJCF order
# (FL/FR/RL/RR x hip/thigh/calf).
def _pose(hip_abd: float, thigh: float, calf: float) -> np.ndarray:
    """Build a 12-d joint-angle vector with symmetric L/R hip abduction."""
    return np.array(
        [
            -hip_abd, thigh, calf,   # FL
            +hip_abd, thigh, calf,   # FR
            -hip_abd, thigh, calf,   # RL
            +hip_abd, thigh, calf,   # RR
        ],
        dtype=np.float32,
    )


# Standing default (matches GO2_DEFAULT_JOINT_POS).
_STAND = _pose(0.10, 0.90, -1.80)
# Tuck: hips straight, thighs pulled forward, calves curled.
_TUCK = _pose(0.00, 1.40, -2.60)


# Keyframe schedule: (time_s, target_pose).
# The policy linearly interpolates between consecutive keyframes; before
# the first keyframe it holds keyframe[0]; after the last it holds keyframe[-1].
KEYFRAMES: list[tuple[float, np.ndarray]] = [
    (0.00, _STAND),   # match initial pose to avoid a discontinuity
    (0.10, _TUCK),    # quick pull to tuck
    (0.80, _TUCK),    # hold tuck briefly
    (1.60, _STAND),   # push up to standing
    (3.00, _STAND),   # hold standing
]


@dataclass
class ScriptedGetUpConfig:
    policy_dt: float = 0.02  # GO2_PHYSICS_DT * GO2_POLICY_DECIMATION = 0.005 * 4
    # Optional override of the keyframe schedule (kept as a knob; defaults
    # to the module-level KEYFRAMES if None).
    keyframes: Optional[list[tuple[float, np.ndarray]]] = None
    # Clip the per-step joint-target delta so PD doesn't explode if a
    # keyframe is far from the current joint position. Per-step in radians.
    max_delta_per_step: float = 0.20


class ScriptedGetUpPolicy:
    """Stateful callable: maps obs[N, obs_dim] -> action[N, 12].

    obs_dim is exposed for the PhysicsCat FR-Net adapter check; we accept
    either 42 (hind_sit-style) or 46 (FR-Net) but only use the joint_pos
    portion, so either is fine.
    """

    # Match the v4b/hind_sit obs contract: 42-dim.
    obs_dim: int = 42

    def __init__(self, cfg: Optional[ScriptedGetUpConfig] = None):
        self.cfg = cfg or ScriptedGetUpConfig()
        self._keyframes = self.cfg.keyframes or KEYFRAMES
        self._t: float = 0.0
        # For inv-action conversion: target = default + scale * action
        #                         -> action = (target - default) / scale
        self._default = GO2_DEFAULT_JOINT_POS.astype(np.float32)
        self._scale = float(GO2_ACTION_SCALE)
        # Track previous emitted target so we can rate-limit step deltas.
        self._prev_target = self._default.copy()

    # --- PhysicsCat-facing reset hook ----------------------------------- #
    def reset(self) -> None:
        """Reset the phase clock. Called by PhysicsCat when get_up activates."""
        self._t = 0.0
        self._prev_target = self._default.copy()

    # --- Keyframe interpolation ----------------------------------------- #
    def _target_at(self, t: float) -> np.ndarray:
        kf = self._keyframes
        if t <= kf[0][0]:
            return kf[0][1].copy()
        if t >= kf[-1][0]:
            return kf[-1][1].copy()
        for i in range(1, len(kf)):
            t0, p0 = kf[i - 1]
            t1, p1 = kf[i]
            if t <= t1:
                alpha = (t - t0) / max(t1 - t0, 1e-6)
                return ((1.0 - alpha) * p0 + alpha * p1).astype(np.float32)
        return kf[-1][1].copy()

    # --- Callable -------------------------------------------------------- #
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """obs ignored except for batch dim; returns same-batch action."""
        target = self._target_at(self._t)

        # Rate-limit the per-step delta to keep PD torques sane.
        delta = target - self._prev_target
        max_d = self.cfg.max_delta_per_step
        delta = np.clip(delta, -max_d, +max_d)
        target = self._prev_target + delta
        self._prev_target = target

        action = (target - self._default) / self._scale
        # Advance phase clock.
        self._t += self.cfg.policy_dt

        # Match batch dim of input.
        batch = obs.shape[0] if obs.ndim >= 2 else 1
        out = np.tile(action[None, :], (batch, 1)).astype(np.float32)
        return out
