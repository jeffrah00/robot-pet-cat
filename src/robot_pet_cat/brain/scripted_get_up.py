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

Keyframe design (recovery from belly-down prone pose):
  We assume the cat is on its belly (chassis on the floor at z~0.05),
  oriented right-side-up. This is a feedback-free open-loop recovery:
  joint targets alone cannot reorient the body, but they CAN push the
  chassis up off the floor if the legs are loaded against the ground
  symmetrically.

  t=0.0..0.4s   hold PRONE: hip splayed wide (~0.30), thighs heavily
                rotated forward (1.5 rad), calves deeply folded (-2.6).
                This positions feet at the chassis level, beside/below
                the hips, with knees tucked up. PD damps initial motion.
  t=0.4..1.8s   ramp to PRESS: knees straighten (calf -2.6 -> -1.4),
                thighs rotate back from forward (1.5 -> 0.7), hips come
                in slightly (0.30 -> 0.15). The straightening knees push
                the feet DOWN against the floor, lifting the chassis UP.
                Slow ramp (1.4s) keeps PD torques moderate; rate limit
                caps per-step delta.
  t=1.8..2.8s   settle into STAND: hip 0.10, thigh 0.90, calf -1.80.
  t>=2.8s       hold standing default forever.

The key physical idea: starting at PRONE with feet planted, gradually
straightening the knees while rotating the thighs back applies an
upward force at the foot contacts that lifts the chassis. This only
works if the body starts roughly belly-down (axis aligned with world
up). Side-lying or upside-down recovery requires closed-loop logic
that this script does NOT have.
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
# Prone: belly-down, legs splayed wide, knees fully folded so feet sit
# at the chassis level beside the hips. This is the assumed starting
# pose -- the render script should initialize the cat in this config.
_PRONE = _pose(0.30, 1.50, -2.60)
# Press: knees significantly extended, thighs rotated back toward
# standing, hips coming inward. Driving from PRONE to PRESS with feet
# planted produces an upward push at the foot contacts.
_PRESS = _pose(0.15, 0.70, -1.40)


# Keyframe schedule: (time_s, target_pose). Linear interpolation
# between consecutive keyframes; before the first it holds
# keyframe[0], after the last it holds keyframe[-1].
KEYFRAMES: list[tuple[float, np.ndarray]] = [
    (0.00, _PRONE),   # match prone init pose, no PD discontinuity
    (0.40, _PRONE),   # hold briefly so settling damps any init motion
    (1.80, _PRESS),   # slow ramp lifts the chassis off the floor
    (2.80, _STAND),   # settle into normal standing pose
    (8.00, _STAND),   # hold
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
        """Reset the phase clock. Called by PhysicsCat when get_up activates.

        _prev_target is seeded with the first keyframe's pose so the
        rate-limiter doesn't clip the t=0 output (we want to emit the first
        keyframe exactly on the first call -- the upstream caller should have
        already initialized joint positions to match that pose)."""
        self._t = 0.0
        self._prev_target = self._keyframes[0][1].copy()

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
