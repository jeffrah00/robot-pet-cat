"""Intrinsic reward streams the brain optimizes against.

Each reward is a function of (scene_state, cat_state, mood). The brain composes
them with mood-modulated weights in `compute_composite_reward` below and the
resulting scalar is what PPO sees.

v1 (existing): ComfortReward + PlayReward + CuriosityReward.
v2 additions (docs/brain_design_v2.md, 2026-05-19):
  - HoldBonusReward (pause-as-default)
  - VantageReward, AmbushReward, PreyTrackingReward, SocialDistanceReward (stubs)

CuriosityReward (ICM-style) is implemented as a stateful torch module: see the
class for the API. Unlike Comfort/Play/Hold it is NOT a pure function of
(scene_state, cat_state); the caller (BrainEnv) supplies the per-step
intrinsic reward via the `curiosity_value` kwarg of `compute_composite_reward`.
This keeps the composition layer stateless while letting the curiosity model
own its prev-feature state and its trainable weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from robot_pet_cat.scene import SceneEntity, SceneState

    from .mood import Mood


# --------------------------------------------------------------------------- #
# Shared cat-state shape passed into reward.compute()
# --------------------------------------------------------------------------- #


@dataclass
class CatState:
    """Privileged proprioceptive snapshot the brain has of its own body.

    The kinematic scaffold env populates this from KinematicCat; a future
    physics-backed env would populate it from the trunk body's qpos+qvel.
    """

    xy: np.ndarray  # shape (2,) world-frame
    yaw: float
    body_height: float
    speed: float  # |horizontal velocity|
    active_skill: str | None  # name of currently-active skill, or None
    time_in_skill: float  # seconds since the current skill was selected


# --------------------------------------------------------------------------- #
# ICM transition record + curiosity module
# --------------------------------------------------------------------------- #


@dataclass
class Transition:
    """One (s_t, a_t, s_{t+1}) sample for ICM updates.

    `feat_t` / `feat_tp1` are pre-encoded numpy feature vectors of shape
    (obs_dim,) -- BrainEnv is responsible for projecting its obs dict into a
    fixed-shape vector. `action` is a discrete index in [0, n_actions).
    """

    feat_t: np.ndarray
    action: int
    feat_tp1: np.ndarray


class CuriosityReward:
    """ICM-style intrinsic curiosity (Pathak et al. 2017).

    This class differs from the other rewards in this file in three ways:

      1. Stateful -- it owns three small torch modules (feature encoder,
         forward model, inverse model) and their joint Adam optimizer.
      2. Step-driven, not pure -- the reward depends on (feat_t, action,
         feat_tp1), which the composer (compute_composite_reward) doesn't
         have. The expected flow is: BrainEnv calls intrinsic_reward(...)
         itself each step and passes the scalar into
         compute_composite_reward(curiosity_value=...).
      3. Learnable -- call .update(transitions) periodically to step the
         forward+inverse losses. Without calls to .update(), the intrinsic
         reward signal is noise around an untrained prediction error and
         won't meaningfully decay on revisited states.

    The .compute(...) method is intentionally a hard error -- the
    (scene_state, cat, mood) signature would invite the silent bug where the
    env stops passing a precomputed value and gets a zero curiosity term.

    Args:
      obs_dim: dimensionality of the feature vector the env will feed in.
      n_actions: size of the discrete action set (action_space_n on
        BrainEnv).
      feature_dim: encoder output size (default 64).
      hidden: MLP hidden width (default 128).
      beta: weight on the inverse loss in the combined update objective.
        0.2 is the value from the Pathak paper.
      reward_scale: multiplier applied to the L2 prediction error before
        returning as a reward. Pathak uses eta=0.5 in the paper.
      lr: Adam learning rate for the joint forward+inverse update.
      device: torch device string. CPU is fine for the prototyping env.
      seed: optional manual seed for reproducible init.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        feature_dim: int = 64,
        hidden: int = 128,
        beta: float = 0.2,
        reward_scale: float = 0.5,
        lr: float = 1e-3,
        device: str = "cpu",
        seed: Optional[int] = None,
    ) -> None:
        # Lazy torch import so a bare `from .rewards import CuriosityReward`
        # succeeds without torch installed; instantiation is what needs it.
        import torch

        from ._icm import FeatureEncoder, ForwardModel, InverseModel

        if seed is not None:
            torch.manual_seed(int(seed))

        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.feature_dim = int(feature_dim)
        self.beta = float(beta)
        self.reward_scale = float(reward_scale)
        self.device = torch.device(device)

        self.encoder = FeatureEncoder(obs_dim, feature_dim, hidden=hidden).to(self.device)
        self.forward_model = ForwardModel(feature_dim, n_actions, hidden=hidden).to(self.device)
        self.inverse_model = InverseModel(feature_dim, n_actions, hidden=hidden).to(self.device)

        params = (
            list(self.encoder.parameters())
            + list(self.forward_model.parameters())
            + list(self.inverse_model.parameters())
        )
        self.optimizer = torch.optim.Adam(params, lr=float(lr))
        self._torch = torch

    # ------------------------------------------------------------------ #
    # Step-time reward
    # ------------------------------------------------------------------ #

    def intrinsic_reward(
        self,
        feat_t: np.ndarray,
        action: int,
        feat_tp1: np.ndarray,
    ) -> float:
        """No-grad ICM reward for one transition.

        reward = reward_scale * 0.5 * ||phi(s_{t+1}) - phi_hat(s_{t+1})||^2
        """
        torch = self._torch
        if not 0 <= int(action) < self.n_actions:
            raise ValueError(
                f"action {action} out of range [0, {self.n_actions}) for ICM"
            )
        from ._icm import action_one_hot

        with torch.no_grad():
            s_t = self._to_tensor(feat_t).unsqueeze(0)
            s_tp1 = self._to_tensor(feat_tp1).unsqueeze(0)
            a = torch.tensor([int(action)], dtype=torch.long, device=self.device)
            phi_t = self.encoder(s_t)
            phi_tp1 = self.encoder(s_tp1)
            a_oh = action_one_hot(a, self.n_actions)
            phi_hat = self.forward_model(phi_t, a_oh)
            err = 0.5 * (phi_tp1 - phi_hat).pow(2).sum(dim=-1)  # shape (1,)
            return float(self.reward_scale * err.item())

    # ------------------------------------------------------------------ #
    # Learning
    # ------------------------------------------------------------------ #

    def update(self, transitions: Sequence[Transition]) -> dict:
        """Run one SGD step on a batch of transitions.

        Loss = (1 - beta) * forward_loss + beta * inverse_loss.

        Returns {"loss", "forward", "inverse"} scalars for logging.
        """
        torch = self._torch
        from ._icm import action_one_hot

        if len(transitions) == 0:
            return {"loss": 0.0, "forward": 0.0, "inverse": 0.0}

        s_t = torch.stack([self._to_tensor(tr.feat_t) for tr in transitions])
        s_tp1 = torch.stack([self._to_tensor(tr.feat_tp1) for tr in transitions])
        a = torch.tensor(
            [int(tr.action) for tr in transitions], dtype=torch.long, device=self.device
        )

        phi_t = self.encoder(s_t)
        phi_tp1 = self.encoder(s_tp1)
        a_oh = action_one_hot(a, self.n_actions)

        phi_hat = self.forward_model(phi_t, a_oh)
        # phi_tp1 is detached on the forward-loss path: encoder grads flow
        # only through the inverse model. This is what gives ICM its
        # "ignore-uncontrollable-stuff" property (Pathak 2017 section 2.2).
        forward_loss = 0.5 * (phi_hat - phi_tp1.detach()).pow(2).sum(dim=-1).mean()

        a_logits = self.inverse_model(phi_t, phi_tp1)
        inverse_loss = torch.nn.functional.cross_entropy(a_logits, a)

        loss = (1.0 - self.beta) * forward_loss + self.beta * inverse_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "forward": float(forward_loss.item()),
            "inverse": float(inverse_loss.item()),
        }

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #

    def state_dict(self) -> dict:
        return {
            "encoder": self.encoder.state_dict(),
            "forward_model": self.forward_model.state_dict(),
            "inverse_model": self.inverse_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, sd: dict) -> None:
        self.encoder.load_state_dict(sd["encoder"])
        self.forward_model.load_state_dict(sd["forward_model"])
        self.inverse_model.load_state_dict(sd["inverse_model"])
        self.optimizer.load_state_dict(sd["optimizer"])

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _to_tensor(self, x: np.ndarray):
        torch = self._torch
        arr = np.asarray(x, dtype=np.float32)
        if arr.shape != (self.obs_dim,):
            raise ValueError(
                f"feature vector shape {arr.shape} does not match obs_dim {self.obs_dim}"
            )
        return torch.from_numpy(arr).to(self.device)

    # ------------------------------------------------------------------ #
    # API guard: compute() is not the right entry point for ICM
    # ------------------------------------------------------------------ #

    def compute(self, *args, **kwargs) -> float:
        """Guarded against accidental use through the (scene, cat, mood) API."""
        raise TypeError(
            "CuriosityReward.compute(scene, cat, mood) is not supported. "
            "ICM is stateful and needs (feat_t, action, feat_tp1). Use "
            "intrinsic_reward(...) instead and pass the result as "
            "curiosity_value= to compute_composite_reward."
        )


@dataclass
class ComfortReward:
    """Dense reward for being in a 'good resting spot'.

    Three subterms summed with the dataclass weights:

      elevated: cat is within the XY footprint of an entity tagged 'elevated'
                AND its body height is at the entity's top surface (tolerance).
      soft:     same predicate but for entities tagged 'soft'. A cat on the
                couch (both flags) gets elevated + soft simultaneously.
      warm:     gaussian proximity score to the nearest entity tagged 'warm',
                regardless of whether the cat is 'on' it.

    Subterms are in [0, 1]; composite is in [0, sum(weights)].
    """

    elevated_weight: float = 0.4
    soft_weight: float = 0.4
    warm_weight: float = 0.2
    footprint_padding_m: float = 0.05
    height_tolerance_m: float = 0.15
    warm_sigma_m: float = 1.0

    # Hard-coded for v0 living-room couch. A future revision should read
    # half-size from scene metadata so the reward generalizes.
    _COUCH_HALF_SIZE_XY = (0.6, 0.3)

    def compute(self, scene_state: "SceneState", cat: CatState) -> float:
        elev = self._on_elevated(scene_state, cat)
        soft = self._on_soft(scene_state, cat)
        warm = self._warmth_proximity(scene_state, cat)
        return (
            self.elevated_weight * elev
            + self.soft_weight * soft
            + self.warm_weight * warm
        )

    def _on_entity(self, ent: "SceneEntity", cat: CatState) -> float:
        dx = float(cat.xy[0] - ent.pos_xyz[0])
        dy = float(cat.xy[1] - ent.pos_xyz[1])
        hx, hy = self._COUCH_HALF_SIZE_XY
        in_xy = (
            abs(dx) <= hx + self.footprint_padding_m
            and abs(dy) <= hy + self.footprint_padding_m
        )
        if not in_xy:
            return 0.0
        if abs(cat.body_height - 0.30) > self.height_tolerance_m:
            return 0.0
        return 1.0

    def _on_elevated(self, scene_state: "SceneState", cat: CatState) -> float:
        best = 0.0
        for ent in scene_state.filter(elevated=True):
            best = max(best, self._on_entity(ent, cat))
        return best

    def _on_soft(self, scene_state: "SceneState", cat: CatState) -> float:
        best = 0.0
        for ent in scene_state.filter(soft=True):
            best = max(best, self._on_entity(ent, cat))
        return best

    def _warmth_proximity(self, scene_state: "SceneState", cat: CatState) -> float:
        best = 0.0
        for ent in scene_state.filter(warm=True):
            dx = float(cat.xy[0] - ent.pos_xyz[0])
            dy = float(cat.xy[1] - ent.pos_xyz[1])
            d = float(np.hypot(dx, dy))
            score = float(np.exp(-0.5 * (d / self.warm_sigma_m) ** 2))
            best = max(best, score)
        return best


@dataclass
class PlayReward:
    """Reward from interaction with movable play_target objects.

    reward = speed_term + causal_term, where:
      speed_term:  ball speed (m/s) while ball is within paw range of cat,
                   scaled by play_speed_weight. Distant balls contribute 0.
      causal_term: fixed bonus when a swat-class skill is active AND a
                   play_target is within paw range. Cheapest possible
                   causal-credit -- a co-occurrence detector, not a model.
    """

    paw_range_m: float = 0.4
    play_speed_weight: float = 1.0
    causal_bonus: float = 0.5
    causal_skills: tuple = ("swat",)

    def compute(self, scene_state: "SceneState", cat: CatState) -> float:
        total = 0.0
        for ent in scene_state.filter(play_target=True):
            dx = float(cat.xy[0] - ent.pos_xyz[0])
            dy = float(cat.xy[1] - ent.pos_xyz[1])
            d = float(np.hypot(dx, dy))
            if d > self.paw_range_m:
                continue
            if ent.velocity_xyz is not None:
                speed = float(np.linalg.norm(ent.velocity_xyz))
                total += self.play_speed_weight * speed
            if cat.active_skill in self.causal_skills:
                total += self.causal_bonus
        return total


# --------------------------------------------------------------------------- #
# v2 additions
# --------------------------------------------------------------------------- #


@dataclass
class HoldBonusReward:
    """Pause-as-default. Small positive reward when the cat is doing nothing
    interesting AND no nearby *moving* stimulus is pulling for action.

    Reward when ALL of:
      (a) active_skill is in hold_skills, or None (no skill yet)
      (b) cat.speed < speed_threshold
      (c) no entity flagged play_target is within saliency_range_m AND
          moving faster than min_target_speed_mps

    The min_target_speed_mps gate is important: real cats don't watch
    motionless objects. A still ball is not a salient stimulus, so it
    shouldn't suppress the hold bonus. This was a v0 bug -- hold fired
    zero times because the ball was always within 0.6 m of the parked cat.

    Calibrate `bonus` against per-step task rewards. Target ~0.2-0.4 of
    episode ticks rewarded for cat-like stillness (cats are still ~80%
    of real time, but in this v0 scene most steps have something to do).
    """

    bonus: float = 0.01
    speed_threshold: float = 0.05
    saliency_range_m: float = 0.6
    min_target_speed_mps: float = 0.05
    hold_skills: tuple = ("sit", "lie_down", "crouch", "look_at")

    def compute(self, scene_state: "SceneState", cat: CatState) -> float:
        if cat.active_skill is not None and cat.active_skill not in self.hold_skills:
            return 0.0
        if cat.speed > self.speed_threshold:
            return 0.0
        for ent in scene_state.filter(play_target=True):
            dx = float(cat.xy[0] - ent.pos_xyz[0])
            dy = float(cat.xy[1] - ent.pos_xyz[1])
            d = float(np.hypot(dx, dy))
            if d > self.saliency_range_m:
                continue
            # Still targets are not salient: a motionless ball doesn't
            # suppress hold. Only moving play_targets count.
            if ent.velocity_xyz is None:
                # No velocity info -> conservative: treat as salient.
                return 0.0
            target_speed = float(np.linalg.norm(ent.velocity_xyz[:2]))
            if target_speed >= self.min_target_speed_mps:
                return 0.0
        return float(self.bonus)


@dataclass
class VantageReward:
    """Stub (v2). Bonus for being at higher z than typical."""

    def compute(self, *args, **kwargs) -> float:
        raise NotImplementedError("v2 -- see docs/brain_design_v2.md")


@dataclass
class AmbushReward:
    """Stub (v2). Bonus for elevated + line-of-sight to play_target."""

    def compute(self, *args, **kwargs) -> float:
        raise NotImplementedError("v2 -- see docs/brain_design_v2.md")


@dataclass
class PreyTrackingReward:
    """Stub (v2). Bonus for sustained gaze on a moving object."""

    def compute(self, *args, **kwargs) -> float:
        raise NotImplementedError("v2 -- see docs/brain_design_v2.md")


@dataclass
class SocialDistanceReward:
    """Stub (v2). Bonus for middle distance from tagged human entity."""

    def compute(self, *args, **kwargs) -> float:
        raise NotImplementedError("v2 -- see docs/brain_design_v2.md")


# --------------------------------------------------------------------------- #
# Composite reward
# --------------------------------------------------------------------------- #


@dataclass
class CompositeRewardConfig:
    """Weights for the composite reward.

    v1 weights (curiosity, comfort, play) are MULTIPLIED by Mood.weights().
    hold_w is NOT mood-modulated -- pause-as-default applies regardless of
    mood (sleepy and alert cats both reward stillness when nothing's up).

    Defaults are tuned per the v0 baseline findings:
    - curiosity_w = 0.05: ICM is on by default; novelty actively pushes
      against attractor collapse (memory: cats-are-stochastic).
    - comfort_w  = 1.0
    - play_w     = 0.3: dropped from 1.0; v0 showed play dominated 30x.
    - hold_w     = 1.0: hold fires more reliably now that still play
      targets don't trigger saliency (memory: brain-v0-baseline-findings).
    """

    curiosity_w: float = 0.05
    comfort_w: float = 1.0
    play_w: float = 0.3
    hold_w: float = 1.0


def compute_composite_reward(
    scene_state: "SceneState",
    cat: CatState,
    mood: "Mood",
    *,
    comfort: ComfortReward | None = None,
    play: PlayReward | None = None,
    hold: HoldBonusReward | None = None,
    cfg: CompositeRewardConfig | None = None,
    curiosity_value: float = 0.0,
) -> dict:
    """Returns the composite reward AND its per-term breakdown.

    The dict is the point: cat-vibe tuning means watching which terms drive
    behavior moment to moment. PPO reads out['total']; logging reads the rest.

    `curiosity_value` is the precomputed ICM intrinsic reward for this step
    (a scalar from `CuriosityReward.intrinsic_reward(...)`). It is NOT
    computed inside this function because ICM is stateful and needs
    (feat_t, action, feat_tp1) which the composer doesn't have. Default 0.0
    means "no curiosity term active" -- match this with cfg.curiosity_w=0
    if you want the term reliably off.
    """
    if comfort is None:
        comfort = ComfortReward()
    if play is None:
        play = PlayReward()
    if hold is None:
        hold = HoldBonusReward()
    if cfg is None:
        cfg = CompositeRewardConfig()

    curiosity_mood_w, comfort_mood_w, play_mood_w = mood.weights()

    r_comfort = comfort.compute(scene_state, cat)
    r_play = play.compute(scene_state, cat)
    r_hold = hold.compute(scene_state, cat)
    r_curiosity = float(curiosity_value)

    total = (
        cfg.curiosity_w * curiosity_mood_w * r_curiosity
        + cfg.comfort_w * comfort_mood_w * r_comfort
        + cfg.play_w * play_mood_w * r_play
        + cfg.hold_w * r_hold
    )
    return {
        "total": float(total),
        "comfort": float(r_comfort),
        "play": float(r_play),
        "hold": float(r_hold),
        "curiosity": float(r_curiosity),
        "_mood_w_curiosity": float(curiosity_mood_w),
        "_mood_w_comfort": float(comfort_mood_w),
        "_mood_w_play": float(play_mood_w),
    }
