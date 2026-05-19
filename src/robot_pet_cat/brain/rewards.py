"""Intrinsic reward streams the brain optimizes against.

Each reward is a function of (scene_state, cat_state, mood). The brain composes
them with mood-modulated weights in `compute_composite_reward` below and the
resulting scalar is what PPO sees.

v1 (existing): ComfortReward + PlayReward + CuriosityReward.
v2 additions (docs/brain_design_v2.md, 2026-05-19):
  - HoldBonusReward (pause-as-default)
  - VantageReward, AmbushReward, PreyTrackingReward, SocialDistanceReward (stubs)

CuriosityReward (ICM-style) remains NotImplementedError -- it needs a
trainable forward model. Prototype with curiosity_w=0 until that lands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

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
# v1 rewards: curiosity, comfort, play
# --------------------------------------------------------------------------- #


@dataclass
class CuriosityReward:
    """ICM-style intrinsic curiosity (Pathak et al. 2017). Stub for now."""

    embed_dim: int = 64

    def compute(self, *args, **kwargs) -> float:
        raise NotImplementedError(
            "ICM curiosity needs a trainable forward model. Run brain "
            "prototyping with curiosity_w=0 until that lands."
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
        # Body height check: trunk should be ~standing height when "on" the
        # surface. Real "is cat on top" needs contact detection; we don't have
        # that in the kinematic scaffold.
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
    causal_skills: tuple[str, ...] = ("swat",)

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
    interesting AND no nearby stimulus is pulling for action.

    Reward when ALL of:
      (a) active_skill is in hold_skills, or None (no skill yet)
      (b) cat.speed < speed_threshold
      (c) no entity flagged play_target is within saliency_range_m

    Calibrate `bonus` against per-step task rewards. Target ~0.7-0.85 of
    episode ticks rewarded for cat-like stillness.
    """

    bonus: float = 0.01
    speed_threshold: float = 0.05
    saliency_range_m: float = 0.6
    hold_skills: tuple[str, ...] = ("sit", "lie_down", "crouch", "look_at")

    def compute(self, scene_state: "SceneState", cat: CatState) -> float:
        if cat.active_skill is not None and cat.active_skill not in self.hold_skills:
            return 0.0
        if cat.speed > self.speed_threshold:
            return 0.0
        for ent in scene_state.filter(play_target=True):
            dx = float(cat.xy[0] - ent.pos_xyz[0])
            dy = float(cat.xy[1] - ent.pos_xyz[1])
            d = float(np.hypot(dx, dy))
            if d <= self.saliency_range_m:
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
    """

    curiosity_w: float = 0.0
    comfort_w: float = 1.0
    play_w: float = 1.0
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
) -> dict:
    """Returns the composite reward AND its per-term breakdown.

    The dict is the point: cat-vibe tuning means watching which terms drive
    behavior moment to moment. PPO reads out['total']; logging reads the rest.
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
    r_curiosity = 0.0  # stubbed; mult by 0 keeps us out of NotImplementedError

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
