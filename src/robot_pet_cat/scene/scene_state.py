"""Extract a structured scene_state dict from a MuJoCo model + data.

The brain's reward streams (curiosity, comfort, play) all want to know:
  - what entities exist
  - where each one is (world frame)
  - what semantic flags apply to it (elevated / soft / warm / play_target)
  - for movable entities: velocity, so play reward can credit cat-caused motion

This module is *the* boundary between MuJoCo internals and brain logic. The
brain never touches mj_model or mj_data directly; it consumes the
`SceneState` dict produced here. That keeps the brain backend-agnostic (we
could swap MuJoCo for Isaac Lab later without rewriting brain/).

Semantic flags live separately from the MJCF (in `LIVING_ROOM_V0_FLAGS` below)
because they are *project decisions*, not physics. Putting "is this surface
warm?" in the XML would mix two concerns and make flag iteration painful.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np


# --------------------------------------------------------------------------- #
# Flag + entity dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SemanticFlags:
    """Boolean tags attached to an entity. Used by reward streams."""

    elevated: bool = False
    soft: bool = False
    warm: bool = False
    play_target: bool = False

    def any(self) -> bool:
        return self.elevated or self.soft or self.warm or self.play_target


@dataclass
class SceneEntity:
    """A named thing in the world that the brain might care about.

    `kind` distinguishes physically-simulated bodies from sites (regions). A
    site has no mass or collision; the brain treats it as a tagged region.
    """

    name: str
    kind: str  # "body" | "site"
    pos_xyz: np.ndarray  # shape (3,)
    velocity_xyz: np.ndarray | None  # shape (3,), None for sites and static bodies
    flags: SemanticFlags


@dataclass
class SceneState:
    """The full snapshot the brain rewards consume each tick."""

    entities: list[SceneEntity] = field(default_factory=list)

    # Convenience indices built on construction.
    by_name: dict[str, SceneEntity] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.by_name = {e.name: e for e in self.entities}

    # Reward streams typically want to filter by flag, e.g. "every entity with
    # play_target=True" or "every entity with both elevated and soft".
    def filter(
        self,
        *,
        elevated: bool | None = None,
        soft: bool | None = None,
        warm: bool | None = None,
        play_target: bool | None = None,
    ) -> list[SceneEntity]:
        out: list[SceneEntity] = []
        for e in self.entities:
            if elevated is not None and e.flags.elevated != elevated:
                continue
            if soft is not None and e.flags.soft != soft:
                continue
            if warm is not None and e.flags.warm != warm:
                continue
            if play_target is not None and e.flags.play_target != play_target:
                continue
            out.append(e)
        return out


# --------------------------------------------------------------------------- #
# Flag configuration for the v0 living-room scene
# --------------------------------------------------------------------------- #


# Maps (kind, name) -> flags. Kept here, NOT in the MJCF, so we can iterate on
# flag definitions without re-saving XML.
LIVING_ROOM_V0_FLAGS: dict[tuple[str, str], SemanticFlags] = {
    # Couch is an elevated soft surface — comfort hot spot.
    ("body", "couch"): SemanticFlags(elevated=True, soft=True),
    # Ball is a small movable object — play hot spot.
    ("body", "ball"): SemanticFlags(play_target=True),
    # Window region is a tagged floor patch — warm comfort spot.
    ("site", "window_region"): SemanticFlags(warm=True),
}


# Entities that are physically free-floating (have a free joint), so the brain
# should track their velocity for play-reward causal credit.
MOVABLE_BODIES: frozenset[str] = frozenset({"ball"})


# --------------------------------------------------------------------------- #
# Extractor
# --------------------------------------------------------------------------- #


def extract_scene_state(
    mj_model: Any,
    mj_data: Any,
    flag_config: dict[tuple[str, str], SemanticFlags] | None = None,
    movable_bodies: Iterable[str] | None = None,
) -> SceneState:
    """Walk mj_model's named bodies + sites and build a SceneState.

    Args:
      mj_model: a mujoco.MjModel (we read names + counts from it)
      mj_data:  a mujoco.MjData stepped against `mj_model`. We read xpos and
                qvel from it.
      flag_config: maps (kind, name) -> SemanticFlags. Defaults to the v0
                   living-room scene config (LIVING_ROOM_V0_FLAGS). Pass your
                   own to override.
      movable_bodies: names of bodies with a free joint whose velocity should
                      be reported. Defaults to MOVABLE_BODIES.

    Returns:
      SceneState with one entity per (body or site) that has at least one flag
      set. Bodies without any flag are dropped to keep the dict small — they
      don't influence brain rewards.
    """
    if flag_config is None:
        flag_config = LIVING_ROOM_V0_FLAGS
    if movable_bodies is None:
        movable_bodies = MOVABLE_BODIES
    movable_bodies = frozenset(movable_bodies)

    entities: list[SceneEntity] = []

    # --- bodies ----------------------------------------------------------
    nbody = int(mj_model.nbody)
    body_xpos = np.asarray(mj_data.xpos).reshape(nbody, 3)
    # Build a per-body linear velocity (cvel[3:6] is linear in world frame for
    # body-level com velocity; for free joints it's the trunk velocity).
    body_cvel = (
        np.asarray(mj_data.cvel).reshape(nbody, 6)
        if hasattr(mj_data, "cvel") and mj_data.cvel.size
        else np.zeros((nbody, 6), dtype=np.float64)
    )

    for body_id in range(nbody):
        name = _body_name(mj_model, body_id)
        if name is None or name == "world":
            continue
        flags = flag_config.get(("body", name))
        if flags is None or not flags.any():
            continue
        vel = (
            body_cvel[body_id, 3:6].astype(np.float32, copy=True)
            if name in movable_bodies
            else None
        )
        entities.append(
            SceneEntity(
                name=name,
                kind="body",
                pos_xyz=body_xpos[body_id].astype(np.float32, copy=True),
                velocity_xyz=vel,
                flags=flags,
            )
        )

    # --- sites -----------------------------------------------------------
    nsite = int(getattr(mj_model, "nsite", 0))
    if nsite:
        site_xpos = np.asarray(mj_data.site_xpos).reshape(nsite, 3)
        for site_id in range(nsite):
            name = _site_name(mj_model, site_id)
            if name is None:
                continue
            flags = flag_config.get(("site", name))
            if flags is None or not flags.any():
                continue
            entities.append(
                SceneEntity(
                    name=name,
                    kind="site",
                    pos_xyz=site_xpos[site_id].astype(np.float32, copy=True),
                    velocity_xyz=None,
                    flags=flags,
                )
            )

    return SceneState(entities=entities)


# --------------------------------------------------------------------------- #
# Name lookup helpers — MuJoCo's Python bindings spell this inconsistently
# across versions; centralize the dance here.
# --------------------------------------------------------------------------- #


def _body_name(mj_model: Any, body_id: int) -> str | None:
    # Modern bindings: int(mj_model.body(body_id).name) works via a property.
    # Fall back to the C-style id2name if needed.
    try:
        n = mj_model.body(body_id).name
        return n if n else None
    except Exception:
        pass
    try:
        import mujoco  # local import so the module imports without mujoco for static analysis

        n = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        return n if n else None
    except Exception:
        return None


def _site_name(mj_model: Any, site_id: int) -> str | None:
    try:
        n = mj_model.site(site_id).name
        return n if n else None
    except Exception:
        pass
    try:
        import mujoco

        n = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SITE, site_id)
        return n if n else None
    except Exception:
        return None
