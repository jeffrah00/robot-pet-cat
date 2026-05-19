"""Observation -> ICM feature vector helper.

The brain's gym/dict obs is heterogeneous (mood vector + scalars + scene_state
object). ICM needs a fixed-shape float vector. This module owns the projection
so BrainEnv and tests share one definition.

v0 vector (15 dim):
  - mood[0..5]                                                       (6)
  - cat_xy (world frame)                                             (2)
  - cos(yaw), sin(yaw)                                               (2)
  - body_height                                                      (1)
  - speed                                                            (1)
  - time_in_skill, clipped to 10s and rescaled to [0,1]              (1)
  - nearest play_target relative position (dx, dy) or zeros          (2)

If the scene has no play_target the last two slots are zero. The vector is
stable across episodes and scenes -- changing this shape is a breaking
change for any trained ICM checkpoint.
"""

from __future__ import annotations

import math

import numpy as np

CURIOSITY_OBS_DIM_V0 = 15


def obs_dict_to_curiosity_vec(obs: dict) -> np.ndarray:
    """Project a BrainEnv obs dict into the v0 curiosity feature vector."""
    mood = np.asarray(obs["mood"], dtype=np.float32).reshape(-1)
    if mood.shape[0] != 6:
        raise ValueError(f"expected mood of length 6, got {mood.shape[0]}")
    cat_xy = np.asarray(obs["cat_xy"], dtype=np.float32).reshape(-1)
    if cat_xy.shape[0] != 2:
        raise ValueError(f"expected cat_xy of length 2, got {cat_xy.shape[0]}")
    yaw = float(obs["cat_yaw"])
    body_h = float(obs["cat_body_height"])
    speed = float(obs["cat_speed"])
    time_in_skill = float(obs.get("time_in_skill", 0.0))
    t_clip = max(0.0, min(time_in_skill, 10.0)) / 10.0

    # Nearest play_target relative pos. SceneState carries the entities.
    dx = dy = 0.0
    scene_state = obs.get("scene_state")
    if scene_state is not None:
        play_targets = scene_state.filter(play_target=True)
        if play_targets:
            ent = min(
                play_targets,
                key=lambda e: float(
                    np.hypot(e.pos_xyz[0] - cat_xy[0], e.pos_xyz[1] - cat_xy[1])
                ),
            )
            dx = float(ent.pos_xyz[0] - cat_xy[0])
            dy = float(ent.pos_xyz[1] - cat_xy[1])

    vec = np.concatenate(
        [
            mood,                                # 6
            cat_xy,                              # 2
            np.asarray([math.cos(yaw), math.sin(yaw)], dtype=np.float32),  # 2
            np.asarray([body_h, speed, t_clip, dx, dy], dtype=np.float32),  # 5
        ],
        dtype=np.float32,
    )
    assert vec.shape == (CURIOSITY_OBS_DIM_V0,), vec.shape
    return vec
