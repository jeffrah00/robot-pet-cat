"""Tests for the v0 living-room scene + scene_state extractor."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco", reason="mujoco not installed in this env")

from robot_pet_cat.scene import (  # noqa: E402
    LIVING_ROOM_V0_FLAGS,
    SceneEntity,
    SceneState,
    SemanticFlags,
    extract_scene_state,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LIVING_ROOM_XML = REPO_ROOT / "assets" / "scenes" / "living_room.xml"


@pytest.fixture(scope="module")
def living_room_model_data():
    if not LIVING_ROOM_XML.is_file():
        pytest.skip(f"missing {LIVING_ROOM_XML}")
    m = mujoco.MjModel.from_xml_path(str(LIVING_ROOM_XML))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


class TestXMLLoads:
    def test_loads(self, living_room_model_data) -> None:
        m, _ = living_room_model_data
        # Floor body + couch + ball = 3 bodies (world counts as body 0 in
        # mujoco, so nbody is 3 = world + couch + ball).
        assert m.nbody == 3
        assert m.nsite == 1


class TestExtractor:
    def test_returns_three_entities(self, living_room_model_data) -> None:
        m, d = living_room_model_data
        s = extract_scene_state(m, d)
        assert len(s.entities) == 3

    def test_couch_has_elevated_soft_flags(self, living_room_model_data) -> None:
        m, d = living_room_model_data
        s = extract_scene_state(m, d)
        couch = s.by_name["couch"]
        assert couch.kind == "body"
        assert couch.flags.elevated is True
        assert couch.flags.soft is True
        assert couch.flags.warm is False
        assert couch.flags.play_target is False
        # Static body → no velocity tracked.
        assert couch.velocity_xyz is None
        # XML says pos="2.0 0.0 0.20".
        assert np.allclose(couch.pos_xyz, [2.0, 0.0, 0.20], atol=1e-3)

    def test_ball_is_movable_and_tagged_play_target(self, living_room_model_data) -> None:
        m, d = living_room_model_data
        s = extract_scene_state(m, d)
        ball = s.by_name["ball"]
        assert ball.kind == "body"
        assert ball.flags.play_target is True
        # Movable → velocity vector present (length 3, all zeros at rest).
        assert ball.velocity_xyz is not None
        assert ball.velocity_xyz.shape == (3,)
        assert np.allclose(ball.velocity_xyz, [0.0, 0.0, 0.0], atol=1e-5)

    def test_window_region_is_a_site_with_warm_flag(self, living_room_model_data) -> None:
        m, d = living_room_model_data
        s = extract_scene_state(m, d)
        win = s.by_name["window_region"]
        assert win.kind == "site"
        assert win.flags.warm is True
        assert win.velocity_xyz is None
        assert np.allclose(win.pos_xyz[:2], [-2.0, 0.0], atol=1e-3)

    def test_filter_by_play_target(self, living_room_model_data) -> None:
        m, d = living_room_model_data
        s = extract_scene_state(m, d)
        names = [e.name for e in s.filter(play_target=True)]
        assert names == ["ball"]

    def test_filter_by_elevated_and_soft(self, living_room_model_data) -> None:
        m, d = living_room_model_data
        s = extract_scene_state(m, d)
        names = [e.name for e in s.filter(elevated=True, soft=True)]
        assert names == ["couch"]

    def test_filter_excludes_unmatched(self, living_room_model_data) -> None:
        m, d = living_room_model_data
        s = extract_scene_state(m, d)
        assert s.filter(elevated=True, warm=True) == []  # no entity has both

    def test_custom_flag_config_overrides_default(self, living_room_model_data) -> None:
        """Passing a different flag config retags entities; un-flagged ones drop."""
        m, d = living_room_model_data
        custom = {
            ("body", "couch"): SemanticFlags(warm=True),  # promote couch to warm only
            ("body", "ball"): SemanticFlags(),  # untag ball → dropped
        }
        s = extract_scene_state(m, d, flag_config=custom)
        names = sorted(e.name for e in s.entities)
        assert names == ["couch"]  # ball untagged, window not in config, only couch remains


class TestSemanticFlags:
    def test_any_returns_false_when_all_off(self) -> None:
        assert SemanticFlags().any() is False

    def test_any_returns_true_when_one_on(self) -> None:
        assert SemanticFlags(elevated=True).any() is True
        assert SemanticFlags(play_target=True).any() is True


class TestSceneStateContainer:
    def test_by_name_is_built_automatically(self) -> None:
        entities = [
            SceneEntity(
                name="x",
                kind="body",
                pos_xyz=np.zeros(3, dtype=np.float32),
                velocity_xyz=None,
                flags=SemanticFlags(soft=True),
            )
        ]
        s = SceneState(entities=entities)
        assert "x" in s.by_name
        assert s.by_name["x"].flags.soft is True
