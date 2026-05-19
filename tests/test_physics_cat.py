"""Smoke tests for the PhysicsCat path of BrainEnv.

These tests cover the wiring -- the merged scene compiles, PhysicsCat finds
all the joints/sensors it needs, BrainEnv switches cleanly between the
kinematic and physics backends -- without requiring a real walker checkpoint.
A `mock_policy` callable stands in for model_6400.pt; we trust the actual
locomotion to be validated on the pod via scripts/smoke_test_physics_cat.py.

Skipped unconditionally if the Go2 menagerie assets are not findable, so
sandboxes without the menagerie clone still pass the suite.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from robot_pet_cat.skills.skill_policy import LocomotionCommand


# Skip the whole module if menagerie isn't reachable -- the merged scene
# can't be compiled without it.
def _menagerie_available() -> bool:
    try:
        from robot_pet_cat.motion.go2_base import _menagerie_go2_dir
        _menagerie_go2_dir()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _menagerie_available(),
    reason="mujoco_menagerie/unitree_go2 not on this machine; "
    "run scripts/fetch_menagerie.py or set MUJOCO_PLAYGROUND_MENAGERIE.",
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _mock_zero_policy(obs: np.ndarray) -> np.ndarray:
    return np.zeros((obs.shape[0], 12), dtype=np.float32)


class _PatchLoader:
    """Swap load_go2_walker_policy in both modules so BrainEnv
    (use_physics_cat=True) doesn't actually try to load a file from disk."""

    def __enter__(self):
        import robot_pet_cat.brain.go2_policy as gp
        import robot_pet_cat.brain.physics_cat as pc

        fake = lambda *a, **kw: _mock_zero_policy
        self._gp = gp
        self._pc = pc
        self._orig_gp = gp.load_go2_walker_policy
        self._orig_pc = pc.load_go2_walker_policy
        gp.load_go2_walker_policy = fake
        pc.load_go2_walker_policy = fake
        return self

    def __exit__(self, *exc):
        self._gp.load_go2_walker_policy = self._orig_gp
        self._pc.load_go2_walker_policy = self._orig_pc


# --------------------------------------------------------------------------- #
# Scene compilation
# --------------------------------------------------------------------------- #


class TestSceneCompilation:
    def test_merged_scene_compiles(self) -> None:
        """The living-room-with-go2 XML compiles with menagerie assets."""
        import mujoco
        from robot_pet_cat.brain.env import DEFAULT_PHYSICS_SCENE_XML
        from robot_pet_cat.motion.go2_base import get_assets

        xml = DEFAULT_PHYSICS_SCENE_XML.read_text()
        model = mujoco.MjModel.from_xml_string(xml, assets=get_assets())

        # Robot: 1 base + 12 leg bodies.
        body_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            for i in range(model.nbody)
        ]
        assert "base" in body_names
        assert "couch" in body_names
        assert "ball" in body_names

        # 12 actuators for the Go2 legs.
        assert model.nu == 12


# --------------------------------------------------------------------------- #
# PhysicsCat directly
# --------------------------------------------------------------------------- #


class TestPhysicsCat:
    def _build(self):
        import mujoco
        from robot_pet_cat.brain.env import DEFAULT_PHYSICS_SCENE_XML
        from robot_pet_cat.brain.physics_cat import PhysicsCat, PhysicsCatConfig
        from robot_pet_cat.motion.go2_base import get_assets

        model = mujoco.MjModel.from_xml_string(
            DEFAULT_PHYSICS_SCENE_XML.read_text(), assets=get_assets()
        )
        data = mujoco.MjData(model)
        cat = PhysicsCat(model, _mock_zero_policy, PhysicsCatConfig())
        return model, data, cat

    def test_reset_places_base_correctly(self) -> None:
        model, data, cat = self._build()
        cat.reset(data, xy=(0.5, -0.3), yaw=1.0, body_height=0.32)
        assert cat.xy[0] == pytest.approx(0.5, abs=1e-5)
        assert cat.xy[1] == pytest.approx(-0.3, abs=1e-5)
        assert cat.body_height == pytest.approx(0.32, abs=1e-5)
        assert cat.yaw == pytest.approx(1.0, abs=1e-4)

    def test_observation_is_47_dim(self) -> None:
        model, data, cat = self._build()
        cat.reset(data)
        obs = cat._build_observation(data, np.zeros(3, dtype=np.float32))
        assert obs.shape == (47,)
        # Projected gravity at rest pose should point mostly down in body
        # frame (the base is approximately level).
        proj_g = obs[3:6]
        assert proj_g[2] < -0.95
        # Joint pos diffs should be ~0 at reset.
        assert np.max(np.abs(obs[11:23])) < 1e-4

    def test_step_advances_physics(self) -> None:
        model, data, cat = self._build()
        cat.reset(data)
        # Mock policy emits zero action -> joints get held at defaults via
        # PD controller. The robot should NOT teleport.
        cmd = LocomotionCommand(vx=0.0, vy=0.0, yaw_rate=0.0, gait="stand")
        for _ in range(5):
            cat.step(data, cmd, dt=0.05)
        # Allowed some drift (gravity pulls the unbalanced robot) but not
        # to outside the room.
        assert abs(cat.xy[0]) < 1.0
        assert abs(cat.xy[1]) < 1.0
        # Should still be above the floor.
        assert cat.body_height > 0.05

    def test_locomotion_command_to_twist_honors_stand(self) -> None:
        model, data, cat = self._build()
        cmd_stand = LocomotionCommand(vx=1.0, vy=0.0, yaw_rate=0.0, gait="stand")
        twist = cat._command_to_twist(cmd_stand)
        assert np.allclose(twist, [0.0, 0.0, 0.0])

    def test_locomotion_command_clamps_speed(self) -> None:
        model, data, cat = self._build()
        cmd_fast = LocomotionCommand(vx=5.0, vy=0.0, yaw_rate=0.0, gait="trot")
        twist = cat._command_to_twist(cmd_fast)
        # Default config clamps linear speed to 1.0 m/s.
        assert np.hypot(twist[0], twist[1]) == pytest.approx(1.0, abs=1e-5)

    def test_phase_clock_advances_only_when_moving(self) -> None:
        model, data, cat = self._build()
        cat.reset(data)
        # Stand: phase clock stays at 0.
        cat.step(data, LocomotionCommand(gait="stand"), dt=0.05)
        assert cat._phase_time == 0.0
        # Walk: phase clock advances.
        cat.step(data, LocomotionCommand(vx=0.5, gait="trot"), dt=0.05)
        assert cat._phase_time > 0.0


# --------------------------------------------------------------------------- #
# BrainEnv wiring
# --------------------------------------------------------------------------- #


class TestBrainEnvPhysicsBackend:
    def test_use_physics_cat_requires_policy_path(self) -> None:
        from robot_pet_cat.brain.env import BrainEnv, BrainEnvConfig

        with pytest.raises(ValueError, match="walker_policy_path"):
            BrainEnv(BrainEnvConfig(use_physics_cat=True))

    def test_reset_and_step_with_mock_policy(self) -> None:
        from robot_pet_cat.brain.env import BrainEnv, BrainEnvConfig

        with _PatchLoader():
            env = BrainEnv(
                BrainEnvConfig(
                    use_physics_cat=True, walker_policy_path="/dev/null"
                )
            )
            obs = env.reset()
            assert obs["cat_xy"].shape == (2,)
            # Couch + ball + window region survive in the merged scene.
            scene_names = {e.name for e in obs["scene_state"].entities}
            assert {"couch", "ball", "window_region"}.issubset(scene_names)

            # Step a handful of times; should not raise.
            for action in [0, 1, 1, 0]:
                obs, r, term, trunc, info = env.step(action)
            assert isinstance(r, float)
            assert env.episode_step == 4

    def test_kinematic_path_still_works(self) -> None:
        """Sanity: no menagerie required, no policy required."""
        from robot_pet_cat.brain.env import BrainEnv, BrainEnvConfig

        env = BrainEnv(BrainEnvConfig(use_physics_cat=False))
        obs = env.reset()
        assert obs["cat_xy"].shape == (2,)
        obs, r, term, trunc, info = env.step(0)
        assert isinstance(r, float)
