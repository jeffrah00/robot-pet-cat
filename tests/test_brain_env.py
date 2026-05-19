"""Tests for the BrainEnv scaffold (src/robot_pet_cat/brain/env.py).

Auto-skips if mujoco isn't installed, since the env loads an MJCF scene.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco", reason="mujoco not installed in this env")

from robot_pet_cat.brain.attractor import (  # noqa: E402
    Attractor,
    ModePolicy,
    ModePolicyConfig,
)
from robot_pet_cat.brain.env import (  # noqa: E402
    HOLD_ACTION,
    BrainEnv,
    BrainEnvConfig,
    KinematicCat,
    pick_default_goal,
)
from robot_pet_cat.skills.skill_policy import LocomotionCommand  # noqa: E402


# --------------------------------------------------------------------------- #
# KinematicCat
# --------------------------------------------------------------------------- #


class TestKinematicCat:
    def test_forward_velocity_integrates_x_in_world(self) -> None:
        cat = KinematicCat()
        cmd = LocomotionCommand(vx=1.0, vy=0.0, yaw_rate=0.0)
        cat.step(cmd, dt=0.5)
        assert cat.xy[0] == pytest.approx(0.5, abs=1e-6)
        assert cat.xy[1] == pytest.approx(0.0, abs=1e-6)

    def test_yaw_rotates_velocity_into_world(self) -> None:
        """Cat facing +y, body-frame vx=1 => world +y motion."""
        cat = KinematicCat(yaw=np.pi / 2)
        cmd = LocomotionCommand(vx=1.0, vy=0.0)
        cat.step(cmd, dt=1.0)
        assert cat.xy[0] == pytest.approx(0.0, abs=1e-5)
        assert cat.xy[1] == pytest.approx(1.0, abs=1e-5)

    def test_yaw_wraps_to_pi_range(self) -> None:
        cat = KinematicCat(yaw=3.0)
        cmd = LocomotionCommand(yaw_rate=1.0)
        cat.step(cmd, dt=1.0)
        # 4.0 rad would be >pi; should wrap.
        assert -np.pi <= cat.yaw < np.pi

    def test_body_height_passthrough(self) -> None:
        cat = KinematicCat()
        cat.step(LocomotionCommand(body_height=0.15), dt=0.05)
        assert cat.body_height == 0.15


# --------------------------------------------------------------------------- #
# pick_default_goal
# --------------------------------------------------------------------------- #


class TestPickDefaultGoal:
    def test_static_poses_get_no_goal(self) -> None:
        env = BrainEnv()
        env.reset()
        scene = env._extract_scene_state()
        cat_xy = env.cat.xy
        for name in ("sit", "lie_down", "crouch"):
            assert pick_default_goal(name, scene, cat_xy) is None

    def test_walk_to_targets_nearest_play_target(self) -> None:
        env = BrainEnv()
        env.reset()
        scene = env._extract_scene_state()
        goal = pick_default_goal("walk_to", scene, env.cat.xy)
        # Ball is at (0, 1) in the v0 scene.
        assert tuple(goal.target_xy) == pytest.approx((0.0, 1.0), abs=1e-3)


# --------------------------------------------------------------------------- #
# BrainEnv basics
# --------------------------------------------------------------------------- #


class TestBrainEnvBasics:
    def test_action_space_n_equals_one_plus_skills(self) -> None:
        env = BrainEnv()
        assert env.action_space_n == 1 + len(env.skill_names)

    def test_reset_returns_observation_dict(self) -> None:
        env = BrainEnv()
        obs = env.reset()
        for key in ("mood", "cat_xy", "cat_yaw", "cat_body_height", "cat_speed",
                    "active_skill_idx", "time_in_skill", "t_sim", "scene_state"):
            assert key in obs, f"missing obs key {key}"

    def test_hold_does_nothing_first_tick(self) -> None:
        env = BrainEnv()
        env.reset()
        obs, r, term, trunc, info = env.step(HOLD_ACTION)
        assert info["dispatched"] == "hold"
        assert obs["active_skill_idx"] == -1
        assert obs["cat_xy"].tolist() == [0.0, 0.0]

    def test_walk_to_moves_cat(self) -> None:
        env = BrainEnv()
        env.reset()
        # walk_to is action index 1 (SKILL_NAMES[0] = walk_to).
        for _ in range(20):
            env.step(1)
        assert np.linalg.norm(env.cat.xy) > 0.05

    def test_jump_to_falls_back_to_hold(self) -> None:
        """jump_to raises NotImplementedError; env should catch + clear."""
        env = BrainEnv()
        env.reset()
        jump_to_idx = 1 + env.skill_names.index("jump_to")
        obs, r, term, trunc, info = env.step(jump_to_idx)
        # active_skill cleared after catch
        assert env.active_skill_name is None

    def test_max_steps_truncates(self) -> None:
        env = BrainEnv(BrainEnvConfig(max_steps_per_episode=3))
        env.reset()
        for i in range(3):
            obs, r, term, trunc, info = env.step(HOLD_ACTION)
        assert trunc is True
        assert term is False


# --------------------------------------------------------------------------- #
# Mode policy wiring
# --------------------------------------------------------------------------- #


class TestModePolicyWiring:
    def test_no_mode_policy_means_no_action_mask_in_obs(self) -> None:
        env = BrainEnv()
        obs = env.reset()
        assert "mode" not in obs
        assert "action_mask" not in obs

    def test_with_mode_policy_obs_carries_mode_and_mask(self) -> None:
        mp = ModePolicy(ModePolicyConfig(initial_mode=Attractor.RESTING))
        env = BrainEnv(BrainEnvConfig(mode_policy=mp))
        obs = env.reset()
        assert obs["mode"] == "resting"
        assert obs["action_mask"].shape == (env.action_space_n,)

    def test_forbidden_action_falls_back_to_hold(self) -> None:
        mp = ModePolicy(ModePolicyConfig(initial_mode=Attractor.PLAYING))
        env = BrainEnv(BrainEnvConfig(mode_policy=mp))
        env.reset()
        sit_idx = 1 + env.skill_names.index("sit")  # sit not in PLAYING mode
        obs, r, term, trunc, info = env.step(sit_idx)
        assert info.get("dispatch_masked") is True
        assert info["dispatched"] == "hold"
        assert info["dispatched_raw"] == "sit"

    def test_allowed_action_passes_through(self) -> None:
        mp = ModePolicy(ModePolicyConfig(initial_mode=Attractor.PLAYING))
        env = BrainEnv(BrainEnvConfig(mode_policy=mp))
        env.reset()
        swat_idx = 1 + env.skill_names.index("swat")  # swat IS in PLAYING
        obs, r, term, trunc, info = env.step(swat_idx)
        assert info.get("dispatch_masked") is None or info["dispatch_masked"] is False
        assert info["dispatched"] == "swat"


# --------------------------------------------------------------------------- #
# Swat-near-ball impulse (no-physics hack)
# --------------------------------------------------------------------------- #


class TestSwatImpulse:
    def test_swat_near_ball_imparts_velocity(self) -> None:
        env = BrainEnv(BrainEnvConfig(
            initial_cat_xy=(0.0, 0.95),  # 5cm from ball at (0, 1)
            swat_ball_impulse_mps=1.0,
        ))
        env.reset()
        swat_idx = 1 + env.skill_names.index("swat")
        obs, r, term, trunc, info = env.step(swat_idx)
        # Ball velocity should be nonzero in y direction (cat south of ball -> push north).
        ball = obs["scene_state"].by_name["ball"]
        assert ball.velocity_xyz is not None
        assert np.linalg.norm(ball.velocity_xyz[:2]) > 0.1
