"""BrainEnv -- scaffold env for prototyping the Tier 3 brain without GPU.

This env lets the brain (mode policy, body policy, gaze policy, reward
streams) be developed and unit-tested WITHOUT a trained Tier 1 motor cortex
or physical simulation:

  - Tier 1 is stubbed by KinematicCat: a LocomotionCommand is integrated
    directly into world-frame position over dt. No joint torques, no
    contact, no balance.
  - Tier 2 skills are dispatched normally via SkillRegistry.
  - Mood drifts normally.
  - Scene comes from assets/scenes/living_room.xml (cat is NOT included;
    KinematicCat is the cat).
  - Rewards via brain.rewards.compute_composite_reward.

Action space:
  - Index 0 is HOLD (pause-as-default): do not switch active_skill,
    continue emitting the prior pose (or neutral stand if none).
  - Indices 1..N map to SKILL_NAMES[i-1].

Optional behavioral-attractor layer: if cfg.mode_policy is set, every step
ticks the mode policy and incoming actions are masked to the active mode's
permissible skills (actions outside the mask fall back to HOLD).

Optional ICM curiosity (cfg.curiosity_enabled): the env instantiates a
CuriosityReward, projects each obs dict to a fixed-shape feature vector
(see brain/curiosity_feat.py), and:
  - feeds the intrinsic reward into compute_composite_reward(curiosity_value=)
  - exposes the per-step (feat_t, action, feat_tp1) Transition in
    info["curiosity_transition"] for a trainer to batch into .update()
With curiosity_enabled=False (default) the env is byte-identical in its
reward dict to the pre-curiosity behavior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from robot_pet_cat.scene import (
    LIVING_ROOM_V0_FLAGS,
    SceneState,
    extract_scene_state,
)
from robot_pet_cat.skills.skill_policy import LocomotionCommand
from robot_pet_cat.skills.skill_registry import SKILL_NAMES, SkillRegistry

from .attractor import Attractor, ModePolicy
from .curiosity_feat import CURIOSITY_OBS_DIM_V0, obs_dict_to_curiosity_vec
from .mood import Mood, MoodConfig
from .rewards import (
    CatState,
    ComfortReward,
    CompositeRewardConfig,
    CuriosityReward,
    HoldBonusReward,
    PlayReward,
    Transition,
    compute_composite_reward,
)

DEFAULT_SCENE_XML = (
    Path(__file__).resolve().parents[3] / "assets" / "scenes" / "living_room.xml"
)
DEFAULT_PHYSICS_SCENE_XML = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "go1_scenes"
    / "living_room_with_go1.xml"
)

HOLD_ACTION = 0


# --------------------------------------------------------------------------- #
# KinematicCat -- stub Tier 1
# --------------------------------------------------------------------------- #


@dataclass
class KinematicCat:
    """Integrates LocomotionCommand into world-frame state. No physics."""

    xy: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    yaw: float = 0.0
    body_height: float = 0.30
    last_speed: float = 0.0

    def reset(
        self,
        xy: tuple = (0.0, 0.0),
        yaw: float = 0.0,
        body_height: float = 0.30,
    ) -> None:
        self.xy = np.asarray(xy, dtype=np.float32)
        self.yaw = float(yaw)
        self.body_height = float(body_height)
        self.last_speed = 0.0

    def step(self, cmd: LocomotionCommand, dt: float) -> None:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        vx_world = cmd.vx * c - cmd.vy * s
        vy_world = cmd.vx * s + cmd.vy * c
        self.xy = self.xy + np.asarray([vx_world, vy_world], dtype=np.float32) * dt
        self.yaw = float(self.yaw + cmd.yaw_rate * dt)
        self.yaw = (self.yaw + math.pi) % (2.0 * math.pi) - math.pi
        self.body_height = float(cmd.body_height)
        self.last_speed = float(math.hypot(cmd.vx, cmd.vy))


# --------------------------------------------------------------------------- #
# Default skill-target picker
# --------------------------------------------------------------------------- #


def _nearest_entity(scene_state: SceneState, cat_xy, **flag_filter):
    candidates = scene_state.filter(**flag_filter)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda e: float(np.hypot(e.pos_xyz[0] - cat_xy[0], e.pos_xyz[1] - cat_xy[1])),
    )


def pick_default_goal(skill_name: str, scene_state: SceneState, cat_xy: np.ndarray):
    """Choose a sensible goal for a skill from scene + cat position.

    2026-05-25: post-Go1 pivot, every skill in the canonical set is
    either a static posture (crouch, lie_belly, get_up) or pure
    velocity locomotion (walk_normal, walk_slow). None of them need an
    xy target or a per-entity goal — None is the right answer for all.
    2026-05-27: lie_side removed.
    """
    del scene_state, cat_xy  # unused after pivot
    if skill_name in ("get_up", "walk_normal", "walk_slow",
                       "crouch", "lie_belly", "stay"):
        return None
    raise KeyError(f"no default goal recipe for skill {skill_name!r}")


# --------------------------------------------------------------------------- #
# BrainEnvConfig
# --------------------------------------------------------------------------- #


@dataclass
class BrainEnvConfig:
    scene_xml: Path = DEFAULT_SCENE_XML
    dt_s: float = 0.05
    max_steps_per_episode: int = 400
    initial_cat_xy: tuple = (0.0, 0.0)
    initial_cat_yaw: float = 0.0
    initial_body_height: float = 0.30
    flag_config: Optional[dict] = None
    movable_bodies: tuple = ("ball",)
    mode_policy: Optional[ModePolicy] = None
    # Soft action mask under attractor modes. Probability of letting an
    # out-of-mode action through INSTEAD of forcing it to HOLD. 0.0 ==
    # original hard-mask behavior (v0). ~0.3 lets the cat occasionally
    # do something off-mode -- per memory cats-are-stochastic, this is
    # more cat-like than a hard gate.
    mode_soft_pass: float = 0.3
    # RNG seed for the mask-softening coin flip. None => use env state.
    mode_soft_rng_seed: Optional[int] = None
    # Stochastic action-repeat / decision-period. After each decision,
    # the env samples a new period uniformly in
    # [decision_period_min_steps, decision_period_max_steps] and uses
    # that as the gap until the next decision. In between it repeats
    # the last-dispatched action. Defaults of min=max=1 preserve the
    # original per-step behavior. Range ~[40, 300] with dt=0.05 means
    # the cat commits to each skill for 2-15 sim-seconds (mean ~8.5s),
    # with the *variability itself* coming from a uniform draw -- no
    # metronome, no enforcement, just a stochastic decision cadence.
    decision_period_min_steps: int = 1
    decision_period_max_steps: int = 1
    # Deprecated: enforcement-style skill commitment. Kept for
    # backwards compat; prefer the decision_period_* knobs. 0 = off.
    min_skill_duration_s: float = 0.0
    # ICM curiosity wiring. Off by default -- when enabled, the env owns a
    # CuriosityReward, projects obs->feature each step, feeds intrinsic
    # reward to the composer, and emits Transitions in info for a trainer
    # to batch into .update(). curiosity_w stays controlled via
    # CompositeRewardConfig.curiosity_w (so wiring vs. signal weight are
    # separate knobs).
    curiosity_enabled: bool = False
    curiosity_seed: Optional[int] = None
    curiosity_lr: float = 1.0e-3
    curiosity_feature_dim: int = 64
    curiosity_hidden: int = 128
    curiosity_beta: float = 0.2
    curiosity_reward_scale: float = 0.5
    curiosity_device: str = "cpu"
    # Tier 1 backend. When False (default), Tier 1 is stubbed by
    # KinematicCat (the legacy fast-iter path). When True, BrainEnv compiles
    # the living-room-with-go2 scene and drives the Go2 with a trained
    # walker policy (PhysicsCat). The mode switch is binary at construction
    # time -- no live swapping.
    use_physics_cat: bool = False
    # Path to the merged scene XML used in physics mode. The default
    # living_room_with_go1.xml lives under data/go1_scenes/ so its
    # <include file="go1.xml"/> resolves through go1_base.get_assets().
    physics_scene_xml: Path = DEFAULT_PHYSICS_SCENE_XML
    # Path to the trained Unitree velocity walker policy. Accepts an
    # .onnx file (preferred), a .pt RSL-RL checkpoint, or a directory
    # containing either. Required when use_physics_cat=True.
    walker_policy_path: Optional[Path] = None
    # Paths to optional specialty policies (42-dim obs, no command/phase).
    hind_sit_policy_path: Optional[Path] = None
    get_up_policy_path: Optional[Path] = None
    # Additional mjlab Go1 checkpoints (all derived from Mjlab-*-Flat-Unitree-Go1).
    walker_slow_policy_path: Optional[Path] = None
    crouch_policy_path: Optional[Path] = None
    lie_belly_policy_path: Optional[Path] = None
    # Spawn height for the Go2 base body at reset. 0.32 matches the
    # init_state z in unitree_rl_mjlab's training config.
    physics_spawn_z: float = 0.32
    # Cat-eye camera observation. When True (and use_physics_cat=True), BrainEnv
    # renders a tiny cat_eye frame each step and adds "cat_eye_img" (H x W float32
    # [0,1] grayscale) to the obs dict. This is the raw pixel signal that feeds
    # the v1 curiosity feature vector and the visual obs space.
    use_cat_eye_obs: bool = False
    cat_eye_render_width: int = 16
    cat_eye_render_height: int = 16


# --------------------------------------------------------------------------- #
# BrainEnv
# --------------------------------------------------------------------------- #


class BrainEnv:
    """Scaffold env: scene + KinematicCat + skills + mood + rewards."""

    def __init__(self, cfg: Optional[BrainEnvConfig] = None) -> None:
        import mujoco  # lazy import

        self.cfg = cfg or BrainEnvConfig()
        self._mujoco = mujoco

        # Tier 1 backend: physics walker or kinematic stub.
        if self.cfg.use_physics_cat:
            if self.cfg.walker_policy_path is None:
                raise ValueError(
                    "use_physics_cat=True requires walker_policy_path "
                    "(path to .onnx, .pt, or run directory)."
                )
            # Compile merged scene with menagerie's Go2 assets in-memory.
            from robot_pet_cat.brain.physics_cat import (
                PhysicsCat,
                PhysicsCatConfig,
                inject_cat_eye_camera,
            )
            from robot_pet_cat.motion.go1_base import get_assets

            xml_str = Path(self.cfg.physics_scene_xml).read_text()
            assets = inject_cat_eye_camera(get_assets())
            self.mj_model = mujoco.MjModel.from_xml_string(xml_str, assets=assets)
            self.mj_data = mujoco.MjData(self.mj_model)
            self.cat = PhysicsCat(
                self.mj_model,
                self.cfg.walker_policy_path,
                PhysicsCatConfig(spawn_z=self.cfg.physics_spawn_z),
                hind_sit_policy=self.cfg.hind_sit_policy_path,
                get_up_policy=self.cfg.get_up_policy_path,
                walker_slow_policy=self.cfg.walker_slow_policy_path,
                crouch_policy=self.cfg.crouch_policy_path,
                lie_belly_policy=self.cfg.lie_belly_policy_path,
            )
        else:
            self.mj_model = mujoco.MjModel.from_xml_path(str(self.cfg.scene_xml))
            self.mj_data = mujoco.MjData(self.mj_model)
            self.cat = KinematicCat()

        # Optional cat-eye observer (physics mode only).
        self._cat_eye_renderer = None
        self._cat_eye_cam_id: int = -1
        if self.cfg.use_cat_eye_obs and self.cfg.use_physics_cat:
            self._cat_eye_renderer = mujoco.Renderer(
                self.mj_model,
                width=self.cfg.cat_eye_render_width,
                height=self.cfg.cat_eye_render_height,
            )
            self._cat_eye_cam_id = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "cat_eye"
            )

        self.mood = Mood(MoodConfig())
        self.registry = SkillRegistry()
        self.mode_policy: Optional[ModePolicy] = self.cfg.mode_policy
        self.flag_config = self.cfg.flag_config or LIVING_ROOM_V0_FLAGS
        self.movable_bodies = set(self.cfg.movable_bodies)
        self.r_comfort = ComfortReward()
        self.r_play = PlayReward()
        self.r_hold = HoldBonusReward()
        self.r_cfg = CompositeRewardConfig()

        # Optional ICM curiosity.
        self.curiosity: Optional[CuriosityReward] = None
        if self.cfg.curiosity_enabled:
            self.curiosity = CuriosityReward(
                obs_dim=CURIOSITY_OBS_DIM_V0,
                n_actions=self.action_space_n,
                feature_dim=self.cfg.curiosity_feature_dim,
                hidden=self.cfg.curiosity_hidden,
                beta=self.cfg.curiosity_beta,
                reward_scale=self.cfg.curiosity_reward_scale,
                lr=self.cfg.curiosity_lr,
                device=self.cfg.curiosity_device,
                seed=self.cfg.curiosity_seed,
            )
        self._prev_feat_vec: Optional[np.ndarray] = None

        # Runtime state
        self.active_skill_name: Optional[str] = None
        self.time_in_skill: float = 0.0
        self.t_sim: float = 0.0
        self.episode_step: int = 0
        self.last_reward_breakdown: dict = {}
        # Soft-mask RNG. Independent of policy stochasticity so the
        # softened mask is reproducible per env seed.
        self._mode_soft_rng = np.random.default_rng(self.cfg.mode_soft_rng_seed)
        # Action-repeat state. _steps_until_decision counts down each
        # step; when <=0 we consult the policy's input action and
        # sample a new period in [min_steps, max_steps]. Initialized
        # to 0 so step 0 is always a decision.
        self._last_action: int = HOLD_ACTION
        self._steps_until_decision: int = 0
        # Joint id lookup for free-joint bodies
        self._free_joint_qvel_offset: dict = {}
        for body_name in self.movable_bodies:
            jname = self._find_free_joint_for_body(body_name)
            if jname is not None:
                jid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                if jid >= 0:
                    self._free_joint_qvel_offset[body_name] = int(self.mj_model.jnt_dofadr[jid])

    @property
    def action_space_n(self) -> int:
        """0 = hold; 1..N = skill[i-1]."""
        return 1 + len(self.registry)

    @property
    def skill_names(self):
        return SKILL_NAMES

    def reset(self) -> dict:
        if self.cfg.use_physics_cat:
            # PhysicsCat owns mj_resetData / mj_forward and the leg pose.
            self.cat.reset(
                self.mj_data,
                xy=self.cfg.initial_cat_xy,
                yaw=self.cfg.initial_cat_yaw,
                body_height=self.cfg.physics_spawn_z,
            )
        else:
            self._mujoco.mj_resetData(self.mj_model, self.mj_data)
            self._mujoco.mj_forward(self.mj_model, self.mj_data)
            self.cat.reset(
                xy=self.cfg.initial_cat_xy,
                yaw=self.cfg.initial_cat_yaw,
                body_height=self.cfg.initial_body_height,
            )
        self.active_skill_name = None
        self.time_in_skill = 0.0
        self.t_sim = 0.0
        self.episode_step = 0
        self.last_reward_breakdown = {}
        self._last_action = HOLD_ACTION
        self._steps_until_decision = 0
        obs = self._observe()
        # Seed the prev-feature buffer so the first step has a feat_t.
        if self.curiosity is not None:
            self._prev_feat_vec = obs_dict_to_curiosity_vec(obs)
        else:
            self._prev_feat_vec = None
        return obs

    def step(self, action: int):
        """Returns (obs, reward, terminated, truncated, info)."""
        info: dict = {}
        if not 0 <= action < self.action_space_n:
            raise ValueError(f"action {action} out of range [0, {self.action_space_n})")

        # Stochastic action-repeat. Most steps, we ignore the policy's
        # input and repeat the last-dispatched action. On decision steps
        # (chosen at random intervals), we accept the new input and
        # sample the next gap. With min=max=1 this is a no-op (every
        # step is a decision step).
        period_min = self.cfg.decision_period_min_steps
        period_max = self.cfg.decision_period_max_steps
        if period_min > 1 or period_max > 1:
            if self._steps_until_decision <= 0:
                # Decision step: use the input action, sample next gap.
                self._last_action = int(action)
                if period_min >= period_max:
                    self._steps_until_decision = int(period_min)
                else:
                    self._steps_until_decision = int(
                        self._mode_soft_rng.integers(period_min, period_max + 1)
                    )
                info["action_decision_step"] = True
            else:
                action = self._last_action
                info["action_repeated"] = True
            self._steps_until_decision -= 1

        # Optional mode-mask. If active mode forbids the chosen skill, fall
        # back to HOLD -- but only with probability (1 - mode_soft_pass), so
        # the cat occasionally still does an off-mode action. See memory
        # cats-are-stochastic: a hard gate is wrong for cat-vibe.
        if self.mode_policy is not None:
            mask = self.mode_policy.allowed_action_mask(self.action_space_n, SKILL_NAMES)
            if not bool(mask[action]):
                info["dispatched_raw"] = (
                    "hold" if action == HOLD_ACTION else SKILL_NAMES[action - 1]
                )
                # Hard-blocked skills ignore soft-pass entirely. A cat in
                # RESTING mode never walks across the room regardless of the
                # stochastic gate -- that breaks mode semantics completely.
                _skill_name = SKILL_NAMES[action - 1] if action != HOLD_ACTION else None
                _hard_blocked = (
                    _skill_name is not None
                    and self.mode_policy.is_hard_blocked(_skill_name)
                )
                if _hard_blocked:
                    info["dispatch_masked"] = True
                    info["dispatch_hard_blocked"] = True
                    action = HOLD_ACTION
                elif float(self.cfg.mode_soft_pass) >= 1.0:
                    # Soft-pass disabled the gate entirely
                    info["dispatch_masked"] = False
                    info["dispatch_soft_passed"] = True
                elif self._mode_soft_rng.random() < float(self.cfg.mode_soft_pass):
                    # Let this out-of-mode action through this time
                    info["dispatch_masked"] = False
                    info["dispatch_soft_passed"] = True
                else:
                    info["dispatch_masked"] = True
                    action = HOLD_ACTION

        if action == HOLD_ACTION:
            # 2026-05-26: HOLD action now binds to the `stay` skill so the
            # body holds its current pose AND the HUD / skill tracking layer
            # show "stay" (the visible behavior) rather than the brain-layer
            # "(hold)" no-op. The brain still emits action 0 as before; the
            # env interprets it as "dispatch stay" instead of "no skill".
            new_skill_name = "stay"
            info["dispatched"] = "hold"
        else:
            new_skill_name = SKILL_NAMES[action - 1]
            info["dispatched"] = new_skill_name

        # Skill-commitment gate: a non-HOLD skill, once dispatched, persists
        # for at least cfg.min_skill_duration_s before the policy can switch
        # to a DIFFERENT non-HOLD skill. HOLD always interrupts.
        # See memory cats-are-stochastic — high entropy at the per-step
        # policy level looks twitchy without this commitment.
        if (
            self.cfg.min_skill_duration_s > 0.0
            and self.active_skill_name is not None
            and new_skill_name is not None  # HOLD is allowed to interrupt
            and new_skill_name != self.active_skill_name
            and self.time_in_skill < self.cfg.min_skill_duration_s
        ):
            info["dispatch_committed_to"] = self.active_skill_name
            info["dispatch_overridden"] = new_skill_name
            new_skill_name = self.active_skill_name  # force-continue

        # Skill switch
        if new_skill_name is not None and new_skill_name != self.active_skill_name:
            if self.active_skill_name is not None:
                self.registry.reset(self.active_skill_name)
            self.registry.reset(new_skill_name)
            self.active_skill_name = new_skill_name
            self.time_in_skill = 0.0
    
        # Get command this tick
        cmd = self._command_from_active_skill()

        if self.cfg.use_physics_cat:
            if self.active_skill_name == "jump_to":
                self._maybe_apply_jump_impulse()
            self.cat.step(self.mj_data, cmd, self.cfg.dt_s)
        else:
            # Kinematic stub: integrate the command directly, then fake
            # ball motion via qvel injection + manual decay + mj_forward
            # (no mj_step is called in this path).
            self.cat.step(cmd, self.cfg.dt_s)
            self._decay_ball_velocity(decay_per_s=0.9)
            self._mujoco.mj_forward(self.mj_model, self.mj_data)

        self.time_in_skill += self.cfg.dt_s
        self.t_sim += self.cfg.dt_s
        self.episode_step += 1

        # Update mood and (if present) mode policy
        self.mood.update(self.cfg.dt_s)
        if self.mode_policy is not None:
            self.mode_policy.tick(self.cfg.dt_s, self.mood)

        # New observation
        new_obs = self._observe()

        # ICM: compute intrinsic reward (no-grad) and emit transition
        curiosity_value = 0.0
        if self.curiosity is not None and self._prev_feat_vec is not None:
            feat_tp1 = obs_dict_to_curiosity_vec(new_obs)
            curiosity_value = self.curiosity.intrinsic_reward(
                self._prev_feat_vec, int(action), feat_tp1
            )
            info["curiosity_transition"] = Transition(
                feat_t=self._prev_feat_vec.copy(),
                action=int(action),
                feat_tp1=feat_tp1.copy(),
            )
            self._prev_feat_vec = feat_tp1

        # Reward
        scene_state = self._extract_scene_state()
        cat_state = self._cat_state_for_rewards()
        breakdown = compute_composite_reward(
            scene_state,
            cat_state,
            self.mood,
            comfort=self.r_comfort,
            play=self.r_play,
            hold=self.r_hold,
            cfg=self.r_cfg,
            curiosity_value=curiosity_value,
        )
        self.last_reward_breakdown = breakdown
        reward = breakdown["total"]

        truncated = self.episode_step >= self.cfg.max_steps_per_episode
        terminated = False
        info["reward_breakdown"] = breakdown
        return new_obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #

    def _observe(self) -> dict:
        scene_state = self._extract_scene_state()
        obs = {
            "mood": self.mood.state.copy(),
            "cat_xy": self.cat.xy.copy(),
            "cat_yaw": float(self.cat.yaw),
            "cat_body_height": float(self.cat.body_height),
            "cat_speed": float(self.cat.last_speed),
            "active_skill_idx": (
                self.registry.index_of(self.active_skill_name)
                if self.active_skill_name is not None
                else -1
            ),
            "time_in_skill": float(self.time_in_skill),
            "t_sim": float(self.t_sim),
            "scene_state": scene_state,
        }
        if self.mode_policy is not None:
            obs["mode"] = self.mode_policy.current_mode.value
            obs["action_mask"] = self.mode_policy.allowed_action_mask(
                self.action_space_n, SKILL_NAMES
            )
        # Cat-eye pixel observation (optional, physics mode only).
        if self._cat_eye_renderer is not None and self._cat_eye_cam_id >= 0:
            self._cat_eye_renderer.update_scene(
                self.mj_data, camera=self._cat_eye_cam_id
            )
            img_rgb = self._cat_eye_renderer.render()  # H x W x 3  uint8
            obs["cat_eye_img"] = (img_rgb.mean(axis=2) / 255.0).astype(np.float32)
        return obs

    def _extract_scene_state(self) -> SceneState:
        return extract_scene_state(
            self.mj_model,
            self.mj_data,
            flag_config=self.flag_config,
            movable_bodies=self.movable_bodies,
        )

    def _cat_state_for_rewards(self) -> CatState:
        return CatState(
            xy=self.cat.xy.copy(),
            yaw=self.cat.yaw,
            body_height=self.cat.body_height,
            speed=self.cat.last_speed,
            active_skill=self.active_skill_name,
            time_in_skill=self.time_in_skill,
        )

    # ------------------------------------------------------------------ #
    # Skill dispatch
    # ------------------------------------------------------------------ #

    def _command_from_active_skill(self) -> LocomotionCommand:
        if self.active_skill_name is None:
            # HOLD (action 0) = no skill active. Freeze in current pose
            # rather than driving the walker to its default standing pose
            # (which moves the legs noticeably). Equivalent to dispatching
            # `stay` without committing the brain to that skill name --
            # active_skill_name stays None for the HUD / reward layer.
            return LocomotionCommand(gait="stay")
        scene_state = self._extract_scene_state()
        goal = pick_default_goal(self.active_skill_name, scene_state, self.cat.xy)
        skill_obs = self._skill_obs_dict()
        try:
            return self.registry.step(self.active_skill_name, skill_obs, goal)
        except NotImplementedError:
            # jump_to until trained. Fall back to neutral stand and clear active.
            self.active_skill_name = None
            self.time_in_skill = 0.0
            return LocomotionCommand(gait="stand")

    def _skill_obs_dict(self) -> dict:
        return {
            "root_xy": self.cat.xy.copy(),
            "root_yaw": float(self.cat.yaw),
            "t_sim": float(self.t_sim),
            "dt": float(self.cfg.dt_s),
        }

    # ------------------------------------------------------------------ #
    # Ball state hacks
    # ------------------------------------------------------------------ #

    def _maybe_apply_jump_impulse(self) -> None:
        """Intercept JumpTo LAUNCH phase: inject ballistic impulse into Go2 base."""
        from robot_pet_cat.skills.jump_to import JumpTo
        skill = self.registry.get("jump_to")
        if not isinstance(skill, JumpTo) or not skill.launch_requested:
            return
        if self.cfg.use_physics_cat:
            self.cat.apply_launch_impulse(self.mj_data, skill.target_xyz)
        else:
            # KinematicCat: teleport directly onto the platform.
            tx, ty, tz = skill.target_xyz
            self.cat.xy = np.array([tx, ty], dtype=np.float32)
            self.cat.body_height = tz
        skill.advance_to_airborne()


    def _decay_ball_velocity(self, decay_per_s: float = 0.9) -> None:
        """Apply rolling friction to all ball joints each step."""
        import mujoco
        for jnt_id in range(self.mj_model.njnt):
            jnt_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id) or ""
            if not jnt_name.startswith("ball_") or not jnt_name.endswith("_joint"):
                continue
            vadr = self.mj_model.jnt_dofadr[jnt_id]
            vel = self.mj_data.qvel[vadr : vadr + 3]
            speed = float(np.linalg.norm(vel))
            if speed > 1e-4:
                friction = min(speed, self.cfg.ball_rolling_friction * self.cfg.physics_dt)
                self.mj_data.qvel[vadr : vadr + 3] = vel * (1.0 - friction / speed)
            else:
                self.mj_data.qvel[vadr : vadr + 3] = 0.0

    def _find_free_joint_for_body(self, body_name: str) -> str:
        """Return the name of the free joint attached to body_name, or empty string."""
        import mujoco
        body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            return ""
        for j in range(self.mj_model.njnt):
            if self.mj_model.jnt_bodyid[j] == body_id and self.mj_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                return mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        return ""
