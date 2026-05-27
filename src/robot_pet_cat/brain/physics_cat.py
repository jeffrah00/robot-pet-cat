"""PhysicsCat -- Tier 1 motor cortex backed by a trained Go2 walker policy.

This is the drop-in replacement for KinematicCat. Same external interface
(`xy`, `yaw`, `body_height`, `last_speed`, `reset()`, `step(cmd, dt)`) but
underneath it actually drives the Go2 in MuJoCo:

  - Builds joint / body / sensor index lookups against the merged
    living-room-with-go2 scene.
  - Each PhysicsCat.step():
      1. Splits the brain dt (default 0.05 s) into physics substeps of
         walker dt 0.005 s.
      2. Every `decimation` (4) physics steps, builds the 48-dim observation
         and calls the walker policy to get the next 12-d action.
      3. Writes joint position targets (default + 0.25*action) into mj_data.ctrl.
      4. Calls mujoco.mj_step.
      5. After all substeps, refreshes cached pose properties.

LocomotionCommand mapping:
    cmd.vx / cmd.vy / cmd.yaw_rate go directly into the policy's
    `command` channel. cmd.gait == "stand" forces the command to zero
    so the policy enters its stand pose. cmd.body_height and cmd.head_*
    aren't part of the velocity walker's obs and are ignored at Tier 1
    (Tier 2 keyframes or a later head policy will handle them).

Reset:
    qpos for the base free joint = [x, y, z_spawn, quat_from_yaw(yaw)]
    qpos for each of the 12 leg joints = GO2_DEFAULT_JOINT_POS[i]
    qvel cleared. last_action cleared. phase clock cleared.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from robot_pet_cat.skills.skill_policy import LocomotionCommand

from .go2_policy import (
    GO2_ACTION_SCALE,
    GO2_DEFAULT_JOINT_POS,
    GO2_JOINT_NAMES,
    GO2_PHASE_PERIOD,
    GO2_PHASE_STAND_THRESHOLD,
    GO2_PHYSICS_DT,
    GO2_POLICY_DECIMATION,
    WalkerPolicy,
    load_go2_walker_policy,
    load_go1_walker_policy,
    load_hind_sit_policy,
    load_get_up_policy,
)


# Default base height at spawn (matches Unitree training init_state, m).
DEFAULT_BASE_SPAWN_Z = 0.278

# Name of the Go1 base body in menagerie's go1.xml.
GO1_BASE_BODY = "trunk"


def inject_cat_eye_camera(assets: dict[str, bytes]) -> dict[str, bytes]:
    """Patch the menagerie go1.xml in `assets` to mount a cat_eye camera
    onto the Go1 trunk body. Returns the same dict (mutated).

    MuJoCo's standalone XML does not let an outer file `re-open` an included
    body by name -- duplicate <body name="base"> raises XML Error. To attach
    a body-mounted POV camera, we inject the <camera> as the first child of
    the existing <body name="base"...> tag inside go1.xml's content as
    held in the assets dict, then compile from_xml_string with the patched
    asset.

    Idempotent: if the patched asset already has a cat_eye camera, no-op.
    """
    import re

    KEY = "go1.xml"
    if KEY not in assets:
        return assets
    text = assets[KEY].decode("utf-8")
    if "cat_eye" in text:
        return assets  # already patched

    # Pose: forward of trunk (+x by 0.30), slightly above (+z by 0.05).
    # xyaxes maps the camera's local X (right) and Y (up) so the camera's
    # -Z (look direction) is along the body's +x.
    cam_block = (
        '\n    <camera name="cat_eye" pos="0.30 0 0.05" '
        'xyaxes="0 -1 0 0 0 1" fovy="70"/>\n'
        '    <site name="nose" pos="0.30 0 0.0" size="0.01" '
        'type="sphere" group="2" rgba="1 0.5 0.5 0.6"/>'
    )
    new_text, n = re.subn(
        r'(<body name="trunk"[^>]*>)',
        r'\1' + cam_block,
        text,
        count=1,
    )
    if n == 0:
        raise RuntimeError(
            "Could not find <body name=\"trunk\" ...> in go1.xml to "
            "inject cat_eye camera."
        )
    assets[KEY] = new_text.encode("utf-8")
    return assets


@dataclass
class PhysicsCatConfig:
    """Knobs that don't depend on the model handle."""

    spawn_z: float = DEFAULT_BASE_SPAWN_Z
    action_scale: float = GO2_ACTION_SCALE
    decimation: int = GO2_POLICY_DECIMATION
    physics_dt: float = GO2_PHYSICS_DT
    phase_period: float = GO2_PHASE_PERIOD
    phase_stand_threshold: float = GO2_PHASE_STAND_THRESHOLD
    # Optional command-magnitude clamp before passing to the policy. Goes
    # well with Tier 2 skills that may emit unrealistic numbers during
    # debugging. None == no clamp.
    max_linear_speed: Optional[float] = 1.0
    max_yaw_rate: Optional[float] = 2.0


class PhysicsCat:
    """A Tier-1 motor cortex backed by the Unitree Go2 velocity walker."""

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        mj_model,
        policy: WalkerPolicy | str | Path,
        cfg: Optional[PhysicsCatConfig] = None,
        hind_sit_policy: "WalkerPolicy | str | Path | None" = None,
        get_up_policy: "WalkerPolicy | str | Path | None" = None,
        walker_slow_policy: "WalkerPolicy | str | Path | None" = None,
        crouch_policy: "WalkerPolicy | str | Path | None" = None,
        lie_belly_policy: "WalkerPolicy | str | Path | None" = None,
        lie_side_policy: "WalkerPolicy | str | Path | None" = None,
    ) -> None:
        import mujoco

        self._mujoco = mujoco
        self.cfg = cfg or PhysicsCatConfig()
        self.mj_model = mj_model

        # Resolve the walker policy. Accept either a pre-built callable or
        # a path -- the latter is the common case from configs.
        if callable(policy):
            self.policy: WalkerPolicy = policy
        else:
            self.policy = load_go1_walker_policy(policy)

        # Optional hind_sit policy: dispatched when a skill emits
        # gait="hind_sit". Same actuator pipeline as the walker (action *
        # 0.25 + default joint pos), but with a 42-dim obs that lacks the
        # walker's command + phase channels.
        if hind_sit_policy is None:
            self.hind_sit_policy = None
        elif callable(hind_sit_policy):
            self.hind_sit_policy = hind_sit_policy
        else:
            self.hind_sit_policy = load_hind_sit_policy(hind_sit_policy)

        # Optional get_up policy: dispatched when a skill emits gait="get_up".
        # Same 42-dim obs as hind_sit (no command/phase channels).
        if get_up_policy is None:
            self.get_up_policy = None
        elif callable(get_up_policy):
            self.get_up_policy = get_up_policy
        else:
            self.get_up_policy = load_get_up_policy(get_up_policy)

        # Optional slow walker: 48-dim like the normal walker but trained
        # with a slower command range (Mjlab-Velocity-Slow-Unitree-Go1).
        if walker_slow_policy is None:
            self.walker_slow_policy = None
        elif callable(walker_slow_policy):
            self.walker_slow_policy = walker_slow_policy
        else:
            self.walker_slow_policy = load_go1_walker_policy(walker_slow_policy)

        # Posture policies: 42-dim, same FR/FL/RR/RL MJCF joint order as
        # get_up, same action_scale=0.6 and settle_steps=25. Forks of the
        # getup task with different reward targets (mjlab_playground
        # _extra_tasks.py: Mjlab-{Crouch,LieBelly,LieSide}-Flat-Unitree-Go1).
        def _maybe_load_posture(p):
            if p is None: return None
            if callable(p): return p
            return load_get_up_policy(p)
        self.crouch_policy = _maybe_load_posture(crouch_policy)
        self.lie_belly_policy = _maybe_load_posture(lie_belly_policy)
        self.lie_side_policy = _maybe_load_posture(lie_side_policy)
        # ---- Index lookups against the compiled MuJoCo model ----------- #
        # Base body and its free-joint qpos/qvel offsets.
        base_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, GO1_BASE_BODY)
        if base_id < 0:
            raise RuntimeError(
                f"Go1 base body {GO1_BASE_BODY!r} not in compiled model. "
                "Did the scene <include> go1.xml?"
            )
        self._base_body_id = base_id
        # Find the free joint attached to the base body. There should be
        # exactly one.
        self._base_qpos_adr: int = -1
        self._base_qvel_adr: int = -1
        for j in range(int(mj_model.njnt)):
            if (
                mj_model.jnt_bodyid[j] == base_id
                and mj_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
            ):
                self._base_qpos_adr = int(mj_model.jnt_qposadr[j])
                self._base_qvel_adr = int(mj_model.jnt_dofadr[j])
                break
        if self._base_qpos_adr < 0:
            raise RuntimeError("No free joint on Go1 trunk body in compiled model.")

        # Leg joint qpos / qvel addresses, in GO2_JOINT_NAMES order.
        self._joint_qpos_adr = np.zeros(12, dtype=np.int32)
        self._joint_qvel_adr = np.zeros(12, dtype=np.int32)
        for i, name in enumerate(GO2_JOINT_NAMES):
            jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"Go2 joint {name!r} not in compiled model.")
            self._joint_qpos_adr[i] = int(mj_model.jnt_qposadr[jid])
            self._joint_qvel_adr[i] = int(mj_model.jnt_dofadr[jid])

        # Actuator ids, in the same joint order. Actuator names in
        # menagerie's go1.xml match the joint names without "_joint":
        # FR_hip_joint -> FR_hip. Try both spellings.
        self._actuator_ids = np.zeros(12, dtype=np.int32)
        for i, jname in enumerate(GO2_JOINT_NAMES):
            short = jname.replace("_joint", "")
            for cand in (jname, short):
                aid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, cand)
                if aid >= 0:
                    self._actuator_ids[i] = aid
                    break
            else:
                raise RuntimeError(
                    f"No actuator for joint {jname!r} (tried {jname}, {short})."
                )

        # Sensor lookups for IMU. base_ang_vel uses 'gyro'; projected
        # gravity is computed from the IMU site's z-axis (upvector).
        self._gyro_adr = self._sensor_adr("gyro", required=True)
        self._upvec_adr = self._sensor_adr("upvector", required=True)
        # Local linvel for last_speed reporting.
        self._linvel_adr = self._sensor_adr("local_linvel", required=False)

        # Runtime caches.
        self._last_action = np.zeros(12, dtype=np.float32)
        self._phase_time = 0.0
        # Pose cache, refreshed after each step.
        self._xy = np.zeros(2, dtype=np.float32)
        self._yaw = 0.0
        self._z = float(self.cfg.spawn_z)
        self._last_speed = 0.0
        # Last command issued (for the policy's `command` channel).
        self._last_cmd_arr = np.zeros(3, dtype=np.float32)
        # Track which policy was last active so we can zero last_action when
        # switching between walker and hind_sit (the two policies have
        # different action distributions; a stale last_action is OOD).
        self._last_active_policy = "walker"
        # Held joint targets for gait="stay" -- snapshotted on
        # activation, written every tick thereafter.
        self._stay_targets = np.zeros(12, dtype=np.float32)
        # Settle-step counter for the get_up policy (trained at 8 Hz,
        # settle_steps=25 vs walker decimation=4 at 50 Hz).
        self._get_up_substep_counter: int = 0

        # ---- v7/FR-Net adapter: foot-geom IDs for predicted_next_contact ---- #
        # When get_up_policy.obs_dim == 46 the policy expects 4 extra contact
        # bits appended (FL, FR, RL, RR). At training time those came from a
        # learned MassContactPredictor; at deployment we substitute the actual
        # foot-ground contacts from MuJoCo (the predictor was trained to
        # match those, so this is the limiting case of perfect prediction).
        # Geom-name lookup is best-effort with several common Go2 naming
        # variants; if none are found the adapter falls back to a zeros vector
        # rather than crashing.
        self._foot_geom_ids: list[int] = []
        foot_aliases = [
            ("FL_foot", "FL"), ("FR_foot", "FR"),
            ("RL_foot", "RL"), ("RR_foot", "RR"),
        ]
        for name, short in foot_aliases:
            gid = -1
            for cand in (name, name + "_collision", short + "_foot_collision", short):
                gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, cand)
                if gid >= 0:
                    break
            self._foot_geom_ids.append(int(gid))
        # Build the world-geom id list once (id=0 is typically the floor).
        # We don't strictly need to filter on it: a foot is "in contact"
        # whenever it's in any contact, which during normal locomotion is
        # nearly always with the floor.

    # ------------------------------------------------------------------ #
    # KinematicCat-compatible properties
    # ------------------------------------------------------------------ #

    @property
    def xy(self) -> np.ndarray:
        return self._xy.copy()

    @property
    def yaw(self) -> float:
        return float(self._yaw)

    @property
    def body_height(self) -> float:
        return float(self._z)

    @property
    def last_speed(self) -> float:
        return float(self._last_speed)

    # ------------------------------------------------------------------ #
    # Reset / step
    # ------------------------------------------------------------------ #

    def reset(
        self,
        mj_data,
        xy: tuple = (0.0, 0.0),
        yaw: float = 0.0,
        body_height: float = DEFAULT_BASE_SPAWN_Z,
    ) -> None:
        """Snap the Go2 to a clean pose. Call BEFORE the first env step."""
        mujoco = self._mujoco
        mujoco.mj_resetData(self.mj_model, mj_data)

        # Base free joint qpos = [x, y, z, qw, qx, qy, qz]
        qpos = mj_data.qpos
        qvel = mj_data.qvel
        adr = self._base_qpos_adr
        qpos[adr + 0] = float(xy[0])
        qpos[adr + 1] = float(xy[1])
        qpos[adr + 2] = float(body_height)
        qw = math.cos(0.5 * yaw)
        qz = math.sin(0.5 * yaw)
        qpos[adr + 3] = qw
        qpos[adr + 4] = 0.0
        qpos[adr + 5] = 0.0
        qpos[adr + 6] = qz
        # Clear base velocity
        vadr = self._base_qvel_adr
        qvel[vadr : vadr + 6] = 0.0

        # Default joint angles + zero joint velocities
        for i in range(12):
            qpos[self._joint_qpos_adr[i]] = float(GO2_DEFAULT_JOINT_POS[i])  # default standing pose
            qvel[self._joint_qvel_adr[i]] = 0.0

        # Clear ctrl too (otherwise stale targets from last episode survive).
        mj_data.ctrl[:] = 0.0
        for i in range(12):
            mj_data.ctrl[self._actuator_ids[i]] = float(GO2_DEFAULT_JOINT_POS[i])  # default standing pose target

        mujoco.mj_forward(self.mj_model, mj_data)

        self._last_action = np.zeros(12, dtype=np.float32)
        self._last_cmd_arr = np.zeros(3, dtype=np.float32)
        self._phase_time = 0.0
        self._refresh_pose_cache(mj_data)

    def step(self, mj_data, cmd: LocomotionCommand, dt: float) -> None:
        """Advance physics by `dt`, driving the Go2 via the appropriate policy.

        Routing:
          - If cmd.gait == "hind_sit" AND a hind_sit policy is loaded, the
            42-dim hind_sit obs is built and dispatched to that policy.
          - Otherwise the 47-dim walker obs is built and dispatched to the
            walker policy. `cmd.vx/vy/yaw_rate` go into its command channel;
            cmd.gait == "stand" zeroes that channel.
        """
        mujoco = self._mujoco

        # Determine which policy is active this tick. Posture policies
        # (get_up/crouch/lie_belly/lie_side/hind_sit) are 42-dim and share
        # the get_up dispatch path (8Hz settle, scale=0.6, perm to phys order).
        # Walker policies (normal + slow) are 48-dim, decimation=4 (50Hz).
        posture_policies = {
            "hind_sit": self.hind_sit_policy,
            "get_up": self.get_up_policy,
            "crouch": self.crouch_policy,
            "lie_belly": self.lie_belly_policy,
            "lie_side": self.lie_side_policy,
        }
        use_posture = (
            cmd.gait in posture_policies and posture_policies[cmd.gait] is not None
        )
        use_slow_walker = (
            cmd.gait == "walk_slow" and self.walker_slow_policy is not None
        )
        use_stay = (cmd.gait == "stay")
        # Back-compat names used downstream.
        use_hind_sit = (cmd.gait == "hind_sit") and (self.hind_sit_policy is not None)
        use_get_up = use_posture  # all posture policies follow the get_up dispatch shape
        if use_posture:
            active = cmd.gait
        elif use_slow_walker:
            active = "walker_slow"
        elif use_stay:
            active = "stay"
        else:
            active = "walker"

        # When switching between policies, zero last_action: the two heads
        # have different action distributions and feeding the wrong one's
        # stale output is out-of-distribution. Additionally, if the
        # newly-active policy implements .reset() (e.g. ScriptedGetUpPolicy),
        # call it so any internal phase clock restarts at t=0 on activation.
        if active != self._last_active_policy:
            self._last_action = np.zeros(12, dtype=np.float32)
            self._last_active_policy = active
            if active in posture_policies:
                self._get_up_substep_counter = 0  # restart settle clock (shared)
                for _i in range(12):  # zero actuator targets for clean handoff
                    mj_data.ctrl[self._actuator_ids[_i]] = 0.0
                _newp = posture_policies[active]
                if _newp is not None and hasattr(_newp, "reset"):
                    _newp.reset()
            elif active == "stay":
                # Snapshot current joint configuration; held forever after.
                self._stay_targets = np.array(
                    [mj_data.qpos[self._joint_qpos_adr[i]] for i in range(12)],
                    dtype=np.float32,
                )

        # Map LocomotionCommand -> 3-dim twist (only used by walker path).
        cmd_arr = self._command_to_twist(cmd)
        self._last_cmd_arr = cmd_arr

        # go1_getup was trained with settle_steps=25 (8 Hz policy rate).
        # Walker/hind_sit use decimation=4 (50 Hz). Use a settle counter.
        GET_UP_SETTLE_STEPS = 25
        n_substeps = max(1, int(round(dt / self.cfg.physics_dt)))
        for substep in range(n_substeps):
            # Re-query the active policy at the correct rate.
            # mjlab Go1-Getup trains with decimation=4 (50 Hz), same as walker.
            # settle_steps=25 is the training-time reset-window mask, NOT
            # a decimation. Earlier code queried posture policies at 8 Hz.
            query = (substep % self.cfg.decimation == 0)
            if query and use_stay:
                # Held-pose bypass: no policy call, no obs build.
                for _i in range(12):
                    mj_data.ctrl[self._actuator_ids[_i]] = float(self._stay_targets[_i])
            elif query:
                if use_posture:
                    # All posture policies use the 42/46-dim get_up obs shape.
                    obs = self._build_posture_observation(mj_data, posture_policies[active])
                    action = posture_policies[active](obs[None, :])[0]
                elif use_slow_walker:
                    obs = self._build_observation(mj_data, cmd_arr)
                    action = self.walker_slow_policy(obs[None, :])[0]
                else:
                    obs = self._build_observation(mj_data, cmd_arr)
                    action = self.policy(obs[None, :])[0]
                self._last_action = np.asarray(action, dtype=np.float32)
                # If cmd.joint_targets is set, bypass policy (manual PD skills).
                if cmd.joint_targets is not None:
                    targets = np.asarray(cmd.joint_targets, dtype=np.float32)
                elif use_posture:
                    # mjlab posture policies (get_up + crouch/lie_belly/lie_side)
                    # use SettleRelativeJointPositionAction:
                    #   target = current_pos + raw_action * 0.6 - encoder_bias
                    # encoder_bias=0 at deployment. Earlier code wrote
                    # `target = 0.6 * action` (absolute from zero) which
                    # pushed every joint toward ~0 rad on each query.
                    _GO1_PERM = self._GO1_TO_PHYS_PERM
                    qpos_now = mj_data.qpos
                    targets = np.zeros(12, dtype=np.float32)
                    for _k in range(12):
                        phys_i = _GO1_PERM[_k]
                        targets[phys_i] = (
                            float(qpos_now[self._joint_qpos_adr[phys_i]])
                            + 0.6 * self._last_action[_k]
                        )
                else:
                    # Walker policy returns actions in Go1 MJCF order
                    # (FR,FL,RR,RL); PhysicsCat actuator_ids are in
                    # GO2_JOINT_NAMES order (FL,FR,RL,RR). Permute action
                    # *and* default offsets so the per-leg targets land on
                    # the right actuators. (Same fix the get_up branch got
                    # in commits 31ad1954 + 0b278766; the walker path had
                    # been silently writing MJCF-order actions into
                    # phys-order actuators, swapping every L/R pair.)
                    _GO1_PERM = self._GO1_TO_PHYS_PERM
                    targets = np.zeros(12, dtype=np.float32)
                    for _k in range(12):
                        targets[_GO1_PERM[_k]] = (
                            GO2_DEFAULT_JOINT_POS[_GO1_PERM[_k]]
                            + self.cfg.action_scale * self._last_action[_k]
                        )
                for i in range(12):
                    mj_data.ctrl[self._actuator_ids[i]] = float(targets[i])
            if use_posture:
                self._get_up_substep_counter += 1

            mujoco.mj_step(self.mj_model, mj_data)
            # Advance phase clock by one physics substep so that between
            # consecutive policy queries (decimation=4 substeps) the phase
            # advances by decimation * physics_dt = 4 * 0.005 = 0.02 s,
            # matching the training step_dt.  The old code advanced by the
            # full brain dt (0.05 s) *outside* the loop, running 2.5x too fast.
            cmd_norm = float(np.linalg.norm(cmd_arr))
            if cmd_norm >= self.cfg.phase_stand_threshold:
                self._phase_time = (self._phase_time + self.cfg.physics_dt) % self.cfg.phase_period
            else:
                self._phase_time = 0.0

        self._refresh_pose_cache(mj_data)

    # ------------------------------------------------------------------ #
    # Observation construction
    # ------------------------------------------------------------------ #

    def _compute_foot_contacts(self, mj_data) -> np.ndarray:
        """Return per-foot 0/1 contact state, in FL/FR/RL/RR order.

        Walks mj_data.contact[:ncon] and marks a foot active whenever its
        cached geom id appears in either side of any contact. If foot geoms
        weren't found at init time this returns zeros (the policy will see
        the FR-Net features as 0.0, which is what the predictor outputs in
        a fully-airborne pose anyway).
        """
        out = np.zeros(4, dtype=np.float32)
        if not any(g >= 0 for g in self._foot_geom_ids):
            return out
        ncon = int(mj_data.ncon)
        for i in range(ncon):
            c = mj_data.contact[i]
            g1 = int(c.geom1); g2 = int(c.geom2)
            for fi, fg in enumerate(self._foot_geom_ids):
                if fg < 0:
                    continue
                if g1 == fg or g2 == fg:
                    out[fi] = 1.0
        return out

    def _build_posture_observation(self, mj_data, policy) -> np.ndarray:
        """Posture obs sized to the target policy's obs_dim. All 5 posture
        heads (get_up, crouch, lie_belly, lie_side, hind_sit) were trained
        on the same 42-dim Go1 MJCF-order shape; v7+ get_up adds a 4-dim
        FR-Net predicted_next_contact tail to make 46."""
        base = self._build_go1_getup_observation(mj_data)
        target_dim = getattr(policy, "obs_dim", 42)
        if target_dim == 42:
            return base
        if target_dim == 46:
            contacts = self._compute_foot_contacts(mj_data)
            return np.concatenate([base, contacts]).astype(np.float32)
        raise RuntimeError(
            f"posture policy has unsupported obs_dim={target_dim}; expected 42 or 46"
        )

    def _build_get_up_observation(self, mj_data) -> np.ndarray:
        """Backwards-compatible wrapper."""
        base = self._build_go1_getup_observation(mj_data)  # 42-dim, Go1 MJCF order
        target_dim = getattr(self.get_up_policy, "obs_dim", 42)
        if target_dim == 42:
            return base
        if target_dim == 46:
            contacts = self._compute_foot_contacts(mj_data)
            return np.concatenate([base, contacts]).astype(np.float32)
        raise RuntimeError(
            f"get_up policy has unsupported obs_dim={target_dim}; expected 42 or 46"
        )

    # Joint permutation: Go1 MJCF natural order (FR,FL,RR,RL)
    # -> PhysicsCat GO2_JOINT_NAMES order (FL,FR,RL,RR).
    # Self-inverse permutation (swapping front/rear leg pairs).
    _GO1_TO_PHYS_PERM = (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)

    def _build_go1_getup_observation(self, mj_data) -> np.ndarray:
        """42-dim obs for the mjlab Unitree-Go1-Getup-Flat policy.

        Go1 MJCF has no keyframe -> default_joint_pos == 0 -> joint_pos = raw qpos.
        mjlab iterates joints in MJCF order: FR, FL, RR, RL.
        PhysicsCat _joint_qpos_adr is in GO2_JOINT_NAMES order: FL, FR, RL, RR.
        _GO1_TO_PHYS_PERM corrects the indexing so obs matches training.

        Layout: base_ang_vel(3) + proj_grav(3) + joint_pos(12)
                + joint_vel(12) + last_action(12) = 42
        """
        xmat = np.asarray(mj_data.xmat[self._base_body_id]).reshape(3, 3)
        projected_gravity = -xmat[2, :]
        base_ang_vel = mj_data.sensordata[self._gyro_adr : self._gyro_adr + 3]

        p = self._GO1_TO_PHYS_PERM
        qpos = mj_data.qpos
        # mjlab joint_pos_rel(biased=True) = joint_pos_biased - default_joint_pos.
        # Go1 EntityCfg.init_state.joint_pos sets a non-zero default
        # (hip +/-0.1, thigh 0.9, calf -1.8 == GO2_DEFAULT_JOINT_POS).
        # encoder_bias=0 at deployment.
        joint_pos = np.array(
            [qpos[self._joint_qpos_adr[p[i]]] - GO2_DEFAULT_JOINT_POS[p[i]]
             for i in range(12)],
            dtype=np.float32,
        )
        qvel = mj_data.qvel
        joint_vel = np.array(
            [qvel[self._joint_qvel_adr[p[i]]] for i in range(12)],
            dtype=np.float32,
        )
        obs = np.concatenate(
            [
                np.asarray(base_ang_vel, dtype=np.float32),
                np.asarray(projected_gravity, dtype=np.float32),
                joint_pos,
                joint_vel,
                self._last_action,  # raw policy output, in Go1 MJCF order
            ]
        ).astype(np.float32)
        return obs

    def _build_hind_sit_observation(self, mj_data) -> np.ndarray:
        """Build the 42-dim observation expected by the hind_sit policy.

        Matches the get_up actor obs (no command, no phase, no height_scan):
            base_ang_vel(3) + projected_gravity(3) + joint_pos(12)
                + joint_vel(12) + last_action(12)
        """
        base_ang_vel = mj_data.sensordata[self._gyro_adr : self._gyro_adr + 3]
        xmat = np.asarray(mj_data.xmat[self._base_body_id]).reshape(3, 3)
        projected_gravity = -xmat[2, :]

        qpos = mj_data.qpos
        joint_pos = np.array(
            [qpos[self._joint_qpos_adr[i]] - GO2_DEFAULT_JOINT_POS[i] for i in range(12)],
            dtype=np.float32,
        )
        qvel = mj_data.qvel
        joint_vel = np.array(
            [qvel[self._joint_qvel_adr[i]] for i in range(12)],
            dtype=np.float32,
        )
        last_action = self._last_action

        # 47-dim walker obs: base_ang_vel(3)+proj_grav(3)+cmd(3)+phase(2)+joint_pos(12)+joint_vel(12)+action(12)
        # 42-dim: base_ang_vel(3)+proj_grav(3)+joint_pos(12)+joint_vel(12)+action(12)
        obs = np.concatenate(
            [
                np.asarray(base_ang_vel, dtype=np.float32),
                np.asarray(projected_gravity, dtype=np.float32),
                joint_pos,
                joint_vel,
                last_action,
            ]
        ).astype(np.float32)
        return obs

    def _build_observation(self, mj_data, cmd_arr: np.ndarray) -> np.ndarray:
        """Build 48-dim obs for Go1 flat velocity walker.

        Layout: base_ang_vel(3)+proj_grav(3)+cmd(3)+phase(2)
                +joint_pos(12)+joint_vel(12)+action(12)+body_height_cmd(1) = 48

        Matches the velocity_mob *actor* obs used during training.
        base_lin_vel is critic-only and must NOT appear here.
        phase is a 2-dim sin/cos clock zeroed when the command norm is
        below phase_stand_threshold (matches training env behaviour).
        body_height_cmd is unused in robot-pet-cat so it is held at 0.

        Joint dims (joint_pos, joint_vel, last_action) are emitted in Go1
        MJCF order (FR,FL,RR,RL) -- what mjlab iterated when training the
        walker. PhysicsCat looks up indices in GO2_JOINT_NAMES order
        (FL,FR,RL,RR), so _GO1_TO_PHYS_PERM is applied at read time. The
        action-application path is the inverse permutation; see the
        walker branch of step().
        """
        # rotation matrix for proj_grav and (unused here) lin_vel
        xmat = np.asarray(mj_data.xmat[self._base_body_id]).reshape(3, 3)

        # 1. base_ang_vel (3) -- gyro reading.
        base_ang_vel = mj_data.sensordata[self._gyro_adr : self._gyro_adr + 3]

        # 2. projected_gravity (3): -R[2,:] (third ROW of xmat)
        projected_gravity = -xmat[2, :]

        # 3. command (3): the velocity twist [vx, vy, yaw_rate].
        command = cmd_arr

        # 4. phase (2): sin/cos clock, zeroed when standing (matches training).
        cmd_norm = float(np.linalg.norm(cmd_arr))
        if cmd_norm < self.cfg.phase_stand_threshold:
            phase = np.zeros(2, dtype=np.float32)
        else:
            phase_frac = self._phase_time / self.cfg.phase_period
            phase = np.array(
                [
                    np.sin(2.0 * np.pi * phase_frac),
                    np.cos(2.0 * np.pi * phase_frac),
                ],
                dtype=np.float32,
            )

        # 5. joint_pos (12): q - q_default, in Go1 MJCF order.
        p = self._GO1_TO_PHYS_PERM
        qpos = mj_data.qpos
        joint_pos = np.array(
            [qpos[self._joint_qpos_adr[p[i]]] - GO2_DEFAULT_JOINT_POS[p[i]]
             for i in range(12)],
            dtype=np.float32,
        )

        # 6. joint_vel (12), in Go1 MJCF order.
        qvel = mj_data.qvel
        joint_vel = np.array(
            [qvel[self._joint_qvel_adr[p[i]]] for i in range(12)],
            dtype=np.float32,
        )

        # 7. last_action (12), already in Go1 MJCF order (raw policy output).
        last_action = self._last_action

        # 8. body_height_cmd (1): height offset command -- not used, hold at 0.
        body_height_cmd = np.zeros(1, dtype=np.float32)

        # 48-dim: base_ang_vel(3)+proj_grav(3)+cmd(3)+phase(2)+joint_pos(12)+joint_vel(12)+action(12)+body_height_cmd(1)
        obs = np.concatenate(
            [
                np.asarray(base_ang_vel, dtype=np.float32),
                np.asarray(projected_gravity, dtype=np.float32),
                command,
                phase,
                joint_pos,
                joint_vel,
                last_action,
                body_height_cmd,
            ]
        ).astype(np.float32)
        return obs

    def _command_to_twist(self, cmd: LocomotionCommand) -> np.ndarray:
        """Convert a LocomotionCommand into the walker's 3-d twist channel.

        Honors the "stand" gait by zeroing the twist (so the policy enters
        its stand-mask branch). Clamps to optional config maxima.
        """
        if cmd.gait == "stand":
            return np.zeros(3, dtype=np.float32)
        vx, vy, wz = float(cmd.vx), float(cmd.vy), float(cmd.yaw_rate)
        if self.cfg.max_linear_speed is not None:
            mag = math.hypot(vx, vy)
            if mag > self.cfg.max_linear_speed and mag > 1e-9:
                scale = self.cfg.max_linear_speed / mag
                vx *= scale
                vy *= scale
        if self.cfg.max_yaw_rate is not None:
            wz = max(-self.cfg.max_yaw_rate, min(self.cfg.max_yaw_rate, wz))
        return np.array([vx, vy, wz], dtype=np.float32)

    def _refresh_pose_cache(self, mj_data) -> None:
        """Refresh xy / yaw / z / last_speed from mj_data after a step."""
        adr = self._base_qpos_adr
        qpos = mj_data.qpos
        self._xy = np.array([qpos[adr + 0], qpos[adr + 1]], dtype=np.float32)
        self._z = float(qpos[adr + 2])
        # Quaternion (w, x, y, z) -> yaw (Z rotation).
        qw = float(qpos[adr + 3])
        qx = float(qpos[adr + 4])
        qy = float(qpos[adr + 5])
        qz = float(qpos[adr + 6])
        # yaw = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy^2 + qz^2))
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        self._yaw = math.atan2(siny_cosp, cosy_cosp)
        # Linear speed -- prefer local_linvel sensor if present, else compute
        # from base qvel.
        if self._linvel_adr >= 0:
            lv = mj_data.sensordata[self._linvel_adr : self._linvel_adr + 3]
            self._last_speed = float(math.hypot(lv[0], lv[1]))
        else:
            vadr = self._base_qvel_adr
            self._last_speed = float(math.hypot(mj_data.qvel[vadr + 0], mj_data.qvel[vadr + 1]))

    def apply_launch_impulse(
        self,
        mj_data,
        target_xyz: tuple,
    ) -> None:
        """One-shot: set base qvel to a ballistic arc aimed at target_xyz.

        Called by BrainEnv when JumpTo.launch_requested is True. After this
        call BrainEnv's normal cat.step() continues, integrating the impulse
        through all physics substeps this tick and onward.

        Physics:
          v_z = sqrt(2*g*(dz + margin))  -- enough to clear platform top
          v_xy = direction * (d_xy / t_land)  -- land approximately on target
        """
        g = 9.81
        adr = self._base_qpos_adr
        vadr = self._base_qvel_adr

        cur_x = float(mj_data.qpos[adr + 0])
        cur_y = float(mj_data.qpos[adr + 1])
        cur_z = float(mj_data.qpos[adr + 2])

        tx, ty, tz = target_xyz
        dz = max(0.0, tz - cur_z) + 0.08  # 8 cm clearance above platform
        dz = min(dz, 0.5)
        vz = math.sqrt(2.0 * g * dz)
        t_peak = vz / g
        t_land = 2.0 * t_peak

        dx, dy = tx - cur_x, ty - cur_y
        d_xy = math.hypot(dx, dy)
        if d_xy > 1e-3:
            vx = dx / max(t_land, 0.01)
            vy = dy / max(t_land, 0.01)
        else:
            vx, vy = 0.0, 0.0

        mj_data.qvel[vadr + 0] = vx
        mj_data.qvel[vadr + 1] = vy
        mj_data.qvel[vadr + 2] = vz
        mj_data.qvel[vadr + 3] = 0.0
        mj_data.qvel[vadr + 4] = 0.0
        mj_data.qvel[vadr + 5] = 0.0

    def _sensor_adr(self, name: str, required: bool = True) -> int:
        mujoco = self._mujoco
        sid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sid < 0:
            if required:
                raise RuntimeError(f"Sensor {name!r} not in compiled model.")
            return -1
        return int(self.mj_model.sensor_adr[sid])
