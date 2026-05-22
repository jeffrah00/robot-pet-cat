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
    load_hind_sit_policy,
    load_get_up_policy,
)


# Default base height at spawn (matches Unitree training init_state, m).
DEFAULT_BASE_SPAWN_Z = 0.32

# Name of the Go2 base body in menagerie's go2_mjx.xml.
GO2_BASE_BODY = "base_link"


def inject_cat_eye_camera(assets: dict[str, bytes]) -> dict[str, bytes]:
    """Patch the menagerie go2_mjx.xml in `assets` to mount a cat_eye camera
    onto the Go2 base body. Returns the same dict (mutated).

    MuJoCo's standalone XML does not let an outer file `re-open` an included
    body by name -- duplicate <body name="base"> raises XML Error. To attach
    a body-mounted POV camera, we inject the <camera> as the first child of
    the existing <body name="base"...> tag inside go2_mjx.xml's content as
    held in the assets dict, then compile from_xml_string with the patched
    asset.

    Idempotent: if the patched asset already has a cat_eye camera, no-op.
    """
    import re

    KEY = "go2_mjx.xml"
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
        r'(<body name="base"[^>]*>)',
        r'\1' + cam_block,
        text,
        count=1,
    )
    if n == 0:
        raise RuntimeError(
            "Could not find <body name=\"base\" ...> in go2_mjx.xml to "
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
            self.policy = load_go2_walker_policy(policy)

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
        # ---- Index lookups against the compiled MuJoCo model ----------- #
        # Base body and its free-joint qpos/qvel offsets.
        base_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, GO2_BASE_BODY)
        if base_id < 0:
            raise RuntimeError(
                f"Go2 base body {GO2_BASE_BODY!r} not in compiled model. "
                "Did the scene <include> go2_mjx.xml?"
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
            raise RuntimeError("No free joint on Go2 base body in compiled model.")

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
        # menagerie's go2_mjx.xml match the joint names without "_joint":
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
            qpos[self._joint_qpos_adr[i]] = float(GO2_DEFAULT_JOINT_POS[i])
            qvel[self._joint_qvel_adr[i]] = 0.0

        # Clear ctrl too (otherwise stale targets from last episode survive).
        mj_data.ctrl[:] = 0.0
        for i in range(12):
            mj_data.ctrl[self._actuator_ids[i]] = float(GO2_DEFAULT_JOINT_POS[i])

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

        # Determine which policy is active this tick.
        use_hind_sit = (cmd.gait == "hind_sit") and (self.hind_sit_policy is not None)
        use_get_up = (cmd.gait == "get_up") and (self.get_up_policy is not None)
        active = "hind_sit" if use_hind_sit else ("get_up" if use_get_up else "walker")

        # When switching between policies, zero last_action: the two heads
        # have different action distributions and feeding the wrong one's
        # stale output is out-of-distribution.
        if active != self._last_active_policy:
            self._last_action = np.zeros(12, dtype=np.float32)
            self._last_active_policy = active

        # Map LocomotionCommand -> 3-dim twist (only used by walker path).
        cmd_arr = self._command_to_twist(cmd)
        self._last_cmd_arr = cmd_arr

        n_substeps = max(1, int(round(dt / self.cfg.physics_dt)))
        for substep in range(n_substeps):
            # Re-query the active policy every `decimation` physics substeps.
            if substep % self.cfg.decimation == 0:
                if use_hind_sit:
                    obs = self._build_hind_sit_observation(mj_data)
                    action = self.hind_sit_policy(obs[None, :])[0]
                elif use_get_up:
                    obs = self._build_hind_sit_observation(mj_data)
                    action = self.get_up_policy(obs[None, :])[0]
                else:
                    obs = self._build_observation(mj_data, cmd_arr)
                    action = self.policy(obs[None, :])[0]
                self._last_action = np.asarray(action, dtype=np.float32)
                # Joint targets = default + scale * action, in MJCF order.
                # Both policies use the same action_scale and default pose.
                targets = GO2_DEFAULT_JOINT_POS + self.cfg.action_scale * self._last_action
                for i in range(12):
                    mj_data.ctrl[self._actuator_ids[i]] = float(targets[i])

            mujoco.mj_step(self.mj_model, mj_data)

        # Advance the phase clock by dt (only when moving, like the env).
        cmd_norm = float(np.linalg.norm(cmd_arr))
        if cmd_norm >= self.cfg.phase_stand_threshold:
            self._phase_time = (self._phase_time + dt) % self.cfg.phase_period
        else:
            self._phase_time = 0.0

        self._refresh_pose_cache(mj_data)

    # ------------------------------------------------------------------ #
    # Observation construction
    # ------------------------------------------------------------------ #

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
        """Build the 47-dim obs the flat velocity walker was trained against."""
        # 1. base_ang_vel (3) -- gyro reading.
        base_ang_vel = mj_data.sensordata[self._gyro_adr : self._gyro_adr + 3]

        # 2. projected_gravity (3): world gravity (0,0,-1) expressed in
        # body frame. xmat is R_world<-body (column i = i-th body axis in
        # world coords). To rotate a world-frame vector into body coords,
        # premultiply by R^T, so projected_gravity = R^T @ [0,0,-1] =
        # -R[2,:] (third ROW of xmat, NOT third column).
        xmat = np.asarray(mj_data.xmat[self._base_body_id]).reshape(3, 3)
        projected_gravity = -xmat[2, :]

        # 3. command (3): the velocity twist [vx, vy, yaw_rate].
        command = cmd_arr

        # 4. phase (2): [sin, cos] of 2*pi * (t/period), gated by command
        # magnitude.
        cmd_norm = float(np.linalg.norm(command))
        if cmd_norm < self.cfg.phase_stand_threshold:
            phase = np.zeros(2, dtype=np.float32)
        else:
            theta = 2.0 * math.pi * (self._phase_time / self.cfg.phase_period)
            phase = np.array([math.sin(theta), math.cos(theta)], dtype=np.float32)

        # 5. joint_pos (12): q - q_default
        qpos = mj_data.qpos
        joint_pos = np.array(
            [qpos[self._joint_qpos_adr[i]] - GO2_DEFAULT_JOINT_POS[i] for i in range(12)],
            dtype=np.float32,
        )

        # 6. joint_vel (12)
        qvel = mj_data.qvel
        joint_vel = np.array(
            [qvel[self._joint_qvel_adr[i]] for i in range(12)],
            dtype=np.float32,
        )

        # 7. last_action (12)
        last_action = self._last_action

        # 47-dim: base_ang_vel(3)+proj_grav(3)+cmd(3)+phase(2)+joint_pos(12)+joint_vel(12)+action(12)
        obs = np.concatenate(
            [
                np.asarray(base_ang_vel, dtype=np.float32),
                np.asarray(projected_gravity, dtype=np.float32),
                command,
                phase,
                joint_pos,
                joint_vel,
                last_action,
            ]
        ).astype(np.float32)
        return obs

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

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
        # Cap dz to prevent runaway vz if cur_z is weird (e.g. physics instability).
        dz = min(dz, 0.5)  # max 50 cm effective rise -- gives vz ~ 3.1 m/s max
        vz = math.sqrt(2.0 * g * dz)
        t_peak = vz / g
        t_land = 2.0 * t_peak  # approximate total time of flight

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
        # Zero angular velocity for a clean arc.
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
