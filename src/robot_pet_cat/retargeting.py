"""2D cat keypoints -> Go2 reference trajectory.

Phase 2a MVP. The hard part of "learn cat-like motion from video" is not pose
detection (off-the-shelf models exist) but morphology retargeting: a real cat
skeleton and Go2's skeleton don't match. This module focuses on the retarget,
with a swappable perception front-end.

Pipeline:
    JSON of 2D keypoints in image space
                |
                v
    side-view lift to 3D world (y=depth=0 by assumption)
                |
                v
    root pose (x, z, pitch) from withers+hips
                |
                v
    per-leg foot target in body frame
                |
                v
    closed-form 2-link IK to Go2 (hip_y, knee); hip_x set to 0
                |
                v
    qpos[T, 19] for Go2  ->  .npz file

Plug-in points for real perception (replace `load_keypoints_json`):
  - SuperAnimal-Quadruped (DeepLabCut)
  - MMPose AnimalPose
  - BARC (3D, removes the side-view lift step)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Skeleton constants
# ---------------------------------------------------------------------------

# Cat keypoint indices, fixed for this project.
KP_NAMES = [
    "nose",        # 0
    "withers",     # 1 (top of shoulders)
    "hips",        # 2 (top of pelvis)
    "tail_base",   # 3
    "fl_paw",      # 4 (front-left paw)
    "fr_paw",      # 5 (front-right paw)
    "rl_paw",      # 6 (rear-left paw)
    "rr_paw",      # 7 (rear-right paw)
    "fl_elbow",    # 8 (optional)
    "fr_elbow",    # 9
    "rl_knee",     # 10
    "rr_knee",     # 11
]
N_KP = len(KP_NAMES)


@dataclass(frozen=True)
class Go2Kinematics:
    """Approximate Go2 link lengths and hip origins in body frame.

    Sourced from the Unitree Go2 URDF. Numbers are coarse but consistent
    enough for AMP's distribution-matching purposes — we are not doing exact
    motion tracking.
    """

    thigh_len_m: float = 0.213
    calf_len_m: float = 0.213
    nominal_height_m: float = 0.30  # standing trunk height above floor

    # Hip joint origins in body frame (x=forward, y=left, z=up).
    fl_hip: tuple[float, float, float] = (+0.1934, +0.0465, 0.0)
    fr_hip: tuple[float, float, float] = (+0.1934, -0.0465, 0.0)
    rl_hip: tuple[float, float, float] = (-0.1934, +0.0465, 0.0)
    rr_hip: tuple[float, float, float] = (-0.1934, -0.0465, 0.0)

    # Joint limits (rad).
    hip_y_min: float = -1.05
    hip_y_max: float = +1.05
    knee_min:  float = -2.70
    knee_max:  float = -0.60


GO2 = Go2Kinematics()

# Mapping: leg id -> (paw keypoint index, hip origin attribute name).
LEGS = [
    ("FL", 4, GO2.fl_hip),
    ("FR", 5, GO2.fr_hip),
    ("RL", 6, GO2.rl_hip),
    ("RR", 7, GO2.rr_hip),
]

# qpos layout: 7 root (xyz quat) + 12 actuated joints.
# Joint order matches the Go2 MJCF: FL_hip, FL_thigh, FL_calf, FR_..., RL_..., RR_...
QPOS_DIM = 7 + 12
ROOT_DIM = 7


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


@dataclass
class KeypointClip:
    fps: float
    scale_m_per_px: float
    floor_y_px: float
    image_height_px: float
    keypoints: np.ndarray  # (T, N_KP, 2) image-space xy

    @property
    def n_frames(self) -> int:
        return self.keypoints.shape[0]

    @property
    def dt(self) -> float:
        return 1.0 / self.fps


def load_keypoints_json(path: Path) -> KeypointClip:
    """Load a clip in our project JSON format.

    Schema:
      {
        "fps": 30.0,
        "scale_m_per_px": 0.005,
        "floor_y_px": 600,
        "image_height_px": 720,
        "keypoint_names": [...],          # for documentation/sanity check
        "frames": [
          {"t": 0.0, "keypoints": [[px, py], ...]},  # length == N_KP
          ...
        ]
      }
    """
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(
            f"{path} is not a project keypoint JSON "
            f"(top level is {type(data).__name__}, expected dict)"
        )
    if data.get("keypoint_names") not in (None, KP_NAMES):
        raise ValueError(
            f"keypoint_names mismatch in {path}; expected {KP_NAMES}"
        )
    if "frames" not in data:
        raise ValueError(f"{path} is missing required 'frames' field")
    frames = data["frames"]
    kps = np.array([f["keypoints"] for f in frames], dtype=np.float64)
    if kps.ndim != 3 or kps.shape[1:] != (N_KP, 2):
        raise ValueError(
            f"keypoints in {path} should be (T, {N_KP}, 2); got {kps.shape}"
        )
    return KeypointClip(
        fps=float(data["fps"]),
        scale_m_per_px=float(data["scale_m_per_px"]),
        floor_y_px=float(data["floor_y_px"]),
        image_height_px=float(data["image_height_px"]),
        keypoints=kps,
    )


# ---------------------------------------------------------------------------
# 2D -> 3D lift (side-view assumption)
# ---------------------------------------------------------------------------


def lift_to_world(clip: KeypointClip) -> np.ndarray:
    """Side-view lift: image (px, py) -> world (x_forward, y_depth=0, z_up).

    The camera is assumed perpendicular to the cat's sagittal plane.
    Image x maps to world x (forward). Image y is inverted and offset to make
    floor_y_px correspond to z=0.

    Returns:
        world_kps: (T, N_KP, 3)
    """
    px = clip.keypoints[..., 0]
    py = clip.keypoints[..., 1]
    x_w = px * clip.scale_m_per_px
    y_w = np.zeros_like(x_w)
    z_w = (clip.floor_y_px - py) * clip.scale_m_per_px
    return np.stack([x_w, y_w, z_w], axis=-1)


# ---------------------------------------------------------------------------
# Root pose from cat skeleton
# ---------------------------------------------------------------------------


def root_pose_per_frame(world_kps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate Go2 trunk pose from cat withers + hips.

    Strategy:
      - Trunk center = midpoint of (withers, hips), at Go2's nominal height
        above the floor (we trust the cat's ground clearance roughly).
      - Pitch = angle of the (hips -> withers) vector in the xz plane.
      - Yaw and roll forced to 0 (side view doesn't constrain them).

    Returns:
        root_xyz: (T, 3)
        root_pitch: (T,) radians
    """
    withers = world_kps[:, 1, :]
    hips = world_kps[:, 2, :]
    midpoint = 0.5 * (withers + hips)

    # Pitch from spine direction (hips -> withers).
    spine = withers - hips
    pitch = np.arctan2(spine[:, 2], spine[:, 0])  # z over x in side-view plane

    # Use cat's body height as a soft prior on Go2's trunk height.
    # We keep the cat's x/z but force z to be GO2.nominal_height_m at minimum
    # so the trunk doesn't sink below the floor.
    z = np.maximum(midpoint[:, 2], GO2.nominal_height_m)
    root_xyz = np.stack([midpoint[:, 0], midpoint[:, 1], z], axis=-1)
    return root_xyz, pitch


def pitch_to_quat(pitch: np.ndarray) -> np.ndarray:
    """Roll=Yaw=0, only pitch. Returns (T, 4) [w, x, y, z] quaternions."""
    half = 0.5 * pitch
    w = np.cos(half)
    y = np.sin(half)  # rotation about world y (depth axis)
    x = np.zeros_like(w)
    z = np.zeros_like(w)
    return np.stack([w, x, y, z], axis=-1)


def quat_inverse(q: np.ndarray) -> np.ndarray:
    """Inverse of a unit quaternion."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([w, -x, -y, -z], axis=-1)


def quat_rotate_vec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v (..., 3) by quaternion q (..., 4)."""
    qw = q[..., 0:1]
    qv = q[..., 1:4]
    t = 2.0 * np.cross(qv, v)
    return v + qw * t + np.cross(qv, t)


# ---------------------------------------------------------------------------
# Per-leg IK
# ---------------------------------------------------------------------------


def leg_ik(foot_in_body: np.ndarray, hip_origin: tuple[float, float, float]) -> tuple[float, float, float]:
    """Closed-form 2-link IK for one Go2 leg.

    hip_x roll forced to 0 (side-view lift has no lateral information).

    Args:
        foot_in_body: (3,) foot target relative to body origin in body frame
        hip_origin: hip joint origin in body frame
    Returns:
        (hip_x, hip_y, knee) in radians
    """
    hx, hy, hz = hip_origin
    fx = foot_in_body[0] - hx
    # Side-view: y is degenerate, treat foot as in the xz plane.
    fz = foot_in_body[2] - hz

    l1 = GO2.thigh_len_m
    l2 = GO2.calf_len_m

    r = math.hypot(fx, fz)
    r_max = l1 + l2 - 1e-3
    r_min = abs(l1 - l2) + 1e-3
    r = max(min(r, r_max), r_min)

    # Knee (interior angle of the thigh-foot-calf triangle).
    cos_knee_inner = (l1 * l1 + l2 * l2 - r * r) / (2 * l1 * l2)
    cos_knee_inner = max(-1.0, min(1.0, cos_knee_inner))
    knee_inner = math.acos(cos_knee_inner)
    # Go2 convention: knee=0 -> straight; negative -> bent (calf swings back).
    knee = -(math.pi - knee_inner)

    # Hip pitch.
    # alpha = angle of foot direction from straight-down (-z), measured toward +x.
    alpha = math.atan2(fx, -fz)
    cos_thigh_inner = (l1 * l1 + r * r - l2 * l2) / (2 * l1 * r)
    cos_thigh_inner = max(-1.0, min(1.0, cos_thigh_inner))
    thigh_inner = math.acos(cos_thigh_inner)
    hip_y = alpha + thigh_inner  # Knee bends backward -> thigh leads the calf.

    hip_y = max(GO2.hip_y_min, min(GO2.hip_y_max, hip_y))
    knee = max(GO2.knee_min, min(GO2.knee_max, knee))
    return 0.0, hip_y, knee


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def retarget_clip(clip: KeypointClip) -> dict[str, np.ndarray]:
    """Run the full retarget. Returns a dict ready to np.savez_compressed."""
    world_kps = lift_to_world(clip)  # (T, N_KP, 3)
    root_xyz, root_pitch = root_pose_per_frame(world_kps)
    root_quat = pitch_to_quat(root_pitch)

    T = clip.n_frames
    joints = np.zeros((T, 12), dtype=np.float64)

    # For each frame, transform paw world positions into body frame and IK.
    inv_q = quat_inverse(root_quat)
    for t in range(T):
        for leg_idx, (_, paw_kp, hip_origin) in enumerate(LEGS):
            foot_world = world_kps[t, paw_kp]
            foot_body = quat_rotate_vec(inv_q[t], foot_world - root_xyz[t])
            _hx, hy, knee = leg_ik(foot_body, hip_origin)
            j = leg_idx * 3
            joints[t, j + 0] = 0.0       # hip_x
            joints[t, j + 1] = hy        # hip_y
            joints[t, j + 2] = knee      # knee

    qpos = np.zeros((T, QPOS_DIM), dtype=np.float64)
    qpos[:, 0:3] = root_xyz
    qpos[:, 3:7] = root_quat
    qpos[:, 7:] = joints

    # Velocities via finite differences (matched to clip dt).
    qvel = np.zeros_like(qpos)
    if T >= 2:
        qvel[1:] = (qpos[1:] - qpos[:-1]) / clip.dt
        qvel[0] = qvel[1]

    return {
        "qpos": qpos,
        "qvel": qvel,
        "dt": np.array([clip.dt]),
        "fps": np.array([clip.fps]),
    }


def write_npz(out_path: Path, data: dict[str, np.ndarray]) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **data)
    return out_path


def process_directory(in_dir: Path, out_dir: Path) -> list[Path]:
    """Find every .json in `in_dir`, retarget it, write a .npz in `out_dir`.

    Returns the list of written paths.
    """
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    written = []
    skipped = []
    for json_path in sorted(in_dir.glob("*.json")):
        try:
            clip = load_keypoints_json(json_path)
        except (ValueError, KeyError) as e:
            # Likely a sidecar / metadata json (e.g., from DLC). Skip silently.
            skipped.append((json_path.name, str(e).split(chr(10))[0][:80]))
            continue
        data = retarget_clip(clip)
        out_path = out_dir / f"{json_path.stem}.npz"
        write_npz(out_path, data)
        written.append(out_path)
    if skipped:
        print(f"[retarget] skipped {len(skipped)} non-schema json(s):")
        for name, reason in skipped:
            print(f"           {name}: {reason}")
    return written


# ---------------------------------------------------------------------------
# Synthetic test fixture generator
# ---------------------------------------------------------------------------


def synth_cat_walk(fps: float = 30.0, duration_s: float = 1.0, seed: int = 0) -> dict:
    """Generate a synthetic side-view cat-walk keypoint sequence as a dict
    ready to json.dump. The cat trots at ~0.5 m/s with a diagonal-pair gait.

    Useful as a smoke-test fixture so the pipeline runs end-to-end with no
    real perception model installed.
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fps)) + 1
    ts = np.linspace(0.0, duration_s, n)

    # Image canvas + scale. 1280x720, 0.005 m/px = 6.4m wide, 3.6m tall.
    image_height_px = 720
    floor_y_px = 600
    scale_m_per_px = 0.005

    # Body geometry in pixels.
    hips_px_0 = np.array([300, 480])   # initial hips position in image
    body_len_px = 170                   # withers - hips along x
    body_h_px = 120                     # standing trunk height above floor
    head_offset_px = np.array([60, -30])  # nose relative to withers
    tail_offset_px = np.array([-40, 0])   # tail base relative to hips
    elbow_offset_px = np.array([0, 30])
    knee_offset_px = np.array([0, 30])

    forward_speed_px_per_s = 100        # 0.5 m/s
    gait_freq_hz = 2.0                  # 2 step cycles per second
    paw_lift_px = 25
    paw_swing_px = 50

    keypoints = np.zeros((n, N_KP, 2), dtype=np.float64)

    for i, t in enumerate(ts):
        # Hips + withers translate forward over time.
        hips = hips_px_0 + np.array([forward_speed_px_per_s * t, 0.0])
        withers = hips + np.array([body_len_px, -10.0])
        nose = withers + head_offset_px
        tail_base = hips + tail_offset_px

        # Gait: diagonal pairs. Phase a = FL & RR; phase b = FR & RL.
        phase_a = np.sin(2 * np.pi * gait_freq_hz * t)
        phase_b = np.sin(2 * np.pi * gait_freq_hz * t + np.pi)

        def paw_pos(base_xy: np.ndarray, phase: float) -> np.ndarray:
            # Foot stays on the floor for half the cycle (phase < 0),
            # lifts and swings forward for the other half (phase > 0).
            lift = max(0.0, phase) * paw_lift_px
            swing = phase * paw_swing_px
            return base_xy + np.array([swing, -lift])

        # Stance bases (where each paw sits when on the floor).
        fl_base = withers + np.array([0, floor_y_px - withers[1]])
        fr_base = fl_base
        rl_base = hips + np.array([0, floor_y_px - hips[1]])
        rr_base = rl_base

        fl_paw = paw_pos(fl_base, phase_a)
        fr_paw = paw_pos(fr_base, phase_b)
        rl_paw = paw_pos(rl_base, phase_b)
        rr_paw = paw_pos(rr_base, phase_a)

        # Elbows/knees: midpoint along the limb, slightly forward of straight.
        fl_elbow = 0.5 * (fl_base + withers) + elbow_offset_px
        fr_elbow = fl_elbow
        rl_knee = 0.5 * (rl_base + hips) + knee_offset_px
        rr_knee = rl_knee

        # Tiny annotator-style jitter so two runs aren't bit-identical.
        jitter = rng.normal(0.0, 0.5, size=(N_KP, 2))

        keypoints[i] = np.stack([
            nose,
            withers,
            hips,
            tail_base,
            fl_paw,
            fr_paw,
            rl_paw,
            rr_paw,
            fl_elbow,
            fr_elbow,
            rl_knee,
            rr_knee,
        ]) + jitter

    frames = [
        {"t": float(t), "keypoints": keypoints[i].tolist()}
        for i, t in enumerate(ts)
    ]
    return {
        "fps": float(fps),
        "scale_m_per_px": float(scale_m_per_px),
        "floor_y_px": float(floor_y_px),
        "image_height_px": float(image_height_px),
        "keypoint_names": KP_NAMES,
        "source": "synthetic",
        "frames": frames,
    }


def write_synth_walk(out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(synth_cat_walk(), indent=2))
    return out_path
