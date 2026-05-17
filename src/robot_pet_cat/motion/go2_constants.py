"""Constants for the Unitree Go2 locomotion env.

This is the Go2 analog of mujoco_playground's go1_constants. We need a separate
module because:
  - Go2's menagerie XML names foot SITES differently than Go1 (FL_foot vs FL),
    even though the foot GEOMS happen to share the same FL/FR/RL/RR names.
  - Go2's home keyframe differs slightly from Go1's (symmetric hips at 0.0
    instead of +/-0.1, base height 0.27 vs 0.278).
  - We point the asset loader at unitree_go2 in menagerie, not unitree_go1.

The joint names (FL_hip_joint, FL_thigh_joint, FL_calf_joint, ...) are identical
across Go1 and Go2, so any reward logic that references qpos slices by joint
order keeps working unchanged.

We ship our own scene XML in data/go2_scenes/ rather than reusing mujoco_playground's
Go1 scene; the latter would pull in Go1 meshes and home pose.
"""

from __future__ import annotations

from pathlib import Path

# Path to our hand-rolled Go2 scene XML (sits alongside the package so wheels
# pick it up). The XML in turn pulls Go2 meshes from mujoco_menagerie via the
# get_assets() override on Go2Env.
ROOT_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "go2_scenes"
)
FEET_ONLY_FLAT_TERRAIN_XML = ROOT_PATH / "scene_mjx_feetonly_flat_terrain.xml"


def task_to_xml(task_name: str) -> Path:
    """Map a task name to its scene XML. For now only flat terrain exists;
    rough terrain and getup variants can be added later by mirroring the Go1
    set."""
    mapping = {
        "flat_terrain": FEET_ONLY_FLAT_TERRAIN_XML,
        "joystick_flat_terrain": FEET_ONLY_FLAT_TERRAIN_XML,
    }
    if task_name not in mapping:
        raise ValueError(
            f"Unknown Go2 task {task_name!r}. Known: {sorted(mapping)}"
        )
    return mapping[task_name]


# --- Names that appear inside the compiled MJCF -----------------------------
# These must match the names in scene_mjx_feetonly_flat_terrain.xml (and the
# included go2_mjx.xml from menagerie).

# Go2 menagerie names the IMU site "imu" (same as Go1).
IMU_SITE = "imu"

# Go2 root body in menagerie is "base" (Go1 calls it "trunk"). If you swap
# in a different Go2 XML and it errors at startup, this is the first thing
# to verify with: `mujoco.MjModel.from_xml_path(...).body("base")`.
ROOT_BODY = "base"

# Order MUST match the order in which feet sites appear in qpos / sensors so
# that downstream code (rewards, swing-phase tracking) indexes consistently
# across Go1 and Go2.
FEET_SITES = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]

# Foot collision geoms. Go2 menagerie names these FR/FL/RR/RL, matching Go1.
FEET_GEOMS = ["FR", "FL", "RR", "RL"]

# Per-foot named sensors emitted by our scene XML.
FEET_POS_SENSOR = [f"{f}_pos" for f in ["FR", "FL", "RR", "RL"]]
FEET_GLOBAL_LINVEL_SENSOR = [f"{f}_global_linvel" for f in ["FR", "FL", "RR", "RL"]]
FEET_FLOOR_FOUND_SENSOR = [f"{f}_floor_found" for f in ["FR", "FL", "RR", "RL"]]

# Body-frame and world-frame sensors (names mirror Go1's scene XML).
GYRO_SENSOR = "gyro"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
UPVECTOR_SENSOR = "upvector"
FORWARDVECTOR_SENSOR = "forwardvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
POSITION_SENSOR = "position"
ORIENTATION_SENSOR = "orientation"
