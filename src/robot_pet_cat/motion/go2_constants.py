"""Constants for the Unitree Go2 locomotion env.

This is the Go2 analog of mujoco_playground's go1_constants. We need a separate
module because:
  - Go2's menagerie XML names foot SITES differently than Go1 (FL_foot vs FL),
    even though the foot GEOMS happen to share the same FL/FR/RL/RR names.
  - Go2's home keyframe differs slightly from Go1's (symmetric hips at 0.0
    instead of +/-0.1, base height 0.27 vs 0.278).
  - We point the asset loader at unitree_go2 in menagerie, not unitree_go1.

Joint names (FL_hip_joint, FL_thigh_joint, FL_calf_joint, ...) are identical
across Go1 and Go2 in menagerie, so any reward logic that references qpos
slices by joint order keeps working unchanged.
"""

from __future__ import annotations

from pathlib import Path

ROOT_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "go2_scenes"
)
FEET_ONLY_FLAT_TERRAIN_XML = ROOT_PATH / "scene_mjx_feetonly_flat_terrain.xml"


def task_to_xml(task_name: str) -> Path:
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

# Go2 menagerie names the IMU site "imu" (same as Go1).
IMU_SITE = "imu"

# Go2 menagerie names the root body "base" (Go1 calls it "trunk").
ROOT_BODY = "base"

# Foot SITES as menagerie's go2_mjx.xml declares them. The "_foot" suffix
# matters: Go1's joystick task constructs per-foot sensor names on the fly
# as f"{site}_global_linvel", so whatever we put here is also the prefix
# every per-foot sensor below must use.
FEET_SITES = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]

# Foot collision GEOMS. menagerie's Go2 names these FR/FL/RR/RL, matching Go1.
FEET_GEOMS = ["FR", "FL", "RR", "RL"]

# Per-foot sensor names: must be exactly {site}_{kind} for every site in
# FEET_SITES, so Go1's constructed lookups resolve.
FEET_POS_SENSOR = [f"{s}_pos" for s in FEET_SITES]
FEET_GLOBAL_LINVEL_SENSOR = [f"{s}_global_linvel" for s in FEET_SITES]
FEET_FLOOR_FOUND_SENSOR = [f"{s}_floor_found" for s in FEET_SITES]

# Body-frame and world-frame sensors (names match Go1's scene XML and
# menagerie's go2_mjx.xml).
GYRO_SENSOR = "gyro"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
UPVECTOR_SENSOR = "upvector"
FORWARDVECTOR_SENSOR = "forwardvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
POSITION_SENSOR = "position"
ORIENTATION_SENSOR = "orientation"
