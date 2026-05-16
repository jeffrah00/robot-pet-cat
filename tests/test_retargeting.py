"""End-to-end test for the keypoint -> Go2 retargeting pipeline.

Uses the synthetic walk fixture (`synth_cat_walk`) so the test runs without
requiring a real cat clip or pose model. Validates that:

  - the pipeline produces qpos with the right shape and dtype
  - root height stays positive (cat doesn't sink through the floor)
  - joint angles respect Go2's documented limits
  - finite-difference velocities are consistent with positions

These are sanity checks, not motion-quality checks. AMP will judge motion
quality from the discriminator side once we wire that up in Phase 2b.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from robot_pet_cat.retargeting import (
    GO2,
    KP_NAMES,
    N_KP,
    QPOS_DIM,
    load_keypoints_json,
    process_directory,
    retarget_clip,
    synth_cat_walk,
    write_synth_walk,
)


def _make_clip(tmp_path: Path) -> Path:
    out = tmp_path / "synth_walk.json"
    write_synth_walk(out)
    return out


def test_fixture_schema(tmp_path):
    path = _make_clip(tmp_path)
    data = json.loads(path.read_text())
    assert data["keypoint_names"] == KP_NAMES
    assert len(data["frames"][0]["keypoints"]) == N_KP
    assert data["fps"] == 30.0
    # 1-second clip at 30 fps => 31 frames (inclusive endpoints).
    assert len(data["frames"]) == 31


def test_loader_shape(tmp_path):
    path = _make_clip(tmp_path)
    clip = load_keypoints_json(path)
    assert clip.keypoints.shape == (31, N_KP, 2)
    assert clip.fps == 30.0
    assert clip.dt == pytest.approx(1.0 / 30.0)


def test_pipeline_output_shape(tmp_path):
    path = _make_clip(tmp_path)
    clip = load_keypoints_json(path)
    data = retarget_clip(clip)

    assert data["qpos"].shape == (clip.n_frames, QPOS_DIM)
    assert data["qvel"].shape == (clip.n_frames, QPOS_DIM)
    assert data["qpos"].dtype == np.float64


def test_root_above_floor(tmp_path):
    """Trunk z should be >= GO2.nominal_height_m everywhere."""
    path = _make_clip(tmp_path)
    clip = load_keypoints_json(path)
    qpos = retarget_clip(clip)["qpos"]
    root_z = qpos[:, 2]
    assert (root_z >= GO2.nominal_height_m - 1e-6).all(), root_z


def test_joint_limits_respected(tmp_path):
    """Per-leg joint angles should fall within Go2's documented limits."""
    path = _make_clip(tmp_path)
    clip = load_keypoints_json(path)
    qpos = retarget_clip(clip)["qpos"]

    joints = qpos[:, 7:]  # (T, 12)
    for leg in range(4):
        j = leg * 3
        hip_x = joints[:, j + 0]
        hip_y = joints[:, j + 1]
        knee  = joints[:, j + 2]
        assert (hip_x == 0).all(), "MVP forces hip_x=0"
        assert (hip_y >= GO2.hip_y_min - 1e-6).all()
        assert (hip_y <= GO2.hip_y_max + 1e-6).all()
        assert (knee  >= GO2.knee_min  - 1e-6).all()
        assert (knee  <= GO2.knee_max  + 1e-6).all()


def test_quaternion_unit_norm(tmp_path):
    path = _make_clip(tmp_path)
    clip = load_keypoints_json(path)
    qpos = retarget_clip(clip)["qpos"]
    quat = qpos[:, 3:7]
    norms = np.linalg.norm(quat, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_velocity_is_finite_diff_of_position(tmp_path):
    """qvel[1:] should equal finite difference of qpos for the non-root joints."""
    path = _make_clip(tmp_path)
    clip = load_keypoints_json(path)
    data = retarget_clip(clip)
    qpos = data["qpos"]
    qvel = data["qvel"]
    dt = clip.dt
    expected = (qpos[1:] - qpos[:-1]) / dt
    np.testing.assert_allclose(qvel[1:], expected, atol=1e-9)


def test_process_directory_writes_npz(tmp_path):
    """End-to-end: dir of JSONs -> dir of NPZs."""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    write_synth_walk(in_dir / "synth_walk.json")
    written = process_directory(in_dir, out_dir)
    assert len(written) == 1
    assert written[0] == out_dir / "synth_walk.npz"
    assert written[0].exists()

    loaded = np.load(written[0])
    assert "qpos" in loaded.files
    assert "qvel" in loaded.files
    assert loaded["qpos"].shape == (31, QPOS_DIM)


def test_synth_seed_is_deterministic():
    """Same seed -> identical keypoints; different seeds -> different keypoints."""
    a = synth_cat_walk(seed=0)
    b = synth_cat_walk(seed=0)
    c = synth_cat_walk(seed=1)
    arr_a = np.array(a["frames"][0]["keypoints"])
    arr_b = np.array(b["frames"][0]["keypoints"])
    arr_c = np.array(c["frames"][0]["keypoints"])
    np.testing.assert_array_equal(arr_a, arr_b)
    assert not np.array_equal(arr_a, arr_c)
