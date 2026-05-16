# Retargeting: 2D cat keypoints to Go2 reference trajectories

This doc covers the Phase 2a MVP pipeline that lives in
`src/robot_pet_cat/retargeting.py`. It takes a JSON of 2D keypoints from a
side-view cat video and produces a Go2 `.npz` reference trajectory consumable
by the AMP trainer.

The MVP intentionally separates perception (your problem) from retargeting
(this module's problem). Drop-in perception models are listed at the bottom.

## End-to-end use

```bash
# 1. (Optional) regenerate the synthetic-walk fixture for sanity check
python -c "from robot_pet_cat.retargeting import write_synth_walk; \
  write_synth_walk('data/motion_clips_raw/synth_walk.json')"

# 2. Run the retarget on every .json in the input directory
rpc retarget --clips data/motion_clips_raw --out data/motion_clips
```

Each input `clip.json` produces `clip.npz` containing:

| key   | shape    | meaning                                    |
| ----- | -------- | ------------------------------------------ |
| qpos  | (T, 19)  | Go2 joint positions per frame              |
| qvel  | (T, 19)  | finite-difference velocities at clip dt    |
| dt    | (1,)     | timestep in seconds                        |
| fps   | (1,)     | source clip frame rate                     |

`qpos` layout: `[x, y, z, qw, qx, qy, qz, FL_hip_x, FL_hip_y, FL_knee,
FR_hip_x, FR_hip_y, FR_knee, RL_..., RR_...]`.

## Keypoint JSON format

```json
{
  "fps": 30.0,
  "scale_m_per_px": 0.005,
  "floor_y_px": 600,
  "image_height_px": 720,
  "keypoint_names": [
    "nose", "withers", "hips", "tail_base",
    "fl_paw", "fr_paw", "rl_paw", "rr_paw",
    "fl_elbow", "fr_elbow", "rl_knee", "rr_knee"
  ],
  "frames": [
    {"t": 0.0,   "keypoints": [[100,50], [80,30], ...]},
    {"t": 0.033, "keypoints": [...]},
    ...
  ]
}
```

Coordinates are image-space pixels with origin at top-left (standard OpenCV).
`scale_m_per_px` is the size of one pixel in world meters — measure this once
per clip by knowing the cat's body length (typical housecat: 0.45 m withers-
to-hips). `floor_y_px` is the image y-coordinate of the floor at the cat's
position; this defines world z=0.

The `fl_elbow`, `fr_elbow`, `rl_knee`, `rr_knee` keypoints are optional today
(the MVP IK ignores them). They become useful when we upgrade to a 6-DOF
per-leg target or when we need to disambiguate ill-posed paw positions.

## What the pipeline does, step by step

```
keypoint json
     |
     v
[1] lift_to_world    side-view assumption: image y -> world z, image x ->
                    world x, world y forced to 0
     |
     v
[2] root_pose_per_frame    trunk center = midpoint(withers, hips);
                           pitch from (hips -> withers) angle;
                           trunk z clipped >= GO2.nominal_height_m
     |
     v
[3] per-leg IK     for each frame and each of the 4 legs:
                     foot_target_world = paw_kp lifted to 3D
                     foot_target_body  = quat_inv(root) * (foot - root)
                     (hip_x, hip_y, knee) = closed-form IK,
                                            hip_x forced to 0 (MVP)
     |
     v
[4] assemble qpos = [root_xyz, root_quat, 12 joint angles]
                    + qvel from finite differences
     |
     v
.npz file
```

### MVP simplifications

The pipeline trades retargeting fidelity for "works end-to-end with one
keypoint detector and no manual mocap." Specific compromises:

1. **Side-view assumption.** The lift assumes camera y-axis is perpendicular
   to the cat's sagittal plane. Off-axis footage will produce systematic
   errors in z and miss lateral foot motion entirely.
2. **`hip_x` (leg roll) is zero.** Side view gives no lateral information.
   Left/right paws differ only by where their hip joints sit in body frame.
   Real cats have non-zero leg splay; we lose that.
3. **Yaw and roll are zero.** Only pitch is recovered from the spine.
4. **Cat-to-Go2 scale is implicit.** We use `GO2.nominal_height_m` as a soft
   floor for trunk height, so a too-tall cat ends up with fully-extended legs
   (knee saturated at `GO2.knee_max`). Real fix: compute a per-clip scale
   factor from the cat's withers-to-hips distance.
5. **No contact constraints.** The IK places paws at whatever world position
   the keypoint says, even if that's underground or floating. Real fix:
   project paws onto a contact plane during stance phases.

These all become problems if you read `qpos.npy` and try to play it back in
MuJoCo expecting a faithful reconstruction. They are *not* problems for AMP,
which uses these clips as a distribution to match, not trajectories to track.

## Validating with the synthetic fixture

The module includes `synth_cat_walk()`, a procedural 1-second cat trot. The
fixture is committed at `data/motion_clips_raw/synth_walk.json` and is the
basis of all the tests in `tests/test_retargeting.py`:

```bash
PYTHONPATH=src pytest tests/test_retargeting.py -v --basetemp=/tmp/pytest-rpc
```

(The `--basetemp=/tmp/...` flag dodges a pytest cleanup quirk on the
Windows-mounted workspace; on a Linux/macOS box you can drop it.)

If you change anything in the math, those tests catch regressions before the
output looks subtly wrong on real clips.

## Plugging in real perception

Replace the JSON-loading step with one of these:

### SuperAnimal-Quadruped (DeepLabCut)

Best off-the-shelf option for animal pose in May 2026. Universal model
covering cats, dogs, horses; no training needed.

```python
import deeplabcut as dlc
config = dlc.create_pretrained_project(
    ProjectName="cat-superanimal",
    Experimenter="jeffrah00",
    videos=["data/motion_clips_raw/your_clip.mp4"],
    model="superanimal_quadruped",
)
dlc.analyze_videos(config, ...)
# Then read the resulting .h5 and translate to our keypoint JSON schema.
```

Pros: works on cats out of the box; 39 keypoints, more than we need.
Cons: heavyweight install (`deeplabcut` pulls in tensorflow).

### MMPose AnimalPose

```python
from mmpose.apis import init_model, inference_topdown
model = init_model(
    config="configs/animal_2d_keypoint/animalpose-hrnet-w48.py",
    checkpoint="...",
)
results = inference_topdown(model, img)
# results[0].pred_instances.keypoints -> (1, 20, 2)
```

Pros: lighter than DLC; well-maintained.
Cons: 20-keypoint format needs mapping to our 12-keypoint convention.

### BARC (3D, future upgrade)

If we want real 3D pose without the side-view assumption, BARC fits a
parametric cat mesh (SMAL-derived) to a video.

- Page: https://barc.is.tue.mpg.de/
- Replaces both `load_keypoints_json` and `lift_to_world`. The retargeting
  step still applies but operates on 3D positions directly.
- This is the right upgrade once the MVP is working end-to-end.

## Where this goes next

Phase 2b builds the AMP trainer that consumes these `.npz` files. The
discriminator is trained on `(qpos[t], qpos[t+1])` transitions, so the format
above is exactly what it needs. The trainer is in
`src/robot_pet_cat/motion/amp_trainer.py` (currently a stub).
