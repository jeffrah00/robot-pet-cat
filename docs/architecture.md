# Architecture

## Two-tier policy split

We separate *how to move* from *what to do*.

- **Low level (50 Hz)** — a PPO policy trained in sim takes a velocity command
  `(vx, vy, ωz)` plus proprioception (joint angles/velocities, IMU) and outputs
  joint torques. This is the standard quadruped locomotion problem; lots of
  prior art.
- **High level (5 Hz)** — a fine-tuned SmolVLA takes the cat's onboard camera
  frame + recent history + a goal token (e.g. `"explore"`, `"rest"`, `"follow"`)
  and outputs a high-level command vector: velocity setpoint, gait choice,
  head/tail pose. This is what we learn from cat videos.

Why split? Cat videos contain no joint-torque labels. Web data tells us **what a
cat does** in different situations; it cannot tell us **how to balance**.

## Data flow at inference

```
camera frame  ─┐
goal token    ─┼──► SmolVLA ──► HighLevelCommand ──► PPO policy ──► torques ──► MuJoCo
proprioception ┘                                         ▲
                                                         │
                                          (proprioception loop runs at 50 Hz,
                                           VLA is queried at 5 Hz; command is
                                           held between updates)
```

## Data flow at training

Two independent pipelines:

```
Phase 2 (sim-only):                Phase 3+4 (web + sim):
  random init                        cat videos
      │                                  │
      ▼                                  ▼
  PPO in MuJoCo                    yt-dlp + filter
      │                                  │
      ▼                                  ▼
  Go2 locomotion policy           pose extraction (MMPose)
  (HF: <user>/go2-locomotion)            │
                                         ▼
                                   behavior labels +
                                   inferred high-level cmds
                                         │
                                         ▼
                                   SmolVLA fine-tune
                                   (HF: <user>/smolvla-cat)
```

## Inferring high-level commands from video

This is the trickiest step. We don't have ground truth, so we approximate:

1. Run pose estimation per frame → 2D keypoints (head, spine, four paws, tail).
2. Fit a simple kinematic prior to recover smoothed 3D body velocity and
   orientation.
3. Use velocity magnitude + gait-frequency analysis to label gait (stand /
   walk / trot / crouch / leap).
4. Use head pitch from keypoints directly; same for tail base angle.
5. The clustered behavior class (sit, groom, pounce, etc.) becomes the goal
   token that conditions training.

This is noisy. That's fine — the VLA learns the distribution, not exact labels.

## Sim → real (much later)

Out of scope for v1. Notes for future-us:
- Domain randomization in Phase 2 already covers friction/mass/latency.
- For real Go2, hardware deployment uses Unitree's SDK; the locomotion policy
  needs to be exported to TorchScript and the action latency budget shrinks.
- The VLA would run on a host laptop and stream commands over UDP.
