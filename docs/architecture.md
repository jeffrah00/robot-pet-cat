# Architecture

The cat is a three-tier learning stack. The split is deliberate: each tier has
a different training signal, a different reward function, and a different time
constant.

## Three tiers

### Tier 1 — motion (50 Hz)

An AMP-trained PPO policy. Inputs: proprioception (joint angles, joint
velocities, IMU) + a low-dim command vector (velocity setpoint, gait flag,
head pose, body height). Outputs: joint torques.

What this tier learns: *how to move like a cat*. The style discriminator
biases the policy distribution toward cat motion clips — slow weight-shifts,
soft landings, settling motions when stationary, anticipatory crouches.

It does NOT learn what to do. It just makes any commanded motion look catty.

### Tier 2 — skills (10-20 Hz)

A library of subgoal-conditioned policies. Each takes the robot observation +
a small structured goal and emits the Tier-1 command vector. Examples:

- `WalkTo(target_xy)` — emit velocity commands pointing at the target
- `Sit()` — emit a held crouched-low command
- `JumpTo(surface_xyz)` — curriculum-learned jump sequence
- `Swat(object_xyz)` — approach + paw-lift command sequence

Skills are RL-trained with skill-specific task reward + the shared AMP style
reward inherited from Tier 1. They look cat-like because Tier 1 makes
everything look cat-like.

### Tier 3 — brain (0.3-1 Hz)

A high-level RL policy that picks which skill to invoke. Inputs: first-person
camera frame (or privileged scene state during training) + last-skill identity
+ a six-dim mood latent. Outputs: skill_id (categorical) + target (continuous).

The brain optimizes a *composite* reward:

```
total_reward(t) = w_curiosity(mood) * R_curiosity(t)
                + w_comfort(mood)   * R_comfort(t)
                + w_play(mood)      * R_play(t)
```

Each weight is a sigmoid function of one mood-latent dimension. The mood
itself drifts under Ornstein-Uhlenbeck dynamics over minutes of sim time.
This is what produces session-to-session behavioral variation.

## Data flow at inference

```
        camera frame ─┐
        scene state  ─┼─► brain (0.3-1 Hz) ─► skill_id, target
        mood latent  ─┘                        │
                                               ▼
                                   skill (10-20 Hz) ─► velocity, gait, head pose
                                                       │
                                                       ▼
                                          motion (50 Hz) ─► joint torques
                                                            │
                                                            ▼
                                                       MuJoCo
```

Each tier runs at its own rate. Tier outputs are held between updates.

## Why this produces "cat-like" behavior rather than "scripted robot"

Three claims, one per tier:

1. **Motion looks like a cat** because the AMP style discriminator scored
   cat clips during training. There are no scripted joint trajectories. The
   policy *generates* the gait, conditioned on the style.

2. **Behavior emerges, doesn't get listed** because Tier 3 is plain RL, not a
   finite state machine. The cat sits by the window *because* the comfort
   reward gradient there is positive, not because we wrote
   `if near_window: sit()`. Add a fireplace and the cat will start sitting
   near the fireplace too without code changes.

3. **Behavior shifts over a session** because the mood latent drifts. Same
   cat, same room, different behavior at minute 1 vs minute 30.

## What we don't claim

- We don't claim cat-accurate. We claim cat-styled.
- We don't claim domain transfer to real Go2. That's a later phase.
- We don't claim every emergent behavior will be desirable. Reward shaping
  takes iteration. Expect a phase of "the cat will not stop jumping" before
  things settle.

## References

- AMP: [Peng et al. 2021](https://xbpeng.github.io/projects/AMP/) for the
  style-prior training recipe
- Quadruped-from-mocap retargeting: [Peng et al. 2020](https://xbpeng.github.io/projects/Robotic_Imitation/index.html)
- Curiosity reward: [Pathak et al. 2017 ICM](https://arxiv.org/abs/1705.05363)
  and [Burda et al. 2018](https://arxiv.org/abs/1808.04355) for the
  large-scale study of curiosity-driven learning
- Jump skill: [Atanassov et al. 2024](https://arxiv.org/abs/2401.16337)

## Sim → real (much later)

Out of scope for v1. Notes for future-us:
- AMP-trained policies tend to transfer better than vanilla PPO because the
  style prior implicitly regularizes toward smoother motions
- Domain randomization in Tier 1 (friction, mass, latency) is the standard
  trick; do this before any hardware attempt
- The mood latent stays in software; only the joint commands go to hardware
