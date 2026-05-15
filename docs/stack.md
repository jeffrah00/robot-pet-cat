# Stack rationale

Quick justification for every piece we picked, and what we explicitly rejected.
Updated for the three-tier (motion / skills / brain) architecture — see
[`docs/architecture.md`](architecture.md) and [`docs/cat-behavior.md`](cat-behavior.md).

## Simulator: MuJoCo + MuJoCo Playground

**Picked because:**
- Free, open source, mature physics.
- MJX (MuJoCo on JAX) gives GPU-parallel rollouts at Isaac-Lab-class throughput
  without the install pain.
- MuJoCo Playground ships pre-built quadruped tasks (`Go2Joystick`) that we
  build the AMP env on top of.
- Rendering quality is plain, but our brain reads scene state + (later)
  privileged features, not photoreal pixels. No need for Isaac.

**Rejected:**
- **Isaac Sim / Lab.** Heavier install, NVIDIA-only, higher VRAM floor, slower
  iteration. Only worth it if/when we add a VLM-as-policy head that needs
  photoreal pixels — Phase 7+ problem.
- **Genesis.** Promising, ecosystem still thin.
- **PyBullet.** Older, less accurate.

## Robot model: Unitree Go2

**Picked because:**
- Open MJCF in `mujoco_menagerie`, maintained by DeepMind.
- ~15 kg, closer to cat proportions than ANYmal or Spot.
- Largest body of community RL work to draw on.

**Rejected:**
- **Spot.** Closed sim assets.
- **ANYmal.** Too big (~30 kg, 0.7 m tall) — gait won't read as catty.
- **MIT Mini Cheetah.** Community models less reliable than Unitree's.

## Low-level controller: AMP-trained PPO

**Picked because:**
- AMP ([Peng et al. 2021](https://xbpeng.github.io/projects/AMP/)) is the
  validated recipe for "make this policy move like the things in this clip
  set." Quadruped-from-animal-mocap version was demonstrated in
  [Peng et al. 2020](https://xbpeng.github.io/projects/Robotic_Imitation/index.html).
- Style discriminator handles morphology mismatch gracefully — we're matching
  distribution, not trajectory.
- Cat-style motion at Tier 1 means every skill at Tier 2 inherits cat-feel
  without us doing anything skill-specific.

**Rejected:**
- **Vanilla PPO with hand-shaped reward.** Looks generic-quadruped.
- **DeepMimic per-clip imitation.** Too brittle, requires high-quality clips
  we don't have.
- **Behavior cloning from cat video.** No action labels exist; the actions
  would have to be retargeted joint positions anyway, which is the AMP recipe
  in disguise but harder.

## Mid-level skills: subgoal-conditioned RL

**Picked because:**
- Each skill is its own short PPO run on top of the AMP backbone — a few hours
  on a single GPU. Cheap.
- Skills compose well: `walk_to(target)` then `sit` then `swat(ball)` works
  without explicit chaining logic.
- The jump skill specifically benefits from the curriculum recipe in
  [Atanassov et al. 2024](https://arxiv.org/abs/2401.16337) — well-known and
  reproducible.

**Rejected:**
- **One monolithic policy.** Harder to debug, longer to train, harder to
  swap individual capabilities.
- **Scripted skill primitives.** That's what makes Sony Aibo feel like a vending
  machine. Skills are learned so they look fluid.

## High-level brain: RL with intrinsic rewards + mood latent

**Picked because:**
- Curiosity + comfort + play are well-validated reward structures in their own
  right; we're composing, not inventing.
- The mood latent is a tiny addition (~50 lines) that buys huge perceived
  variation in behavior.
- Sampling temperature gives "coherent unpredictability" — cat-feel.

**Rejected:**
- **State machine / behavior tree.** Feels scripted because it is scripted.
- **Pure imitation from cat behavior labels.** Would require huge labeled
  dataset we don't have, and the result still wouldn't have mood drift.
- **VLM-as-policy from day one.** Big, slow, hard to train, doesn't add
  cat-feel — defer to Phase 7+.

## Cloud compute: RunPod (primary), Lambda Labs (backup)

**Picked because:**
- RunPod has the cheapest A100 spot rates (~$0.40-0.80/hr).
- Per-minute billing matches our bursty workload.
- Lambda is more reliable for long runs — fallback when spot preemption
  hurts.

**Rejected:**
- **AWS / GCP / Azure.** Pricier, more setup.
- **Vast.ai.** Cheaper, but node quality varies.
- **Colab.** Session timeouts kill RL.

## Storage: Hugging Face Hub only

**Picked because:**
- Models (motion checkpoint, every skill, brain) go to HF.
- Motion clips dataset is tiny (~100 MB even for 30 clips). HF handles it.
- No raw video archive needed in this architecture — we don't ingest hours of
  cat video, just curate ~30 clips.

**Rejected:**
- **Cloudflare R2 / Backblaze B2.** Was in the previous plan when we expected
  tens of GB of raw video. Architecture pivot eliminated that need.
- **S3, GDrive, Dropbox.** Same reasons as before.

## Experiment tracking: Weights & Biases

**Picked because:** unlimited free personal tier, good RL rollout video logging,
universal in the field.

**Rejected:** MLflow (heavier), TensorBoard (no remote), Neptune (smaller).
