# Stack rationale

Quick justification for every piece we picked, and what we explicitly rejected.

## Simulator: MuJoCo + MuJoCo Playground

**Picked because:**
- Free, open source, mature physics. Used by every serious quadruped lab.
- MJX (MuJoCo on JAX) gives GPU-parallel rollouts — Isaac-Lab-level throughput
  without the Isaac install pain.
- MuJoCo Playground ships pre-built quadruped tasks (`Go1Joystick`, `Go2Joystick`)
  with sane reward shaping and domain randomization. We don't reinvent it.
- Runs on a single A100 spot instance for the budgets we have.

**Rejected:**
- **Isaac Sim / Lab.** Photoreal vision is great, but: heavy install, NVIDIA-only,
  high VRAM floor, slower to iterate. We don't need photoreal pixels until the VLA
  is doing pixel-level inference from sim images — and even then we can swap in.
- **Genesis.** Promising but the ecosystem is thin. Few pretrained quadruped
  assets, fewer worked examples. Revisit in 6 months.
- **PyBullet.** Older, less accurate, smaller community now.

## Robot model: Unitree Go2

**Picked because:**
- Open MJCF lives in `mujoco_menagerie`, well-maintained by DeepMind.
- Small (~15 kg) — closer to cat proportions than ANYmal or Spot.
- Largest body of community RL work to draw on.

**Rejected:**
- **Boston Dynamics Spot.** Closed sim assets, no official MuJoCo model.
- **ANYmal.** Excellent model, but big (~30 kg, 0.7 m tall) and overpowered for a
  cat — gait will feel wrong.
- **MIT Mini Cheetah.** Cute, but the public MuJoCo model is community-maintained
  and less reliable than Unitree's.

## Low-level controller: PPO via Brax / MuJoCo Playground

**Picked because:**
- Standard RL recipe for quadruped locomotion; lots of reference runs to compare to.
- Brax is JAX-native, batches well on a single GPU, ~50M env steps in 4–6 hours.

**Rejected:**
- **Diffusion policy / world models** for low-level control. Cool, but overkill for
  flat-ground locomotion. Save complexity for the high-level VLA.

## High-level policy: SmolVLA (LeRobot)

**Picked because:**
- ~450M params — fine-tunable on a single A100 in a few hours. Fits the budget.
- Hugging Face LeRobot has a solid fine-tuning recipe, datasets format, and
  community.
- Vision + language + action is exactly the input/output shape we want.

**Rejected (for now, not forever):**
- **NVIDIA GR00T N1.5.** Tightly coupled to humanoid pretraining and Isaac
  workflows. Possible to adapt to quadruped but more friction than SmolVLA.
- **π0 / OpenPi (Physical Intelligence).** Much larger and more capable, but
  expensive to fine-tune. Plan to revisit in Phase 5 if SmolVLA's ceiling is hit.
- **OpenVLA-7B.** Robot-arm-pretrained; less useful for legged motion. Big.
- **End-to-end one-model VLA.** Cute in theory; in practice we get much more bang
  for our buck by separating *how to move* (RL) from *what to do* (VLA).

## Cloud compute: RunPod (primary), Lambda Labs (backup)

**Picked because:**
- RunPod has the cheapest A100 spot rates as of mid-2026 (~$0.40–0.80/hr).
- Per-minute billing, easy template UI, simple SSH.
- Lambda Labs is more reliable for >12-hour runs — keep it as fallback when spot
  preemption hurts.

**Rejected:**
- **AWS / GCP / Azure.** Higher prices, more setup overhead.
- **Vast.ai.** Even cheaper, but variability in node quality is annoying.
- **Colab Pro+.** Per-session timeouts kill long RL runs.

## Storage: Hugging Face Hub + Cloudflare R2

**Picked because:**
- HF is the right home for the *processed* dataset and the trained models —
  everyone in the community already pulls from there.
- R2's **no-egress** pricing makes raw video (tens of GB) painless to move around.

**Rejected:**
- **S3.** Egress fees eat budget when we re-download raw clips.
- **Google Drive / Dropbox.** Not API-friendly for ML pipelines.
- **HF for raw video.** Bloats the dataset repo and the LFS quotas don't love it.

## Experiment tracking: Weights & Biases

**Picked because:** free personal tier is unlimited, has great video logging for
RL rollouts, every ML person can read a W&B dashboard.

**Rejected:** MLflow (heavier self-host), TensorBoard (no remote logging), Neptune
(comparable to W&B but smaller).
