# Roadmap

Phased plan from zero to a cat-acting quadruped in sim. Each phase has a concrete
"done" signal so you know when to move on.

---

## Phase 0 — Accounts & infrastructure (½ day)

**Goal:** every external service we need exists and is wired up.

- [ ] All accounts in [`ACCOUNTS.md`](ACCOUNTS.md) created and verified
- [ ] GitHub repo created and this scaffold pushed
- [ ] Hugging Face username noted, `huggingface-cli login` works locally
- [ ] RunPod (or Lambda) account funded with $20 to start
- [ ] W&B account created, API key saved
- [ ] Cloudflare R2 bucket created (or Backblaze B2)

**Done when:** `huggingface-cli whoami` and `wandb login --verify` both succeed.

---

## Phase 1 — Sim smoke test (1–2 days)

**Goal:** Unitree Go2 stands up in MuJoCo and can be controlled with random joint targets.

- [ ] Pull Unitree Go2 MJCF from `mujoco_menagerie`
- [ ] Build a minimal `MujocoEnv` wrapper in `src/robot_pet_cat/sim/`
- [ ] Render a 5-second rollout locally (headless on cloud, GUI locally if available)
- [ ] Add a pytest that loads the model and steps the sim for 100 frames

**Done when:** `pytest tests/test_sim.py` is green and you have a rendered MP4 of the robot
flailing around.

---

## Phase 2 — Low-level locomotion (1–2 weeks)

**Goal:** PPO policy that takes `(vx, vy, ωz)` velocity commands and produces stable
walking on flat ground.

- [ ] Set up MuJoCo Playground's quadruped locomotion env (`joystick` task) as a starting
      point rather than building from scratch
- [ ] First training run: 50M env steps, 1× A100, ~4–6 hours, ~$5–10 on RunPod spot
- [ ] Evaluate: can it track forward/backward/strafe/turn commands at 0.3–1.0 m/s?
- [ ] Push the trained checkpoint to Hugging Face under `<you>/go2-locomotion`
- [ ] Domain randomization pass (friction, mass, latency) for sim-to-real-ish robustness

**Done when:** the policy follows a scripted velocity sequence on video without falling.

---

## Phase 3 — Cat video dataset (1 week, parallel with Phase 2)

**Goal:** ~10–50 hours of curated cat-behavior video with per-clip behavior labels and
extracted 2D pose.

- [ ] Build `data/youtube_scraper.py` using `yt-dlp` against a curated list of cat
      channels (Pexels, public-domain footage, CC-licensed YouTube channels)
- [ ] Filter clips: single cat, mostly visible, ≥3 seconds, decent lighting
- [ ] Run pose estimation with [MMPose AnimalPose](https://github.com/open-mmlab/mmpose)
      or [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) — outputs 2D keypoints
- [ ] Cluster behaviors using pose-feature embeddings → label classes
      (sit, walk, run, pounce, groom, lie, stretch, jump)
- [ ] Push the processed dataset to Hugging Face as `<you>/cat-behaviors`

**Done when:** dataset is on HF Hub with a datasheet and at least 8 behavior classes
each with ≥100 clips.

**Licensing note:** Only ingest clips you're allowed to use — CC-BY, CC0, Pexels,
public-domain, or your own footage. Skip random YouTube.

---

## Phase 4 — High-level behavior policy (2–3 weeks)

**Goal:** a VLA that takes the cat-robot's camera frame + a goal token
("rest", "explore", "play", "follow human") and emits a high-level command stream
that the locomotion controller can track.

- [ ] Fine-tune SmolVLA on cat dataset, where actions are *high-level* command vectors
      (velocity setpoint, gait style, head/tail pose) inferred from extracted pose
- [ ] Compare against a simpler baseline: a small MLP/transformer with the same inputs
- [ ] Wire VLA output → locomotion controller in `src/robot_pet_cat/vla/runner.py`
- [ ] Closed-loop eval in MuJoCo: spawn cat in a room, give it goal tokens, score
      behavior-class match against a held-out video set

**Done when:** human eval (n=5) judges the cat's behavior under each goal token as
"cat-like" more than 60% of the time.

---

## Phase 5 — Polish & demo (1 week)

- [ ] Reward shaping for cat-specific quirks (slow blinks, tail twitches, paw lifts)
- [ ] Multi-room environment with a "human" agent for follow/greet behaviors
- [ ] Recorded demo video pushed to a public HF Space
- [ ] Blog post / README walkthrough

---

## Out of scope (for now)

- **Real hardware.** No Go2 purchase yet — sim-only until the policy is convincing.
- **Photoreal vision.** Stick to MuJoCo's default rendering. Move to Isaac Sim only if
  the VLA needs photoreal training images.
- **Petting / touch.** Tactile is a much harder sensing stack; defer.
- **Sound.** Meowing is a separate generative model; defer.

---

## Budget estimate (Phase 0–4, $100–$500/mo tier)

| Phase | Service | Cost |
|------:|---------|-----:|
| 0 | Account setup | $0 |
| 1 | Local / free CPU | $0 |
| 2 | RunPod A100 spot, ~30 hr | ~$25 |
| 3 | Storage R2 100 GB + small GPU for pose | ~$15 |
| 4 | RunPod A100 spot, ~50 hr | ~$45 |
| — | HF Pro (optional, private datasets) | $9/mo |
| — | W&B | $0 (free tier) |
| **Total to a working demo** | | **~$100–$150 one-time + small monthly** |

Numbers assume aggressive use of spot/preemptible instances. Sustained on-demand could
3–5× this.
