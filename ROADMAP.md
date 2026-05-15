# Roadmap

Phased plan to get from zero to a cat-acting quadruped in sim. Each phase has a
concrete "done" signal. Target: 6-10 weeks of evenings, ~$150-300 total cloud
spend.

The architecture is a three-tier learning stack:

1. **motion/** — AMP-trained low-level controller giving cat-style gait.
2. **skills/** — subgoal-conditioned mid-level policies (walk_to, sit, swat,
   jump_to, ...).
3. **brain/** — high-level RL policy that picks skills based on visual context
   and a slow-drifting mood latent. This is where cat personality lives.

See [`docs/cat-behavior.md`](docs/cat-behavior.md) for why this composition
produces emergent cat-like behavior rather than scripted robot-pet behavior.

---

## Phase 0 — Accounts & infrastructure (½ day)

**Goal:** every external service we need is set up and verified.

- [ ] All accounts in [`ACCOUNTS.md`](ACCOUNTS.md) created and verified
- [ ] GitHub repo created and this scaffold pushed
- [ ] Hugging Face username noted, `huggingface-cli login` works locally
- [ ] RunPod (or Lambda) account funded with $20 to start
- [ ] W&B account created, API key saved

**Done when:** `huggingface-cli whoami` and `wandb login --verify` both succeed.

---

## Phase 1 — Sim smoke test (1-2 days)

**Goal:** Unitree Go2 stands up in MuJoCo and can be controlled with random
joint targets.

- [ ] Pull Unitree Go2 MJCF from `mujoco_menagerie` as a git submodule
- [ ] Step the env headlessly and render an MP4 of the robot flailing around
- [ ] `pytest tests/test_sim.py` green

**Done when:** the rendered MP4 exists.

---

## Phase 2 — Motion: cat-style low-level controller (2-3 weeks)

**Goal:** an AMP-trained PPO policy that walks/turns/stops on flat ground with
cat-style gait, posture, and balance recovery — *not* a generic Go2 trot.

The discriminator that defines "cat-style" is trained against a small set of
retargeted cat motion clips. We need ~10-30 short clips total; this is a one-day
effort, not the multi-week dataset operation the previous plan described.

- [ ] Collect ~10-30 short cat motion clips (CC-licensed Pexels / Pixabay /
      Wikimedia / your own footage). 3-10 sec each. Variety: walking, sitting
      down, stretching, crouching, leaping.
- [ ] Extract 3D pose with [BARC](https://barc.is.tue.mpg.de/) or
      [SMAL](https://smal.is.tue.mpg.de/). MMPose AnimalPose as 2D fallback.
- [ ] Retarget pose sequences onto the Go2 skeleton via IK. Use Peng et al.
      2020's [motion_imitation](https://github.com/erwincoumans/motion_imitation)
      retargeting code as a starting point — adjust skeleton mapping for the
      cat skeleton. Quality bar is low: style not fidelity.
- [ ] Save retargeted clips as `.npz` reference trajectories in `data/motion_clips/`
- [ ] Train AMP: PPO with task reward (stay-upright + velocity-tracking) plus
      style reward from the AMP discriminator
- [ ] First training run: ~50M env steps, 1× A100, ~6-8 hours, ~$10-15 on
      RunPod spot
- [ ] Push checkpoint to `jeffrah00/go2-cat-motion` on Hugging Face

**Done when:** the policy follows a scripted velocity sequence on video, doesn't
fall, and a human who's never seen the project says "that looks more like a cat
walking than a Boston Dynamics dog."

**References:**
- [Peng et al. 2021 "AMP: Adversarial Motion Priors"](https://xbpeng.github.io/projects/AMP/)
- [Peng et al. 2020 "Learning Agile Robotic Locomotion Skills by Imitating Animals"](https://xbpeng.github.io/projects/Robotic_Imitation/index.html)
- [motion_imitation code](https://github.com/erwincoumans/motion_imitation)

---

## Phase 3 — Skills: mid-level subgoal-conditioned policies (2-3 weeks)

**Goal:** a small library of skills, each inheriting the AMP style prior, that
the brain can compose. Trained one at a time on top of the Phase 2 controller.

Start with the four that unlock the demo scenarios:

- [ ] `walk_to(target_xy)` — velocity-command wrapper, mostly the AMP policy
- [ ] `sit` — lower haunches, hold pose
- [ ] `jump_to(surface_xyz)` — curriculum-learned jump using the recipe in
      [Atanassov et al. 2024](https://arxiv.org/abs/2401.16337) "Curriculum-Based
      Reinforcement Learning for Quadrupedal Jumping". Trains in ~1 day on a
      single A100.
- [ ] `swat(object_xyz)` — approach + front-paw lift + nudge. Reward shaped on
      ball post-contact velocity. ~6-8 hours of training.

Then the ambient/idle skills (each trains fast, ~2-4 hours):

- [ ] `lie_down`, `stretch`, `groom`, `crouch`, `look_at`

Each skill exports a `Skill` subclass under `src/robot_pet_cat/skills/`,
checkpoint pushed to `jeffrah00/go2-cat-skills/<skill>`.

**Done when:** every skill runs in isolation in MuJoCo and the corresponding
test in `tests/test_skills.py` passes.

---

## Phase 4 — Household scene (1 week, can overlap with Phase 3)

**Goal:** a MuJoCo room the cat can live in, with the semantic affordance tags
the brain needs.

- [ ] Build `assets/scenes/living_room.xml`: floor, walls, a window (with sun
      patch), a couch (low table approximation is fine), a cat tree, a few
      movable balls/toys, optional plushie
- [ ] Source meshes from CC-BY Sketchfab + Polycam scans + the
      [ManiSkill object library](https://github.com/haosulab/ManiSkill)
- [ ] Tag each surface with semantic flags: `soft`, `elevated`, `warm`,
      `play_target`, `window`. These get exposed in the env's `scene_state` dict
- [ ] Add a "human" capsule that can walk on scripted paths for follow/avoid tests

**Done when:** the cat can be spawned in the room and the env exposes a
`scene_state` dict the brain can read.

---

## Phase 5 — Brain: high-level RL with curiosity, comfort, play, mood (2-3 weeks)

**Goal:** the cat picks skills on its own, in a way that looks cat-like across
multiple sessions in the same room.

- [ ] Implement the three intrinsic reward streams in
      `src/robot_pet_cat/brain/rewards.py`:
      - **Curiosity** — ICM-style forward-model prediction error
        ([Pathak et al. 2017](https://arxiv.org/abs/1705.05363))
      - **Comfort** — dense function of (elevated × soft × warm × low-acceleration)
      - **Play** — proportional to nearby movable-object velocity, with causal
        credit when the cat caused the velocity change
- [ ] Implement the mood latent in `src/robot_pet_cat/brain/mood.py` — already
      scaffolded; tune the OU drift and the sigmoid sharpness so transitions
      between sleepy/alert feel gradual (not flicker-y)
- [ ] Train the brain policy with PPO + composite reward, with mood-modulated
      weights. ~20M env steps, 1× A100, ~6-8 hours, ~$15-20
- [ ] Push checkpoint to `jeffrah00/go2-cat-brain`

**Done when:** in a fresh room with a ball, a couch, and a window, the cat
exhibits all three target behaviors *without being told to*: sits by the window
for ≥30 sec in some sessions, swats the ball in some sessions, jumps on the
couch in some sessions. Different sessions produce different sequences.

---

## Phase 6 — Polish & demo (1 week)

- [ ] Tune the mood schedule so the cat is roughly:
      "morning playful → midday lounging → evening alert"
- [ ] Add ear/tail twitch idle motion (small additive joint noise during
      stationary skills — this is the one place a tiny scripted touch helps)
- [ ] Record a 60-second demo video showing all three target behaviors plus
      ambient cat-vibe motion
- [ ] Push to a public Hugging Face Space
- [ ] Write a blog post / README walkthrough

---

## Out of scope

- **Real hardware.** No physical Go2 — sim only.
- **Photoreal vision.** MuJoCo default rendering is sufficient. Move to Isaac
  Lab only if a later VLM-as-policy upgrade needs photoreal pixels.
- **VLA / VLM as policy.** The Phase 5 brain is plain RL, not a fine-tuned VLM.
  Adding a small VLM head (Moondream / Qwen2-VL-2B) is a Phase 7+ upgrade once
  the RL brain works.
- **Tactile sensing, sound, multi-cat interaction.** All deferred.

---

## Budget estimate

| Phase | Service | Cost |
|------:|---------|-----:|
| 0 | Account setup | $0 |
| 1 | Local CPU | $0 |
| 2 | RunPod A100 spot, ~50 hr | ~$30-50 |
| 3 | RunPod A100 spot, ~80 hr | ~$50-80 |
| 4 | Local + asset purchases | ~$0-20 |
| 5 | RunPod A100 spot, ~30 hr | ~$20-35 |
| 6 | RunPod for demo renders | ~$5 |
| — | HF Pro (optional) | $9/mo |
| — | W&B free tier | $0 |
| **Total to demo** | | **~$120-200** one-time |

Aggressive spot usage assumed. Sustained on-demand could 2-3× this.
