# robot-pet-cat

A simulated robot pet cat — quadruped robot body, learned cat-style motion +
mid-level skills + a curiosity-driven brain that picks what to do based on a
slow-drifting mood. The goal is cat-*feel*, not cat-accuracy.

There's no commercial cat-shaped robot, so we use the Unitree Go2 inside
MuJoCo. Behaviors like "sits by the window," "swats the ball," and "jumps on
the couch" are not scripted — they emerge from the reward structure plus a
mood latent that biases what the cat wants to do moment by moment.

See [`docs/cat-behavior.md`](docs/cat-behavior.md) for why this composition
produces cat-feel instead of robot-feel.

## Status

Early scaffold. Not yet runnable end-to-end. See [`ROADMAP.md`](ROADMAP.md)
for the plan and [`ACCOUNTS.md`](ACCOUNTS.md) for the external services you
need first.

## Architecture (three tiers, each learned)

```
                ┌───────────────────────────────────────────────┐
                │  brain  (RL + mood latent, 0.3-1 Hz)          │
                │  vision + scene + mood  →  skill + target     │
                └─────────────────────┬─────────────────────────┘
                                      │
                                      ▼
                ┌───────────────────────────────────────────────┐
                │  skills  (subgoal-conditioned RL, 10-20 Hz)   │
                │  goal  →  velocity / gait / head pose         │
                └─────────────────────┬─────────────────────────┘
                                      │
                                      ▼
                ┌───────────────────────────────────────────────┐
                │  motion  (AMP-trained PPO, 50 Hz)             │
                │  cat-style gait, posture, balance recovery    │
                └─────────────────────┬─────────────────────────┘
                                      │
                                      ▼
                ┌───────────────────────────────────────────────┐
                │  MuJoCo + Unitree Go2 + household scene       │
                └───────────────────────────────────────────────┘
```

Tier 1 makes everything look catty. Tier 2 is a small library of competences.
Tier 3 picks what to do, with curiosity / comfort / play rewards weighted by
mood.

## Stack

- **Simulator:** [MuJoCo](https://mujoco.org) + [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) (MJX for GPU rollouts)
- **Robot:** Unitree Go2 (open MJCF in `mujoco_menagerie`)
- **Tier 1 motion:** AMP ([Peng et al. 2021](https://xbpeng.github.io/projects/AMP/)) on top of PPO
- **Tier 2 skills:** subgoal-conditioned PPO; jump uses the [Atanassov 2024](https://arxiv.org/abs/2401.16337) curriculum
- **Tier 3 brain:** PPO with ICM curiosity ([Pathak 2017](https://arxiv.org/abs/1705.05363)) + dense comfort + play + mood latent
- **Cloud compute:** RunPod (A100 spot) or Lambda Labs
- **Storage:** Hugging Face Hub (checkpoints + retargeted motion clips)
- **Tracking:** Weights & Biases

See [`docs/stack.md`](docs/stack.md) for the rationale and what was rejected.

## Quickstart (once accounts are set up)

```bash
# 1. clone and install
git clone git@github.com:jeffrah00/robot-pet-cat.git
cd robot-pet-cat
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. log in
huggingface-cli login
wandb login

# 3. smoke-test the sim (CPU is fine)
python -m robot_pet_cat.cli sim --steps 200

# 4. train Tier 1 motion (needs GPU; see scripts/setup_cloud.sh first)
bash scripts/train_motion.sh
```

## Layout

```
robot-pet-cat/
├── src/robot_pet_cat/        Python package
│   ├── sim/                  MuJoCo env wrappers, robot loading
│   ├── motion/               Tier 1 — AMP-trained cat-style locomotion
│   ├── skills/               Tier 2 — subgoal-conditioned skill policies
│   └── brain/                Tier 3 — high-level RL + mood + intrinsic rewards
├── configs/                  YAML hyperparams for each tier
│   ├── motion/cat_amp.yaml
│   ├── skills/               (per-skill configs added as we go)
│   └── brain/cat_brain.yaml
├── scripts/                  Setup, training, motion-clip extraction
├── data/                     Retargeted motion clips (gitignored)
├── checkpoints/              Model weights (gitignored, push to HF)
├── notebooks/                Exploration / analysis
├── docs/                     Architecture, cat-behavior, stack rationale
└── tests/                    Unit + integration tests
```

## License

MIT — see [`LICENSE`](LICENSE).
