# robot-pet-cat

A simulated robot pet cat — quadruped robot body, vision-language-action (VLA) policy
brain, behavior trained from cat videos collected off the web.

This project lives entirely in simulation for now. There is no commercial cat-shaped
robot, so we use an open quadruped model (Unitree Go2) inside MuJoCo, and we layer a
fine-tuned VLA on top to make it act cat-like (lounging, walking, pouncing, grooming,
greeting, sleeping).

## Status

Early scaffold. Not yet runnable end-to-end. See [`ROADMAP.md`](ROADMAP.md) for the plan
and [`ACCOUNTS.md`](ACCOUNTS.md) for the external services you need before code can run.

## Architecture (two-tier policy)

```
                ┌───────────────────────────────────────────────┐
                │  High-level VLA  (SmolVLA, fine-tuned on cat) │
                │  vision + language goal  →  behavior command  │
                └─────────────────────┬─────────────────────────┘
                                      │  velocity, heading, pose
                                      ▼
                ┌───────────────────────────────────────────────┐
                │  Low-level locomotion controller  (RL/PPO)    │
                │  velocity command  →  joint torques           │
                └─────────────────────┬─────────────────────────┘
                                      │  torques
                                      ▼
                ┌───────────────────────────────────────────────┐
                │  MuJoCo + Unitree Go2 model                   │
                └───────────────────────────────────────────────┘
```

The split matters because web cat videos do not have joint-torque labels. We learn
*how to move* from RL in sim, and we learn *what a cat would do* from videos.

## Stack

- **Simulator:** [MuJoCo](https://mujoco.org) + [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) (MJX for GPU parallel rollouts)
- **Robot:** Unitree Go2 (open MJCF model)
- **Low-level policy:** PPO (Brax/JAX or PyTorch + RSL-RL)
- **High-level policy:** [SmolVLA](https://huggingface.co/lerobot/smolvla_base) (LeRobot) — upgradeable to π0 / OpenPi later
- **Cloud compute:** RunPod or Lambda Labs (A100/H100 spot)
- **Cloud storage:** Hugging Face Hub (datasets + models) + Cloudflare R2 (raw video)
- **Experiment tracking:** Weights & Biases (free tier)

See [`docs/stack.md`](docs/stack.md) for why each piece was chosen.

## Quickstart (once accounts are set up)

```bash
# 1. clone and install
git clone git@github.com:<you>/robot-pet-cat.git
cd robot-pet-cat
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. log in to services
huggingface-cli login
wandb login

# 3. smoke-test the sim (CPU)
python -m robot_pet_cat.sim.mujoco_env --render

# 4. train low-level locomotion (needs GPU)
bash scripts/train_locomotion.sh
```

## Layout

```
robot-pet-cat/
├── src/robot_pet_cat/        Python package
│   ├── sim/                  MuJoCo env wrappers, robot loading
│   ├── locomotion/           Low-level PPO trainer + inference
│   ├── vla/                  High-level VLA runner, fine-tuning
│   └── data/                 Cat-video scraping + pose extraction
├── configs/                  YAML hyperparams for each stage
├── scripts/                  One-shot shell scripts (training, data collection)
├── data/                     Raw + processed datasets (gitignored)
├── checkpoints/              Model weights (gitignored, push to HF Hub)
├── notebooks/                Exploration / analysis
├── docs/                     Architecture and design notes
└── tests/                    Unit + integration tests
```

## License

MIT — see [`LICENSE`](LICENSE).
