# robot-pet-cat

A three-tier learning stack that turns a simulated Unitree Go1 quadruped into
something that *acts like a cat*. No external command input, no scripted
schedule — just a small skill library and a mood-driven brain that decides
what to do moment by moment.

**Paper:** [`paper/paper.pdf`](paper/paper.pdf) (arXiv submission in progress)
**Project page:** https://jeffrah00.github.io/robot-pet-cat/
**License:** MIT

<p align="center">
  <em>10-minute autonomous rollout:</em><br>
  <code>renders/brain_4skill_10min.mp4</code>
</p>

---

## The stack

```
brain    (Tier 3) — PPO + 6-D mood latent + 6-mode attractor, ~0.5 Hz
   ↓ skill ID
skills   (Tier 2) — { walk, crouch, lie_belly, stay } + scripted get_up
   ↓ velocity / joint targets
motion   (Tier 1) — Go1 velocity-tracking walker (mjlab PPO), 50 Hz
   ↓ joint torques
MuJoCo + Unitree Go1
```

Each tier is independently trainable. The contribution is the composition:

1. Tier 1 is a generic mjlab walker — cat-feel does not live here.
2. Tier 2 is intentionally small (4 learned + 1 scripted). Earlier versions
   carried 15 skills; we kept only the ones that *couldn't be faked*.
3. Tier 3 is where personality lives: a 6-dim Ornstein–Uhlenbeck mood latent
   biases reward weights, and an attractor mask gates the skill set by
   behavioral mode (resting / observing / stalking / playing / grooming /
   exploring). A stochastic decision period turns the metronomic 2 Hz PPO
   into something that switches at irregular, cat-like intervals.

See [`paper/paper.tex`](paper/paper.tex) for the full method and ablations.

---

## Quickstart

```bash
git clone https://github.com/jeffrah00/robot-pet-cat.git
cd robot-pet-cat
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Download released weights (~5 MB total)
python scripts/download_weights.py

# Run a 30-second autonomous rollout
python scripts/render_brain_3d.py --checkpoint checkpoints/brain/brain_4skill_v1.zip \
    --duration 30 --output renders/demo.mp4

# Or roll out a single skill
python scripts/render_brain_3d.py --scripted lie_belly --duration 20 \
    --output renders/lie_belly_demo.mp4
```

Tested with Python 3.11, MuJoCo 3.5.0, PyTorch 2.4 (cu124), Stable-Baselines3
2.3.

---

## Released checkpoints

| File | Purpose |
|---|---|
| `models/go1_walker_v0.pt`              | Tier 1 — velocity-tracking walker |
| `models/mjlab_go1_walker_normal.pt`    | Tier 2 — forward walking gait |
| `models/mjlab_go1_crouch.pt`           | Tier 2 — stable crouch posture |
| `models/mjlab_go1_lie_belly.pt`        | Tier 2 — belly-down loaf pose |
| `src/robot_pet_cat/brain/scripted_get_up.py` | Tier 2 — scripted PD `get_up` (no checkpoint) |
| `checkpoints/brain/brain_4skill_v1.zip` | Tier 3 — PPO brain, Discrete(5), ~152k env steps |

`stay` is not a separate checkpoint — it re-uses the Tier 1 walker with the
velocity command pinned to zero.

All weights are in the `models/` folder on GitHub.

---

## Repository layout

```
robot-pet-cat/
├── src/robot_pet_cat/
│   ├── motion/         Tier 1 — walker, AMP scaffolding (unused), Go1 base
│   ├── skills/         Tier 2 — registry + each skill class
│   ├── brain/          Tier 3 — env, attractor, mood, rewards, runner
│   ├── scene/          Scene state, play targets, future vision hooks
│   └── sim/            MuJoCo env wrappers
├── configs/            YAML hyperparams for each tier
├── scripts/            Train + render + diagnostics
├── checkpoints/        Released weights (populated by download script)
├── renders/            Reference rollouts (mp4)
├── paper/              LaTeX source for the arXiv submission
├── docs/               Architecture and design notes
└── tests/              Unit + integration
```

---

## Reproducing the paper

```bash
# Tier 1: ~30 min on a single A100
bash scripts/train_motion.sh

# Tier 2: ~15 min per skill on a single A100
python -m robot_pet_cat.skills.train --skill walk
python -m robot_pet_cat.skills.train --skill crouch
python -m robot_pet_cat.skills.train --skill lie_belly
python -m robot_pet_cat.skills.train --skill stay

# Tier 3: ~3 h on CPU, no GPU needed
python -m robot_pet_cat.brain.runner --total-timesteps 20_000_000
```

The paper's 10-minute rollout (`renders/brain_4skill_10min.mp4`) is
reproduced by:

```bash
python scripts/render_brain_3d.py \
    --checkpoint checkpoints/brain/brain_4skill_v1.zip \
    --duration 600 --output renders/brain_4skill_10min.mp4
```

---

## What we tried and removed

Documented in [`paper/paper.tex` §5](paper/paper.tex) and in the project page.
Short version:

- **AMP imitation track** — `cat_imitation_v1.pkl` produced zero forward
  motion under velocity commands. Cat-feel moved up to Tier 3.



---

## Citation

```bibtex
@article{rah2026brain,
  title   = {How to Make a Robot Pet Cat: Part 1 --- Brain},
  author  = {Rah, Jeffrey},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## License

MIT. See 