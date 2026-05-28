# Release v0.1 — first public release

**Date:** 2026-05-28

## Released artifacts

### Code
- Full source under `src/robot_pet_cat/` (MIT).
- Reproducible scripts under `scripts/` and configs under `configs/`.

### Weights (`checkpoints/`)
| File | Size | Notes |
|---|---|---|
| `walker/go1_walker_v0.pt` | ~1.5 MB | Tier 1 — Go1 mjlab velocity-tracking walker, 10k PPO iters |
| `skills/walk_v1.pt` | ~160 KB | Tier 2 — forward walking gait |
| `skills/crouch_v1.pt` | ~160 KB | Tier 2 — stable crouch posture |
| `skills/lie_belly_v1.pt` | ~160 KB | Tier 2 — belly-down loaf pose |
| `skills/stay_v1.pt` | ~160 KB | Tier 2 — freeze current joint config |
| `brain/brain_4skill_v1.zip` | 160 KB | Tier 3 — SB3 PPO, Discrete(5) action space |

All weights mirrored to https://huggingface.co/jeffrah00/robot-pet-cat.

### Renders
- `renders/brain_4skill_10min.mp4` — headline 10-min autonomous rollout
- Per-skill MP4s under `renders/` (see project page)

### Paper
- `paper/paper.tex` + `paper/references.bib` — arXiv source

## What's NOT in this release
- AMP imitation track checkpoints (deprecated; see paper §5).
- The failed get_up RL variants (v1–v10, see memory log).
- Vision encoder / head-mounted camera (scaffolded but not used).

## Known limitations
- Sim-only. No hardware transfer demonstrated.
- Brain decision is proprioception + scene-state, not vision.
- Cat-feel evaluated by visual inspection; no controlled user study.

## Reproducing
See README §"Reproducing the paper" for full commands.
