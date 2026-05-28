"""Roll out a brain checkpoint headlessly and produce the Table 1 stats:
mode dwell-times (mean +/- std) and skill-dispatch counts per attractor mode.

Usage:
    python scripts/measure_rollout.py \
        --checkpoint checkpoints/brain/brain_4skill_v1.zip \
        --duration 600 \
        --out renders/table1_stats.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Local imports — assumes running from the repo root with the venv activated.
from stable_baselines3 import PPO

from robot_pet_cat.brain.env import BrainEnv, BrainEnvConfig


MODES_ORDER = ["resting", "observing", "stalking", "playing", "grooming", "exploring"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="checkpoints/brain/brain_4skill_v1.zip")
    p.add_argument("--duration", type=float, default=600.0,
                   help="Rollout length in seconds of sim time (default 600 = 10 min).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="renders/table1_stats.txt",
                   help="Write the human-readable table + raw JSON here.")
    args = p.parse_args()

    cfg = BrainEnvConfig()
    env = BrainEnv(cfg=cfg)
    obs, _info = env.reset(seed=args.seed)

    dt = env.cfg.dt_s  # seconds per brain step
    n_steps = max(1, int(round(args.duration / dt)))
    print(f"[measure] dt={dt:.3f}s  n_steps={n_steps}  target {args.duration:.0f}s")

    model = PPO.load(args.checkpoint, env=env, device="cpu")

    mode_segments: list[tuple[float, float, str]] = []  # (t_start, t_end, mode)
    skill_by_mode: dict[str, Counter] = defaultdict(Counter)
    cur_mode: str | None = None
    cur_mode_start = 0.0
    prev_skill: str | None = None

    for step in range(n_steps):
        t = step * dt
        # Mode bookkeeping
        mode_obj = getattr(env, "mode_policy", None)
        mode_name = mode_obj.current_mode.value if (mode_obj is not None and mode_obj.current_mode is not None) else None
        if mode_name != cur_mode:
            if cur_mode is not None:
                mode_segments.append((cur_mode_start, t, cur_mode))
            cur_mode = mode_name
            cur_mode_start = t

        # Sample action
        try:
            action, _ = model.predict(obs, deterministic=False)
        except Exception as e:
            print(f"[measure] policy.predict failed at step {step}: {e}", file=sys.stderr)
            raise

        obs, _r, terminated, truncated, _info = env.step(action)

        # Skill dispatch bookkeeping
        skill_name = getattr(env, "active_skill_name", None)
        if skill_name is not None and skill_name != prev_skill:
            bucket = cur_mode if cur_mode is not None else "<no_mode>"
            skill_by_mode[bucket][skill_name] += 1
            prev_skill = skill_name

        if terminated or truncated:
            obs, _info = env.reset(seed=args.seed + step)
            prev_skill = None

    # Close out the final mode segment
    if cur_mode is not None:
        mode_segments.append((cur_mode_start, n_steps * dt, cur_mode))

    # Aggregate dwell-times per mode
    mode_dwells: dict[str, list[float]] = defaultdict(list)
    for t0, t1, name in mode_segments:
        mode_dwells[name].append(t1 - t0)

    # Render table
    lines: list[str] = []
    lines.append(f"Rollout: checkpoint={args.checkpoint} duration={args.duration:.0f}s dt={dt:.3f}s steps={n_steps}")
    lines.append("")
    lines.append(f"{'Mode':12s} | {'n':>3s} | {'mean (s)':>9s} | {'std (s)':>8s} | {'total (s)':>10s} | skill dispatches")
    lines.append("-" * 95)
    for mode in MODES_ORDER:
        dwells = mode_dwells.get(mode, [])
        if dwells:
            mean = float(np.mean(dwells))
            std = float(np.std(dwells))
            n = len(dwells)
            tot = float(np.sum(dwells))
        else:
            mean = std = tot = 0.0
            n = 0
        top = ", ".join(f"{s}:{c}" for s, c in skill_by_mode.get(mode, Counter()).most_common(5))
        lines.append(f"{mode:12s} | {n:3d} | {mean:9.1f} | {std:8.1f} | {tot:10.1f} | {top}")

    other_modes = sorted(set(mode_dwells) - set(MODES_ORDER))
    for mode in other_modes:
        dwells = mode_dwells[mode]
        mean = float(np.mean(dwells)); std = float(np.std(dwells)); n = len(dwells); tot = float(np.sum(dwells))
        top = ", ".join(f"{s}:{c}" for s, c in skill_by_mode.get(mode, Counter()).most_common(5))
        lines.append(f"{mode:12s} | {n:3d} | {mean:9.1f} | {std:8.1f} | {tot:10.1f} | {top}")

    table = "\n".join(lines)
    print(table)

    # Persist raw data alongside the table for paper editing.
    payload = {
        "checkpoint": args.checkpoint,
        "duration_s": args.duration,
        "dt_s": dt,
        "n_steps": n_steps,
        "mode_segments": mode_segments,
        "mode_dwells": dict(mode_dwells),
        "skill_dispatches_by_mode": {k: dict(v) for k, v in skill_by_mode.items()},
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table + "\n\n--- JSON ---\n" + json.dumps(payload, indent=2, default=str))
    print(f"\n[measure] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
