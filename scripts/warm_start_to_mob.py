#!/usr/bin/env python3
"""Pad a 47-dim Go2-Flat walker checkpoint to a 48-dim Go2-Flat-MoB init.

Reads an rsl_rl-style checkpoint, finds the first Linear layer of the actor
and critic, inserts a zero column at the position where the new
body_height_cmd dimension lives in the observation, and writes the modified
checkpoint to disk. Result is hot-swappable into mjlab train resume.

The MoB v1 actor obs is (Flat task, no height_scan):
  base_ang_vel(3) + projected_gravity(3) + command(was 3, now 4) +
  phase(2) + joint_pos(12) + joint_vel(12) + actions(12) = 47 -> 48
The new dim is inserted at column index 9 (after the existing 3 command dims).

The critic obs is wider but command sits at the same position (9), so the
same insert applies. base_lin_vel + foot_height/air_time/contact/forces all
come AFTER the actor terms.

Usage:
  python3 warm_start_to_mob.py --inspect <checkpoint.pt>
    # Prints state dict shapes; use first to verify key names.

  python3 warm_start_to_mob.py \\
      --input  /workspace/.../model_6400.pt \\
      --output /workspace/.../model_6400_mob_init.pt \\
      --insert-col 9

Then resume MoB training with:
  python3 scripts/train.py Unitree-Go2-Flat-MoB \\
      --agent.resume True \\
      --agent.load-run <dir-containing-model_6400_mob_init.pt> \\
      --agent.load-checkpoint model_6400_mob_init.pt

If the actor / critic layout differs from this default the script aborts
with a clear message. Use --inspect first when in doubt.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch


# Candidate regexes for the FIRST linear layer of an MLP head.
# rsl_rl ActorCritic typically names them like `actor.0.weight` (Sequential)
# or `actor.layers.0.weight` (custom). We match both styles.
FIRST_LINEAR_PATTERNS = [
    re.compile(r"^actor\.0\.weight$"),
    re.compile(r"^actor\.layers\.0\.weight$"),
    re.compile(r"^actor\.mlp\.0\.weight$"),
    re.compile(r"^actor\.net\.0\.weight$"),
    re.compile(r"^critic\.0\.weight$"),
    re.compile(r"^critic\.layers\.0\.weight$"),
    re.compile(r"^critic\.mlp\.0\.weight$"),
    re.compile(r"^critic\.net\.0\.weight$"),
]

# Normalizer state keys (EmpiricalNormalization in rsl_rl).
NORM_PATTERNS = {
    "actor_mean": [re.compile(r"^obs_normalizer\.mean$"), re.compile(r"^actor_normalizer\.mean$")],
    "actor_var":  [re.compile(r"^obs_normalizer\.var$"),  re.compile(r"^actor_normalizer\.var$")],
    "critic_mean":[re.compile(r"^critic_normalizer\.mean$"), re.compile(r"^privileged_obs_normalizer\.mean$")],
    "critic_var": [re.compile(r"^critic_normalizer\.var$"),  re.compile(r"^privileged_obs_normalizer\.var$")],
}


def find_first_linear_keys(sd: dict) -> tuple[str | None, str | None]:
    """Return (actor_key, critic_key) for the first Linear weight, or None each."""
    actor_key = None
    critic_key = None
    for k in sd:
        if actor_key is None and any(p.match(k) for p in FIRST_LINEAR_PATTERNS if "actor" in p.pattern):
            actor_key = k
        if critic_key is None and any(p.match(k) for p in FIRST_LINEAR_PATTERNS if "critic" in p.pattern):
            critic_key = k
    return actor_key, critic_key


def find_norm_keys(sd: dict) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for label, patterns in NORM_PATTERNS.items():
        out[label] = None
        for k in sd:
            if any(p.match(k) for p in patterns):
                out[label] = k
                break
    return out


def insert_zero_col(W: torch.Tensor, col: int) -> torch.Tensor:
    """W: [out, in]. Insert a column of zeros at position `col`. Returns [out, in+1]."""
    out, in_dim = W.shape
    assert col <= in_dim, f"col={col} > in_dim={in_dim}"
    new = torch.zeros(out, in_dim + 1, dtype=W.dtype, device=W.device)
    new[:, :col] = W[:, :col]
    new[:, col + 1:] = W[:, col:]
    return new


def insert_at_position(v: torch.Tensor, col: int, value: float) -> torch.Tensor:
    """v: [N]. Insert `value` at position col. Returns [N+1]."""
    n = v.shape[0]
    new = torch.empty(n + 1, dtype=v.dtype, device=v.device)
    new[:col] = v[:col]
    new[col] = value
    new[col + 1:] = v[col:]
    return new


def inspect(path: Path) -> None:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    print(f"=== checkpoint: {path}")
    if isinstance(ck, dict):
        for k, v in ck.items():
            if isinstance(v, dict):
                print(f"  [{k}] (dict, {len(v)} keys)")
                for kk, vv in list(v.items())[:50]:
                    shape = tuple(vv.shape) if hasattr(vv, "shape") else type(vv).__name__
                    print(f"    {kk}: {shape}")
                if len(v) > 50:
                    print(f"    ... ({len(v) - 50} more keys)")
            else:
                print(f"  [{k}] {type(v).__name__}")
    else:
        print(f"  (root is {type(ck).__name__})")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inspect", type=Path, help="Print checkpoint structure and exit.")
    p.add_argument("--input", type=Path, help="Input .pt checkpoint.")
    p.add_argument("--output", type=Path, help="Output .pt checkpoint.")
    p.add_argument("--insert-col", type=int, default=9,
                   help="Obs-vector column index to insert the zero column at. "
                        "Default 9 (after the 3-dim command, for Flat task).")
    args = p.parse_args()

    if args.inspect:
        inspect(args.inspect)
        return 0

    if not args.input or not args.output:
        print("ERROR: --input and --output required (or --inspect).", file=sys.stderr)
        return 2

    ck = torch.load(args.input, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict):
        print(f"ERROR: expected dict checkpoint, got {type(ck).__name__}", file=sys.stderr)
        return 2

    # rsl_rl checkpoints typically have a 'model_state_dict' sub-dict.
    sd_key = None
    for cand in ("model_state_dict", "state_dict", "policy"):
        if cand in ck and isinstance(ck[cand], dict):
            sd_key = cand
            break
    if sd_key is None:
        # Maybe the root IS the state dict.
        sd = ck
    else:
        sd = ck[sd_key]

    actor_key, critic_key = find_first_linear_keys(sd)
    if actor_key is None and critic_key is None:
        print("ERROR: no actor/critic first-Linear weight found. Run with --inspect.", file=sys.stderr)
        return 2
    print(f"[detected] actor first Linear: {actor_key}")
    print(f"[detected] critic first Linear: {critic_key}")

    col = args.insert_col

    if actor_key is not None:
        W = sd[actor_key]
        new = insert_zero_col(W, col)
        sd[actor_key] = new
        print(f"[ok] padded {actor_key}: {tuple(W.shape)} -> {tuple(new.shape)}")
        # Also pad the BIAS? No — bias shape is [out], unchanged. Skip.

    if critic_key is not None:
        W = sd[critic_key]
        new = insert_zero_col(W, col)
        sd[critic_key] = new
        print(f"[ok] padded {critic_key}: {tuple(W.shape)} -> {tuple(new.shape)}")

    # Pad normalizers (mean: insert 0, var: insert 1 -> identity transform).
    norm_keys = find_norm_keys(sd)
    for label, key in norm_keys.items():
        if key is None:
            continue
        v = sd[key]
        is_var = label.endswith("_var")
        fill = 1.0 if is_var else 0.0
        new = insert_at_position(v, col, fill)
        sd[key] = new
        print(f"[ok] padded {key}: ({v.shape[0]},) -> ({new.shape[0]},) fill={fill}")

    if sd_key is not None:
        ck[sd_key] = sd

    # Reset optimizer state — the param shapes changed, so old optimizer state is invalid.
    if "optimizer_state_dict" in ck:
        del ck["optimizer_state_dict"]
        print("[ok] dropped optimizer_state_dict (param shapes changed)")

    # Reset iter so resume starts curriculum from step 0 — we WANT to retrain.
    if "iter" in ck:
        ck["iter"] = 0
        print("[ok] reset iter=0")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, args.output)
    print(f"[ok] wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
