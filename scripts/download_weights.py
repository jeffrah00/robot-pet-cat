"""Download released robot-pet-cat weights from Hugging Face.

Usage:
    python scripts/download_weights.py
    python scripts/download_weights.py --target checkpoints/
    python scripts/download_weights.py --only brain

Mirrors:
    Primary:  https://huggingface.co/jeffrah00/robot-pet-cat
    Fallback: GitHub Release v0.1 (https://github.com/jeffrah00/robot-pet-cat/releases)

All released weights are MIT-licensed; see LICENSE.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

# (dest_relpath, sha256 placeholder, hf_filename)
# All weights are MIT-licensed; see LICENSE in the repo.
MANIFEST = [
    # Tier 1 walker
    ("models/go1_walker_v0.pt",            "TBD", "go1_walker_v0.pt"),
    # Tier 2 mjlab Go1 skill checkpoints
    ("models/mjlab_go1_walker_normal.pt",  "TBD", "mjlab_go1_walker_normal.pt"),
    ("models/mjlab_go1_crouch.pt",         "TBD", "mjlab_go1_crouch.pt"),
    ("models/mjlab_go1_lie_belly.pt",      "TBD", "mjlab_go1_lie_belly.pt"),
    # NOTE: `stay` reuses the walker with v_cmd=0; no separate checkpoint.
    # `get_up` is scripted PD, not a checkpoint.
    # Tier 3 brain
    ("checkpoints/brain/brain_4skill_v1.zip", "TBD", "brain_4skill_v1.zip"),
]

HF_REPO = "jeffrah00/robot-pet-cat"
HF_BASE = f"https://huggingface.co/{HF_REPO}/resolve/main"
GH_RELEASE_BASE = "https://github.com/jeffrah00/robot-pet-cat/releases/download/v0.1"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  -> {url}")
    with urllib.request.urlopen(url) as resp, dest.open("wb") as out:
        out.write(resp.read())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", default="checkpoints",
                   help="Destination directory (default: checkpoints/).")
    p.add_argument("--only", choices=["walker", "skills", "brain"], default=None,
                   help="Download a single subset.")
    p.add_argument("--verify", action="store_true",
                   help="Re-verify sha256 of already-downloaded files.")
    args = p.parse_args()

    root = Path(args.target).resolve()
    root.mkdir(parents=True, exist_ok=True)

    targets = MANIFEST
    if args.only == "walker":
        targets = [m for m in MANIFEST if m[0] == "models/go1_walker_v0.pt"]
    elif args.only == "skills":
        targets = [m for m in MANIFEST if m[0].startswith("models/mjlab_go1_")]
    elif args.only == "brain":
        targets = [m for m in MANIFEST if m[0].startswith("checkpoints/brain/")]

    for relpath, sha, hf_name in targets:
        dest = root / relpath
        if dest.exists() and not args.verify:
            print(f"[skip] {relpath} (exists)")
            continue

        # Try HF first; fall back to a GitHub release.
        urls = [f"{HF_BASE}/{hf_name}", f"{GH_RELEASE_BASE}/{hf_name}"]
        ok = False
        for url in urls:
            try:
                download(url, dest)
                ok = True
                break
            except Exception as e:  # noqa: BLE001
                print(f"  ! {url}: {e}")
        if not ok:
            print(f"FAIL: could not fetch {relpath}", file=sys.stderr)
            return 1

        # Verify 