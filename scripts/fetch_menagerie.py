#!/usr/bin/env python3
"""Clone mujoco_menagerie into the local cache so Go2 env compile works.

mujoco_playground's pip wheel doesn't bundle the menagerie meshes -- they
live in a separate repo and the playground clones them on-demand. We do
the same on our side so make_go2_joystick_env() can find unitree_go2/.

Idempotent: if the clone already exists, just verifies the Go2 directory.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

CACHE = Path.home() / ".cache" / "mujoco_menagerie"
URL = "https://github.com/google-deepmind/mujoco_menagerie.git"


def main() -> int:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    if (CACHE / ".git").is_dir():
        print(f"[fetch_menagerie] already present: {CACHE}")
    else:
        print(f"[fetch_menagerie] cloning {URL} -> {CACHE}")
        subprocess.check_call(["git", "clone", "--depth", "1", URL, str(CACHE)])

    go2 = CACHE / "unitree_go2"
    if not (go2 / "go2_mjx.xml").is_file():
        print(f"[fetch_menagerie] ERROR: {go2}/go2_mjx.xml missing")
        return 1
    print(f"[fetch_menagerie] OK: {go2}/go2_mjx.xml exists")
    return 0


if __name__ == "__main__":
    sys.exit(main())
