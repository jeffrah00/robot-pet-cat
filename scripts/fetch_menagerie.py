#!/usr/bin/env python3
"""Clone mujoco_menagerie + place it where mujoco_playground expects.

Two things need to happen for Go2 env compile to work:

  1. Our Go2 wrapper needs menagerie/unitree_go2/ reachable -- we cache it
     at ~/.cache/mujoco_menagerie/ and find it via go2_base._menagerie_go2_dir.

  2. Internally, our wrapper instantiates mujoco_playground's Go1 Joystick
     class to reuse its task code (rewards, observation flatten). The Go1
     constructor compiles Go1's own MJCF, and the Go1 XMLs ship with
     hardcoded relative mesh paths like
         "../../../../../../mujoco_menagerie/unitree_go1/assets/trunk.stl"
     that assume the playground repo sits next to a menagerie checkout.
     When you pip install mujoco_playground, that relative path resolves
     to <site-packages>/../mujoco_menagerie/ -- a directory that doesn't
     exist. We satisfy it with a symlink to our cache.

After running this script the Go1 menagerie compile will succeed, and our
Go2 model swap can proceed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

CACHE = Path.home() / ".cache" / "mujoco_menagerie"
URL = "https://github.com/google-deepmind/mujoco_menagerie.git"


def _clone_cache() -> None:
    """Clone menagerie into ~/.cache if not already present."""
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    if (CACHE / ".git").is_dir():
        print(f"[fetch_menagerie] already present: {CACHE}")
        return
    print(f"[fetch_menagerie] cloning {URL} -> {CACHE}")
    subprocess.check_call(["git", "clone", "--depth", "1", URL, str(CACHE)])


def _link_into_playground_layout() -> None:
    """Make our cache discoverable via Go1 XML's hardcoded relative paths.

    Go1's MJCF resolves "../../../../../../mujoco_menagerie/..." starting from
    site-packages/mujoco_playground/_src/locomotion/go1/xmls/, i.e. it expects
    a directory at <site-packages-parent>/mujoco_menagerie/.

    On some installs the path is actually <repo-root>/mujoco_menagerie/ next to
    the playground source. We probe both and symlink (or print) as needed.
    """
    try:
        import mujoco_playground
    except ImportError:
        print("[fetch_menagerie] mujoco_playground not installed; skipping symlink step")
        return

    pkg_root = Path(mujoco_playground.__file__).resolve().parent
    site_packages = pkg_root.parent  # site-packages/

    # Compute the directory that the Go1 XML's "../../../../../../mujoco_menagerie"
    # resolves to. From .../mujoco_playground/_src/locomotion/go1/xmls/some.xml
    # six "../" lands at site-packages.parent. Then append mujoco_menagerie.
    expected = site_packages.parent / "mujoco_menagerie"

    # Also try "next to site-packages" as a fallback some layouts use.
    alt = site_packages / "mujoco_menagerie"

    for target in (expected, alt):
        if target.exists() or target.is_symlink():
            print(f"[fetch_menagerie] already exists: {target}")
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(CACHE, target_is_directory=True)
            print(f"[fetch_menagerie] symlinked {target} -> {CACHE}")
        except OSError as e:
            # Read-only filesystems, no symlink perms, etc. Print a hint
            # rather than silently fail.
            print(f"[fetch_menagerie] could not symlink {target}: {e}")
            print(f"[fetch_menagerie]   workaround: cp -r {CACHE} {target}")


def main() -> int:
    _clone_cache()

    go2 = CACHE / "unitree_go2"
    go1 = CACHE / "unitree_go1"
    if not (go2 / "go2_mjx.xml").is_file():
        print(f"[fetch_menagerie] ERROR: {go2}/go2_mjx.xml missing")
        return 1
    if not go1.is_dir():
        print(f"[fetch_menagerie] WARNING: {go1} missing (Go1 compile will fail)")
    else:
        print(f"[fetch_menagerie] OK: {go1}/ and {go2}/ both present")

    _link_into_playground_layout()
    print("[fetch_menagerie] done. Re-run scripts/smoke_test_go2_env.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
