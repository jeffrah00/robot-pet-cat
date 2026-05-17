#!/usr/bin/env python3
"""Clone mujoco_menagerie + place it where mujoco_playground expects.

Three things need to happen for Go2 env compile to work:

  1. Our Go2 wrapper needs menagerie/unitree_go2/ reachable -- we cache it
     at ~/.cache/mujoco_menagerie/ and find it via go2_base._menagerie_go2_dir.

  2. menagerie tracks *.stl with Git LFS. A plain `git clone` pulls 130-byte
     pointer files instead of the real binary meshes; mujoco then fails with
     "Error opening file ... .stl". We install LFS support and run
     `git lfs pull` after cloning. We also verify a sample STL is real binary
     (not an LFS pointer) so the failure mode is loud, not silent.

  3. Internally, our wrapper instantiates mujoco_playground's Go1 Joystick
     class to reuse its task code (rewards, observation flatten). The Go1
     constructor compiles Go1's own MJCF, and the Go1 XMLs ship with
     hardcoded relative mesh paths
         "../../../../../../mujoco_menagerie/unitree_go1/assets/trunk.stl"
     that assume the playground repo sits next to a menagerie checkout.
     We satisfy that with a symlink to our cache.

After running this script the Go1 menagerie compile will succeed, and our
Go2 model swap can proceed.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

CACHE = Path.home() / ".cache" / "mujoco_menagerie"
URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def _have_git_lfs() -> bool:
    return shutil.which("git-lfs") is not None or shutil.which("git") and (
        subprocess.run(["git", "lfs", "version"], capture_output=True).returncode == 0
    )


def _clone_cache() -> None:
    """Clone menagerie into ~/.cache if not already present.

    Uses GIT_LFS_SKIP_SMUDGE=1 for the clone (so we don't double-download)
    then explicitly `git lfs pull` so the failure mode is clean if LFS
    isn't available.
    """
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    if (CACHE / ".git").is_dir():
        print(f"[fetch_menagerie] already present: {CACHE}")
        return

    if not _have_git_lfs():
        print("[fetch_menagerie] WARNING: git-lfs not installed; STL meshes "
              "will be LFS pointer files (mujoco will refuse to load them).")
        print("[fetch_menagerie]   Install: `apt install git-lfs` then `git lfs install`")

    print(f"[fetch_menagerie] cloning {URL} -> {CACHE}")
    env = {"GIT_LFS_SKIP_SMUDGE": "1"}
    subprocess.check_call(
        ["git", "clone", "--depth", "1", URL, str(CACHE)],
        env={**__import__("os").environ, **env},
    )


def _pull_lfs() -> None:
    """Run `git lfs pull` inside the cache so STL pointers become real bytes."""
    if not _have_git_lfs():
        print("[fetch_menagerie] skipping `git lfs pull` (git-lfs missing)")
        return
    print(f"[fetch_menagerie] git lfs pull in {CACHE}")
    subprocess.check_call(["git", "lfs", "install", "--local"], cwd=str(CACHE))
    subprocess.check_call(["git", "lfs", "pull"], cwd=str(CACHE))


def _verify_lfs_resolved() -> bool:
    """Spot-check a Go1 mesh: if it starts with the LFS pointer prefix,
    LFS smudging never ran and the file is just a 130-byte text pointer."""
    sample = CACHE / "unitree_go1" / "assets" / "thigh.stl"
    if not sample.is_file():
        print(f"[fetch_menagerie] sample {sample} missing; skipping LFS check")
        return False
    with sample.open("rb") as f:
        head = f.read(64)
    if head.startswith(LFS_POINTER_PREFIX):
        print(f"[fetch_menagerie] ERROR: {sample} is an LFS pointer "
              f"({sample.stat().st_size} bytes). Run: cd {CACHE} && git lfs pull")
        return False
    print(f"[fetch_menagerie] OK: {sample} is real binary ({sample.stat().st_size} bytes)")
    return True


def _link_into_playground_layout() -> None:
    """Make our cache discoverable via Go1 XML's hardcoded relative paths."""
    try:
        import mujoco_playground
    except ImportError:
        print("[fetch_menagerie] mujoco_playground not installed; skipping symlink step")
        return

    pkg_root = Path(mujoco_playground.__file__).resolve().parent
    site_packages = pkg_root.parent

    expected = site_packages.parent / "mujoco_menagerie"
    alt = site_packages / "mujoco_menagerie"

    for target in (expected, alt):
        if target.exists() or target.is_symlink():
            # If it's a symlink to our cache already, good. If it's a stale
            # symlink to nowhere, drop and recreate.
            if target.is_symlink() and not target.resolve().exists():
                target.unlink()
            else:
                print(f"[fetch_menagerie] already exists: {target}")
                continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(CACHE, target_is_directory=True)
            print(f"[fetch_menagerie] symlinked {target} -> {CACHE}")
        except OSError as e:
            print(f"[fetch_menagerie] could not symlink {target}: {e}")
            print(f"[fetch_menagerie]   workaround: cp -r {CACHE} {target}")


def main() -> int:
    _clone_cache()
    _pull_lfs()

    go2 = CACHE / "unitree_go2"
    go1 = CACHE / "unitree_go1"
    if not (go2 / "go2_mjx.xml").is_file():
        print(f"[fetch_menagerie] ERROR: {go2}/go2_mjx.xml missing")
        return 1
    if not go1.is_dir():
        print(f"[fetch_menagerie] WARNING: {go1} missing (Go1 compile will fail)")
    else:
        print(f"[fetch_menagerie] OK: {go1}/ and {go2}/ both present")

    if not _verify_lfs_resolved():
        return 2

    _link_into_playground_layout()
    print("[fetch_menagerie] done. Re-run scripts/smoke_test_go2_env.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
