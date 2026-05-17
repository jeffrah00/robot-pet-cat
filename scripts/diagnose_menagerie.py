#!/usr/bin/env python3
"""Tell me exactly where mujoco_playground's Go1 is looking for menagerie.

We're getting "Error opening file ../../../../../../mujoco_menagerie/..." even
after fetching/symlinking, which means our model of where playground looks
is wrong. Rather than guess again, dump the ground truth:

  - What's in ~/.cache/mujoco_menagerie? (and is the LFS pull effective?)
  - Where does `import mujoco_playground` resolve to?
  - Where does Go1's base.py compute its menagerie path?
  - What does Go1's get_assets() return (keys + counts)?
  - Do the symlinks we placed actually point anywhere useful?

Run this on RunPod and paste the output back so I can fix the real path.
"""

from __future__ import annotations

import sys
from pathlib import Path


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    section("1. ~/.cache/mujoco_menagerie state")
    cache = Path.home() / ".cache" / "mujoco_menagerie"
    print(f"cache: {cache}  exists={cache.exists()}")
    if cache.exists():
        go1 = cache / "unitree_go1"
        go2 = cache / "unitree_go2"
        for d in (go1, go2):
            print(f"  {d}: exists={d.exists()}")
            if d.exists():
                xmls = sorted(p.name for p in d.glob("*.xml"))
                print(f"    xmls: {xmls}")
                assets = d / "assets"
                if assets.is_dir():
                    n_stl = len(list(assets.glob("*.stl")))
                    print(f"    assets/: {n_stl} STL files")
                    sample = assets / "trunk.stl"
                    if sample.exists():
                        sz = sample.stat().st_size
                        with sample.open("rb") as f:
                            head = f.read(48)
                        is_lfs = head.startswith(b"version https://git-lfs")
                        print(f"    assets/trunk.stl: {sz} bytes, lfs_pointer={is_lfs}")

    section("2. mujoco_playground installation")
    try:
        import mujoco_playground
        pkg = Path(mujoco_playground.__file__).resolve().parent
        print(f"package root: {pkg}")
        print(f"  exists={pkg.exists()}")
        print(f"site-packages: {pkg.parent}")
        print(f"site-packages parent: {pkg.parent.parent}")
        for cand in (
            pkg.parent.parent / "mujoco_menagerie",
            pkg.parent / "mujoco_menagerie",
            pkg / "mujoco_menagerie",
            pkg / "_src" / "mujoco_menagerie",
        ):
            print(f"  candidate {cand}: exists={cand.exists()}  "
                  f"is_symlink={cand.is_symlink()}  "
                  f"resolves_to={cand.resolve() if cand.exists() or cand.is_symlink() else 'N/A'}")
    except ImportError as e:
        print(f"FAIL: {e}")
        return 1

    section("3. Go1 base module internals")
    try:
        from mujoco_playground._src.locomotion.go1 import base as go1_base
        print(f"go1.base.__file__: {go1_base.__file__}")
        for attr in ("MENAGERIE_PATH", "MENAGERIE_ROOT", "ROOT_PATH",
                     "MJX_ROOT_PATH", "ASSETS_PATH", "GO1_ROOT_PATH"):
            if hasattr(go1_base, attr):
                v = getattr(go1_base, attr)
                print(f"  {attr} = {v!r}")
                try:
                    p = Path(str(v))
                    print(f"    exists={p.exists()}  resolves_to={p.resolve() if p.exists() else 'N/A'}")
                except Exception as e:
                    print(f"    (not a path: {e})")
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback; traceback.print_exc()

    section("4. Go1 constants module")
    try:
        from mujoco_playground._src.locomotion.go1 import go1_constants
        print(f"go1_constants.__file__: {go1_constants.__file__}")
        for name in dir(go1_constants):
            if name.isupper() and not name.startswith("_"):
                v = getattr(go1_constants, name)
                if isinstance(v, (str, Path)):
                    print(f"  {name} = {v!r}")
    except Exception as e:
        print(f"FAIL: {e}")

    section("5. Go1 get_assets() (what mujoco actually gets)")
    try:
        from mujoco_playground._src.locomotion.go1 import base as go1_base
        if hasattr(go1_base, "get_assets"):
            assets = go1_base.get_assets()
            print(f"len={len(assets)}")
            keys = list(assets.keys())
            for k in keys[:20]:
                print(f"  key: {k!r}  ({len(assets[k])} bytes)")
            if len(keys) > 20:
                print(f"  ... and {len(keys) - 20} more")
        else:
            print("go1_base has no get_assets() attribute")
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback; traceback.print_exc()

    section("6. What Go1 XML actually contains for trunk.stl")
    try:
        from mujoco_playground._src.locomotion.go1 import go1_constants
        for name in dir(go1_constants):
            if "xml" in name.lower() or "path" in name.lower():
                v = getattr(go1_constants, name)
                if isinstance(v, (str, Path)) and str(v).endswith(".xml"):
                    p = Path(v)
                    if p.exists():
                        text = p.read_text()
                        for line in text.splitlines():
                            if "trunk.stl" in line or "meshdir" in line:
                                print(f"  {p.name}: {line.strip()}")
    except Exception as e:
        print(f"FAIL: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
