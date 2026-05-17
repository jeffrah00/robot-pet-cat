#!/usr/bin/env python3
"""Tell me exactly where mujoco_playground's Go1 is looking for menagerie."""

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
        for d in (cache / "unitree_go1", cache / "unitree_go2"):
            print(f"  {d}: exists={d.exists()}")
            if not d.exists():
                continue
            xmls = sorted(p.name for p in d.glob("*.xml"))
            print(f"    xmls: {xmls}")
            for sub in sorted(d.iterdir()):
                if sub.is_dir():
                    files = sorted(p.name for p in sub.iterdir() if p.is_file())
                    by_ext: dict[str, int] = {}
                    for f in files:
                        ext = Path(f).suffix.lower()
                        by_ext[ext] = by_ext.get(ext, 0) + 1
                    print(f"    {sub.name}/: {len(files)} files by ext: {by_ext}")
                    for f in files[:5]:
                        p = sub / f
                        sz = p.stat().st_size
                        with p.open("rb") as fh:
                            head = fh.read(48)
                        is_lfs = head.startswith(b"version https://git-lfs")
                        print(f"      {f}: {sz} bytes, lfs_pointer={is_lfs}")

    section("2. mujoco_playground installation")
    try:
        import mujoco_playground
        pkg = Path(mujoco_playground.__file__).resolve().parent
        print(f"package root: {pkg}")
        print(f"site-packages: {pkg.parent}")
        print(f"site-packages parent: {pkg.parent.parent}")
        for cand in (
            pkg.parent.parent / "mujoco_menagerie",
            pkg.parent / "mujoco_menagerie",
            pkg / "mujoco_menagerie",
            pkg / "_src" / "mujoco_menagerie",
        ):
            resolved = cand.resolve() if cand.exists() or cand.is_symlink() else "N/A"
            print(f"  candidate {cand}: exists={cand.exists()}  "
                  f"is_symlink={cand.is_symlink()}  resolves_to={resolved}")
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
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback; traceback.print_exc()

    section("4. Go1 constants module path-like attrs")
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
            for k in list(assets.keys())[:30]:
                print(f"  key: {k!r}  ({len(assets[k])} bytes)")
        else:
            print("go1_base has no get_assets() attribute")
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback; traceback.print_exc()

    section("6. Go1 XMLs on disk: mesh / meshdir lines")
    try:
        from mujoco_playground._src.locomotion.go1 import go1_constants
        go1_dir = Path(go1_constants.__file__).resolve().parent
        print(f"go1 source dir: {go1_dir}")
        for xml in sorted(go1_dir.rglob("*.xml")):
            rel = xml.relative_to(go1_dir)
            text = xml.read_text(errors="replace")
            hits = []
            for ln in text.splitlines():
                if "trunk" in ln or "meshdir" in ln or "<mesh " in ln:
                    hits.append(ln.strip())
            if hits:
                print(f"  {rel}:")
                for h in hits[:8]:
                    print(f"    {h}")
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback; traceback.print_exc()

    return 0


if __name__ == "__main__":
    sys.exit(main())
