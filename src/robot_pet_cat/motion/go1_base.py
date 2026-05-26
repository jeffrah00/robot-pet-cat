"""Go1 asset loader -- parallel to go2_base but for menagerie's unitree_go1."""

from __future__ import annotations

import os
from pathlib import Path

# data/go1_scenes (scene XMLs that include go1.xml)
ROOT_PATH = Path(__file__).resolve().parents[3] / "data" / "go1_scenes"


def _menagerie_go1_dir() -> Path:
    """Locate mujoco_menagerie's unitree_go1 directory at runtime.

    Search order:
    1. third_party/ inside the project repo (present on pods after git clone)
    2. mujoco_playground module MENAGERIE_PATH / MENAGERIE_ROOT attributes
    3. MUJOCO_PLAYGROUND_MENAGERIE env-var override
    4. ~/.cache/mujoco_menagerie fallback
    """
    # 1. project third_party (most reliable on RunPod)
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "third_party" / "mujoco_menagerie" / "unitree_go1"
        if (candidate / "go1.xml").is_file():
            return candidate

    # 2. mujoco_playground module attributes
    candidates: list[Path] = []
    for mod_name in (
        "mujoco_playground._src.locomotion.go1.base",
        "mujoco_playground._src.locomotion.go1.go1_constants",
        "mujoco_playground._src.locomotion.unitree_go1.base",
        "mujoco_playground._src.mjx_env",
        "mujoco_playground._src.menagerie",
    ):
        try:
            mod = __import__(mod_name, fromlist=["MENAGERIE_PATH"])
        except ImportError:
            continue
        mp = getattr(mod, "MENAGERIE_PATH", None) or getattr(mod, "MENAGERIE_ROOT", None)
        if mp is not None:
            candidates.append(Path(mp) / "unitree_go1")

    # 3. env-var override
    env_override = os.environ.get("MUJOCO_PLAYGROUND_MENAGERIE")
    if env_override:
        candidates.append(Path(env_override) / "unitree_go1")

    # 4. ~/.cache fallback
    candidates.append(Path.home() / ".cache" / "mujoco_menagerie" / "unitree_go1")

    for c in candidates:
        if (c / "go1.xml").is_file():
            return c

    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Could not locate mujoco_menagerie/unitree_go1.\n"
        "Tried:\n  " + tried + "\n\n"
        "Fix: ensure third_party/mujoco_menagerie is present (git submodule update --init)."
    )


def get_assets() -> dict[str, bytes]:
    """Return {filename: bytes} for all assets needed to compile the Go1 scene.

    Includes:
    - All XML / STL / OBJ / PNG / JPG files from ROOT_PATH (go1_scenes/)
    - All XML / STL / OBJ / PNG / JPG files from unitree_go1 menagerie dir
    """
    assets: dict[str, bytes] = {}

    # scene XMLs (living_room_with_go1.xml, sensor_feet.xml, etc.)
    if ROOT_PATH.is_dir():
        for p in ROOT_PATH.iterdir():
            if p.is_file() and p.suffix in {".xml", ".stl", ".obj"}:
                assets[p.name] = p.read_bytes()

    # robot model + meshes
    go1_dir = _menagerie_go1_dir()
    for p in go1_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".xml", ".stl", ".obj", ".png", ".jpg"}:
            assets.setdefault(p.name, p.read_bytes())

    return assets
