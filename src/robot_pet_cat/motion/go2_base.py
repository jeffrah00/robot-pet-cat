"""Go2Env base class -- the Go2 analog of mujoco_playground's Go1Env."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from robot_pet_cat.motion import go2_constants as consts


def _menagerie_go2_dir() -> Path:
    """Locate mujoco_menagerie's unitree_go2 directory at runtime."""
    import os

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
            candidates.append(Path(mp) / "unitree_go2")

    env_override = os.environ.get("MUJOCO_PLAYGROUND_MENAGERIE")
    if env_override:
        candidates.append(Path(env_override) / "unitree_go2")

    candidates.append(Path.home() / ".cache" / "mujoco_menagerie" / "unitree_go2")

    try:
        import mujoco_playground

        pkg_root = Path(mujoco_playground.__file__).resolve().parent
        for sub in (
            pkg_root / "mujoco_menagerie" / "unitree_go2",
            pkg_root / "_src" / "mujoco_menagerie" / "unitree_go2",
            pkg_root.parent / "mujoco_menagerie" / "unitree_go2",
        ):
            candidates.append(sub)
    except ImportError:
        pass

    for c in candidates:
        if (c / "go2_mjx.xml").is_file():
            return c

    if os.environ.get("ROBOT_PET_CAT_AUTO_FETCH_MENAGERIE") == "1":
        return _fetch_menagerie_go2()

    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Could not locate mujoco_menagerie/unitree_go2.\n"
        "Tried:\n  " + tried + "\n\n"
        "Fix: python scripts/fetch_menagerie.py"
    )


def _fetch_menagerie_go2() -> Path:
    import subprocess

    cache = Path.home() / ".cache" / "mujoco_menagerie"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not (cache / ".git").is_dir():
        subprocess.check_call(
            ["git", "clone", "--depth", "1",
             "https://github.com/google-deepmind/mujoco_menagerie.git", str(cache)]
        )
    go2 = cache / "unitree_go2"
    if not (go2 / "go2_mjx.xml").is_file():
        raise FileNotFoundError(f"Fetched menagerie but {go2}/go2_mjx.xml missing.")
    return go2


def get_assets() -> dict[str, bytes]:
    """Return the asset dict mujoco needs to compile our Go2 scene from a string.

    IMPORTANT: mujoco's MjModel.from_xml_string rejects duplicate basenames
    in the assets dict ("Repeated file name in assets dict: foo.stl"). It
    keys lookups by basename, so we MUST emit each file under its basename
    only -- never under both a relative path and a basename.
    """
    assets: dict[str, bytes] = {}

    for p in consts.ROOT_PATH.iterdir():
        if p.is_file() and p.suffix in {".xml", ".stl", ".obj"}:
            assets[p.name] = p.read_bytes()

    go2_dir = _menagerie_go2_dir()
    for p in go2_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".xml", ".stl", ".obj", ".png", ".jpg"}:
            assets.setdefault(p.name, p.read_bytes())

    return assets


def _go1_mesh_injection(menagerie_root: Path) -> dict[str, bytes]:
    """Read Go1 meshes and key them by basename for mujoco's assets dict.

    mujoco_playground's Go1 get_assets() returns XMLs + PNG textures but NOT
    the mesh files. Go1 XMLs reference meshes via either meshdir+filename or
    long relative paths; mujoco resolves both by stripping to basename. So we
    just inject {basename: bytes} for every Go1 STL/OBJ.

    Must not emit the same basename twice -- mujoco raises
    "Repeated file name in assets dict" on duplicates.
    """
    extras: dict[str, bytes] = {}
    go1_assets = menagerie_root / "unitree_go1" / "assets"
    if not go1_assets.is_dir():
        return extras
    for mesh in go1_assets.iterdir():
        if not mesh.is_file() or mesh.suffix.lower() not in {".stl", ".obj"}:
            continue
        extras.setdefault(mesh.name, mesh.read_bytes())
    return extras


def make_go2_joystick_env(env_config: Any = None):
    """Build a Joystick env on the Go2 robot.

    1. Patch Go1's module-level get_assets() to ALSO return the mesh files
       (it normally returns only XMLs + PNGs, which makes mujoco compile
       fail in pip-only installs). Revert the patch after Joystick().
    2. Instantiate Go1 Joystick (Go1 MJCF compiles successfully now).
    3. Swap the env's mj_model / mjx_model for our Go2 compile.
    """
    import mujoco
    from mujoco import mjx

    Joystick = _find_go1_joystick_class()
    go1_base = _find_go1_base_module()

    menagerie_root = _menagerie_go2_dir().parent
    extras = _go1_mesh_injection(menagerie_root)
    if not extras:
        raise FileNotFoundError(
            f"No Go1 meshes under {menagerie_root}/unitree_go1/assets/. "
            "Run `python scripts/fetch_menagerie.py` (with git-lfs installed)."
        )

    original_get_assets = getattr(go1_base, "get_assets", None)
    if original_get_assets is None:
        raise RuntimeError(
            f"Could not find get_assets on {go1_base.__name__}; "
            "playground API has changed and our injection strategy needs updating."
        )

    def patched_get_assets():
        d = dict(original_get_assets())
        # setdefault so the original wins on collisions (XMLs/PNGs already
        # exist in Go1's dict; we only add genuinely missing meshes).
        for k, v in extras.items():
            d.setdefault(k, v)
        return d

    go1_base.get_assets = patched_get_assets
    try:
        env = Joystick() if env_config is None else Joystick(config=env_config)
    finally:
        go1_base.get_assets = original_get_assets

    scene_xml_str = consts.FEET_ONLY_FLAT_TERRAIN_XML.read_text()
    assets = get_assets()
    mj_model = mujoco.MjModel.from_xml_string(scene_xml_str, assets=assets)
    mjx_model = mjx.put_model(mj_model)
    for attr, value in (
        ("_mj_model", mj_model),
        ("mj_model", mj_model),
        ("_mjx_model", mjx_model),
        ("mjx_model", mjx_model),
    ):
        if hasattr(env, attr):
            try:
                setattr(env, attr, value)
            except AttributeError:
                pass

    if hasattr(env, "_post_init"):
        env._post_init()

    return env


def _find_go1_joystick_class():
    candidate_paths = [
        "mujoco_playground._src.locomotion.go1.joystick",
        "mujoco_playground.locomotion.go1.joystick",
        "mujoco_playground._src.locomotion.unitree_go1.joystick",
    ]
    last_err: Exception | None = None
    for mod_name in candidate_paths:
        try:
            mod = __import__(mod_name, fromlist=["Joystick"])
            return mod.Joystick
        except (ImportError, AttributeError) as e:
            last_err = e
    raise ImportError(
        f"Could not find mujoco_playground's Go1 Joystick class. "
        f"Tried: {candidate_paths}. Last error: {last_err}"
    )


def _find_go1_base_module():
    candidate_paths = [
        "mujoco_playground._src.locomotion.go1.base",
        "mujoco_playground.locomotion.go1.base",
        "mujoco_playground._src.locomotion.unitree_go1.base",
    ]
    last_err: Exception | None = None
    for mod_name in candidate_paths:
        try:
            return __import__(mod_name, fromlist=["get_assets"])
        except ImportError as e:
            last_err = e
    raise ImportError(
        f"Could not find mujoco_playground's Go1 base module. "
        f"Tried: {candidate_paths}. Last error: {last_err}"
    )
