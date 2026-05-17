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
    in the assets dict. Use basename-only keys.
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
    the mesh files. We inject {basename: bytes} for every Go1 STL/OBJ.
    Basename-only -- mujoco rejects duplicates.
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


def _force_jax_backend(env_config: Any) -> Any:
    """Override env_config.impl to 'jax' to avoid a mujoco/warp ABI mismatch.

    Recent mujoco.mjx defaults Go1's config to impl='warp', and the warp
    version typically installed alongside it doesn't have
    `warp.types.warp_type_to_np_dtype`. The JAX backend is the standard
    CPU/GPU path with no warp dependency, so we just force it.
    """
    if env_config is None:
        return env_config
    if hasattr(env_config, "impl"):
        try:
            env_config.impl = "jax"
        except (AttributeError, TypeError):
            pass
    return env_config


def _resolve_default_config(Joystick) -> Any:
    """Get Joystick's default_config without instantiating it -- the
    constructor is what's failing, so we can't go env.default_config()."""
    import importlib

    mod = importlib.import_module(Joystick.__module__)
    if hasattr(mod, "default_config") and callable(mod.default_config):
        return mod.default_config()
    for name in ("default_config", "DEFAULT_CONFIG"):
        attr = getattr(Joystick, name, None)
        if callable(attr):
            try:
                return attr()
            except TypeError:
                pass
        elif attr is not None:
            return attr
    return None


def make_go2_joystick_env(env_config: Any = None):
    """Build a Joystick env on the Go2 robot.

    1. Patch Go1's get_assets() to ALSO return the mesh files (pip install
       ships only XMLs + PNGs, no meshes).
    2. Resolve default config and force impl='jax' to dodge the mujoco/warp
       ABI mismatch in mjx.put_model.
    3. Instantiate Go1 Joystick (now compiles).
    4. Swap to our Go2 mj_model / mjx_model.
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
            f"Could not find get_assets on {go1_base.__name__}."
        )

    def patched_get_assets():
        d = dict(original_get_assets())
        for k, v in extras.items():
            d.setdefault(k, v)
        return d

    # Force jax backend at config-creation time so __init__ sees the right impl.
    if env_config is None:
        env_config = _resolve_default_config(Joystick)
    env_config = _force_jax_backend(env_config)

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
