"""Go2Env base class -- the Go2 analog of mujoco_playground's Go1Env.

Architecturally this is a thin subclass of the Go1 base env. The reason that
works is:
  - Joint names (FL_hip_joint, FL_thigh_joint, FL_calf_joint, ...) are
    *identical* between Go1 and Go2 in menagerie. Anything that indexes
    qpos / qvel / ctrl by joint name keeps working.
  - The foot collision geom names (FL/FR/RL/RR) also happen to match.
  - We swap in our own scene XML (with the right asset references) and override
    get_assets() so the compiled MJCF pulls Go2 meshes from menagerie/unitree_go2.
  - Site naming differs (Go2 uses *_foot) but the only consumers of foot site
    names are in our scene XML's sensor block -- which we wrote against the
    Go2 names directly -- so this never reaches the Python layer.

If you swap menagerie versions and something breaks, the failure mode is
usually "site/body/geom not found"; in that case verify ROOT_BODY, FEET_SITES,
and FEET_GEOMS in go2_constants against the current menagerie XML.
"""

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
        "Fix by one of:\n"
        "  (a) `python scripts/fetch_menagerie.py`\n"
        "  (b) `ROBOT_PET_CAT_AUTO_FETCH_MENAGERIE=1 ...`\n"
        "  (c) Set $MUJOCO_PLAYGROUND_MENAGERIE."
    )


def _fetch_menagerie_go2() -> Path:
    import subprocess

    cache = Path.home() / ".cache" / "mujoco_menagerie"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not (cache / ".git").is_dir():
        print(f"[go2_base] cloning mujoco_menagerie -> {cache}")
        subprocess.check_call(
            [
                "git", "clone", "--depth", "1",
                "https://github.com/google-deepmind/mujoco_menagerie.git",
                str(cache),
            ]
        )
    go2 = cache / "unitree_go2"
    if not (go2 / "go2_mjx.xml").is_file():
        raise FileNotFoundError(f"Fetched menagerie but {go2}/go2_mjx.xml missing.")
    return go2


def get_assets() -> dict[str, bytes]:
    """Return the asset dict mujoco needs to compile our Go2 scene from a string."""
    assets: dict[str, bytes] = {}

    for p in consts.ROOT_PATH.iterdir():
        if p.is_file() and p.suffix in {".xml", ".stl", ".obj"}:
            assets[p.name] = p.read_bytes()

    go2_dir = _menagerie_go2_dir()
    for p in go2_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".xml", ".stl", ".obj", ".png", ".jpg"}:
            rel = p.relative_to(go2_dir).as_posix()
            assets[rel] = p.read_bytes()
            assets.setdefault(p.name, p.read_bytes())

    return assets


def _go1_mesh_injection(menagerie_root: Path) -> dict[str, bytes]:
    """Read Go1 meshes and key them under the relative paths the Go1 XML uses.

    Critical context: mujoco_playground's Go1 get_assets() returns only
    XMLs + PNG textures -- no meshes. The mesh references in the Go1 XML
    look like
        <mesh file="../../../../../../mujoco_menagerie/unitree_go1/assets/trunk.stl"/>
    When mujoco.MjModel.from_xml_string can't find that key in the assets
    dict it falls back to opening from disk, which fails in pip installs.

    We fix it by augmenting the assets dict with those exact relative-path
    keys (plus a basename fallback since mujoco's lookup tries that too).
    Same trick for .obj if menagerie ships meshes that way.
    """
    extras: dict[str, bytes] = {}
    go1_assets = menagerie_root / "unitree_go1" / "assets"
    if not go1_assets.is_dir():
        return extras
    rel_prefix = "../../../../../../mujoco_menagerie/unitree_go1/assets/"
    for mesh in go1_assets.iterdir():
        if not mesh.is_file() or mesh.suffix.lower() not in {".stl", ".obj"}:
            continue
        data = mesh.read_bytes()
        extras[rel_prefix + mesh.name] = data
        extras.setdefault(mesh.name, data)
    return extras


def make_go2_joystick_env(env_config: Any = None):
    """Build a Joystick env on the Go2 robot.

    1. Patch Go1's module-level get_assets() to ALSO return the STL meshes
       (it normally returns only XMLs + PNGs, which makes mujoco compile
       fail in pip-only installs). Revert the patch after Joystick().
    2. Instantiate Go1 Joystick (compiles Go1's MJCF successfully now).
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
        d.update(extras)
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
        "Could not find mujoco_playground's Go1 Joystick class. "
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
        "Could not find mujoco_playground's Go1 base module. "
        f"Tried: {candidate_paths}. Last error: {last_err}"
    )
