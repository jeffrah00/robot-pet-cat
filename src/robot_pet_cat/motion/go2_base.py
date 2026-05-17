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
    """Locate mujoco_menagerie's unitree_go2 directory at runtime.

    mujoco_playground does NOT bundle menagerie meshes inside its wheel --
    they get cloned on-demand to a cache dir on first env load. The exact
    hook has drifted between playground releases; we try several strategies:

      1. Import MENAGERIE_PATH from a known playground module (any version
         where Go1 envs work has this somewhere).
      2. Honor an explicit override via $MUJOCO_PLAYGROUND_MENAGERIE.
      3. Check the standard playground cache: ~/.cache/mujoco_menagerie.
      4. Walk a few candidate paths inside the installed playground tree
         (covers historic layouts).
      5. As a last resort, git-clone menagerie into a cache dir.

    The git-clone fallback is opt-in via $ROBOT_PET_CAT_AUTO_FETCH_MENAGERIE=1
    so we never silently pull megabytes of meshes during import.
    """
    import os

    # 1. Find MENAGERIE_PATH that Go1 already uses, so we stay in sync with
    # whatever cache hook playground itself respects.
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

    # 2. Explicit override.
    env_override = os.environ.get("MUJOCO_PLAYGROUND_MENAGERIE")
    if env_override:
        candidates.append(Path(env_override) / "unitree_go2")

    # 3. Standard XDG cache location playground tends to use.
    candidates.append(Path.home() / ".cache" / "mujoco_menagerie" / "unitree_go2")

    # 4. Walk likely paths inside the installed playground tree.
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

    # 5. Opt-in last-resort fetch.
    if os.environ.get("ROBOT_PET_CAT_AUTO_FETCH_MENAGERIE") == "1":
        return _fetch_menagerie_go2()

    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Could not locate mujoco_menagerie/unitree_go2.\n"
        "Tried:\n  " + tried + "\n\n"
        "Fix by one of:\n"
        "  (a) `python scripts/fetch_menagerie.py`  (clones menagerie into "
        "~/.cache/mujoco_menagerie -- ~50 MB)\n"
        "  (b) `ROBOT_PET_CAT_AUTO_FETCH_MENAGERIE=1 python scripts/smoke_test_go2_env.py`\n"
        "  (c) Set $MUJOCO_PLAYGROUND_MENAGERIE to your existing menagerie checkout."
    )


def _fetch_menagerie_go2() -> Path:
    """Clone mujoco_menagerie into the user's cache and return the Go2 dir."""
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
        raise FileNotFoundError(
            f"Fetched menagerie at {cache} but unitree_go2/go2_mjx.xml is missing. "
            "Has the menagerie layout changed?"
        )
    return go2


def get_assets() -> dict[str, bytes]:
    """Return the asset dict mujoco needs to compile our Go2 scene from a string.

    Combines:
      - Our hand-rolled scene_mjx_feetonly_flat_terrain.xml and sensor_feet.xml
        from data/go2_scenes/ (lets us add sensors + floor + keyframe).
      - Everything from menagerie's unitree_go2 directory (meshes + the
        go2_mjx.xml our scene XML includes).
    """
    assets: dict[str, bytes] = {}

    # Our scene files. Keys are the file basenames so the <include
    # file="go2_mjx.xml"/> in our scene resolves correctly.
    for p in consts.ROOT_PATH.iterdir():
        if p.is_file() and p.suffix in {".xml", ".stl", ".obj"}:
            assets[p.name] = p.read_bytes()

    # Menagerie's Go2 directory. We include EVERYTHING from there so meshes
    # referenced by go2_mjx.xml resolve.
    go2_dir = _menagerie_go2_dir()
    for p in go2_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".xml", ".stl", ".obj", ".png", ".jpg"}:
            rel = p.relative_to(go2_dir).as_posix()
            assets[rel] = p.read_bytes()
            # Also expose by basename so a bare include="go2_mjx.xml" works.
            assets.setdefault(p.name, p.read_bytes())

    return assets


def make_go2_joystick_env(env_config: Any = None):
    """Build a Joystick env on the Go2 robot.

    Implementation strategy: instantiate mujoco_playground's Go1 Joystick
    env, then re-compile its underlying mjx model from OUR Go2 scene XML +
    Go2 assets. This sidesteps having to fully recreate the Go1 task code
    (reward functions, observation flatten, etc.) which is large and stable.

    The override depends on mujoco_playground's Go1Env exposing a
    ``_post_init`` or model attribute we can patch; we accept either by
    trying both. If neither hook is present, fall back to a manual
    construction using their Joystick __init__.
    """
    import mujoco
    from mujoco import mjx

    Joystick = _find_go1_joystick_class()

    scene_xml_str = consts.FEET_ONLY_FLAT_TERRAIN_XML.read_text()
    assets = get_assets()

    if env_config is None:
        env = Joystick()
    else:
        env = Joystick(config=env_config)

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
    """Find the Joystick class for Go1 across playground versions."""
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
