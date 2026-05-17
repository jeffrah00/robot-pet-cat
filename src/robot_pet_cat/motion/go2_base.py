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

    mujoco_playground bundles a checkout of mujoco_menagerie; we reuse the
    same one so versions stay aligned. The location has drifted between
    playground releases, so we try a few candidates and raise a helpful
    error if none works.
    """
    from mujoco_playground._src import mjx_env  # noqa: PLC0415

    candidates: list[Path] = []
    # Newer playground releases expose ROOT explicitly.
    if hasattr(mjx_env, "ROOT_PATH"):
        candidates.append(Path(mjx_env.ROOT_PATH) / "mujoco_menagerie" / "unitree_go2")
    # Common layout: <playground>/_src/mujoco_menagerie/unitree_go2/
    mjx_env_dir = Path(mjx_env.__file__).resolve().parent
    candidates.append(mjx_env_dir / "mujoco_menagerie" / "unitree_go2")
    candidates.append(mjx_env_dir.parent / "mujoco_menagerie" / "unitree_go2")
    # Some setups co-locate menagerie under the package root.
    import mujoco_playground  # noqa: PLC0415

    pkg_root = Path(mujoco_playground.__file__).resolve().parent
    candidates.append(pkg_root / "mujoco_menagerie" / "unitree_go2")

    for c in candidates:
        if (c / "go2_mjx.xml").is_file():
            return c

    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Could not locate mujoco_menagerie/unitree_go2. "
        "Tried:\n  " + tried + "\n"
        "If you installed mujoco_playground from source, the menagerie "
        "submodule may not have been initialized. Try: "
        "`git -C $(python -c 'import mujoco_playground, pathlib; "
        "print(pathlib.Path(mujoco_playground.__file__).parent.parent.parent)') "
        "submodule update --init --recursive`."
    )


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
            # Use forward-slash relative path as the key, since MJCF paths
            # are POSIX-style.
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
    import mujoco  # noqa: PLC0415
    from mujoco import mjx  # noqa: PLC0415

    # Locate the Go1 joystick class. Module path varies by version.
    Joystick = _find_go1_joystick_class()

    scene_xml_str = consts.FEET_ONLY_FLAT_TERRAIN_XML.read_text()
    assets = get_assets()

    if env_config is None:
        env = Joystick()
    else:
        env = Joystick(config=env_config)

    # Hot-swap the underlying mujoco model from our Go2 XML.
    mj_model = mujoco.MjModel.from_xml_string(scene_xml_str, assets=assets)
    mjx_model = mjx.put_model(mj_model)
    # Best-effort: most playground envs expose these attribute names.
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

    # Some envs cache derived data in _post_init; re-run it if present so the
    # cached site/body/geom ids reflect the new model.
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
