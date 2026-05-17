"""Register a Go2-flavored Joystick env into mujoco_playground's registry.

mujoco_playground only ships Go1 quadruped envs out of the box (Footstand,
Getup, Handstand, JoystickFlatTerrain, JoystickRoughTerrain). For our cat
project we specifically need Go2, so this module monkey-patches the registry
with "Go2JoystickFlatTerrain" pointing at the Go2 model wrapped around the
Go1 Joystick task code (see go2_base.make_go2_joystick_env for why that's
sound: same joint/geom names).

Importing this module is what triggers registration -- amp_env.py does this
unconditionally before any registry.load() call.

Idempotent: importing twice is a no-op.
"""

from __future__ import annotations

_REGISTERED = False
ENV_NAME = "Go2JoystickFlatTerrain"


def register() -> None:
    """Register Go2JoystickFlatTerrain. Idempotent."""
    global _REGISTERED
    if _REGISTERED:
        return

    from mujoco_playground import registry  # noqa: PLC0415

    from robot_pet_cat.motion.go2_base import make_go2_joystick_env  # noqa: PLC0415

    # mujoco_playground's registry API has shifted between versions. Try the
    # public surface first, then fall back to mutating internal dicts.
    if hasattr(registry, "register_environment"):
        # New-style API (>= ~0.0.7).
        def _default_config():
            from robot_pet_cat.motion.go2_base import _find_go1_joystick_class  # noqa: PLC0415
            J = _find_go1_joystick_class()
            return J().default_config()

        registry.register_environment(
            env_name=ENV_NAME,
            env_class=lambda cfg=None: make_go2_joystick_env(cfg),
            cfg_class=_default_config,
        )
    else:
        # Older API: keep a name -> factory dict and override registry.load.
        # We do this by intercepting the existing load function.
        _patch_legacy_registry(registry)

    _REGISTERED = True


def _patch_legacy_registry(registry) -> None:
    """Wrap registry.load to fall through to our factory for Go2 envs."""
    from robot_pet_cat.motion.go2_base import make_go2_joystick_env  # noqa: PLC0415

    factories = {ENV_NAME: make_go2_joystick_env}

    original_load = registry.load

    def patched_load(name, config=None, *args, **kwargs):
        if name in factories:
            return factories[name](config)
        return original_load(name, config, *args, **kwargs)

    registry.load = patched_load

    # Some code paths discover available envs via registry.ALL_ENVS; advertise
    # ourselves there if the attribute exists.
    if hasattr(registry, "ALL_ENVS"):
        try:
            registry.ALL_ENVS = type(registry.ALL_ENVS)(
                list(registry.ALL_ENVS) + [ENV_NAME]
            )
        except TypeError:
            # ALL_ENVS could be a frozen container; ignore -- registry.load
            # still works via the patched function above.
            pass


# Register on import. This is the whole point of the module.
register()
