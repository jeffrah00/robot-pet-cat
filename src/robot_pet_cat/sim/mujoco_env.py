"""Minimal MuJoCo environment wrapper around the Unitree Go2 model.

This is a Phase 1 smoke test. Once `mujoco_menagerie` is available it loads the
official Go2 scene; otherwise it falls back to a placeholder humanoid so the rest
of the pipeline is still runnable.

The full RL env is built in Phase 2 on top of MuJoCo Playground rather than this
file — this is just enough to verify the install.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# Path used by `mujoco_menagerie`. Phase 1 task: clone the menagerie repo as a
# submodule under `third_party/` and update this constant.
GO2_XML_CANDIDATES = [
    Path("third_party/mujoco_menagerie/unitree_go2/scene.xml"),
    Path("third_party/mujoco_menagerie/unitree_go2/go2.xml"),
]


def find_go2_xml() -> Path | None:
    for p in GO2_XML_CANDIDATES:
        if p.exists():
            return p
    return None


def smoke_test(render: bool = False, steps: int = 1000) -> int:
    """Load the Go2 (or fallback) and step it. Returns exit code."""
    try:
        import mujoco
    except ImportError:
        print("mujoco not installed. Run: pip install -e .")
        return 1

    xml_path = find_go2_xml()
    if xml_path is None:
        # Fallback so Phase 0 still has something to show.
        print(
            "Unitree Go2 model not found. Add the mujoco_menagerie submodule:\n"
            "  git submodule add https://github.com/google-deepmind/mujoco_menagerie "
            "third_party/mujoco_menagerie\n"
            "Falling back to the built-in humanoid demo."
        )
        model = mujoco.MjModel.from_xml_string(_HUMANOID_FALLBACK_XML)
    else:
        print(f"Loading Go2 from {xml_path}")
        model = mujoco.MjModel.from_xml_path(str(xml_path))

    data = mujoco.MjData(model)
    rng = np.random.default_rng(0)

    if render:
        try:
            import mujoco.viewer as viewer
        except ImportError:
            print("viewer not available in this build; running headless.")
            render = False

    if render:
        with viewer.launch_passive(model, data) as v:
            for _ in range(steps):
                data.ctrl[:] = rng.normal(0, 0.05, size=model.nu)
                mujoco.mj_step(model, data)
                v.sync()
    else:
        for _ in range(steps):
            data.ctrl[:] = rng.normal(0, 0.05, size=model.nu)
            mujoco.mj_step(model, data)

    print(f"Stepped {steps} frames; final qpos sum = {float(np.sum(data.qpos)):.3f}")
    return 0


# Tiny fallback so the file is runnable before submodules are cloned.
_HUMANOID_FALLBACK_XML = """
<mujoco>
  <worldbody>
    <geom type="plane" size="5 5 0.1" rgba="0.5 0.5 0.6 1"/>
    <body pos="0 0 1">
      <freejoint/>
      <geom type="capsule" size="0.05 0.2" rgba="0.8 0.2 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""
