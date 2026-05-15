"""Phase 1 smoke test: the package imports and the sim wrapper runs without crashing."""

from __future__ import annotations


def test_package_imports() -> None:
    import robot_pet_cat

    assert robot_pet_cat.__version__


def test_sim_smoke() -> None:
    """The smoke test should run even before the Go2 submodule is cloned —
    it falls back to a built-in fixture in that case."""
    from robot_pet_cat.sim.mujoco_env import smoke_test

    rc = smoke_test(render=False, steps=10)
    assert rc == 0
