#!/usr/bin/env python3
"""Surgical patcher for the Walker-MoB v1 task scaffold.

V1 scope: extend the working Go2 velocity walker with ONE new command dim
(body_height_cmd, an offset from the base 0.30 m trunk height). One new
reward (track_body_height). Everything else inherits from the velocity
task — same observations, same actions, same gait phasing, same runner.

Run from /workspace/unitree_rl_mjlab on the pod. Idempotent — safe to re-run.

Preconditions:
  - The upstream `src/tasks/velocity/` directory exists.
  - Either `src/tasks/velocity_mob/` does not exist (the patcher copies it from
    velocity), or it has already been patched (re-running is a no-op).

Edits, in order:
  1. Bootstrap: copy src/tasks/velocity -> src/tasks/velocity_mob.
     Rename velocity_env_cfg.py -> velocity_mob_env_cfg.py. Drop rl/ (we
     import the existing VelocityOnPolicyRunner from src.tasks.velocity.rl).
     Drop non-go2 robot config subdirs.
  2. src/tasks/velocity_mob/mdp/velocity_command.py: append MobVelocityCommand
     + MobVelocityCommandCfg subclasses. New tensor shape [N, 4]; extra dim
     is body_height_cmd. _resample_command samples it from cfg.ranges.body_height.
  3. src/tasks/velocity_mob/mdp/rewards.py: append track_body_height(env,
     command_name, std, base_height_target).
  4. src/tasks/velocity_mob/velocity_mob_env_cfg.py:
       - rename factory to make_velocity_mob_env_cfg
       - swap UniformVelocityCommandCfg -> MobVelocityCommandCfg in the
         commands dict, add body_height range
       - flip the shadowed `src.tasks.velocity.mdp` import to velocity_mob
       - register the track_body_height reward
  5. src/tasks/velocity_mob/config/go2/env_cfgs.py:
       - rename go2 cfg fns to *_mob_env_cfg
       - update imports + factory call
  6. src/tasks/velocity_mob/config/go2/__init__.py:
       - register Unitree-Go2-Flat-MoB (and Rough-MoB for completeness)

After running, print a summary.

Usage on a runpod:
  cd /workspace/unitree_rl_mjlab && source /workspace/mjlab_venv/bin/activate
  python3 /workspace/robot-pet-cat/scripts/patch_mob_unitree.py
  python3 -c "import src.tasks.velocity_mob.config.go2"  # smoke-import
"""
from __future__ import annotations
import re
import shutil
from pathlib import Path

VELOCITY = Path("src/tasks/velocity")
ROOT = Path("src/tasks/velocity_mob")
ENV_CFG = ROOT / "velocity_mob_env_cfg.py"
CMD_PY = ROOT / "mdp" / "velocity_command.py"
REWARDS_PY = ROOT / "mdp" / "rewards.py"
GO2_ENV_CFGS = ROOT / "config" / "go2" / "env_cfgs.py"
GO2_INIT = ROOT / "config" / "go2" / "__init__.py"

NON_GO2_ROBOTS = ["a2", "as2", "g1", "g1_23dof", "h1_2", "h2", "r1", "x2"]


def bootstrap() -> None:
    """Copy velocity -> velocity_mob; rename env cfg; drop rl/ and pycaches."""
    if ROOT.exists():
        return
    assert VELOCITY.exists(), f"missing {VELOCITY}"
    shutil.copytree(VELOCITY, ROOT)
    src = ROOT / "velocity_env_cfg.py"
    if src.exists():
        src.rename(ENV_CFG)
    rl_dir = ROOT / "rl"
    if rl_dir.exists():
        shutil.rmtree(rl_dir)
    for p in ROOT.rglob("__pycache__"):
        shutil.rmtree(p, ignore_errors=True)
    print(f"[ok] bootstrapped {ROOT} from {VELOCITY}")


def prune_non_go2_robots() -> None:
    cfg_dir = ROOT / "config"
    for robot in NON_GO2_ROBOTS:
        d = cfg_dir / robot
        if d.exists():
            shutil.rmtree(d)
            print(f"[ok] removed {d}")


CMD_APPEND = '''

# -----------------------------------------------------------------------------
# Walker-MoB v1 — body_height_cmd command extension (added by patch_mob_unitree.py).
# -----------------------------------------------------------------------------


class MobVelocityCommand(UniformVelocityCommand):
    """UniformVelocityCommand + a body_height_cmd channel (offset from default base height)."""

    cfg: "MobVelocityCommandCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # Grow the command tensor from [N, 3] -> [N, 4].
        old = self.vel_command_b
        self.vel_command_b = torch.zeros(self.num_envs, 4, device=self.device)
        self.vel_command_b[:, :3] = old

    def _resample_command(self, env_ids) -> None:
        super()._resample_command(env_ids)
        r = torch.empty(len(env_ids), device=self.device)
        self.vel_command_b[env_ids, 3] = r.uniform_(*self.cfg.ranges.body_height)
        # Optional warm-start curriculum: a fraction of envs get body_height=0
        # so the policy sees the "act like the base walker" regime regularly.
        if self.cfg.zero_body_height_envs > 0.0:
            mask = r.uniform_(0.0, 1.0) < self.cfg.zero_body_height_envs
            zero_ids = env_ids[mask]
            self.vel_command_b[zero_ids, 3] = 0.0


from dataclasses import dataclass as _dataclass


@_dataclass(kw_only=True)
class MobVelocityCommandCfg(UniformVelocityCommandCfg):
    """UniformVelocityCommandCfg + body_height range + a warm-start knob."""

    zero_body_height_envs: float = 0.0
    """Fraction of envs whose body_height_cmd is forced to 0 on resample. Set
    to 1.0 for warm-up phase (acts like the original walker), then decay to 0
    over training to enable full sampling."""

    @_dataclass
    class Ranges(UniformVelocityCommandCfg.Ranges):
        body_height: tuple[float, float] = (-0.10, 0.10)

    ranges: "MobVelocityCommandCfg.Ranges"  # type: ignore[assignment]

    def build(self, env) -> MobVelocityCommand:
        return MobVelocityCommand(self, env)
'''


def patch_velocity_command_py() -> None:
    text = CMD_PY.read_text()
    if "class MobVelocityCommand(" in text:
        print(f"[skip] {CMD_PY}: already patched")
        return
    text = text.rstrip() + CMD_APPEND
    CMD_PY.write_text(text)
    print(f"[ok] patched {CMD_PY}")


REWARD_APPEND = '''

# -----------------------------------------------------------------------------
# Walker-MoB v1 — body height tracking reward (added by patch_mob_unitree.py).
# -----------------------------------------------------------------------------


def track_body_height(
    env: "ManagerBasedRlEnv",
    command_name: str = "twist",
    std: float = 0.05,
    base_height_target: float = 0.30,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track commanded base z-height.

    The command tensor has shape [N, 4]; index 3 is the body_height_cmd offset
    (meters) from base_height_target. Reward is a squared-error Gaussian, like
    the existing track_linear_velocity/track_angular_velocity rewards.
    """
    cmd = env.command_manager.get_command(command_name)  # [N, 4]
    desired_z = base_height_target + cmd[:, 3]
    asset: Entity = env.scene[asset_cfg.name]
    actual_z = asset.data.root_link_pos_w[:, 2]
    err = actual_z - desired_z
    return torch.exp(-(err * err) / (std * std))
'''


def patch_rewards_py() -> None:
    text = REWARDS_PY.read_text()
    if "def track_body_height(" in text:
        print(f"[skip] {REWARDS_PY}: already patched")
        return
    text = text.rstrip() + REWARD_APPEND
    REWARDS_PY.write_text(text)
    print(f"[ok] patched {REWARDS_PY}")


def patch_env_cfg() -> None:
    text = ENV_CFG.read_text()
    if "def make_velocity_mob_env_cfg(" in text:
        print(f"[skip] {ENV_CFG}: already patched")
        return

    # 1. Rename factory + docstring.
    text = text.replace(
        "def make_velocity_env_cfg(",
        "def make_velocity_mob_env_cfg(",
    )
    text = text.replace(
        '"""Create base velocity tracking task configuration."""',
        '"""Create base velocity-MoB tracking task configuration (V1: body_height only)."""',
    )

    # 2. Flip the shadowed mdp import to use our local velocity_mob mdp.
    text = text.replace(
        "import src.tasks.velocity.mdp as mdp",
        "import src.tasks.velocity_mob.mdp as mdp",
    )

    # 3. Swap the upstream UniformVelocityCommandCfg import for our subclass.
    # The upstream import is kept around because the dataclass parent is
    # required elsewhere (none in this file — safe to drop entirely).
    text = text.replace(
        "from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg",
        "from src.tasks.velocity_mob.mdp import MobVelocityCommandCfg",
    )
    text = text.replace(
        '"twist": UniformVelocityCommandCfg(',
        '"twist": MobVelocityCommandCfg(',
    )
    text = text.replace(
        "ranges=UniformVelocityCommandCfg.Ranges(",
        "ranges=MobVelocityCommandCfg.Ranges(",
    )

    # 4. Add body_height range to the Ranges block. Insert right after
    # `heading=(-math.pi, math.pi),` (which we know is the last range entry).
    text = text.replace(
        "heading=(-math.pi, math.pi),",
        "heading=(-math.pi, math.pi),\n        body_height=(-0.10, 0.10),",
    )

    # 5. Add the track_body_height reward entry. Insert right before
    # the closing `}` of the rewards dict.
    lines = text.splitlines(keepends=True)
    rew_start = None
    for i, line in enumerate(lines):
        if line.startswith("  rewards = {"):
            rew_start = i
            break
    if rew_start is None:
        raise RuntimeError("rewards dict not found")
    rew_end = None
    for j in range(rew_start + 1, len(lines)):
        if lines[j].rstrip("\n") == "  }":
            rew_end = j
            break
    if rew_end is None:
        raise RuntimeError("rewards closing `}` not found")

    new_reward = (
        '    "track_body_height": RewardTermCfg(\n'
        "      func=mdp.track_body_height,\n"
        "      weight=2.0,\n"
        '      params={"command_name": "twist", "std": 0.05, "base_height_target": 0.30},\n'
        "    ),\n"
    )
    lines = lines[:rew_end] + [new_reward] + lines[rew_end:]
    text = "".join(lines)

    ENV_CFG.write_text(text)
    print(f"[ok] patched {ENV_CFG}")


def patch_go2_env_cfgs() -> None:
    text = GO2_ENV_CFGS.read_text()
    if "def unitree_go2_flat_mob_env_cfg(" in text:
        print(f"[skip] {GO2_ENV_CFGS}: already patched")
        return
    # Repoint factory import.
    text = text.replace(
        "from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg",
        "from src.tasks.velocity_mob.velocity_mob_env_cfg import make_velocity_mob_env_cfg",
    )
    text = text.replace("make_velocity_env_cfg(", "make_velocity_mob_env_cfg(")
    # Swap the cfg type used in the play-mode isinstance cast.
    text = text.replace(
        "from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg",
        "from src.tasks.velocity_mob.mdp import MobVelocityCommandCfg",
    )
    text = text.replace(
        "assert isinstance(twist_cmd, UniformVelocityCommandCfg)",
        "assert isinstance(twist_cmd, MobVelocityCommandCfg)",
    )
    # Rename the two go2 cfg fns.
    text = text.replace(
        "def unitree_go2_rough_env_cfg(",
        "def unitree_go2_rough_mob_env_cfg(",
    )
    text = text.replace(
        "def unitree_go2_flat_env_cfg(",
        "def unitree_go2_flat_mob_env_cfg(",
    )
    # Update internal cross-call (flat -> rough).
    text = text.replace(
        "cfg = unitree_go2_rough_env_cfg(",
        "cfg = unitree_go2_rough_mob_env_cfg(",
    )
    # Update docstrings.
    text = text.replace(
        '"""Create Unitree Go2 rough terrain velocity configuration."""',
        '"""Create Unitree Go2 rough-terrain velocity-MoB configuration."""',
    )
    text = text.replace(
        '"""Create Unitree Go2 flat terrain velocity configuration."""',
        '"""Create Unitree Go2 flat-terrain velocity-MoB configuration."""',
    )
    GO2_ENV_CFGS.write_text(text)
    print(f"[ok] patched {GO2_ENV_CFGS}")


GO2_INIT_NEW = '''from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_go2_flat_mob_env_cfg,
  unitree_go2_rough_mob_env_cfg,
)
from .rl_cfg import unitree_go2_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-Go2-Rough-MoB",
  env_cfg=unitree_go2_rough_mob_env_cfg(),
  play_env_cfg=unitree_go2_rough_mob_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Flat-MoB",
  env_cfg=unitree_go2_flat_mob_env_cfg(),
  play_env_cfg=unitree_go2_flat_mob_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
'''


def patch_go2_init() -> None:
    current = GO2_INIT.read_text()
    if "Unitree-Go2-Flat-MoB" in current:
        print(f"[skip] {GO2_INIT}: already patched")
        return
    GO2_INIT.write_text(GO2_INIT_NEW)
    print(f"[ok] patched {GO2_INIT}")


def main() -> int:
    if not VELOCITY.exists():
        print(
            f"ERROR: cannot find {VELOCITY}. Run this from /workspace/unitree_rl_mjlab.",
            flush=True,
        )
        return 2
    bootstrap()
    prune_non_go2_robots()
    patch_velocity_command_py()
    patch_rewards_py()
    patch_env_cfg()
    patch_go2_env_cfgs()
    patch_go2_init()
    print()
    print("MoB v1 patch applied. Registered tasks:")
    print("  - Unitree-Go2-Flat-MoB")
    print("  - Unitree-Go2-Rough-MoB")
    print()
    print("Smoke-import check:")
    print("  python3 -c 'import src.tasks.velocity_mob.config.go2'")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
