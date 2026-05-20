#!/usr/bin/env python3
"""Surgical patcher for crouch task scaffold.

Run from /workspace/unitree_rl_mjlab on the pod. Idempotent — safe to re-run.

Preconditions:
  - The upstream `src/tasks/velocity/` directory exists.
  - Either `src/tasks/crouch/` does not exist (the patcher copies it from
    velocity), or it has already been patched (re-running is a no-op for
    most steps).

Edits:
  - src/tasks/crouch/crouch_env_cfg.py:
      rename make_velocity_env_cfg -> make_crouch_env_cfg
      add `from . import mdp as crouch_mdp` import
      remove "command" and "phase" ObservationTermCfg entries
      remove the entire `commands: dict[...]` block
      remove "command_vel" CurriculumTermCfg entry
      remove `commands=commands,` kwarg passed to ManagerBasedRlEnvCfg
      remove rewards: track_linear_velocity, track_angular_velocity,
                       pose, foot_gait, foot_clearance, foot_slip,
                       soft_landing, stand_still
      add rewards: base_height_target, feet_in_contact

  - src/tasks/crouch/mdp/rewards.py:
      append base_height_target() and feet_in_contact() functions

  - src/tasks/crouch/config/go2/env_cfgs.py:
      change import: make_velocity_env_cfg -> make_crouch_env_cfg
      rename unitree_go2_rough_env_cfg -> unitree_go2_crouch_env_cfg
      strip cfg.rewards["pose"] customizations (pose reward is removed)
      drop `if play: twist_cmd = cfg.commands["twist"] ...` block
      drop the unitree_go2_flat_env_cfg function (we only need one entry point)

  - src/tasks/crouch/config/go2/__init__.py:
      register a single task: Unitree-Go2-Crouch

After running, the patcher prints a summary of what changed.
"""
from __future__ import annotations
import re
import shutil
from pathlib import Path

VELOCITY = Path("src/tasks/velocity")
ROOT = Path("src/tasks/crouch")
ENV_CFG = ROOT / "crouch_env_cfg.py"
REWARDS_PY = ROOT / "mdp" / "rewards.py"
GO2_ENV_CFGS = ROOT / "config" / "go2" / "env_cfgs.py"
GO2_INIT = ROOT / "config" / "go2" / "__init__.py"

# Non-go2 robot configs to remove from the crouch package (we only need go2).
NON_GO2_ROBOTS = ["a2", "as2", "g1", "g1_23dof", "h1_2", "h2", "r1", "x2"]


def bootstrap_crouch_from_velocity() -> None:
    """Copy src/tasks/velocity -> src/tasks/crouch if crouch doesn't yet exist.

    Renames velocity_env_cfg.py -> crouch_env_cfg.py. Drops the rl/ folder
    (the velocity OnPolicyRunner is imported directly by the crouch go2
    __init__.py).
    """
    if ROOT.exists():
        return
    assert VELOCITY.exists(), f"missing {VELOCITY}"
    shutil.copytree(VELOCITY, ROOT)
    # Rename env cfg
    src = ROOT / "velocity_env_cfg.py"
    if src.exists():
        src.rename(ENV_CFG)
    # Drop rl/ (we use velocity's runner)
    rl_dir = ROOT / "rl"
    if rl_dir.exists():
        shutil.rmtree(rl_dir)
    # Drop pycache
    for p in ROOT.rglob("__pycache__"):
        shutil.rmtree(p, ignore_errors=True)
    print(f"bootstrapped {ROOT} from {VELOCITY}")


def prune_non_go2_robots() -> None:
    """Remove robot config subdirs we don't need (only go2 is in scope)."""
    cfg_dir = ROOT / "config"
    for robot in NON_GO2_ROBOTS:
        d = cfg_dir / robot
        if d.exists():
            shutil.rmtree(d)
            print(f"removed {d}")


def remove_dict_entry(text: str, key: str) -> str:
    """Remove a dict entry like `"key": SomeCfg(...)` followed by `),`.

    The entry is identified by `"<key>": ` at any indentation; the closing
    `),` must appear at the same indentation as the key line. Returns the
    text without the matched block, including the trailing newline.
    """
    lines = text.splitlines(keepends=True)
    needle = f'"{key}":'
    start = None
    indent = None
    for i, line in enumerate(lines):
        stripped = line.lstrip(" ")
        if stripped.startswith(needle):
            start = i
            indent = len(line) - len(stripped)
            break
    if start is None:
        raise RuntimeError(f"dict entry {key!r} not found")
    # Walk forward; entry ends when we see a line whose content is exactly
    # the same indent followed by  '),'  (with optional trailing whitespace),
    # or a one-liner case where the first line itself ends with '),'.
    first_stripped = lines[start].rstrip("\n").rstrip()
    if first_stripped.endswith("),"):
        return "".join(lines[:start] + lines[start + 1 :])
    end = None
    expected = " " * indent + "),"
    for j in range(start + 1, len(lines)):
        if lines[j].rstrip("\n").rstrip() == expected:
            end = j
            break
    if end is None:
        raise RuntimeError(f"closing `),` not found for {key!r}")
    return "".join(lines[:start] + lines[end + 1 :])


def remove_commands_dict(text: str) -> str:
    lines = text.splitlines(keepends=True)
    start = None
    indent = None
    for i, line in enumerate(lines):
        if "commands: dict[str, CommandTermCfg] = {" in line:
            start = i
            indent = len(line) - len(line.lstrip(" "))
            break
    if start is None:
        raise RuntimeError("commands dict not found")
    expected_close = " " * indent + "}"
    end = None
    for j in range(start + 1, len(lines)):
        if lines[j].rstrip("\n") == expected_close:
            end = j
            break
    if end is None:
        raise RuntimeError("commands closing `}` not found")
    return "".join(lines[:start] + lines[end + 1 :])


def patch_env_cfg() -> None:
    text = ENV_CFG.read_text()
    # Idempotency: if the factory is already renamed, assume the rest of
    # the env_cfg patches were already applied and skip.
    if "def make_crouch_env_cfg(" in text:
        print(f"{ENV_CFG}: already patched, skipping")
        return

    # 1. Rename factory.
    text = text.replace(
        "def make_velocity_env_cfg(", "def make_crouch_env_cfg("
    )
    text = text.replace(
        '"""Create base velocity tracking task configuration."""',
        '"""Create base crouch task configuration."""',
    )

    # 2. Add local mdp import (used for crouch-specific rewards).
    # Insert after the upstream mdp import.
    upstream_import = "from mjlab.tasks.velocity import mdp"
    if "from . import mdp as crouch_mdp" not in text:
        text = text.replace(
            upstream_import,
            upstream_import + "\nfrom . import mdp as crouch_mdp",
        )

    # 3. Remove observation terms tied to the velocity command.
    text = remove_dict_entry(text, "command")
    text = remove_dict_entry(text, "phase")

    # 4. Remove the entire commands dict.
    text = remove_commands_dict(text)

    # 5. Remove `commands=commands,` kwarg.
    text = re.sub(r"^\s*commands=commands,\n", "", text, flags=re.MULTILINE)

    # 6. Remove "command_vel" curriculum entry.
    text = remove_dict_entry(text, "command_vel")

    # 7. Remove velocity-specific reward entries.
    for k in [
        "track_linear_velocity",
        "track_angular_velocity",
        "pose",
        "foot_gait",
        "foot_clearance",
        "foot_slip",
        "soft_landing",
        "stand_still",
    ]:
        text = remove_dict_entry(text, k)

    # 8. Insert crouch rewards right before the closing `}` of the rewards
    # dict. The rewards dict opens with `  rewards = {` (2-space indent)
    # and closes with `  }` at the same level.
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

    new_rewards = (
        '    "base_height_target": RewardTermCfg(\n'
        "      func=crouch_mdp.base_height_target,\n"
        "      weight=-2.0,\n"
        "      params={\n"
        '        "asset_cfg": SceneEntityCfg("robot"),\n'
        '        "target_height": 0.15,\n'
        "      },\n"
        "    ),\n"
        '    "feet_in_contact": RewardTermCfg(\n'
        "      func=crouch_mdp.feet_in_contact,\n"
        "      weight=1.0,\n"
        '      params={"sensor_name": "feet_ground_contact"},\n'
        "    ),\n"
    )
    lines = lines[:rew_end] + [new_rewards] + lines[rew_end:]
    text = "".join(lines)

    ENV_CFG.write_text(text)
    print(f"patched {ENV_CFG}")


REWARD_APPEND = '''

# -----------------------------------------------------------------------------
# Crouch-specific rewards (added by patch_crouch.py).
# -----------------------------------------------------------------------------


def base_height_target(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_height: float = 0.15,
) -> torch.Tensor:
    """Penalise squared distance between base z and target_height.

    Returns a positive value (penalty applied via negative weight in cfg).
    """
    asset: Entity = env.scene[asset_cfg.name]
    base_z = asset.data.root_link_pos_w[:, 2]
    return torch.square(base_z - target_height)


def feet_in_contact(
    env: "ManagerBasedRlEnv",
    sensor_name: str,
) -> torch.Tensor:
    """Reward number of feet currently in contact with the ground (0..4).

    Encourages the cat to keep all four feet planted while crouching.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    found = sensor.data.found
    assert found is not None
    # found shape: (B, num_feet, num_slots) -> aggregate to (B, num_feet)
    contact = (found > 0).any(dim=-1).float()
    return contact.sum(dim=-1)
'''


def patch_rewards_py() -> None:
    text = REWARDS_PY.read_text()
    if "def base_height_target(" in text:
        print(f"{REWARDS_PY}: already patched, skipping")
        return
    text = text.rstrip() + REWARD_APPEND
    REWARDS_PY.write_text(text)
    print(f"patched {REWARDS_PY}")


def patch_go2_env_cfgs() -> None:
    text = GO2_ENV_CFGS.read_text()
    # Idempotency: detect already-patched file via the renamed factory.
    if "def unitree_go2_crouch_env_cfg(" in text:
        print(f"{GO2_ENV_CFGS}: already patched, skipping")
        return
    # 1. Update import path.
    text = text.replace(
        "from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg",
        "from src.tasks.crouch.crouch_env_cfg import make_crouch_env_cfg",
    )
    # 2. Rename factory call.
    text = text.replace("make_velocity_env_cfg(", "make_crouch_env_cfg(")
    # 3. Rename outer function.
    text = text.replace(
        "def unitree_go2_rough_env_cfg(",
        "def unitree_go2_crouch_env_cfg(",
    )
    text = text.replace(
        '"""Create Unitree Go2 rough terrain velocity configuration."""',
        '"""Create Unitree Go2 crouch task configuration."""',
    )
    # 4. Strip customisations targeting removed rewards. `pose` opens a dict
    # literal so it needs the block-aware strip; the others are one-liners.
    text = strip_pose_customisation(text)
    for removed in [
        "foot_gait",
        "foot_clearance",
        "foot_slip",
        "track_linear_velocity",
        "track_angular_velocity",
        "stand_still",
        "soft_landing",
    ]:
        text = re.sub(
            r'^\s*cfg\.rewards\["' + re.escape(removed) + r'"\].*\n',
            "",
            text,
            flags=re.MULTILINE,
        )
    # 5. Drop the `if play: twist_cmd = cfg.commands["twist"] ...` block.
    text = strip_play_twist_block(text)
    # 6. Drop the unitree_go2_flat_env_cfg function (we only need the rough
    # entry point for crouch — flat terrain is set inside it).
    text = strip_flat_fn(text)
    # 7. Force flat terrain in the crouch env (crouching doesn't need rough).
    # Append a small block after the existing function's terrain setup that
    # forces flat ground. We'll handle this by inserting just before the
    # `return cfg` of unitree_go2_crouch_env_cfg.
    text = force_flat_terrain(text)
    # 8. Drop the unused UniformVelocityCommandCfg import (twist block gone).
    text = text.replace(
        "from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg\n",
        "",
    )

    GO2_ENV_CFGS.write_text(text)
    print(f"patched {GO2_ENV_CFGS}")


def strip_pose_customisation(text: str) -> str:
    lines = text.splitlines(keepends=True)
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith('cfg.rewards["pose"]'):
            # If it opens a dict literal, skip until the matching closing `}`
            stripped = line.rstrip("\n").rstrip()
            if stripped.endswith("{"):
                indent = len(line) - len(line.lstrip(" "))
                # Skip until we find a line `<indent>}` (the closing brace)
                j = i + 1
                while j < len(lines):
                    if lines[j].rstrip("\n") == " " * indent + "}":
                        break
                    j += 1
                i = j + 1  # skip the closing brace line too
                continue
            # Single-line assignment — just drop it
            i += 1
            continue
        out.append(line)
        i += 1
    return "".join(out)


def strip_play_twist_block(text: str) -> str:
    lines = text.splitlines(keepends=True)
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("if play:"):
            indent = len(line) - len(line.lstrip(" "))
            # Look ahead — if any line inside this if-block references
            # cfg.commands, drop the whole block.
            j = i + 1
            while j < len(lines):
                lstrip = lines[j].lstrip(" ")
                if not lines[j].strip():
                    j += 1
                    continue
                this_indent = len(lines[j]) - len(lstrip)
                if this_indent <= indent:
                    break
                j += 1
            block = "".join(lines[i:j])
            if "cfg.commands" in block:
                i = j  # drop the whole block
                continue
        out.append(line)
        i += 1
    return "".join(out)


def strip_flat_fn(text: str) -> str:
    lines = text.splitlines(keepends=True)
    out = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("def unitree_go2_flat_env_cfg("):
            # Skip until next top-level def or end of file
            j = i + 1
            while j < len