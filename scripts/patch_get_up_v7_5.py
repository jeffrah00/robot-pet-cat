#!/usr/bin/env python3
"""
get_up v7.5 -- FR-Net (paper-faithful gated postural reward).

The arXiv 2509.11504 (FR-Net) paper's Eq. 8 orientation reward is:

  R_orient = w1*g_xy^2 + w2*exp(-(g_z+1)^2/(2 eps^2))
             + w3*exp(-|q-q_stand|^2) * 1{|g_z+1| < eps}

The crucial detail: the postural-convergence term exp(-|q-q_stand|^2) is
GATED by 1{|g_z+1| < eps}, meaning it only activates once the base is
already nearly upright. v7's `hind_legs_extended` reward (and any similar
joint-target reward) fires throughout, so the policy can satisfy "hocks at
target angle" while still on its side -- producing the dog-sit / play-bow
failure mode we've observed.

Changes vs v7:
  * Add hind_legs_extended_gated = original * (g_z < -0.95)
    (gate threshold roughly matches the paper's |g_z+1| < eps with eps ~0.05)
  * Swap env_cfg reward function reference from .hind_legs_extended to
    .hind_legs_extended_gated
  * Boost stand_fully_bonus 15 -> 30 (carries v7.2's idea)
  * Relax fell_over 70 -> 100 deg (carries v7.2's idea)
"""

from __future__ import annotations
import subprocess
import sys
from pathlib import Path

ACTIVE = Path("/workspace/unitree_rl_mjlab/src/tasks/get_up")
V7 = Path("/workspace/robot-pet-cat/scripts/patch_get_up_v7.py")


def apply() -> int:
    rc = subprocess.run(["python3", str(V7)], capture_output=True, text=True)
    if rc.returncode != 0:
        print("ERROR: v7 patcher failed:\n" + rc.stderr, file=sys.stderr)
        return rc.returncode
    print(rc.stdout)

    cfg_path = ACTIVE / "get_up_env_cfg.py"
    rewards_path = ACTIVE / "mdp" / "rewards.py"
    txt = cfg_path.read_text()

    def must_replace(old, new, label):
        nonlocal txt
        if old not in txt:
            raise RuntimeError("anchor not found: " + label)
        if txt.count(old) > 1:
            raise RuntimeError("anchor not unique: " + label)
        txt = txt.replace(old, new)
        print("[edit] " + label)

    must_replace(
        "func=get_up_mdp.stand_fully_bonus,\n            weight=15.0,",
        "func=get_up_mdp.stand_fully_bonus,\n            weight=30.0,",
        "stand_fully_bonus 15 -> 30",
    )
    must_replace(
        "math.radians(70.0)",
        "math.radians(100.0)",
        "fell_over 70deg -> 100deg",
    )
    must_replace(
        "func=get_up_mdp.hind_legs_extended,",
        "func=get_up_mdp.hind_legs_extended_gated,",
        "hind_legs_extended -> hind_legs_extended_gated",
    )

    cfg_path.write_text(txt)

    # Append the gated reward function.
    extra = '\n\n# === v7.5 ============================================================\n'
    extra += 'def hind_legs_extended_gated(env, asset_cfg=_DEFAULT_ASSET_CFG):\n'
    extra += '    """Postural-convergence reward, gated on near-upright base.\n\n'
    extra += '    Per FR-Net paper Eq. 8: the postural target exp(-|q-q_stand|^2)\n'
    extra += '    is multiplied by 1{|g_z+1| < eps}. This prevents the policy from\n'
    extra += '    satisfying the joint-angle target while still tilted, which would\n'
    extra += '    otherwise lock in the dog-sit / play-bow failure mode.\n'
    extra += '    """\n'
    extra += '    asset = env.scene[asset_cfg.name]\n'
    extra += '    jp = asset.data.joint_pos\n'
    extra += '    err = torch.zeros(jp.shape[0], device=jp.device)\n'
    extra += '    for idx, target in zip(_HIND_INDICES_V7, _HIND_TARGETS_V7):\n'
    extra += '        err = err + (jp[:, idx] - target) ** 2\n'
    extra += '    raw = torch.exp(-err)\n'
    extra += '    # Gate: only reward postural match once base is nearly upright.\n'
    extra += '    # projected_gravity_b[:, 2] is approx -1 when upright, +1 when belly-up.\n'
    extra += '    gz = asset.data.projected_gravity_b[:, 2]\n'
    extra += '    near_upright = (gz < -0.95).float()\n'
    extra += '    return raw * near_upright\n'
    rewards_path.write_text(rewards_path.read_text() + extra)
    print("[edit] mdp/rewards.py: hind_legs_extended_gated")

    rc = subprocess.run(
        ["python3", "-c",
         "import ast;"
         "ast.parse(open('" + str(cfg_path) + "').read());"
         "ast.parse(open('" + str(rewards_path) + "').read())"],
        capture_output=True, text=True,
    )
    if rc.returncode != 0:
        print("ERROR: syntax check failed:\n" + rc.stderr, file=sys.stderr)
        return 8

    print("[ok] v7.5 patch applied")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(apply())
    except RuntimeError as e:
        print("ERROR: " + str(e), file=sys.stderr)
        sys.exit(9)
