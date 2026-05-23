#!/usr/bin/env python3
"""
get_up v10c -- v7 + hind-feet-on-ground reward.

Hypothesis: a real cat's recovery from prone is "tuck hind legs under body,
push, lift forequarters". v7 rewards hind-joint angles (postural target) and
final stand bonus, but never directly rewards getting the hind PAWS into
contact with the ground. The policy can satisfy hind_legs_extended while the
hind feet are still in the air (legs extended OUT, not DOWN).

This patch adds a small +2.0 reward for both rear feet (RL, RR) being in
contact with the ground simultaneously. Uses the existing `feet_ground_contact`
sensor (also consumed by v7's FR-Net obs term).

KEEP v7 termination, FR-Net, v4c rewards, target_height 0.30. Only adds
one new reward term.

Usage on a runpod:
  python3 scripts/patch_get_up_v10c.py
  cd /workspace/unitree_rl_mjlab && source /workspace/mjlab_venv/bin/activate
  set -a && source /workspace/robot-pet-cat/.env && set +a
  nohup python3 scripts/train.py Unitree-Go2-GetUp --agent.max-iterations 10000 \
      > /tmp/getup_v10c_train.log 2>&1 &
"""

from __future__ import annotations
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V7 = HERE / "patch_get_up_v7.py"
ACTIVE = Path("/workspace/unitree_rl_mjlab/src/tasks/get_up")

V10C_REWARD_BLOCK = '''

# === v10c (hind-feet-on-ground reward) ==============================
def hind_feet_on_ground(env, sensor_name: str = "feet_ground_contact"):
    """+1.0 when BOTH rear feet (RL=idx 2, RR=idx 3) are in contact.

    Foot index convention: 0=FL, 1=FR, 2=RL, 3=RR (matches FR-Net order
    used by v7's contact_predictor.py).
    """
    sensor = env.scene.sensors[sensor_name]
    data = sensor.data
    for attr in ("in_contact", "is_in_contact", "current_contact_state"):
        v = getattr(data, attr, None)
        if v is not None:
            c = v.float()
            break
    else:
        forces = getattr(data, "net_forces_w_history", None)
        if forces is None:
            return torch.zeros(env.num_envs, device=env.device)
        mag = forces[..., 0, :].norm(dim=-1)
        c = (mag > 1.0).float()
    # both rear feet down
    return c[:, 2] * c[:, 3]
'''


def main() -> int:
    # 1. Apply v7 first.
    rc = subprocess.run(["python3", str(V7)])
    if rc.returncode != 0:
        print(f"ERROR: v7 patcher failed (rc={rc.returncode})", file=sys.stderr)
        return rc.returncode

    rewards_path = ACTIVE / "mdp" / "rewards.py"
    cfg_path = ACTIVE / "get_up_env_cfg.py"

    # 2. Append reward function.
    rewards_path.write_text(rewards_path.read_text() + V10C_REWARD_BLOCK)
    print("[edit] mdp/rewards.py: hind_feet_on_ground")

    # 3. Register in env_cfg. Insert before joint_velocity_penalty (same anchor
    #    v7 uses).
    txt = cfg_path.read_text()
    anchor = '"joint_velocity_penalty": RewardTermCfg('
    insertion = (
        '"hind_feet_on_ground": RewardTermCfg(\n'
        '            func=get_up_mdp.hind_feet_on_ground,\n'
        '            weight=2.0,\n'
        '            params={"sensor_name": "feet_ground_contact"},\n'
        '        ),\n'
        '        '
    )
    if anchor not in txt:
        print(f"ERROR: anchor not found: {anchor!r}", file=sys.stderr)
        return 3
    txt = txt.replace(anchor, insertion + anchor, 1)
    cfg_path.write_text(txt)
    print("[edit] get_up_env_cfg.py: registered hind_feet_on_ground")

    # 4. Syntax check.
    rc = subprocess.run(
        ["python3", "-c",
         "import ast;"
         f"ast.parse(open('{cfg_path}').read());"
         f"ast.parse(open('{rewards_path}').read())"],
        capture_output=True, text=True,
    )
    if rc.returncode != 0:
        print(f"ERROR: syntax check failed:\n{rc.stderr}", file=sys.stderr)
        return 4

    print("[ok] v10c patch applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
