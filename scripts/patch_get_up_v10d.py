#!/usr/bin/env python3
"""
get_up v10d -- v7 + asymmetric actor-critic.

Hypothesis: PPO's value head has trouble estimating returns in fallen states
because proprioception alone doesn't disambiguate "on side", "on back",
"belly-flat", "near-upright". A poor V(s) gives noisy advantages, which
degrades policy gradients on the hardest states (exactly the ones we care
about). Giving the CRITIC privileged info (root pose + contact state) while
keeping the ACTOR at v7's proprioception+FR-Net obs should yield cleaner
advantages without changing the deployed policy's input space.

Mechanism:
  * Add a new critic-only obs term `priv_root_state` that returns
    (root_pos_w[:, 2], root_quat_w, root_lin_vel_w, root_ang_vel_w) = 11-dim
    (height + quat + linvel + angvel) appended to the critic's obs vector.
  * Add a new critic-only obs term `priv_contact_state` that returns the
    4-dim foot-contact state from the feet_ground_contact sensor.
  * Insert BOTH terms into critic_terms only, leaving actor_terms untouched.

KEEP v7 termination, FR-Net, v4c rewards, target_height 0.30, jitter. Only
adds critic-side observations.

Usage on a runpod:
  python3 scripts/patch_get_up_v10d.py
  cd /workspace/unitree_rl_mjlab && source /workspace/mjlab_venv/bin/activate
  set -a && source /workspace/robot-pet-cat/.env && set +a
  nohup python3 scripts/train.py Unitree-Go2-GetUp --agent.max-iterations 10000 \
      > /tmp/getup_v10d_train.log 2>&1 &
"""

from __future__ import annotations
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V7 = HERE / "patch_get_up_v7.py"
ACTIVE = Path("/workspace/unitree_rl_mjlab/src/tasks/get_up")

V10D_OBS_BLOCK = '''

# === v10d (asymmetric critic-only privileged obs) ===================
def priv_root_state(env, asset_cfg=_DEFAULT_ASSET_CFG):
    """Critic-only: base height + quat + linvel + angvel = 11-dim.

    Privileged because deployed policy can't read root pose directly; only
    the critic uses this to estimate V(s) more accurately in fallen states.
    """
    asset = env.scene[asset_cfg.name]
    z = asset.data.root_link_pos_w[:, 2:3]
    quat = asset.data.root_link_quat_w
    lin = asset.data.root_lin_vel_w
    ang = asset.data.root_ang_vel_w
    return torch.cat([z, quat, lin, ang], dim=-1)


def priv_contact_state(env, sensor_name: str = "feet_ground_contact"):
    """Critic-only: 4-dim foot-contact 0/1 state."""
    sensor = env.scene.sensors[sensor_name]
    data = sensor.data
    for attr in ("in_contact", "is_in_contact", "current_contact_state"):
        v = getattr(data, attr, None)
        if v is not None:
            return v.float()
    forces = getattr(data, "net_forces_w_history", None)
    if forces is None:
        return torch.zeros(env.num_envs, 4, device=env.device)
    mag = forces[..., 0, :].norm(dim=-1)
    return (mag > 1.0).float()
'''


def main() -> int:
    # 1. Apply v7 first.
    rc = subprocess.run(["python3", str(V7)])
    if rc.returncode != 0:
        print(f"ERROR: v7 patcher failed (rc={rc.returncode})", file=sys.stderr)
        return rc.returncode

    obs_path = ACTIVE / "mdp" / "observations.py"
    cfg_path = ACTIVE / "get_up_env_cfg.py"

    # 2. Append both critic obs functions.
    obs_path.write_text(obs_path.read_text() + V10D_OBS_BLOCK)
    print("[edit] mdp/observations.py: priv_root_state, priv_contact_state")

    # 3. Register BOTH in critic_terms only. v7 leaves a `  critic_terms = {`
    #    line. Find the FIRST term inside critic_terms and prepend ours.
    #    The structure is roughly:
    #      critic_terms = {
    #        "policy_action": ObservationTermCfg(...),
    #        ...
    #      }
    #    Insert two new terms at the top of the dict.
    txt = cfg_path.read_text()
    anchor = "  critic_terms = {\n"
    if anchor not in txt:
        print(f"ERROR: anchor not found: {anchor!r}", file=sys.stderr)
        return 3
    insertion = (
        '  critic_terms = {\n'
        '    "priv_root_state": ObservationTermCfg(\n'
        '      func=get_up_mdp.priv_root_state,\n'
        '    ),\n'
        '    "priv_contact_state": ObservationTermCfg(\n'
        '      func=get_up_mdp.priv_contact_state,\n'
        '      params={"sensor_name": "feet_ground_contact"},\n'
        '    ),\n'
    )
    # Only replace the first occurrence.
    txt = txt.replace(anchor, insertion, 1)
    cfg_path.write_text(txt)
    print("[edit] get_up_env_cfg.py: prepended priv_root_state + priv_contact_state to critic_terms")

    # 4. Re-export through mdp/__init__.py if needed.
    init_path = ACTIVE / "mdp" / "__init__.py"
    if init_path.exists():
        init_txt = init_path.read_text()
        if "priv_root_state" not in init_txt:
            init_txt += "\nfrom .observations import priv_root_state, priv_contact_state  # v10d\n"
            init_path.write_text(init_txt)
            print("[edit] mdp/__init__.py: re-export v10d symbols")

    # 5. Syntax check.
    rc = subprocess.run(
        ["python3", "-c",
         "import ast;"
         f"ast.parse(open('{cfg_path}').read());"
         f"ast.parse(open('{obs_path}').read())"],
        capture_output=True, text=True,
    )
    if rc.returncode != 0:
        print(f"ERROR: syntax check failed:\n{rc.stderr}", file=sys.stderr)
        return 4

    print("[ok] v10d patch applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
