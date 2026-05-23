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
  * Add critic-only obs `priv_root_state`: base height + quat + body-frame
    linear and angular velocity = 11-dim.
  * Add critic-only obs `priv_contact_state`: 4-dim foot-contact state.

KEEP v7 termination, FR-Net, v4c rewards, target_height 0.30, jitter. Only
adds critic-side observations.

v10d v2 fix: avoid `root_lin_vel_w` / `root_ang_vel_w` (don't exist on
mjlab's EntityData). Use root_lin_vel_b + root_ang_vel_b (body frame),
which v7's contact_predictor.py also touches.

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
    """Critic-only: base height + quat + linvel(body) + angvel(body) = 11-dim.

    Privileged because the deployed policy can't read root pose / world-frame
    velocity directly. Use the attribute names that v7's contact_predictor.py
    already touches (root_link_pos_w, root_lin_vel_b, root_ang_vel_b) and
    avoid root_lin_vel_w which mjlab's EntityData lacks.
    """
    asset = env.scene[asset_cfg.name]
    z = asset.data.root_link_pos_w[:, 2:3]
    quat = getattr(asset.data, "root_link_quat_w", None)
    if quat is None:
        quat = asset.data.root_quat_w
    lin = asset.data.root_lin_vel_b
    ang = asset.data.root_ang_vel_b
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
    rc = subprocess.run(["python3", str(V7)])
    if rc.returncode != 0:
        print("ERROR: v7 patcher failed rc=" + str(rc.returncode), file=sys.stderr)
        return rc.returncode

    obs_path = ACTIVE / "mdp" / "observations.py"
    cfg_path = ACTIVE / "get_up_env_cfg.py"

    obs_path.write_text(obs_path.read_text() + V10D_OBS_BLOCK)
    print("[edit] mdp/observations.py: priv_root_state, priv_contact_state")

    txt = cfg_path.read_text()
    anchor = "  critic_terms = {\n"
    if anchor not in txt:
        print("ERROR: anchor not found: " + repr(anchor), file=sys.stderr)
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
    txt = txt.replace(anchor, insertion, 1)
    cfg_path.write_text(txt)
    print("[edit] get_up_env_cfg.py: prepended priv_root_state + priv_contact_state to critic_terms")

    init_path = ACTIVE / "mdp" / "__init__.py"
    if init_path.exists():
        init_txt = init_path.read_text()
        if "priv_root_state" not in init_txt:
            init_txt += "\nfrom .observations import priv_root_state, priv_contact_state  # v10d\n"
            init_path.write_text(init_txt)
            print("[edit] mdp/__init__.py: re-export v10d symbols")

    rc = subprocess.run(
        ["python3", "-c",
         "import ast;"
         "ast.parse(open('" + str(cfg_path) + "').read());"
         "ast.parse(open('" + str(obs_path) + "').read())"],
        capture_output=True, text=True,
    )
    if rc.returncode != 0:
        print("ERROR: syntax check failed:\n" + rc.stderr, file=sys.stderr)
        return 4

    print("[ok] v10d patch applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
