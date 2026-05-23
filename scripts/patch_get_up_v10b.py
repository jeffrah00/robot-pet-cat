#!/usr/bin/env python3
"""
get_up v10b -- v7 + initial-pose curriculum.

Hypothesis: starting every episode from a fully-fallen pose is too hard for
the policy to learn from cold. With a curriculum that ramps the *severity*
of the initial fall from mild (~0.2x v7 jitter, narrow tilt window) to full
(v7-baseline) over the first 5000 iters, the policy gets a smoother gradient:
it can first learn the "small correction -> stand" trajectory, then the
shaping pulls the initial pose progressively further from upright.

Mechanism:
  * Wraps the existing reset_to_random_fallen_pose with a curriculum coeff
    alpha in [0.2, 1.0] derived from env.common_step_counter.
    - alpha=0.2 -> 20% jitter magnitudes, 20% tilt span
    - alpha=1.0 -> v7 defaults (full severity)
  * Linear ramp from 0.2 -> 1.0 over the first ~5e6 env-steps (~5000 iters
    at 1024 envs * ~24 steps/iter). After that, full v7 difficulty.

KEEP v7 termination, FR-Net, v4c rewards intact. Only the reset distribution
moves under a curriculum.

Usage on a runpod:
  python3 scripts/patch_get_up_v10b.py
  cd /workspace/unitree_rl_mjlab && source /workspace/mjlab_venv/bin/activate
  set -a && source /workspace/robot-pet-cat/.env && set +a
  nohup python3 scripts/train.py Unitree-Go2-GetUp --agent.max-iterations 10000 \
      > /tmp/getup_v10b_train.log 2>&1 &
"""

from __future__ import annotations
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V7 = HERE / "patch_get_up_v7.py"
ACTIVE = Path("/workspace/unitree_rl_mjlab/src/tasks/get_up")

# Curriculum block appended to mdp/events.py. Replaces v7's hardcoded jitter
# magnitudes with alpha-scaled versions, where alpha ramps with training.

V10B_CURRICULUM_BLOCK = '''

# === v10b (initial-pose curriculum) =================================
def _curriculum_alpha_v10b(env) -> float:
    """Ramp 0.2 -> 1.0 over first 5e6 env-steps."""
    step = getattr(env, "common_step_counter", None)
    if step is None:
        return 1.0
    try:
        s = int(step.item()) if hasattr(step, "item") else int(step)
    except Exception:
        return 1.0
    RAMP_END = 5_000_000  # ~5000 iters at typical batch sizes
    if s >= RAMP_END:
        return 1.0
    frac = max(0.0, min(1.0, s / RAMP_END))
    return 0.2 + 0.8 * frac
'''


def main() -> int:
    # 1. Apply v7 first.
    rc = subprocess.run(["python3", str(V7)])
    if rc.returncode != 0:
        print(f"ERROR: v7 patcher failed (rc={rc.returncode})", file=sys.stderr)
        return rc.returncode

    events_path = ACTIVE / "mdp" / "events.py"
    if not events_path.exists():
        print(f"ERROR: {events_path} missing after v7 patch", file=sys.stderr)
        return 2

    txt = events_path.read_text()

    # 2. Append the curriculum helper.
    if "_curriculum_alpha_v10b" not in txt:
        txt += V10B_CURRICULUM_BLOCK

    # 3. Edit v7's jitter to be alpha-scaled. Anchor matches the substring
    #    AFTER the 4-space line prefix; the leading "    " stays in the
    #    file. So new_jp's first line must NOT have leading whitespace --
    #    the file's existing 4 spaces become the indent. Subsequent lines
    #    need their own explicit 4 spaces.
    old_jp = "joint_pos = joint_pos + (torch.rand_like(joint_pos) - 0.5) * 0.6"
    new_jp = (
        "_alpha_v10b = _curriculum_alpha_v10b(env)\n"
        "    joint_pos = joint_pos + (torch.rand_like(joint_pos) - 0.5) * (0.6 * _alpha_v10b)"
    )
    if old_jp not in txt:
        print(f"ERROR: anchor not found: {old_jp!r}", file=sys.stderr)
        return 3
    txt = txt.replace(old_jp, new_jp, 1)

    old_rpy = "rpy_noise = (torch.rand((num_resets, 3), device=device) - 0.5) * 0.4"
    new_rpy = "rpy_noise = (torch.rand((num_resets, 3), device=device) - 0.5) * (0.4 * _alpha_v10b)"
    if old_rpy not in txt:
        print(f"ERROR: anchor not found: {old_rpy!r}", file=sys.stderr)
        return 4
    txt = txt.replace(old_rpy, new_rpy, 1)

    events_path.write_text(txt)
    print("[edit] mdp/events.py: alpha-scaled jitter + curriculum helper")

    # 4. Syntax check.
    rc = subprocess.run(
        ["python3", "-c",
         "import ast; ast.parse(open('" + str(events_path) + "').read())"],
        capture_output=True, text=True,
    )
    if rc.returncode != 0:
        print("ERROR: syntax check failed:\n" + rc.stderr, file=sys.stderr)
        return 5

    print("[ok] v10b patch applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
