#!/usr/bin/env python3
"""
get_up v7.3 -- v7 FR-Net + deeper aux head + critic also sees prediction.

Hypothesis: v7's tiny 30->64->64->4 contact predictor may saturate quickly
and stop providing useful signal once it's roughly calibrated. A wider/deeper
head (128 hidden, 3 layers) has more capacity to model contact dynamics over
the full fallen->stand trajectory, and exposing the prediction to the critic
too (currently only actor sees it) should make the value function aware of
the same auxiliary signal the policy uses.

Changes vs v7:
  * MassContactPredictor: hidden 64 -> 128, depth 2 -> 3 layers
  * predicted_next_contact obs term added to critic_terms (in addition to
    actor_terms)
  * All v7 rewards / pose jitter / termination kept identical
"""

from __future__ import annotations
import shutil
import subprocess
import sys
from pathlib import Path

V3 = Path("/workspace/_v3_baseline/get_up")
ACTIVE = Path("/workspace/unitree_rl_mjlab/src/tasks/get_up")
V7 = Path("/workspace/robot-pet-cat/scripts/patch_get_up_v7.py")


def main() -> int:
    rc = subprocess.run(["python3", str(V7)], capture_output=True, text=True)
    if rc.returncode != 0:
        print("ERROR: v7 patcher failed:\n" + rc.stderr, file=sys.stderr)
        return rc.returncode
    print(rc.stdout)

    # 1. Replace the contact_predictor with a deeper / wider net.
    pred_path = ACTIVE / "mdp" / "contact_predictor.py"
    txt = pred_path.read_text()
    old_net = (
        "    def __init__(self, hidden: int = 64, lr: float = 1e-3, device: str = \"cpu\"):\n"
        "        super().__init__()\n"
        "        self.net = nn.Sequential(\n"
        "            nn.Linear(self.INPUT_DIM, hidden),\n"
        "            nn.ELU(),\n"
        "            nn.Linear(hidden, hidden),\n"
        "            nn.ELU(),\n"
        "            nn.Linear(hidden, self.OUTPUT_DIM),\n"
        "            nn.Sigmoid(),\n"
        "        ).to(device)"
    )
    new_net = (
        "    def __init__(self, hidden: int = 128, lr: float = 1e-3, device: str = \"cpu\"):\n"
        "        super().__init__()\n"
        "        self.net = nn.Sequential(\n"
        "            nn.Linear(self.INPUT_DIM, hidden),\n"
        "            nn.ELU(),\n"
        "            nn.Linear(hidden, hidden),\n"
        "            nn.ELU(),\n"
        "            nn.Linear(hidden, hidden),\n"
        "            nn.ELU(),\n"
        "            nn.Linear(hidden, self.OUTPUT_DIM),\n"
        "            nn.Sigmoid(),\n"
        "        ).to(device)"
    )
    if old_net not in txt:
        print("ERROR: contact_predictor net anchor not found", file=sys.stderr)
        return 4
    pred_path.write_text(txt.replace(old_net, new_net))
    print("[edit] contact_predictor.py: 64x2 -> 128x3")

    # 2. Also register predicted_next_contact as a critic obs term.
    cfg_path = ACTIVE / "get_up_env_cfg.py"
    cfg_txt = cfg_path.read_text()
    # The v7 patcher injects predicted_next_contact at the end of actor_terms.
    # Add the same term at the end of critic_terms. We search for the
    # canonical close brace of critic_terms.
    # The v3 baseline's critic_terms ends with `  }\n\n  events = {` (or
    # similar). To stay anchor-stable across baseline versions we look for
    # the LAST `}` before `events =` and inject before it.
    import re
    m = re.search(r"(critic_terms = \{.*?\n)  \}\n\n  events = \{", cfg_txt, re.S)
    if not m:
        # Fallback: try without "events =" downstream.
        m = re.search(r"(critic_terms = \{.*?\n)  \}\n\n", cfg_txt, re.S)
    if not m:
        print("ERROR: critic_terms close anchor not found", file=sys.stderr)
        return 5
    inject = (
        '    "predicted_next_contact": ObservationTermCfg(\n'
        '      func=get_up_mdp.predicted_next_contact,\n'
        '      params={"sensor_name": "feet_ground_contact"},\n'
        '    ),\n'
        '  }\n'
    )
    # Replace the closing brace of critic_terms with the inject + brace.
    span_start = m.end(1)
    span_end = span_start + len("  }\n")
    cfg_txt = cfg_txt[:span_start] + inject + cfg_txt[span_end:]
    cfg_path.write_text(cfg_txt)
    print("[edit] get_up_env_cfg.py: predicted_next_contact added to critic_terms")

    # 3. Syntax check.
    rc = subprocess.run(
        ["python3", "-c",
         f"import ast;"
         f"ast.parse(open('{cfg_path}').read());"
         f"ast.parse(open('{pred_path}').read())"],
        capture_output=True, text=True,
    )
    if rc.returncode != 0:
        print(f"ERROR: syntax check failed:\n{rc.stderr}", file=sys.stderr)
        return 6

    print("[ok] v7.3 patch applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
