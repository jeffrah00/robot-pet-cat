#!/usr/bin/env python3
"""v8 retry env mod: remove fell_over and illegal_contact terminations, lower time_penalty,
let the cat keep trying after a fall (no soft reset, just don't terminate).
Idempotent: safe to re-run.
"""
import pathlib, re, shutil, datetime
ROOT = pathlib.Path('/workspace/unitree_rl_mjlab/src/tasks/get_up')
BACKUP = pathlib.Path('/tmp/v8_pre_backup')
FILES = [
    ROOT/'get_up_env_cfg.py',
    ROOT/'config/go2/env_cfgs.py',
]
BACKUP.mkdir(exist_ok=True)
for f in FILES:
    bk = BACKUP / f.name
    if not bk.exists():
        shutil.copy(f, bk)
        print(f'backed up {f.name} -> {bk}')
# Edit 1: comment out fell_over termination block in get_up_env_cfg.py
p1 = ROOT/'get_up_env_cfg.py'
txt = p1.read_text()
new = re.sub(
    r'(\s*)"fell_over": TerminationTermCfg\(\s*\n\s*func=mdp\.bad_orientation,\s*\n\s*params=\{[^}]+\},\s*\n\s*\),',
    r'\1# [v8_retry] fell_over termination DISABLED -- cat keeps trying after a fall',
    txt,
)
if new == txt:
    print('WARN: fell_over pattern not matched in', p1)
else:
    p1.write_text(new); print('removed fell_over termination')
# Edit 1b: lower time_penalty weight (don't punish episode time so much)
txt = p1.read_text()
new = re.sub(r'("time_penalty": RewardTermCfg\([^)]*weight=)-0\.05', r'\g<1>-0.005', txt, count=1, flags=re.S)
if new != txt:
    p1.write_text(new); print('lowered time_penalty -0.05 -> -0.005')
else:
    print('WARN: time_penalty weight pattern not matched (already changed?)')
# Edit 2: comment out illegal_contact termination in go2/env_cfgs.py
p2 = ROOT/'config/go2/env_cfgs.py'
txt = p2.read_text()
new = re.sub(
    r'(\s*)cfg\.terminations\["illegal_contact"\] = TerminationTermCfg\(\s*\n\s*func=mdp\.illegal_contact,\s*\n\s*params=\{[^}]+\},\s*\n\s*\)',
    r'\1# [v8_retry] illegal_contact termination DISABLED -- body-ground contact OK during fall',
    txt,
)
if new == txt:
    print('WARN: illegal_contact pattern not matched in', p2)
else:
    p2.write_text(new); print('removed illegal_contact termination')
print('v8 mod complete')
