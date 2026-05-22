# Headless Rendering on RunPod

This is the permanent reference for rendering on headless GPU pods.
Re-written 2026-05-22 after the third time we had to rediscover this.

---

## The only method that works headlessly: `render_brain_3d.py`

`render_brain_3d.py` uses raw MuJoCo (not mjlab), so it can use the OSMesa
software renderer without a display.  Use this for all renders.

### 1. One-time setup (per fresh pod)

```bash
apt-get update -qq && apt-get install -y libosmesa6
```

`libosmesa6` is NOT pre-installed.  `apt-get update` is required first --
the cached package list on fresh pods is stale and the install will fail
with "Unable to locate package" without it.

### 2. Full render command

```bash
cd /workspace/robot-pet-cat
source /workspace/mjlab_venv/bin/activate

MUJOCO_GL=osmesa python scripts/render_brain_3d.py \
  --policy /workspace/unitree_rl_mjlab/logs/rsl_rl/go2_velocity/2026-05-18_22-14-29/policy.onnx \
  --steps 600 \
  --out renders/get_up_v5_eval.mp4 \
  --scripted
```

The process exits cleanly when done.  600 steps takes about 10 minutes at
osmesa CPU speed.

### 3. Argument reference

| Flag | Required | Notes |
|---|---|---|
| `--policy PATH` | YES | Go2 **walker** (locomotion) policy. `.onnx`, `.pt`, or run dir. |
| `--steps N` | no | Default 600 (30s at 20fps). |
| `--out PATH` | no | Default `renders/brain_3d.mp4`. Relative to CWD. |
| `--scripted` | no | Uses hardcoded skill sequence instead of brain policy. |
| `--checkpoint PATH` | no | SB3 PPO zip for the brain (used when not --scripted). |
| `--no-mode-policy` | no | Disables the ModePolicy entirely. |

### 4. Walker policy path

The `--policy` arg is the **Go2 velocity/locomotion walker**, not a skill policy.
Skill policies (get_up, hind_sit) are loaded by the script automatically from
`models/` relative to CWD.

Walker policy lives at:
```
/workspace/unitree_rl_mjlab/logs/rsl_rl/go2_velocity/2026-05-18_22-14-29/policy.onnx
```

All training runs (walker and skill) are stored under `go2_velocity/` regardless
of task name (VelocityOnPolicyRunner always uses this subdir).  To identify the
walker: it is the oldest run from 2026-05-18.  Skill training happened from
2026-05-21 onward.

### 5. Gotchas

- **CWD must be `/workspace/robot-pet-cat`** -- model paths like
  `models/get_up_v5.pt` are resolved relative to CWD.
- **`MUJOCO_GL=osmesa` must be set** or MuJoCo will try to use EGL/GLX and fail.
- **EGL fails** (`'NoneType' has no attribute 'eglQueryString'`) -- the pod GPU
  driver does not expose EGL via PyOpenGL.
- **The renders/ directory is created automatically** by the script, but
  `mkdir -p renders` is safe to run first.

---

## play.py -- DO NOT USE for headless rendering

`play.py` imports `mjlab.envs.ManagerBasedRlEnv` at the top of the file, which
chains through MuJoCo -> PyOpenGL at import time.  This means:

- `MUJOCO_GL=osmesa` **does not work** for play.py because PyOpenGL's OSMesa
  loader fails in this pod environment.
- Xvfb **does not work** because Xvfb is not installed on RunPod pods
  (`apt-get install -y xvfb` gives "Unable to locate package").
- Even with a real display, play.py **never exits** after writing the video
  (~60s in at `videos/play/rl-video-step-0.mp4`).  Must be manually killed.

### play.py isolation command (if you have a real display)

```bash
# From /workspace/unitree_rl_mjlab (NOT robot-pet-cat)
cd /workspace/unitree_rl_mjlab
source /workspace/mjlab_venv/bin/activate

DISPLAY=:99 python3 scripts/play.py Unitree-Go2-GetUp-v5 \
  --checkpoint_file /workspace/robot-pet-cat/models/get_up_v5.pt \
  --video True \
  --num_envs 1 \
  --video_length 400 \
  --viewer native
```

Gotcha: CWD must be `unitree_rl_mjlab`, not `robot-pet-cat`.  Running from the
wrong directory gives `No such file or directory: scripts/play.py`.

---

## Transferring renders to GitHub

The pod sandbox cannot push directly.  Use the GitHub Contents API:

```bash
source /workspace/robot-pet-cat/.env  # provides GITHUB_TOKEN
cd /workspace/robot-pet-cat
python3 scripts/sandbox_push.py
```

Or for a single binary file:
```bash
python3 - << 'EOF'
import base64, json, os, urllib.request
token = os.environ["GITHUB_TOKEN"]
with open("renders/get_up_v5_eval.mp4","rb") as f:
    content = base64.b64encode(f.read()).decode()
data = json.dumps({"message":"render: get_up_v5_eval","content":content}).encode()
req = urllib.request.Request(
    "https://api.github.com/repos/jeffrah00/robot-pet-cat/contents/renders/get_up_v5_eval.mp4",
    data=data, headers={"Authorization":f"token {token}","Content-Type":"application/json"})
urllib.request.urlopen(req)
print("pushed")
EOF
```
