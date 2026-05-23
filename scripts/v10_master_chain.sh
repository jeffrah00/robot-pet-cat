#!/usr/bin/env bash
# v10 master chain: runs v10a/b/c/d sequentially on a single pod.
#
# Each variant:
#   1. Apply patcher (resets ACTIVE from V3 baseline via the v7 sub-patcher,
#      then layers the variant-specific edits)
#   2. Train Unitree-Go2-GetUp for the variant's iter count
#   3. Render mp4 via osmesa (1500 frames = 30 s @ 50fps)
#   4. Copy to /workspace/robot-pet-cat/renders/get_up_<label>.mp4
#   5. Push via sandbox_push.py
#
# Re-runnable: each step is independent and resilient to missing checkpoints.

set -u
exec > /tmp/v10_master.log 2>&1
echo "[master] $(date -u) START"

VENV=/workspace/mjlab_venv/bin/activate
LOGDIR=/workspace/unitree_rl_mjlab/logs/rsl_rl/go2_velocity
SCRIPTS=/workspace/robot-pet-cat/scripts
RPC=/workspace/robot-pet-cat
TODAY=$(date -u +%Y-%m-%d)

cd /workspace/unitree_rl_mjlab && source "$VENV"
set -a && source "$RPC/.env" && set +a

run_variant() {
  local LABEL=$1
  local PATCHER=$2
  local ITERS=$3
  local FINAL=$((ITERS - 1))

  echo
  echo "[master] $(date -u) === $LABEL (patcher=$PATCHER, iters=$ITERS) ==="

  echo "[master] applying $PATCHER"
  python3 "$SCRIPTS/$PATCHER" || { echo "[master] $LABEL patcher FAILED, skipping"; return; }

  # Snapshot existing run dirs (any date) so we can identify the NEW one.
  local PRE_RUNS=$(ls -d $LOGDIR/*/ 2>/dev/null | sort)

  echo "[master] $(date -u) starting train.py ($ITERS iters)"
  python3 scripts/train.py Unitree-Go2-GetUp \
    --agent.max-iterations $ITERS \
    > /tmp/getup_${LABEL}_train.log 2>&1
  local RC=$?
  echo "[master] $(date -u) train.py exit=$RC"

  local POST_RUNS=$(ls -d $LOGDIR/*/ 2>/dev/null | sort)
  local NEW_RUN=$(comm -13 <(echo "$PRE_RUNS") <(echo "$POST_RUNS") | head -1)
  if [ -z "$NEW_RUN" ]; then
    echo "[master] $LABEL no new run dir produced, skipping render"
    return
  fi
  echo "[master] $LABEL run dir: $NEW_RUN"

  # Use last checkpoint (handles early-exit and non-multiple-of-50 final iter)
  local CKPT=$(ls -1v ${NEW_RUN}model_*.pt 2>/dev/null | tail -1)
  if [ -z "$CKPT" ]; then
    echo "[master] $LABEL no checkpoints, skipping render"
    return
  fi
  echo "[master] $LABEL ckpt: $CKPT"

  echo "[master] $(date -u) rendering with osmesa (1500 frames = 30s)"
  PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa python3 scripts/play.py Unitree-Go2-GetUp \
    --checkpoint-file "$CKPT" --video True --video-length 1500 --num-envs 1 \
    > /tmp/getup_${LABEL}_play.log 2>&1 &
  local PPID=$!

  local VID="${NEW_RUN}videos/play/rl-video-step-0.mp4"
  for i in $(seq 1 90); do
    sleep 10
    [ -f "$VID" ] && break
  done
  if [ ! -f "$VID" ]; then
    echo "[master] $LABEL mp4 never appeared, killing play.py"
    kill $PPID 2>/dev/null || true
    return
  fi
  # Wait for size to stabilize, then kill play.py (it never exits cleanly headless).
  sleep 120
  local S1=$(stat -c%s "$VID" 2>/dev/null || echo 0)
  sleep 20
  local S2=$(stat -c%s "$VID" 2>/dev/null || echo 0)
  echo "[master] $LABEL mp4 sizes $S1 -> $S2"
  if [ "$S1" = "$S2" ] && [ "$S1" != "0" ]; then
    kill $PPID 2>/dev/null || true
  fi
  sleep 5

  mkdir -p $RPC/renders
  cp "$VID" "$RPC/renders/get_up_${LABEL}.mp4"
  local FINAL_SZ=$(stat -c%s "$RPC/renders/get_up_${LABEL}.mp4" 2>/dev/null || echo 0)
  echo "[master] $LABEL copied to renders/get_up_${LABEL}.mp4 ($FINAL_SZ bytes)"

  cd $RPC && python3 scripts/sandbox_push.py "renders/get_up_${LABEL}.mp4" \
    -m "get_up ${LABEL} final render" || echo "[master] $LABEL push FAILED"
  cd /workspace/unitree_rl_mjlab

  echo "[master] $(date -u) $LABEL DONE"
}

# v10a -- v7 trained longer (15k iters)
run_variant v10a patch_get_up_v10a.py 15000

# v10b -- v7 + initial-pose curriculum (10k iters; 5k for ramp, 5k at full severity)
run_variant v10b patch_get_up_v10b.py 10000

# v10c -- v7 + hind-feet-on-ground reward (10k iters)
run_variant v10c patch_get_up_v10c.py 10000

# v10d -- v7 + asymmetric critic (10k iters)
run_variant v10d patch_get_up_v10d.py 10000

echo
echo "[master] $(date -u) ALL v10 DONE"
