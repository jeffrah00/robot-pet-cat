#!/usr/bin/env bash
# v10 retry chain: re-runs v10b (curriculum) + v10d (asymmetric critic) after
# their patchers were fixed. v10a + v10c are already done and pushed.
#
# Per-variant: patch -> train -> render -> push.
# Bug fix vs v10_master_chain.sh: rename local PPID=$! to PLAY_PID -- $PPID
# is a reserved bash readonly variable, the original assignment errored
# silently and left orphan play.py processes.

set -u
exec > /tmp/v10_retry.log 2>&1
echo "[retry] $(date -u) START"

VENV=/workspace/mjlab_venv/bin/activate
LOGDIR=/workspace/unitree_rl_mjlab/logs/rsl_rl/go2_velocity
SCRIPTS=/workspace/robot-pet-cat/scripts
RPC=/workspace/robot-pet-cat

cd /workspace/unitree_rl_mjlab && source "$VENV"
set -a && source "$RPC/.env" && set +a

run_variant() {
  local LABEL=$1
  local PATCHER=$2
  local ITERS=$3

  echo
  echo "[retry] $(date -u) === $LABEL (patcher=$PATCHER, iters=$ITERS) ==="
  python3 "$SCRIPTS/$PATCHER" || { echo "[retry] $LABEL patcher FAILED, skipping"; return; }

  local PRE_RUNS=$(ls -d $LOGDIR/*/ 2>/dev/null | sort)
  echo "[retry] $(date -u) starting train.py ($ITERS iters)"
  python3 scripts/train.py Unitree-Go2-GetUp --agent.max-iterations $ITERS \
    > /tmp/getup_${LABEL}_train.log 2>&1
  local RC=$?
  echo "[retry] $(date -u) train.py exit=$RC"

  local POST_RUNS=$(ls -d $LOGDIR/*/ 2>/dev/null | sort)
  local NEW_RUN=$(comm -13 <(echo "$PRE_RUNS") <(echo "$POST_RUNS") | head -1)
  if [ -z "$NEW_RUN" ]; then
    echo "[retry] $LABEL no new run dir produced, skipping render"
    return
  fi
  echo "[retry] $LABEL run dir: $NEW_RUN"

  local CKPT=$(ls -1v ${NEW_RUN}model_*.pt 2>/dev/null | tail -1)
  if [ -z "$CKPT" ]; then
    echo "[retry] $LABEL no checkpoints, skipping render"
    return
  fi
  echo "[retry] $LABEL ckpt: $CKPT"

  echo "[retry] $(date -u) rendering with osmesa (1500 frames = 30s)"
  PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa python3 scripts/play.py Unitree-Go2-GetUp \
    --checkpoint-file "$CKPT" --video True --video-length 1500 --num-envs 1 \
    > /tmp/getup_${LABEL}_play.log 2>&1 &
  local PLAY_PID=$!
  echo "[retry] $LABEL play.py pid: $PLAY_PID"

  local VID="${NEW_RUN}videos/play/rl-video-step-0.mp4"
  for i in $(seq 1 90); do
    sleep 10
    [ -f "$VID" ] && break
  done
  if [ ! -f "$VID" ]; then
    echo "[retry] $LABEL mp4 never appeared, killing play.py ($PLAY_PID)"
    kill -9 $PLAY_PID 2>/dev/null || true
    return
  fi
  sleep 120
  local S1=$(stat -c%s "$VID" 2>/dev/null || echo 0)
  sleep 20
  local S2=$(stat -c%s "$VID" 2>/dev/null || echo 0)
  echo "[retry] $LABEL mp4 sizes $S1 -> $S2"
  # Always kill play.py once the size has stabilized (or after the wait).
  kill -9 $PLAY_PID 2>/dev/null || true
  sleep 5

  mkdir -p $RPC/renders
  cp "$VID" "$RPC/renders/get_up_${LABEL}.mp4"
  local FINAL_SZ=$(stat -c%s "$RPC/renders/get_up_${LABEL}.mp4" 2>/dev/null || echo 0)
  echo "[retry] $LABEL copied to renders/get_up_${LABEL}.mp4 ($FINAL_SZ bytes)"

  cd $RPC && python3 scripts/sandbox_push.py "renders/get_up_${LABEL}.mp4" \
    -m "get_up ${LABEL} final render" || echo "[retry] $LABEL push FAILED"
  cd /workspace/unitree_rl_mjlab

  echo "[retry] $(date -u) $LABEL DONE"
}

run_variant v10b patch_get_up_v10b.py 10000
run_variant v10d patch_get_up_v10d.py 10000

echo
echo "[retry] $(date -u) ALL v10 RETRY DONE"
