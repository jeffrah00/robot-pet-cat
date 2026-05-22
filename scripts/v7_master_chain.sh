#!/usr/bin/env bash
# Master pipeline: runs v7.1, v7.2, v7.3 sequentially. Each variant:
#   1. Apply patcher (resets ACTIVE from V3 baseline, then re-mods)
#   2. Train Unitree-Go2-GetUp for N iters
#   3. Render mp4 via osmesa, kill play.py when mp4 size stable
#   4. Copy to /workspace/robot-pet-cat/renders/get_up_<label>.mp4
#   5. Push via sandbox_push.py
#
# Re-runnable: each step is independent of prior state and resilient to
# missing checkpoints (skips with a logged error rather than crashing the
# whole pipeline).

set -u
exec > /tmp/v7_master.log 2>&1
echo "[master] $(date -u) START"

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
  local FINAL=$((ITERS - 1))

  echo
  echo "[master] $(date -u) === $LABEL (patcher=$PATCHER, iters=$ITERS) ==="

  if [ -n "$PATCHER" ]; then
    echo "[master] applying $PATCHER"
    python3 "$SCRIPTS/$PATCHER" || { echo "[master] $LABEL patcher FAILED, skipping"; return; }
  else
    echo "[master] no patcher (reusing current ACTIVE)"
  fi

  # Snapshot existing log dirs so we can identify the NEW one after training.
  local PRE_RUNS=$(ls -d $LOGDIR/2026-05-22_*/ 2>/dev/null | sort)

  echo "[master] $(date -u) starting train.py ($ITERS iters)"
  python3 scripts/train.py Unitree-Go2-GetUp \
    --agent.max-iterations $ITERS \
    > /tmp/getup_${LABEL}_train.log 2>&1
  local RC=$?
  echo "[master] $(date -u) train.py exit=$RC"

  local POST_RUNS=$(ls -d $LOGDIR/2026-05-22_*/ 2>/dev/null | sort)
  local NEW_RUN=$(comm -13 <(echo "$PRE_RUNS") <(echo "$POST_RUNS") | head -1)
  if [ -z "$NEW_RUN" ]; then
    echo "[master] $LABEL no new run dir produced, skipping render"
    return
  fi
  echo "[master] $LABEL run dir: $NEW_RUN"

  local CKPT="${NEW_RUN}model_${FINAL}.pt"
  if [ ! -f "$CKPT" ]; then
    echo "[master] $LABEL checkpoint $CKPT missing, falling back to last available"
    CKPT=$(ls -1v ${NEW_RUN}model_*.pt 2>/dev/null | tail -1)
    if [ -z "$CKPT" ]; then
      echo "[master] $LABEL no checkpoints, skipping render"
      return
    fi
    echo "[master] $LABEL using fallback ckpt: $CKPT"
  fi

  echo "[master] $(date -u) rendering with osmesa"
  PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa python3 scripts/play.py Unitree-Go2-GetUp \
    --checkpoint-file "$CKPT" --video True --video-length 300 --num-envs 1 \
    > /tmp/getup_${LABEL}_play.log 2>&1 &
  local PPID=$!

  local VID="${NEW_RUN}videos/play/rl-video-step-0.mp4"
  for i in $(seq 1 60); do
    sleep 10
    [ -f "$VID" ] && break
  done
  if [ ! -f "$VID" ]; then
    echo "[master] $LABEL mp4 never appeared, killing play.py"
    kill $PPID 2>/dev/null || true
    return
  fi
  sleep 90
  local S1=$(stat -c%s "$VID" 2>/dev/null || echo 0)
  sleep 15
  local S2=$(stat -c%s "$VID" 2>/dev/null || echo 0)
  echo "[master] $LABEL mp4 sizes $S1 -> $S2"
  if [ "$S1" = "$S2" ] && [ "$S1" != "0" ]; then
    kill $PPID 2>/dev/null || true
  fi
  sleep 5

  mkdir -p $RPC/renders
  cp "$VID" "$RPC/renders/get_up_${LABEL}.mp4"
  echo "[master] $LABEL copied to renders/get_up_${LABEL}.mp4 ($(stat -c%s "$RPC/renders/get_up_${LABEL}.mp4") bytes)"

  cd $RPC && python3 scripts/sandbox_push.py "renders/get_up_${LABEL}.mp4" \
    -m "get_up ${LABEL} render" || echo "[master] $LABEL push FAILED"
  cd /workspace/unitree_rl_mjlab

  echo "[master] $(date -u) $LABEL DONE"
}

# v7.1 -- same patcher, longer training (8k iters)
run_variant v7_1 patch_get_up_v7.py 8000

# v7.2 -- v7 + boosted stand_fully_bonus + relaxed fell_over + feet_under_body
run_variant v7_2 patch_get_up_v7_2.py 5000

# v7.3 -- v7 + deeper 128x3 FR-Net + predicted_next_contact in critic
run_variant v7_3 patch_get_up_v7_3.py 5000

echo
echo "[master] $(date -u) ALL DONE"
