#!/usr/bin/env bash
# Standalone chain for v7.4. 8000 iters.
set -u
exec > /tmp/v7_4_chain.log 2>&1
echo "[v7.4] $(date -u) start"

VENV=/workspace/mjlab_venv/bin/activate
LOGDIR=/workspace/unitree_rl_mjlab/logs/rsl_rl/go2_velocity
SCRIPTS=/workspace/robot-pet-cat/scripts
RPC=/workspace/robot-pet-cat

cd /workspace/unitree_rl_mjlab && source "$VENV"
set -a && source "$RPC/.env" && set +a

LABEL=v7_4
ITERS=8000
FINAL=$((ITERS - 1))

echo "[v7.4] applying patcher"
python3 "$SCRIPTS/patch_get_up_v7_4.py" || { echo "[v7.4] patcher FAILED"; exit 1; }

# Capture dirs already present from today and tomorrow (clock may have crossed midnight)
PRE_RUNS=$(ls -d $LOGDIR/2026-05-2[2-3]_*/ 2>/dev/null | sort)

echo "[v7.4] $(date -u) starting train.py ($ITERS iters)"
python3 scripts/train.py Unitree-Go2-GetUp \
  --agent.max-iterations $ITERS \
  > /tmp/getup_${LABEL}_train.log 2>&1
RC=$?
echo "[v7.4] $(date -u) train.py exit=$RC"

POST_RUNS=$(ls -d $LOGDIR/2026-05-2[2-3]_*/ 2>/dev/null | sort)
NEW_RUN=$(comm -13 <(echo "$PRE_RUNS") <(echo "$POST_RUNS") | head -1)
[ -z "$NEW_RUN" ] && { echo "[v7.4] no new run dir, abort"; exit 2; }
echo "[v7.4] run dir: $NEW_RUN"

CKPT="${NEW_RUN}model_${FINAL}.pt"
if [ ! -f "$CKPT" ]; then
  echo "[v7.4] checkpoint $CKPT missing, using last available"
  CKPT=$(ls -1v ${NEW_RUN}model_*.pt 2>/dev/null | tail -1)
  [ -z "$CKPT" ] && { echo "[v7.4] no ckpts, abort"; exit 3; }
fi

echo "[v7.4] $(date -u) rendering"
PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa python3 scripts/play.py Unitree-Go2-GetUp \
  --checkpoint-file "$CKPT" --video True --video-length 300 --num-envs 1 \
  > /tmp/getup_${LABEL}_play.log 2>&1 &
PPID_PLAY=$!

VID="${NEW_RUN}videos/play/rl-video-step-0.mp4"
for i in $(seq 1 60); do
  sleep 10
  [ -f "$VID" ] && break
done
[ ! -f "$VID" ] && { echo "[v7.4] no mp4, abort"; kill $PPID_PLAY 2>/dev/null; exit 4; }
sleep 90
S1=$(stat -c%s "$VID" 2>/dev/null || echo 0)
sleep 15
S2=$(stat -c%s "$VID" 2>/dev/null || echo 0)
echo "[v7.4] mp4 sizes $S1 -> $S2"
[ "$S1" = "$S2" ] && [ "$S1" != "0" ] && kill $PPID_PLAY 2>/dev/null
sleep 5

mkdir -p $RPC/renders
cp "$VID" "$RPC/renders/get_up_${LABEL}.mp4"
cd $RPC && python3 scripts/sandbox_push.py "renders/get_up_${LABEL}.mp4" \
  -m "get_up v7.4 render (FR-Net + hip-abduction penalty, 8k iters)" || echo "[v7.4] push FAILED"

echo "[v7.4] $(date -u) DONE"
