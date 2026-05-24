#!/usr/bin/env bash
# Launch Walker-MoB v1 training on a RunPod pod.
#
# Assumes the pod has already had `bash scripts/setup_cloud.sh` run on it,
# and that /workspace/unitree_rl_mjlab and /workspace/robot-pet-cat are
# both git clones with the venv at /workspace/mjlab_venv.
#
# Usage:
#   bash /workspace/robot-pet-cat/scripts/launch_mob_v1.sh
#
# Two paths supported:
#   ITERS env var: override iter count (default 3000)
#   WARM_START_FROM env var: path to a model_*.pt to pad+warm-start from.
#                            If unset, trains from scratch (recommended for v1
#                            since the obs-vector column-insert warm-start is
#                            still untested on this codebase).

set -u
exec > >(tee /tmp/mob_v1_launch.log) 2>&1
echo "[mob_v1] $(date -u) START"

ITERS=${ITERS:-3000}
WARM_START_FROM=${WARM_START_FROM:-}
VENV=/workspace/mjlab_venv/bin/activate
MJLAB=/workspace/unitree_rl_mjlab
RPC=/workspace/robot-pet-cat
SCRIPTS=$RPC/scripts
LOGDIR=$MJLAB/logs/rsl_rl/go2_velocity

echo "[mob_v1] iters=$ITERS"
echo "[mob_v1] warm_start_from=${WARM_START_FROM:-<scratch>}"

# 1. Refresh robot-pet-cat from GitHub.
cd "$RPC"
git fetch --all --quiet
git reset --hard origin/main
echo "[mob_v1] robot-pet-cat at $(git rev-parse --short HEAD)"

# 2. Activate venv + load .env (for WANDB_API_KEY etc).
cd "$MJLAB"
source "$VENV"
set -a && source "$RPC/.env" && set +a

# 3. Apply MoB patch (idempotent).
echo "[mob_v1] applying patch ..."
python3 "$SCRIPTS/patch_mob_unitree.py" || { echo "[mob_v1] patch FAILED"; exit 1; }

# 4. Run smoke test.
echo "[mob_v1] running smoke test ..."
python3 "$SCRIPTS/smoketest_mob.py" || { echo "[mob_v1] smoke test FAILED"; exit 1; }

# 5. Warm-start (optional).
RESUME_FLAGS=""
if [ -n "$WARM_START_FROM" ]; then
  if [ ! -f "$WARM_START_FROM" ]; then
    echo "[mob_v1] WARM_START_FROM=$WARM_START_FROM not found"
    exit 1
  fi
  WARM_DIR=/workspace/mob_warmstart
  mkdir -p "$WARM_DIR"
  WARM_PT="$WARM_DIR/model_mob_init.pt"
  echo "[mob_v1] padding checkpoint $WARM_START_FROM -> $WARM_PT"
  python3 "$SCRIPTS/warm_start_to_mob.py" \
      --input "$WARM_START_FROM" \
      --output "$WARM_PT" \
      --insert-col 9 || { echo "[mob_v1] warm-start FAILED"; exit 1; }
  RESUME_FLAGS="--agent.resume True --agent.load-run $WARM_DIR --agent.load-checkpoint model_mob_init.pt"
fi

# 6. Snapshot pre-existing run dirs to detect the new one later.
PRE_RUNS=$(ls -d $LOGDIR/*/ 2>/dev/null | sort)

# 7. Train.
echo "[mob_v1] launching train.py Unitree-Go2-Flat-MoB ($ITERS iters)"
nohup python3 scripts/train.py Unitree-Go2-Flat-MoB \
    --agent.max-iterations "$ITERS" \
    $RESUME_FLAGS \
    > /tmp/mob_v1_train.log 2>&1 &
TRAIN_PID=$!
echo "[mob_v1] train PID=$TRAIN_PID  log=/tmp/mob_v1_train.log"

echo
echo "Monitor with:"
echo "  tail -f /tmp/mob_v1_train.log"
echo "  ls -lt $LOGDIR | head"
echo
echo "When done, render with:"
echo "  python3 scripts/play.py Unitree-Go2-Flat-MoB \\"
echo "      --agent.load-run <new-run-dir> \\"
echo "      --agent.load-checkpoint model_<last>.pt \\"
echo "      --viewer auto --num-envs 1 --episode-length-s 20"
