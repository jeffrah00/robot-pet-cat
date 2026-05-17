#!/usr/bin/env bash
# Install GPU PyTorch into the dlc conda env.
#
# Background: DLC 3.x runs on PyTorch and the conda yaml DLC ships with
# pulls the CPU-only torch wheel. SuperAnimal-Quadruped on a 4090 GPU does
# ~30-60 FPS for HRNet-W32; on CPU it does <1 FPS. Going from CPU to GPU
# is a 50-100x speedup -- bigger than any other change.
#
# This script:
#   1. Reports the current torch + CUDA status in the dlc env.
#   2. If CPU-only, force-reinstalls the cu121 torch wheels (the modern
#      RunPod default; pass CUDA_VER=cu118 if your container is older).
#   3. Verifies that torch.cuda.is_available() is True afterwards.
#
# Usage:
#   bash scripts/fix_dlc_gpu.sh              # auto-detect, default cu121
#   CUDA_VER=cu118 bash scripts/fix_dlc_gpu.sh
#   bash scripts/fix_dlc_gpu.sh --check-only # just report status

set -euo pipefail

CUDA_VER="${CUDA_VER:-cu121}"
CHECK_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --check-only) CHECK_ONLY=1 ;;
    *) echo "unknown arg: $arg"; exit 1 ;;
  esac
done

if command -v micromamba >/dev/null 2>&1; then
  RUNNER="micromamba run -n dlc"
elif command -v conda >/dev/null 2>&1; then
  RUNNER="conda run -n dlc"
else
  echo "ERROR: neither micromamba nor conda is on PATH; can't reach the dlc env"
  exit 2
fi

echo "=== current torch / CUDA in dlc env ==="
$RUNNER python - <<'PY'
import torch
print(f"torch.__version__     = {torch.__version__}")
print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"torch.cuda.device_count() = {torch.cuda.device_count()}")
    print(f"device 0 name         = {torch.cuda.get_device_name(0)}")
    print(f"torch CUDA build      = {torch.version.cuda}")
else:
    print("torch is CPU-only -- this is the slowness.")
PY

if [ "$CHECK_ONLY" -eq 1 ]; then
  exit 0
fi

echo
echo "=== installing GPU torch wheels ($CUDA_VER) ==="
# --no-deps so we don't disturb DLC's other pins. Force-reinstall over the
# CPU wheels.
$RUNNER pip install --upgrade --force-reinstall --no-deps \
  torch torchvision \
  --index-url "https://download.pytorch.org/whl/${CUDA_VER}"

echo
echo "=== verifying ==="
$RUNNER python - <<'PY'
import torch
ok = torch.cuda.is_available()
print(f"torch.__version__     = {torch.__version__}")
print(f"torch.cuda.is_available() = {ok}")
if ok:
    print(f"device 0 name         = {torch.cuda.get_device_name(0)}")
    print(f"torch CUDA build      = {torch.version.cuda}")
    print()
    print("DONE. Re-run scripts/retarget_all.py -- pose extraction should be ~50-100x faster.")
else:
    print()
    print("STILL CPU. Likely causes:")
    print("  - container CUDA driver is older than the wheel's CUDA build.")
    print("    Try CUDA_VER=cu118 bash scripts/fix_dlc_gpu.sh")
    print("  - no GPU visible to this process. nvidia-smi inside the dlc env:")
    print("    $RUNNER nvidia-smi")
    import sys; sys.exit(3)
PY
