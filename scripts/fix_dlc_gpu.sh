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
# Diagnostic only; failures here (e.g. broken torch install) shouldn't abort
# the script -- the install step below is what fixes them. set +e to ignore
# the python exit code, then set -e back.
set +e
$RUNNER python - <<'PY'
try:
    import torch
    print(f"torch.__version__     = {torch.__version__}")
    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"torch.cuda.device_count() = {torch.cuda.device_count()}")
        print(f"device 0 name         = {torch.cuda.get_device_name(0)}")
        print(f"torch CUDA build      = {torch.version.cuda}")
    else:
        print("torch is CPU-only -- this is the slowness.")
except ImportError as e:
    print(f"torch import currently broken: {e}")
    print("will reinstall below.")
PY
set -e

if [ "$CHECK_ONLY" -eq 1 ]; then
  exit 0
fi

echo
echo "=== installing GPU torch wheels ($CUDA_VER) ==="
# Pin torch to 2.3.1 because:
#   - torch 2.4+ ships cuDNN 9.x which has known NOT_INITIALIZED failures
#     on RTX 40-series GPUs with HRNet-style ops (what SuperAnimal uses).
#   - torch 2.3.1 bundles cuDNN 8.9 which is the combo DLC tests against
#     and is solid on 40-series.
# --no-deps so we don't disturb DLC's other pins (numpy, pandas, etc).
# Also uninstall any standalone nvidia-cudnn-cu12 first -- if both
# torch's bundled cuDNN and a standalone one are present, the loader
# picks the wrong one and you get NOT_INITIALIZED.
# Three-step install. --no-deps for torch+torchvision (so we don't
# disturb DLC's pandas/numpy pins), then explicitly install
# nvidia-cudnn-cu12 == 8.9.2.26 (the version torch 2.3.1 binds against).
# Without that package, torch._C fails to load with
#   ImportError: libcudnn.so.8: cannot open shared object file
$RUNNER pip uninstall -y torch torchvision nvidia-cudnn-cu12 nvidia-cudnn-cu11 || true
$RUNNER pip install --upgrade --force-reinstall --no-deps \
  torch==2.3.1 torchvision==0.18.1 \
  --index-url "https://download.pytorch.org/whl/${CUDA_VER}"
# nvidia-cudnn-cu12 lives on regular PyPI, not the pytorch index. Pin to
# 8.9.2.26 because that's what torch 2.3.1's torch._C is built against.
$RUNNER pip install nvidia-cudnn-cu12==8.9.2.26

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
