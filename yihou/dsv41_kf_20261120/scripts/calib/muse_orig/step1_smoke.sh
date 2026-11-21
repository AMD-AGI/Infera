#!/bin/bash
# Step 1 (node host side): one-GPU smoke of the fork json-tool, same env as the server window.
# Usage: bash step1_smoke.sh [gpu=7]   (blocking, <= 15 min). Output: $LCP/out/smoke-<UTC>/
set -euo pipefail
source "$(dirname "$0")/env.sh"
G=${1:-7}
[ -f "$LCP/build/BUILD_OK" ] || { echo "no BUILD_OK; run step 0 first"; exit 1; }
# GPU G must be idle: no process on it (rocm-smi --showpids lists PIDs per GPU on this ROCm)
rocm-smi --showuse -d "$G" 2>/dev/null | grep -E 'GPU use' || true
TS=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$LCP/out/smoke-$TS"
timeout 900 docker run --rm --ulimit core=0 --name "yihou-muse-calib-smoke-$TS" --network host --ipc host \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -v "$NFS:/yihou/runtime" -v "$DISK:/yihou/local" -v "$DISK/home:/root" \
  -e HIP_VISIBLE_DEVICES="$G" -e OUT="/yihou/local/calib_profiler/out/smoke-$TS" -e PYTHONNOUSERSITE=1 \
  -e TMPDIR=/yihou/local/calib_profiler/out/smoke-$TS \
  "$IMG" bash "$C_CP/node/in_container_smoke.sh" 2>&1 | tee "$LCP/out/smoke-$TS/smoke.log"
echo "smoke dir $LCP/out/smoke-$TS"
