#!/bin/bash
# Node host side: one-GPU smoke of the fork json-tool in the serving image (gate: probe mode window PASS).
# Usage: bash kf_smoke.sh [hip_gpu=7]   (blocking, <= 15 min). Output: $LCP/out/smoke-<UTC>/
set -euo pipefail
source "$(dirname "$0")/kf_env.sh"
G=${1:-7}
[ -f "$LCP/build/BUILD_OK" ] || { echo "no BUILD_OK; run kf_build.sh first"; exit 1; }
TS=$(date -u +%Y%m%dT%H%M%SZ); O=$LCP/out/smoke-$TS; mkdir -p "$O"
SRC=$(ls -d "$LCP"/src-*/projects/rocprofiler-sdk | head -1)
timeout 900 docker run --rm --ulimit core=0 --name "yihou-dsv41kf-calib-smoke-$TS" --network host --ipc host \
  --device /dev/kfd --device /dev/dri --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -v /perf_apps:/perf_apps:ro -v "$LCP:$LCP" -e HIP_VISIBLE_DEVICES="$G" -e OUT="$O" -e SDK="$LCP/build" -e CP="$CP" \
  -e SRC="$SRC" -e PYTHONNOUSERSITE=1 -e TMPDIR="$O" -e HOME="$O" \
  --entrypoint bash "$IMAGE" "$CP/node/in_container_smoke.sh" 2>&1 | tee "$O/smoke.log"
echo "smoke dir $O"
