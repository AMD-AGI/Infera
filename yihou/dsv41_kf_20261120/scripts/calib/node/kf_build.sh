#!/bin/bash
# Node host side: unpack the staged fork source (Muse stage, v35 @ae7c95f4) and build the json-tool in a GPU-less
# container of the serving image. Returns at once; poll $LCP/logs/build.log and $LCP/build/BUILD_OK.
set -euo pipefail
source "$(dirname "$0")/kf_env.sh"
mkdir -p "$LCP/logs" "$LCP/out"
docker ps -a --format '{{.Names}}' | grep -q '^yihou-dsv41kf-calib-build$' && { echo "build container exists"; exit 1; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || docker pull "$IMAGE"
(cd "$CP/stage" && sha256sum -c SHA256.kf)
DIG=$(cut -d' ' -f1 "$CP/stage/SHA256.kf"); SUB=src-${DIG:0:16}
if [ ! -f "$LCP/$SUB/.extracted" ]; then
  mkdir -p "$LCP/$SUB" && tar -xzf "$CP/stage/rocprof-sdk-src.tar.gz" -C "$LCP/$SUB" && echo "$DIG" > "$LCP/$SUB/.extracted"
fi
docker run -d --rm --ulimit core=0 --name yihou-dsv41kf-calib-build --network host \
  -v /perf_apps:/perf_apps:ro -v "$LCP:$LCP" -e LCP="$LCP" -e CP="$CP" -e JOBS="${JOBS:-64}" -e SRC_SUB="$SUB" -e SRC_SHA256="$DIG" \
  --entrypoint bash "$IMAGE" "$CP/node/in_container_build.sh" > "$LCP/logs/build-container-id.txt"
( docker logs -f yihou-dsv41kf-calib-build > "$LCP/logs/build.log" 2>&1 & )
echo "started; log $LCP/logs/build.log"
