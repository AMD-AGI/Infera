#!/bin/bash
# Step 0 (node host side). Unpacks the staged source into the node-local yihou dir and starts a detached,
# GPU-less build container from the 1106 image. Returns immediately; poll $LCP/logs/build.log and
# $LCP/build/BUILD_OK.  Usage: bash step0_build.sh [jobs=32]
set -euo pipefail
source "$(dirname "$0")/env.sh"
JOBS=${1:-32}
mkdir -p "$LCP/logs" "$LCP/out"
if docker ps -a --format '{{.Names}}' | grep -q '^yihou-muse-calib-build$'; then
  echo "build container already exists"; exit 1
fi
sha256sum -c <(sed "s#  .*/rocprof-sdk-src.tar.gz#  $CP/stage/rocprof-sdk-src.tar.gz#" "$CP/stage/SHA256")
# Source and build are keyed by the archive digest: a new archive (other FORK_BRANCH / commit) extracts into its own
# src-<digest16>/ and refuses to overwrite a build made from another archive, so BUILD_OK always names its source.
DIG=$(cut -d' ' -f1 "$CP/stage/SHA256"); SUB=src-${DIG:0:16}
if [ -d "$LCP/build" ] && [ "$(cat "$LCP/build/SOURCE_SHA256" 2>/dev/null)" != "$DIG" ]; then
  echo "LCP/build exists but was not made from this archive ($DIG); use a fresh LCP"; exit 1
fi
if [ ! -f "$LCP/$SUB/.extracted" ]; then
  mkdir -p "$LCP/$SUB" && tar -xzf "$CP/stage/rocprof-sdk-src.tar.gz" -C "$LCP/$SUB" && echo "$DIG" > "$LCP/$SUB/.extracted"
fi
docker run -d --rm --ulimit core=0 --name yihou-muse-calib-build --network host \
  -v "$NFS:/yihou/runtime" -v "$DISK:/yihou/local" -e JOBS="$JOBS" -e SRC_SUB="$SUB" -e SRC_SHA256="$DIG" \
  "$IMG" bash "$C_CP/node/in_container_build.sh" > "$LCP/logs/build-container-id.txt"
( docker logs -f yihou-muse-calib-build > "$LCP/logs/build.log" 2>&1 & )
echo "started; log $LCP/logs/build.log"
