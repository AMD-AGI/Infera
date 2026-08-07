#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Build the engine image. Run on the host of each node, or build once and push.
#
# The image comes straight from deploy/docker/Dockerfile.sglang.gfx942. It already
# carries the ROCm hicache fixes and the infera-router binary; do not layer a
# second Dockerfile or runtime patches on top of it.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

DOCKERFILE="${DOCKERFILE:-$REPO/deploy/docker/Dockerfile.sglang.gfx942}"
[[ -f "$DOCKERFILE" ]] || { echo "[build] missing Dockerfile: $DOCKERFILE" >&2; exit 1; }

echo "[build] image=$IMAGE"
echo "[build] dockerfile=$DOCKERFILE"
echo "[build] context=$REPO"
docker build -f "$DOCKERFILE" -t "$IMAGE" "$REPO" "$@"
