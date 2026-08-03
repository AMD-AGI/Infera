#!/usr/bin/env bash
# Build the only supported engine image for this recipe.
#
# The image is built directly from deploy/docker/Dockerfile.sglang.gfx942. Do not
# apply runtime patches or layer a second Dockerfile on top of it.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

INFERA_ROOT="${INFERA_ROOT:-$(cd "$HERE/../.." && pwd)}"
DOCKERFILE="${DOCKERFILE:-$INFERA_ROOT/deploy/docker/Dockerfile.sglang.gfx942}"

[[ -f "$DOCKERFILE" ]] || { echo "[build] missing Dockerfile: $DOCKERFILE" >&2; exit 1; }

echo "[build] image=$IMAGE"
echo "[build] dockerfile=$DOCKERFILE"
echo "[build] context=$INFERA_ROOT"
docker build -f "$DOCKERFILE" -t "$IMAGE" "$INFERA_ROOT" "$@"
