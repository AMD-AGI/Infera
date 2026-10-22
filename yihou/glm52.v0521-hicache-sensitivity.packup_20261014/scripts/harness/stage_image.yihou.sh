#!/usr/bin/env bash
# Run ON the node (detached): pull the v0.5.21 base, build infera-sglang:v0521-yihou-1009 from the
# private clone, verify (mooncake engine.so sha + verify_v0521.sh), record the image ID.
# Usage: stage_image.yihou.sh   (log goes to $WS/logs/stage.<host>.yihou.log via the caller)
set -uo pipefail
C=/shared_nfs/yihou_glm52/agentx44_20261009/infera-src.yihou
BASE=lmsysorg/sglang-rocm:v0.5.21-rocm720-mi35x-20261008
TAG=infera-sglang:v0521-yihou-1009
echo "== $(date -u +%FT%TZ) $(hostname) clone=$(git -C $C rev-parse --short HEAD) dirty=$(git -C $C status --short | wc -l)"
docker pull "$BASE" || { echo "STAGE_FAIL pull"; exit 1; }
echo "== $(date -u +%FT%TZ) build"
docker build --network host --build-arg SGLANG_BASE_IMAGE="$BASE" \
  --build-arg BUILD_MOONCAKE=1 --build-arg REQUIRE_MOONCAKE_CROSS_HOST_ROUTING=1 \
  -f "$C/deploy/docker/Dockerfile.sglang.v0521.yihou" -t "$TAG" "$C"
rc=$?; echo "BUILD_EXIT=$rc"; [[ $rc == 0 ]] || { echo "STAGE_FAIL build"; exit 1; }
echo "== $(date -u +%FT%TZ) verify (CPU-only containers, --network none, no GPU devices)"
docker run --rm --network none --entrypoint bash "$TAG" -c \
  'so=$(python3 -c "import mooncake.engine as e; print(e.__file__)"); echo "ENGINE_SO $so"; echo "ENGINE_SHA $(sha256sum $so | cut -d" " -f1)"; python3 -c "import sglang;print(\"SGLANG\", sglang.__version__)"'
docker run --rm --network none -v /shared_nfs/yihou_glm52/agentx44_20261009/build/tools.yihou:/v:ro \
  --entrypoint bash "$TAG" /v/verify_v0521.sh; echo "VERIFY_EXIT=$?"
echo "IMAGE_ID $(docker image inspect --format '{{.Id}}' "$TAG")"
echo "== $(date -u +%FT%TZ) STAGE_DONE"
