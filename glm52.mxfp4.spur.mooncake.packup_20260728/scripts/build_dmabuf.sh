#!/bin/bash
# Build the network-free sglang+mooncake(dmabuf, HIP-gated) image on a spur node.
# docker 29 uses buildx for `build`; DOCKER_CONFIG must be writable or plugin discovery fails.
set -eux
export DOCKER_CONFIG=/tmp/dockercfg
mkdir -p "$DOCKER_CONFIG"
REPO=/home/yihou/dev/git/infera.yihou.dev
cd "$REPO"
docker buildx build \
  --build-arg SGLANG_BASE_IMAGE=lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x \
  --build-arg BUILD_MOONCAKE_DMABUF=1 \
  -f deploy/docker/Dockerfile.sglang.dmabuf \
  -t glm52-sgl-dmabuf:v1 \
  --load \
  "$REPO"
echo "BUILD_DONE glm52-sgl-dmabuf:v1"
