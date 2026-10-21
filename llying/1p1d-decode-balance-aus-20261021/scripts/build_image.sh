#!/usr/bin/env bash
# Build the PR 210 image (deploy/docker/Dockerfile.sglang at the exported commit) on THIS node. Run through the ssh shim
# from the login host, in the foreground of the srun step (a step's processes die with it).
# Docker Hub: the stored token is expired, so pulls go anonymous (DOCKER_CONFIG with an empty config).
# Usage: build_image.sh <source dir> <tag>
set -euo pipefail
src="${1:?source dir}" tag="${2:?tag}"
export DOCKER_CONFIG=/tmp/liyingli-docker-anon
mkdir -p "$DOCKER_CONFIG" && [[ -s "$DOCKER_CONFIG/config.json" ]] || echo '{}' > "$DOCKER_CONFIG/config.json"
echo "$(date -u +%FT%TZ) $(hostname -s) build $tag from $src"
if docker image inspect "$tag" >/dev/null 2>&1; then
    echo "$(date -u +%FT%TZ) $(hostname -s) $tag already present: $(docker image inspect -f '{{.Id}}' "$tag")"
    exit 0
fi
cd "$src"
docker build -f deploy/docker/Dockerfile.sglang -t "$tag" .
echo "$(date -u +%FT%TZ) $(hostname -s) built $tag: $(docker image inspect -f '{{.Id}}' "$tag")"
