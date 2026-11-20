#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: build the engine image for this kit from the current checkout (deploy/docker/Dockerfile.vllm.dsv41):
#       the vLLM ROCm nightly + this kit's vLLM patches + the GEM KV allocator + infera and its Rust router.
# why : the kit runs infera at THIS commit, not a release wheel; the image records which one (BUILD_INFO, labels).
# how : on a node with docker:   bash build_image.sh [TAG]
#       Prints the tag on success. Copy to another node with `docker save | docker load` (no registry needed).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
commit="$(git -C "$REPO" rev-parse --short=12 HEAD)"
git -C "$REPO" diff --quiet HEAD -- infera rust deploy/docker pyproject.toml || commit="$commit-dirty"
# setuptools-scm's guess-next-dev from the last release tag: v0.3.3 + 7 commits -> 0.3.4.dev7.
desc="$(git -C "$REPO" describe --tags --long --match 'v[0-9]*' 2>/dev/null || echo v0.0.0-0-g0)"
base="${desc%-*-g*}"; base="${base#v}"; n="${desc%-g*}"; n="${n##*-}"
version="${base%.*}.$(( ${base##*.} + 1 )).dev$n"
base_image="$(sed -n 's/^ARG VLLM_BASE_IMAGE=//p' "$REPO/deploy/docker/Dockerfile.vllm.dsv41")"
nightly="$(echo "$base_image" | sed -nE 's/.*:nightly-([0-9a-f]{8}).*/\1/p')"
TAG="${1:-infera-vllm-dsv41:${commit}-nightly-${nightly:-base}}"

echo "[build] infera $version ($commit) on $base_image -> $TAG"
DOCKER_BUILDKIT=1 docker build -f "$REPO/deploy/docker/Dockerfile.vllm.dsv41" \
    --build-arg INFERA_VERSION="$version" --build-arg INFERA_COMMIT="$commit" \
    -t "$TAG" "$REPO"
docker run --rm --entrypoint cat "$TAG" /opt/infera/BUILD_INFO
echo "$TAG"
