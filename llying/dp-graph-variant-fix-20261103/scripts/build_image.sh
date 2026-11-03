#!/usr/bin/env bash
# Build the image every patch here was cut against: Infera fdbafb60, deploy/docker/Dockerfile.sglang with its default
# ARGs (base lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261018 + the Infera sglang patch set). Then check the 11 base
# files against patches/BASE_MD5SUMS and optionally save the image for the other node (`docker load -i SAVE_TAR`).
# Builds of the same commit have produced different image IDs (layers) before; the md5 check is what matters.
# Usage (on a node with docker): INFERA_REPO=<clone of AMD-AGI/Infera> [CTX=<local scratch dir>] [SAVE_TAR=<shared path>.tar] \
#     [DOCKER_CONFIG=<empty dir> for an anonymous base pull] build_image.sh
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$here/.." && pwd)"
TAG="${IMAGE:-infera-sglang:v0522-fdbafb60}"
CTX="${CTX:-/tmp/infera-fdbafb60-ctx-$USER}"
: "${INFERA_REPO:?set INFERA_REPO to a clone of AMD-AGI/Infera that has commit fdbafb60}"
if ! docker image inspect "$TAG" >/dev/null 2>&1; then
    rm -rf "$CTX" && mkdir -p "$CTX"
    git -C "$INFERA_REPO" archive fdbafb60 | tar -x -C "$CTX"
    (cd "$CTX" && DOCKER_BUILDKIT=1 docker build --network host -f deploy/docker/Dockerfile.sglang -t "$TAG" .)
fi
echo "image $TAG $(docker image inspect -f '{{.Id}}' "$TAG")"
files="$(awk '{print $2}' "$PKG/patches/BASE_MD5SUMS" | tr '\n' ' ')"
docker run --rm --network none --entrypoint sh "$TAG" -c "cd /sgl-workspace/sglang/python/sglang/srt && md5sum $files" \
    | sort -k2 | diff - <(sort -k2 "$PKG/patches/BASE_MD5SUMS") && echo "base files: all 11 match patches/BASE_MD5SUMS" \
    || { echo "base files differ from patches/BASE_MD5SUMS (see diff above): the patches do not apply to this image as is"; exit 1; }
if [[ -n "${SAVE_TAR:-}" && ! -s "$SAVE_TAR" ]]; then
    docker save "$TAG" -o "$SAVE_TAR"
    echo "saved $SAVE_TAR ($(du -h "$SAVE_TAR" | cut -f1))"
fi
