#!/usr/bin/env bash
# Build the fix image on this node from the exact base image, then copy it to
# the peer node. Usage: build_image.sh <peer-node>
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
PEER="$1"
BASE=infera-sglang:v0519-yihou-0917-nextnfix-hicache
BASE_ID=sha256:fd7220a57b7d3b58efd875c41f7a9ef46b93469581102d96cbeb6f5451e91d35
TAG=infera-sglang:v0519-yihou-0917-nextnfix-hicache-mcdestpin-ibto18

[[ "$(docker image inspect "$BASE" --format '{{.Id}}')" == "$BASE_ID" ]] ||
    { echo "base $BASE is not $BASE_ID" >&2; exit 1; }

context="$(mktemp -d /tmp/mc-rca-fix-build.XXXXXX)"
trap 'rm -rf "$context"' EXIT
cp "$HERE/Dockerfile" "$HERE"/mooncake-*.diff \
    "$REPO/deploy/docker/scripts/build_mooncake_sglang.sh" "$context/"
docker build --network host -t "$TAG" "$context"

id="$(docker image inspect "$TAG" --format '{{.Id}}')"
docker save "$TAG" | ssh -o BatchMode=yes "$PEER" docker load
[[ "$(ssh -o BatchMode=yes "$PEER" docker image inspect "$TAG" --format '{{.Id}}')" == "$id" ]] ||
    { echo "$PEER: image ID differs from $id" >&2; exit 1; }
echo "$TAG $id"
