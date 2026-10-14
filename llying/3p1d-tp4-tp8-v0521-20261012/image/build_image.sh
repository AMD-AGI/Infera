#!/usr/bin/env bash
# Build the cross-rank image on THIS node (137) from the exact mc4340 base, then push it
# to the peer nodes and check every node carries the same image ID.
# Usage (on crsuse2-m2m-137): build_image.sh [peer-node ...]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
BASE=infera-sglang:v0521-yihou-1010-mc4340
BASE_ID=sha256:e8e2838f4a650584e92f1731cce26d3c7c500724f8befd301ecaac644763e22b
TAG=infera-sglang:v0521-llying-1012-mc4340-destpin

[[ "$(docker image inspect "$BASE" --format '{{.Id}}')" == "$BASE_ID" ]] ||
    { echo "base $BASE is not $BASE_ID" >&2; exit 1; }

echo "build start $(date -u +%FT%TZ) on $(hostname)"
docker build --network host --progress=plain --build-arg BASE_IMAGE="$BASE" \
    -t "$TAG" -f "$HERE/Dockerfile" "$HERE"
id="$(docker image inspect "$TAG" --format '{{.Id}}')"
echo "built $TAG $id $(date -u +%FT%TZ)"

for peer in "$@"; do
    echo "push $peer start $(date -u +%FT%TZ)"
    docker save "$TAG" | ssh -o BatchMode=yes "$peer" docker load
    got="$(ssh -o BatchMode=yes "$peer" docker image inspect "$TAG" --format '{{.Id}}')"
    [[ "$got" == "$id" ]] || { echo "$peer: image ID $got differs from $id" >&2; exit 1; }
    echo "push $peer ok $(date -u +%FT%TZ)"
done
echo "ALL OK $TAG $id"
