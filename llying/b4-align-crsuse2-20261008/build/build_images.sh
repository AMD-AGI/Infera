#!/usr/bin/env bash
# Rebuild the aus B4 image chain on this node from the same pinned inputs, add
# the Mooncake v3 cross-rank fix, and optionally copy the final image to a peer.
# Usage: build_images.sh [peer-node]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
PEER="${1:-}"
AUS_KIT="$REPO/llying/glm52.1p1d.c80.aus.packup_20260922"
TRACE_KIT="$REPO/llying/p8d8-c80-c112-tracing-aus-20260922"
CAMPAIGN="$REPO/llying/router-capacity-campaign-20260928"
RCA_FIX="$REPO/llying/cross-rank-rca-20260921/fix"
INFERA_REV=83e0f6c86cce34718f369d8669b750fa812c62d0
SGLANG_BASE=lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260916@sha256:eef7b70e015c94435d68dd92612229d52324b9e8d5edaac741802e7e78e66eb6
# aus tags, so each layer keeps the name it has in the aus records.
BASE=infera-sglang:v0519-llying-aus-base-83e0f6c8
OVERLAY=infera-sglang:v0519-llying-aus-0922-nextnfix-hicache
REQTRACE=infera-sglang:aus-0922-reqtrace
RADIX=infera-sglang:aus-campaign-radix-20260928
FINAL=infera-sglang:aus-campaign-radix-20260928-mcdestpin-ibto18

step() { echo "== $(date -u +%FT%TZ) $*"; }
ctx="$(mktemp -d /tmp/b4-align-build.XXXXXX)"
trap 'rm -rf "$ctx"' EXIT
# Copy only each layer's inputs: the kits also hold large result trees.
mkdir -p "$ctx/infera" "$ctx/overlay/build" "$ctx/overlay/patches" \
    "$ctx/reqtrace/docker" "$ctx/radix" "$ctx/mc"
git -C "$REPO" archive "$INFERA_REV" .dockerignore pyproject.toml README.md infera rust deploy/docker |
    tar -xf - -C "$ctx/infera"
cp "$AUS_KIT/build/Dockerfile.yihou-overlay" "$AUS_KIT/build/apply_nextn_patch.py" "$ctx/overlay/build/"
cp "$AUS_KIT/patches/pr37152.sources.yihou.diff" "$AUS_KIT/patches/nextn-fusion-fork-4350d37c5.patch" \
    "$ctx/overlay/patches/"
cp "$TRACE_KIT/docker/Dockerfile" "$TRACE_KIT/docker/aus_diag.py" "$TRACE_KIT/docker/apply_patch.py" \
    "$ctx/reqtrace/docker/"
cp "$CAMPAIGN/build/Dockerfile" "$CAMPAIGN/build/p01.patch" "$CAMPAIGN/build/p02.patch" "$ctx/radix/"
cp "$RCA_FIX/Dockerfile" "$RCA_FIX"/mooncake-*.diff "$REPO/deploy/docker/scripts/build_mooncake_sglang.sh" \
    "$ctx/mc/"

step "base $BASE (Infera $INFERA_REV on $SGLANG_BASE)"
docker build --network host --build-arg "SGLANG_BASE_IMAGE=$SGLANG_BASE" \
    -t "$BASE" -f "$ctx/infera/deploy/docker/Dockerfile.sglang" "$ctx/infera"
step "overlay $OVERLAY"
docker build --network host --build-arg "BASE_IMAGE=$BASE" \
    -t "$OVERLAY" -f "$ctx/overlay/build/Dockerfile.yihou-overlay" "$ctx/overlay"
docker run --rm -i --entrypoint python3 "$OVERLAY" - < "$AUS_KIT/scripts/verify_image.py"
step "reqtrace $REQTRACE"
docker build --network host -t "$REQTRACE" -f "$ctx/reqtrace/docker/Dockerfile" "$ctx/reqtrace"
step "decode radix $RADIX"
docker build --network host -t "$RADIX" "$ctx/radix"
step "Mooncake v3 $FINAL"
docker build --network host --build-arg "BASE_IMAGE=$RADIX" -t "$FINAL" "$ctx/mc"

for tag in "$BASE" "$OVERLAY" "$REQTRACE" "$RADIX" "$FINAL"; do
    echo "$tag $(docker image inspect "$tag" --format '{{.Id}}')"
done
id="$(docker image inspect "$FINAL" --format '{{.Id}}')"
[[ -n "$PEER" ]] || { step "done $FINAL $id"; exit 0; }
step "copy $FINAL to $PEER"
docker save "$FINAL" | ssh -o BatchMode=yes "$PEER" docker load
[[ "$(ssh -o BatchMode=yes "$PEER" docker image inspect "$FINAL" --format '{{.Id}}')" == "$id" ]] ||
    { echo "$PEER: image ID differs from $id" >&2; exit 1; }
step "done $FINAL $id"
