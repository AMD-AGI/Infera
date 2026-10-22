#!/usr/bin/env bash
# Task dcpx 2026-10-22: build the destpin image on 135 from the loaded v0522 DCP image, then docker load it on 136.
# Usage (ON crsuse2-m2m-135): bash build.yihou.sh > build.yihou.log 2>&1
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BASE=infera-sglang:v0522-1020-yihou-dcp-mc4340
BASE_ID=sha256:437497f5b1f73203228a5c388ba1004ecce194d1a2050d8b49cfa47280923221
TAG=infera-sglang:v0522-1020-yihou-dcp-mc4340-destpin
PEER=crsuse2-m2m-136
date -Is
[[ "$(docker image inspect "$BASE" --format '{{.Id}}')" == "$BASE_ID" ]] || { echo "base is not $BASE_ID"; exit 1; }
docker build --network host --build-arg BASE_IMAGE=$BASE -t "$TAG" "$HERE"
id="$(docker image inspect "$TAG" --format '{{.Id}}')"
echo "built $TAG $id"; date -Is
docker save "$TAG" | ssh -o BatchMode=yes "$PEER" docker load
pid="$(ssh -o BatchMode=yes "$PEER" docker image inspect "$TAG" --format '{{.Id}}')"
echo "peer $PEER id $pid"; [[ "$pid" == "$id" ]] || echo "WARNING: peer image ID differs"
date -Is; echo BUILD_DONE
