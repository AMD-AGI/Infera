#!/usr/bin/env bash
# Run on crsuse2-m2m-138: stream the mc4340 image from 137 so both nodes carry the same image ID.
set -euo pipefail
IMG=infera-sglang:v0521-yihou-1010-mc4340
WANT=sha256:e8e2838f4a650584e92f1731cce26d3c7c500724f8befd301ecaac644763e22b
echo "start $(date -u +%FT%TZ) on $(hostname)"
ssh -o BatchMode=yes -o ConnectTimeout=10 crsuse2-m2m-137 docker save "$IMG" | docker load
got="$(docker image inspect -f '{{.Id}}' "$IMG")"
echo "loaded $IMG id=$got"
[[ "$got" == "$WANT" ]] && echo "ID MATCH" || { echo "ID MISMATCH (want $WANT)"; exit 1; }
echo "done $(date -u +%FT%TZ)"
