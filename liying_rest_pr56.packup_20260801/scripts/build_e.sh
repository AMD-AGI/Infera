#!/usr/bin/env bash
# Build the branch WITH group E under a distinct tag.
#
# Deliberately NOT the :merged tag: the running merged_run containers on both
# nodes were created from that tag and are the ground-truth reference for this
# line of work. Overwriting it would leave the reference unreproducible.
set -eu
SRC=/root/merged_e_src
rm -rf "$SRC" && mkdir -p "$SRC"
tar xzf /tmp/merged_src.tgz -C "$SRC"
cd "$SRC"
echo "=== building infera/engine-sglang:merged-e on $(hostname) ==="
docker build -f deploy/docker/Dockerfile.sglang -t infera/engine-sglang:merged-e . 2>&1 | tail -40
echo "=== done: $(docker image inspect infera/engine-sglang:merged-e --format '{{.Id}}') ==="
