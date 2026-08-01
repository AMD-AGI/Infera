#!/usr/bin/env bash
# Build the merged engine image ON the node, from the branch source.
#
# Built here rather than shipped in because the point of the exercise is that
# the Dockerfile reproduces the experiment: a 28GB save/load of an image built
# somewhere else would prove only that the tarball survived the trip.
set -eu
SRC=/root/merged_src
rm -rf "$SRC" && mkdir -p "$SRC"
tar xzf /tmp/merged_src.tgz -C "$SRC"
cd "$SRC"
echo "=== building infera/engine-sglang:merged on $(hostname) ==="
docker build -f deploy/docker/Dockerfile.sglang -t infera/engine-sglang:merged . 2>&1 | tail -60
echo "=== done: $(docker image inspect infera/engine-sglang:merged --format '{{.Id}}') ==="
