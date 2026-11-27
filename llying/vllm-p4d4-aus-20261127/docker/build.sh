#!/usr/bin/env bash
# Build a nightly engine image on this node (holder item; docker images are node-local).
#   build.sh SERIES TAG     e.g. build.sh series-C.txt infera-vllm:nightly-de7cf7f8-C
set -uo pipefail
. /perf_apps/liyingli/bench_agentx/vllm-p4d4-aus-20261127/site/env.sh
series=${1:?series} tag=${2:?tag}
# One build per tag per node at a time; an item that needs the image waits here and reuses it.
exec 9> "/tmp/lyl-build-${tag//[:\/]/_}.lock"; flock 9
docker image inspect "$tag" > /dev/null 2>&1 && { echo "$(ts) $tag already on $(hostname -s)"; exit 0; }
[[ "$(git -C "$REF" rev-parse HEAD)" == "$REF_COMMIT" && -z "$(git -C "$REF" status --porcelain)" ]] \
    || { echo "$REF is not a clean checkout of $REF_COMMIT"; exit 1; }
pull_image "$NIGHTLY_BASE" || { echo "pull of $NIGHTLY_BASE failed"; exit 1; }
t0=$(date +%s)
DOCKER_BUILDKIT=1 docker build --progress plain -f "$W/docker/Dockerfile.nightly" --build-context "patches=$W/patches" \
    --build-arg "VLLM_BASE_IMAGE=$NIGHTLY_BASE" --build-arg "PATCH_SERIES=$series" \
    --label "infera.commit=$REF_COMMIT" --label "vllm.patch_series=$series" -t "$tag" "$REF"
rc=$?
echo "$(ts) build $tag ($series) rc=$rc in $(( $(date +%s) - t0 )) s"
[[ $rc == 0 ]] && docker run --rm --entrypoint cat "$tag" /opt/vllm-patches/applied.txt
exit $rc
