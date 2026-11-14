#!/bin/bash
# DRIVER619-20261113 formal delivery, SGLang nightly: on the 6.19 node, the e1ab nightly image e1ab-sglang:1112
# (lmsysorg/sglang-rocm:v0.5.23-rocm720-mi35x-20261109 + Infera 0.3.3 + host libionic entrypoint; built from the copied
# kfd619/nightly/image if missing), Mooncake upstream main + the HipGemAllocator PR built inside it, and the derived
# image e1ab-sglang:1112-hipgem.
OUT=$K/out/nightly-build-$(date -u +%m%dT%H%M)
mkdir -p "$OUT"
srun --jobid=$JOB --overlap -N1 -n1 -w $NODE619 --exclude=$NODE616 bash -c "
docker image inspect e1ab-sglang:1112 >/dev/null 2>&1 ||
    (cd $K/nightly/image && DOCKER_CONFIG=\$(mktemp -d) timeout 5400 docker build --pull -t e1ab-sglang:1112 .) || exit 1
docker run --rm --entrypoint bash -v /perf_apps:/perf_apps e1ab-sglang:1112 $K/mc/build_mc.sh $K/mc/src-main $OUT/main engine || exit 1
bash $K/mc/node_image.sh e1ab-sglang:1112 e1ab-sglang:1112-hipgem $OUT/main/pkg
" > "$OUT/run.log" 2>&1
tail -30 "$OUT/run.log"
