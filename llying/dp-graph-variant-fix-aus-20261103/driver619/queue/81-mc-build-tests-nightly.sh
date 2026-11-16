#!/bin/bash
# DRIVER619-20261113 formal delivery: upstream main + the PR with hip_gem_allocator_test, rdma_hip_kfd_pin_test and TENT,
# built in the nightly image on the 6.19 node (the production image lacks main's CLI11; 60 failed on it).
OUT=$K/out/mc-tests-$(date -u +%m%dT%H%M)
mkdir -p "$OUT" && ln -sfn "$OUT" $K/out/mc-tests-latest
srun --jobid=$JOB --overlap -N1 -n1 -w $NODE619 --exclude=$NODE616 bash -c "
docker image inspect e1ab-sglang:1112 >/dev/null 2>&1 ||
    (cd $K/nightly/image && DOCKER_CONFIG=\$(mktemp -d) timeout 5400 docker build --pull -t e1ab-sglang:1112 .) || exit 1
docker run --rm --entrypoint bash -v /perf_apps:/perf_apps e1ab-sglang:1112 $K/mc/build_mc.sh $K/mc/src-main $OUT/main tests
" > "$OUT/run.log" 2>&1
tail -30 "$OUT/run.log"
