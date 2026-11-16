#!/bin/bash
# DRIVER619-20261113 formal delivery, two Mooncake PRs built in the nightly image on the 6.19 node (the production image
# lacks main's CLI11; 60 failed on it), each with hip_gem_allocator_test, rdma_hip_kfd_pin_test and TENT:
#   main  src-main, PR 1 (feat/hip-gem-allocator: allocator, classic transfer engine, TENT refuses GEM buffers)
#   tent  src-tent, PR 2 on top (feat/hip-gem-allocator-tent: TENT registers them through their dma-buf)
OUT=$K/out/mc-tests-$(date -u +%m%dT%H%M)
mkdir -p "$OUT" && ln -sfn "$OUT" $K/out/mc-tests-latest
cp $K/mc/src-main/REV "$OUT/main.rev"; cp $K/mc/src-tent/REV "$OUT/tent.rev"
srun --jobid=$JOB --overlap -N1 -n1 -w $NODE619 --exclude=$NODE616 bash -c "
docker image inspect e1ab-sglang:1112 >/dev/null 2>&1 ||
    (cd $K/nightly/image && DOCKER_CONFIG=\$(mktemp -d) timeout 5400 docker build --pull -t e1ab-sglang:1112 .) || exit 1
for v in main tent; do
    echo \"######## \$v\"
    docker run --rm --entrypoint bash -v /perf_apps:/perf_apps e1ab-sglang:1112 $K/mc/build_mc.sh $K/mc/src-\$v $OUT/\$v tests
    echo \"build \$v rc \$?\"
done
" > "$OUT/run.log" 2>&1
tail -60 "$OUT/run.log"
