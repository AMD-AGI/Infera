#!/bin/bash
# DRIVER619-20261113 formal delivery, two Mooncake PRs built in the nightly image on the 6.19 node (the production image
# lacks main's CLI11; 60 failed on it), each with hip_gem_allocator_test, rdma_hip_kfd_pin_test and TENT:
#   main    src-main, PR 1 (feat/hip-gem-allocator: allocator, classic transfer engine, TENT refuses GEM buffers)
#   tent    src-tent, PR 2 on top (feat/hip-gem-allocator-tent: TENT registers them through their dma-buf)
#   engine  src-main again the way the image builds Mooncake (no TENT, no tests), derived into e1ab-sglang:1112-hipgem2
#           for 86 / 87 (1112-hipgem from 70 predates the GEM device fix of 11-16)
OUT=$K/out/mc-tests-$(date -u +%m%dT%H%M)
mkdir -p "$OUT" && ln -sfn "$OUT" $K/out/mc-tests-latest
cp $K/mc/src-main/REV "$OUT/main.rev"; cp $K/mc/src-tent/REV "$OUT/tent.rev"
srun --jobid=$JOB --overlap -N1 -n1 -w $NODE619 --exclude=$NODE616 bash -c "
docker image inspect e1ab-sglang:1112 >/dev/null 2>&1 ||
    (cd $K/nightly/image && DOCKER_CONFIG=\$(mktemp -d) timeout 5400 docker build --pull -t e1ab-sglang:1112 .) || exit 1
for v in main:tests tent:tests engine:engine; do
    n=\${v%%:*} m=\${v#*:} s=\${v%%:*}; [ \$n = engine ] && s=main
    echo \"######## \$n\"
    docker run --rm --entrypoint bash -v /perf_apps:/perf_apps e1ab-sglang:1112 $K/mc/build_mc.sh $K/mc/src-\$s $OUT/\$n \$m
    echo \"build \$n rc \$?\"
done
bash $K/mc/node_image.sh e1ab-sglang:1112 e1ab-sglang:1112-hipgem2 $OUT/engine/pkg
" > "$OUT/run.log" 2>&1
tail -60 "$OUT/run.log"
