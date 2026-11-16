#!/bin/bash
# DRIVER619-20261113 formal delivery, nightly only: mc_gem_test.py in e1ab-sglang:1112-hipgem (Mooncake main + the PR, from
# 70) on the 6.19 node, with Mooncake's HIP transport on (same-host transfers to GEM buffers must route to RDMA) and
# MOONCAKE_DISABLE_HIP_DMABUF=1 (GEM buffers must still go through dma-buf). Pairs: 150 / 198 GiB HipGemAllocator KV;
# KFD KV with peer memory (the production path), and KFD KV with dma-buf and main's KFD pin anchor (#4657) on.
OUT=$K/out/mc-nightly-$(date -u +%m%dT%H%M)
mkdir -p "$OUT"
P="0:mcgem:150:1:0:0 1:mcgem:198:1:0:0 2:kfd:150:1:0:0 3:kfd:150:0:1:0"
# The derived image exists only where 70 ran (n04-29); elsewhere derive it from 70's engine package.
srun --jobid=$JOB --overlap -N1 -n1 -w $NODE619 --exclude=$NODE616 bash -c "
docker image inspect e1ab-sglang:1112-hipgem >/dev/null 2>&1 ||
    bash $K/mc/node_image.sh e1ab-sglang:1112 e1ab-sglang:1112-hipgem $K/out/nightly-build-1115T1019/main/pkg" > "$OUT/image.log" 2>&1
srun --jobid=$JOB --overlap -N1 -n1 -w $NODE619 --exclude=$NODE616 env IMAGE=e1ab-sglang:1112-hipgem \
    bash $K/bench/gem_node.sh "$OUT" "$P" > "$OUT/runner.log" 2>&1
cat "$OUT"/*.summary
