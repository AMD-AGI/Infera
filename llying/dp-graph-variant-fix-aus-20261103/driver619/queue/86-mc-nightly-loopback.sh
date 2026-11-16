#!/bin/bash
# DRIVER619-20261113 formal delivery, nightly only: mc_gem_test.py in e1ab-sglang:1112-hipgem (Mooncake main + the PR, from
# 70) on the 6.19 node, with Mooncake's HIP transport on (same-host transfers to GEM buffers must route to RDMA) and
# MOONCAKE_DISABLE_HIP_DMABUF=1 (GEM buffers must still go through dma-buf). Pairs: 150 / 198 GiB HipGemAllocator KV;
# KFD KV with peer memory (the production path), and KFD KV with dma-buf and main's KFD pin anchor (#4657) on.
OUT=$K/out/mc-nightly-$(date -u +%m%dT%H%M)
mkdir -p "$OUT"
P="0:mcgem:150:1:0:0 1:mcgem:198:1:0:0 2:kfd:150:1:0:0 3:kfd:150:0:1:0"
srun --jobid=$JOB --overlap -N1 -n1 -w $NODE619 --exclude=$NODE616 env IMAGE=e1ab-sglang:1112-hipgem \
    bash $K/bench/gem_node.sh "$OUT" "$P" > "$OUT/runner.log" 2>&1
cat "$OUT"/*.summary
