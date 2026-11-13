#!/bin/bash
# DRIVER619-20261113: GEM KV through Mooncake unchanged, with libhip_gem_alloc.so preloaded (answers hipMemGetAddressRange
# and hsa_amd_portable_export_dmabuf for its buffers, refuses ibv_reg_mr there). Pairs: 150 / 198 GiB over RDMA,
# 150 GiB with Mooncake's intra-node HIP transport left on, 40 GiB; on the 6.19 and the 6.16 node at once.
OUT=$K/out/gem-preload-$(date -u +%m%dT%H%M)
mkdir -p "$OUT"
P="0:gem:150:0:0:1 1:gem:198:0:0:1 2:gem:150:0:0:0 3:gem:40:0:0:1"
srun --jobid=$JOB --overlap -N1 -n1 -w $NODE616 bash $K/bench/gem_node.sh "$OUT/build" build > "$OUT/build.log" 2>&1
cat "$OUT/build.log"
for n in $NODE619 $NODE616; do
    srun --jobid=$JOB --overlap -N1 -n1 -w $n bash $K/bench/gem_node.sh "$OUT" "$P" > "$OUT/$n.runner.log" 2>&1 &
done
wait
cat "$OUT"/*.summary
