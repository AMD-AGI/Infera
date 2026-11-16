#!/bin/bash
# DRIVER619-20261113 formal delivery (61 again, after 80): mc_gem_test.py with mooncake.allocator.HipGemAllocator and the patched Mooncake
# (no preload), under the production settings: MOONCAKE_DISABLE_HIP_DMABUF=1 and Mooncake's HIP transport on.
# Pairs: 150 / 198 / 40 GiB GEM KV, and the production KFD KV control (segfaults on 6.19), on both nodes.
OUT=$K/out/mc-native-$(date -u +%m%dT%H%M)
mkdir -p "$OUT"
on() { local n=$1; shift; local x=$NODE616; [ "$n" = "$NODE616" ] && x=$NODE619; srun --jobid=$JOB --overlap -N1 -n1 -w $n --exclude=$x "$@"; }
P="0:mcgem:150:1:0:0 1:mcgem:198:1:0:0 2:kfd:150:1:0:0 3:mcgem:40:1:0:0"
for n in $NODE619 $NODE616; do
    on $n env IMAGE=infera-sglang:v0522-fdbafb60-hipgem bash $K/bench/gem_node.sh "$OUT" "$P" > "$OUT/$n.runner.log" 2>&1 &
done
wait
cat "$OUT"/*.summary
