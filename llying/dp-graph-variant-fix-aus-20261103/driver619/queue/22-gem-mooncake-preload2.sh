#!/bin/bash
# DRIVER619-20261113: 21 again with hipIpcGetMemHandle answered for GEM buffers (Mooncake's HIP transport registration
# failed the whole batch_register_memory in 21) and a GPU copy / read bandwidth check of GEM vs default buffers; the
# production-path KFD pair as transfer reference.
OUT=$K/out/gem-preload2-$(date -u +%m%dT%H%M)
mkdir -p "$OUT"
P="0:gem:150:0:0:1 1:gem:198:0:0:1 2:kfd:150:1:0:1 3:gem:40:0:0:1"
srun --jobid=$JOB --overlap -N1 -n1 -w $NODE616 bash $K/bench/gem_node.sh "$OUT/build" build > "$OUT/build.log" 2>&1
cat "$OUT/build.log"
for n in $NODE619 $NODE616; do
    srun --jobid=$JOB --overlap -N1 -n1 -w $n bash $K/bench/gem_node.sh "$OUT" "$P" > "$OUT/$n.runner.log" 2>&1 &
done
wait
cat "$OUT"/*.summary
