#!/bin/bash
# DRIVER619-20261113 section 6 step 2: the production crash in miniature (KFD KV + Mooncake + new stream) and the
# GEM-backed KV fix, with an RDMA loopback write verified, on the 6.19 and the 6.16 node at once.
OUT=$K/out/gem-$(date -u +%m%dT%H%M)
mkdir -p "$OUT"
for n in $NODE619 $NODE616; do
    srun --jobid=$JOB --overlap -N1 -n1 -w $n bash $K/bench/gem_node.sh "$OUT" > "$OUT/$n.runner.log" 2>&1 &
done
wait
cat "$OUT"/*.summary
