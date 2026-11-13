#!/bin/bash
# DRIVER619-20261113: RDMA write bandwidth A/B on one GPU pair and NIC (target GPU 1, initiator GPU 5, ionic_5), one case
# at a time, three rounds: KFD KV + peer memory (production path), KFD KV + dma-buf, GEM KV + dma-buf; 40 GiB so the
# KFD cases also start on 6.19. Both nodes in parallel (no shared NIC).
OUT=$K/out/rdma-bw-ab-$(date -u +%m%dT%H%M)
mkdir -p "$OUT"
for n in $NODE619 $NODE616; do
    (
        for r in 1 2 3; do
            for c in 1:kfd:40:1:0:1 1:kfd:40:0:0:1 1:gem:40:0:0:1; do
                srun --jobid=$JOB --overlap -N1 -n1 -w $n bash $K/bench/gem_node.sh "$OUT/r$r" "$c" >> "$OUT/$n.runner.log" 2>&1
            done
        done
    ) &
done
wait
grep -h -a '"step": "transferred"' "$OUT"/r*/*initiator.log | python3 -c "
import sys, json, collections
by = collections.defaultdict(list)
for l in sys.stdin:
    d = json.loads(l)
    by[d['tag']].append(d['gb_s'])
for k, v in sorted(by.items()):
    print(k, v)" | tee "$OUT/summary.txt"
