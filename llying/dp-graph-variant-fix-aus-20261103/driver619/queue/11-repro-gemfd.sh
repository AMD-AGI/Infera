#!/bin/bash
# DRIVER619-20261113: GEM cases again, data MRs registered with the GEM BO's own dma-buf fd (hsa export of imported
# memory segfaulted in 10-repro), on both nodes.
OUT=$K/out/repro-gemfd-$(date -u +%m%dT%H%M)
mkdir -p "$OUT"
C="0:gem:dmabuf:40 1:gem:dmabuf:150 2:gem:dmabuf:200 3:gem:dmabuf:150:all"
for n in $NODE619 $NODE616; do
    srun --jobid=$JOB --overlap -N1 -n1 -w $n bash $K/bench/repro_node.sh "$OUT" "$C" > "$OUT/$n.runner.log" 2>&1 &
done
wait
python3 $K/bench/summarize.py "$OUT" | tee "$OUT/summary.md"
