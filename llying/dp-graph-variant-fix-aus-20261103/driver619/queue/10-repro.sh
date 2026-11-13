#!/bin/bash
# DRIVER619-20261113 section 6 step 1 (+ step 2 GEM cases): kfd_pin_repro.py matrix on the 6.19 and the 6.16 node at once.
OUT=$K/out/repro-$(date -u +%m%dT%H%M)
mkdir -p "$OUT"
for n in $NODE619 $NODE616; do
    srun --jobid=$JOB --overlap -N1 -n1 -w $n bash $K/bench/repro_node.sh "$OUT" > "$OUT/$n.runner.log" 2>&1 &
done
wait
python3 $K/bench/summarize.py "$OUT" | tee "$OUT/summary.md"
