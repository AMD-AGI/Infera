#!/usr/bin/env bash
# B3 diagnosis: why SimpleCPUOffloadConnector reads back almost nothing under PP4 (aggregated tier_probe.py).
#   a. eager (lazy_offload false): stores at compute time, so a miss here points at store completion under PP;
#   b. lazy with DEBUG logging of vllm.v1.simple_kv_offload (store events submitted / completed).
# Then the DRAM-only (OffloadingConnector) P4D4 3600 s control.
R=${R:-/home/liyingli/bench_agentx/vllm-trackB-20261118}
cd "$R" || exit 1
rm -f /dev/shm/vllm_offload_*
CONNECTOR=SimpleCPUOffloadConnector LAZY=0 SPEC_NAME=CPUOffloadingSpec OVERLAY=0 CPU_BYTES=6000000000 \
    bash scripts/b2_agg_tier.sh simple-eager
CONNECTOR=SimpleCPUOffloadConnector LAZY=1 SPEC_NAME=CPUOffloadingSpec OVERLAY=0 CPU_BYTES=6000000000 \
    ENV_ADD="VLLM_LOGGING_LEVEL=DEBUG" bash scripts/b2_agg_tier.sh simple-lazy-debug
for t in simple-eager simple-lazy-debug; do
    m=runs/b2-agg-$t
    echo "$t: $(grep -vE '^#' $m/metrics.txt | grep -E '^vllm:external_prefix_cache_hits_total' | awk '{print "external hits " $2}') $(grep 'KV Transfer metrics' $m/engine.log | tail -1 | grep -oE 'simple_kv_offload_[a-z_]+=[0-9]+' | tr '\n' ' ')" | tee -a logs/summary.out
done
grep -E 'Store event [0-9]+ completed|store event|Lazy|lazy' runs/b2-agg-simple-lazy-debug/engine.log | grep -v aiter | head -40 > runs/b2-agg-simple-lazy-debug/store-events.txt
rm -f /dev/shm/vllm_offload_*
RUN_PREFIX=b1 DURATION=3600 bash scripts/p4d4_run.sh r3-3600 score
exit 0
