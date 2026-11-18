#!/usr/bin/env bash
# B3: vLLM SimpleCPUOffloadConnector with lazy_offload (stores GPU blocks as they near eviction, like sglang HiCache
# write_back) in place of OffloadingConnector (write-through). After the DRAM-only 3600 s control (WAIT_RC):
#   1. aggregated PP4 read-back check (tier_probe.py, small GPU cache, 6 GB CPU tier);
#   2. only if it passes 9/9 with no fault: P4D4 = b1-r1 configuration with the offload connector swapped, gate then
#      AgentX C104 3600 s.
R=${R:-/home/liyingli/bench_agentx/vllm-trackB-20261118}
cd "$R" || exit 1
Q=/home/liyingli/bench_agentx/vllm-pd-20261117/queue-n2
if [[ -n "${WAIT_RC:-}" ]]; then
    while [ ! -e "$Q/$WAIT_RC.rc" ]; do sleep 20; ls "$Q" > /dev/null; done
    sleep 30
fi
rm -f /dev/shm/vllm_offload_*
CONNECTOR=SimpleCPUOffloadConnector SPEC_NAME=CPUOffloadingSpec OVERLAY=0 CPU_BYTES=6000000000 \
    bash scripts/b2_agg_tier.sh simple-lazy
if ! grep -q 'up=0 probe=9/9 passed faults=0' runs/b2-agg-simple-lazy/config.txt; then
    echo "$(date -u +%T) b3: aggregated simple-lazy check failed, P4D4 skipped" | tee -a logs/summary.out
    exit 0
fi
OFF_CONNECTOR=SimpleCPUOffloadConnector OFF_EXTRA='{"cpu_bytes_to_use":1250000000000,"lazy_offload":true}' \
    OFF_BACKEND_NAME=vllm-simple RUN_PREFIX=b3 DURATION=3600 bash scripts/p4d4_run2.sh r1-simple-lazy gate score
exit 0
