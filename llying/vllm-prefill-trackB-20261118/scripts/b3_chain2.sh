#!/usr/bin/env bash
# B3 (second try): SimpleCPUOffloadConnector lazy_offload with vLLM #66708 (lazy walk stores every block before the GPU
# evicts it, CPU copies of blocks loaded back to the GPU go first) bind-mounted over vllm/v1/simple_kv_offload/manager.py.
# The unpatched lazy mode read back only 16,704 of ~94k repeated tokens in the aggregated check (eager: all).
#   1. aggregated PP4 read-back check; 2. if 9/9 and no fault: P4D4 gate + AgentX C104 3600 s;
#   3. then the DRAM-only (OffloadingConnector, write-through) 3600 s control.
R=${R:-/home/liyingli/bench_agentx/vllm-trackB-20261118}
cd "$R" || exit 1
V=/usr/local/lib/python3.12/dist-packages/vllm
M=$R/overlay-simple/vllm/v1/simple_kv_offload/manager.py:$V/v1/simple_kv_offload/manager.py:ro
rm -f /dev/shm/vllm_offload_*
CONNECTOR=SimpleCPUOffloadConnector SPEC_NAME=CPUOffloadingSpec OVERLAY=0 CPU_BYTES=6000000000 MOUNT_ADD="$M" \
    bash scripts/b2_agg_tier.sh simple-lazy-66708
if grep -q 'up=0 probe=9/9 passed faults=0' runs/b2-agg-simple-lazy-66708/config.txt; then
    OFF_CONNECTOR=SimpleCPUOffloadConnector OFF_EXTRA='{"cpu_bytes_to_use":1250000000000,"lazy_offload":true}' \
        OFF_BACKEND_NAME=vllm-simple P_MOUNT_ADD="$M" RUN_PREFIX=b3 DURATION=3600 \
        bash scripts/p4d4_run2.sh r2-simple-lazy-66708 gate score
else
    echo "$(date -u +%T) b3: aggregated simple-lazy-66708 check failed, P4D4 skipped" | tee -a logs/summary.out
fi
rm -f /dev/shm/vllm_offload_*
RUN_PREFIX=b1 DURATION=3600 bash scripts/p4d4_run.sh r3-3600 score
exit 0
