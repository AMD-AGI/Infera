#!/usr/bin/env bash
# Second node, after the DRAM-only control: P4D4 AgentX C104 3600 s with SimpleCPUOffloadConnector lazy_offload plus
# vLLM #66708 (the lazy walk also runs when a CPU hit allocates its GPU blocks, sized by the load; CPU copies of blocks
# loaded back go first) bind-mounted over vllm/v1/simple_kv_offload/manager.py. In b3-r4 (unpatched lazy) only 12% of
# the tokens that missed the GPU were found on the CPU tier: a load allocates far more blocks than the per-step walk
# covers, and those blocks leave the GPU without a CPU copy.
R=/home/liyingli/bench_agentx/vllm-trackB-n4-20261118
export R
cd "$R" || exit 1
Q=/home/liyingli/bench_agentx/vllm-trackB-20261118/queue-b
while [ ! -e "$Q/${WAIT_RC:?}.rc" ]; do sleep 20; ls "$Q" > /dev/null; done
sleep 30
V=/usr/local/lib/python3.12/dist-packages/vllm
OFF_CONNECTOR=SimpleCPUOffloadConnector OFF_EXTRA='{"cpu_bytes_to_use":1250000000000,"lazy_offload":true,"num_cpu_blocks":389120}' \
    OFF_BACKEND_NAME=vllm-simple P_MOUNT_ADD="$R/overlay-simple/vllm/v1/simple_kv_offload/manager.py:$V/v1/simple_kv_offload/manager.py:ro" \
    RUN_PREFIX=b3 DURATION=3600 bash scripts/p4d4_run2.sh r5-simple-lazy-66708-n4 score
exit 0
