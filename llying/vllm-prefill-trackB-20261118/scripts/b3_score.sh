#!/usr/bin/env bash
# B3: P4D4 AgentX C104 3600 s with vLLM 0.34.0's SimpleCPUOffloadConnector in lazy mode (write-back: GPU blocks are
# stored as they near eviction), otherwise the b1-r1 configuration (prefill GEM 0.88, 1.25 TB CPU tier). No separate
# gate: the aggregated read-back check passed 9/9 three times; the score's smoke still runs. Waits for WAIT_RC.
R=${R:-/home/liyingli/bench_agentx/vllm-trackB-20261118}
cd "$R" || exit 1
Q=/home/liyingli/bench_agentx/vllm-pd-20261117/queue-n2
if [[ -n "${WAIT_RC:-}" ]]; then
    while [ ! -e "$Q/$WAIT_RC.rc" ]; do sleep 20; ls "$Q" > /dev/null; done
    sleep 30
fi
rm -f /dev/shm/vllm_offload_*
OFF_CONNECTOR=SimpleCPUOffloadConnector OFF_EXTRA='{"cpu_bytes_to_use":1250000000000,"lazy_offload":true}' \
    OFF_BACKEND_NAME=vllm-simple RUN_PREFIX=b3 DURATION=3600 bash scripts/p4d4_run2.sh r3-simple-lazy score
exit 0
