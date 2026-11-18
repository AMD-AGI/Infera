#!/usr/bin/env bash
# Host-memory guard for one P4D4 round: every 15 s, if MemAvailable falls below MIN_GB, stop the engines gracefully
# (docker stop, SIGTERM) instead of letting the OOM killer take a PP worker (11-18 m8b: 1.6 TB CPU tier through the
# pinned allocator reached 2.6 TB and was OOM-killed). Ends when the run's stop file appears.
#   memguard.sh RUN_DIR_NAME [MIN_GB=120]
R=${R:-/home/liyingli/bench_agentx/vllm-pd-20261117}
run=$1 min=${2:-120}
stop=$R/runs/$run/samples.jsonl.stop
log=$R/runs/$run/memguard.log
until [ -d "$R/runs/$run" ]; do sleep 10; done
while [ ! -e "$stop" ]; do
    avail=$(awk '/^MemAvailable:/ {print int($2 / 1048576)}' /proc/meminfo)
    shm=$(awk '/^Shmem:/ {print int($2 / 1048576)}' /proc/meminfo)
    anon=$(awk '/^AnonPages:/ {print int($2 / 1048576)}' /proc/meminfo)
    echo "$(date -u +%T) avail ${avail} GB shmem ${shm} GB anon ${anon} GB" >> "$log"
    if (( avail < min )); then
        echo "$(date -u +%T) MemAvailable ${avail} GB < ${min} GB: stopping the engines" | tee -a "$log"
        docker stop -t 90 vpd-prefill vpd-decode > /dev/null 2>&1
        echo "$(date -u +%m%d-%T) memguard $run: engines stopped at MemAvailable ${avail} GB" >> "$R/logs/summary.out"
        exit 0
    fi
    sleep 15
done
exit 0
