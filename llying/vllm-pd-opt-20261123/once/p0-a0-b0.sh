# Phase 0 in one job (one of the two per-user job slots is taken by another session): 0a (instrumented P4D4, about
# 2 h 10 min) then 0b (decode-only harness, two rounds, about 1 h 30 min). Run by run_once.sbatch (ONCE=p0-a0-b0).
R=${R:-/home/liyingli/bench_agentx/vllm-pd-20261117}   # run_once.sbatch does not export R
cd "$R" || exit 0
# Job 306058 (11-23) ran with R unset: docker created these empty source directories on v3-186's root file system.
if [ -e /overlay-cpupin ] || [ -e /runs/p4d4-a0-instr-m10cfg-3600 ]; then
    docker run --rm -v /:/host --network none --entrypoint sh busybox:1.36 -c \
        'rm -rf /host/overlay-cpupin /host/runs/p4d4-a0-instr-m10cfg-3600; rmdir /host/runs 2> /dev/null; ls -d /host/overlay-cpupin /host/runs 2>&1' \
        > "logs/once-$JOB-cleanup.log" 2>&1
fi
bash once/p4d4-a0-instr-3600.sh > "logs/once-$JOB-a0.log" 2>&1
echo "$(date -u +%T) a0 rc=$?"
docker ps -a --format '{{.Names}}' | grep -E '^vpd-' | xargs -r docker rm -f > /dev/null 2>&1
bash once/db-0b-aa-mnbt.sh > "logs/once-$JOB-b0.log" 2>&1
echo "$(date -u +%T) b0 rc=$?"
