#!/usr/bin/env bash
# Tuning trial r3 (config/round.r3.sh: --disaggregation-decode-allocation-policy
# prefill_complete on both roles). Waits for the r2 follow-up sweep to finish, stops the
# r2 stack, waits for the GPUs to drain, then runs the full pipeline (bring-up at real
# acceptance, two-tier known-answer gate, switch to simulated acceptance, AgentX points).
# Run detached on crsuse2-m2m-137.
set -uo pipefail
R=/home/liyingli/bench_agentx/baseline/Infera/llying/3p1d-tp4-tp8-v0521-20261012
POINTS="${POINTS:-144 160}"
echo "$(date -u +%FT%TZ) waiting for followup_r2.sh to finish"
while pgrep -f "operations/followup_r2.sh" >/dev/null; do sleep 60; done
echo "$(date -u +%FT%TZ) r2 sweep finished; stopping r2"
(cd "$R/scripts/bench-harness" && bash stop.sh "CONFIG=$R/config/config.3p1d.sh" ROUND=r2) \
    > "$R/runtime/runs/r2/stop.log" 2>&1
printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" stop "r2 stopped for r3 (rc=$?)" >> "$R/runtime/runs/r2/status.tsv"
declare -A GPUS=([crsuse2-m2m-135]=0,4,5,6 [crsuse2-m2m-137]=0,1,2,3,4,5,6,7 [crsuse2-m2m-138]=0,1,2,3,4,5,6,7)
for i in $(seq 1 360); do
    busy=""
    for node in "${!GPUS[@]}"; do
        used="$(ssh -o BatchMode=yes -o ConnectTimeout=10 -n "$node" "rocm-smi --showmeminfo vram --json" 2>/dev/null |
            python3 -c '
import json,sys
d=json.load(sys.stdin)
print(max(int(d["card"+g]["VRAM Total Used Memory (B)"])//1048576 for g in sys.argv[1].split(",")))' "${GPUS[$node]}")"
        [[ -n "$used" && "$used" -lt 1024 ]] || busy+=" $node:${used:-?}MiB"
    done
    [[ -z "$busy" ]] && break
    (( i % 6 == 1 )) && echo "$(date -u +%FT%TZ) waiting for VRAM drain:$busy"
    (( i < 360 )) || { echo "$(date -u +%FT%TZ) VRAM did not drain in 60 min:$busy"; exit 1; }
    sleep 10
done
echo "$(date -u +%FT%TZ) GPUs drained; starting r3 POINTS='$POINTS'"
ROUND=r3 POINTS="$POINTS" bash "$R/scripts/run_3p1d.sh" all
