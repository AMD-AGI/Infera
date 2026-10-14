#!/usr/bin/env bash
# Replaces start_r3.sh (2026-10-13 01:35): C128 (19,758) beat C144 (18,001), so the r2
# peak is at or below 128 and followup_r2.sh is running C112. Then:
#   1. if C112 > C128, run C96 on the same r2 stack (bracket the baseline peak first);
#   2. B = best measured r2 concurrency; stop r2, wait for the GPUs to drain, and run the
#      r3 tuning trial (round.r3.sh: decode allocation policy prefill_complete) at B and
#      B+16 with the full pipeline (real-acceptance gate, switch, AgentX).
# Run detached on crsuse2-m2m-137.
set -uo pipefail
R=/home/liyingli/bench_agentx/baseline/Infera/llying/3p1d-tp4-tp8-v0521-20261012
tput() {
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["request_metrics"]["throughput"]["per_gpu"]["total_tput_tps"])' \
        "$R/results/r2/c$1/agentx_conc$1.json" 2>/dev/null || echo 0
}
gt() { python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)' "$1" "$2"; }
echo "$(date -u +%FT%TZ) waiting for followup_r2.sh to finish"
while pgrep -f "operations/followup_r2.sh" >/dev/null; do sleep 60; done
c112="$(tput 112)"; c128="$(tput 128)"
echo "$(date -u +%FT%TZ) followup done: C112=$c112 C128=$c128"
if gt "$c112" "$c128"; then
    echo "$(date -u +%FT%TZ) C112 > C128: running C96 on r2"
    ROUND=r2 POINTS=96 bash "$R/scripts/run_3p1d.sh" bench || echo "C96 failed"
fi
best=0; best_tput=0
for f in "$R"/results/r2/c*/agentx_conc*.json; do
    c="${f##*/agentx_conc}"; c="${c%.json}"; t="$(tput "$c")"
    gt "$t" "$best_tput" && { best="$c"; best_tput="$t"; }
done
POINTS="$best $((best + 16))"
echo "$(date -u +%FT%TZ) r2 best C$best ($best_tput tok/s/GPU); r3 POINTS='$POINTS'"
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
