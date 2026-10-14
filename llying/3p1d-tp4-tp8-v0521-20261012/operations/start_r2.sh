#!/usr/bin/env bash
# r2 = r1 with SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=0 (config default since 2026-10-12 19:00).
# Wait for the GPUs r1 used to drain (HiCache stacks release VRAM slowly), then run the
# whole pipeline. Run detached on crsuse2-m2m-137.
set -uo pipefail
R=/home/liyingli/bench_agentx/baseline/Infera/llying/3p1d-tp4-tp8-v0521-20261012
declare -A GPUS=([crsuse2-m2m-135]=0,4,5,6 [crsuse2-m2m-137]=0,1,2,3,4,5,6,7 [crsuse2-m2m-138]=0,1,2,3,4,5,6,7)
mkdir -p "$R/runtime/runs/r2"
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
echo "$(date -u +%FT%TZ) GPUs drained; starting r2"
ROUND=r2 POINTS="144 160 176" bash "$R/scripts/run_3p1d.sh" all
