#!/usr/bin/env bash
# Aggregated GLM-5.2 TP4 + DP4-attention variants one after another on THIS node's GPUs (default 0-3), each through
# ../../mix-v0522-correctness-20261019/scripts/engine_bisect.sh (etcd + engine, 200 GSM8K prompts at 64 in flight
# straight to the engine, garbled / runaway counts), waiting for those GPUs' VRAM to drain between variants.
# A variant is "NAME [KEY=VALUE ...]": engine_bisect.sh switches (MTP=0, HICACHE=0, ...) or EXTRA_ENV / EXTRA_ARGS.
# Usage: car_bisect_chain.sh SITE_ENV "NAME KEY=VALUE ..." ...
set -uo pipefail
site="$1"
shift
B="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../mix-v0522-correctness-20261019/scripts" && pwd)/engine_bisect.sh"
OUT=/mnt/m2m_nobackup/liyingli_3p1d/mix-v0522/bisect
export SITE="$site" GPUS="${GPUS:-0,1,2,3}" PID_OFFSET="${PID_OFFSET:-30000}"
busy() {
    rocm-smi --showmeminfo vram --csv | awk -F, -v g="$GPUS" 'BEGIN {split(g, a, ","); for (i in a) w["card" a[i]] = 1}
        NR > 1 && ($1 in w) && $2 != "" && 100 * $3 / $2 >= 1 {print $1}'
}
for v in "$@"; do
    read -ra kv <<<"$v"
    name="${kv[0]}"
    until [[ -z "$(busy)" ]]; do sleep 30; done
    echo "$(date -u +%T) $name: ${kv[*]:1}"
    # Variables with spaces (EXTRA_ENV="A=1 B=0") travel as KEY=VALUE words separated by '|'.
    args=()
    for a in "${kv[@]:1}"; do args+=("${a//|/ }"); done
    timeout 5400 bash "$B" "$name" "${args[@]}" >"$OUT/$name.log" 2>&1
    echo "$(date -u +%T) $name exited $?: $(cat "$OUT/$name/runs/$name/verdict.txt" 2>/dev/null)"
    docker rm -f "bisect_$name-engine" "bisect_$name-etcd" >/dev/null 2>&1
done
echo "$(date -u +%T) chain done"
