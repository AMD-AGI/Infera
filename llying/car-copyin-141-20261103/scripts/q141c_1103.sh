#!/usr/bin/env bash
# 11-03 third queue on crsuse2-m2m-141:
#   1. E11 without copy-in, decode graphs pinned to the sparse variant (HiCache off, 200 prompts + 16k x 32);
#   2. E10 + all-reduce capture copy-in with HiCache on (600 prompts + 16k x 64): the late-generation garble seen in the
#      HiCache-on control (3 of 387).
# Usage (on 141): q141c_1103.sh > $L/q141c-1103.out 2>&1
set -uo pipefail
E=/home/liyingli/bench_agentx/baseline/Infera/llying/car-copyin-141-20261103
QC=/home/liyingli/bench_agentx/baseline/Infera/llying/prefill-opt-20261027/scripts/quick_correct.sh
KIT=/home/liyingli/bench_agentx/baseline/2p1d-kit/llr35-141
L=/home/liyingli/bench_agentx/llr141/logs
G=/home/liyingli/bench_agentx/llr141/data/gsm8k_samples_1319.jsonl
OUT=$L/q141c-1103.out
qc() {
    local lab k n0; lab="$(basename "$1" .env)"
    for k in 1 2; do
        n0=$(grep -ac "$lab-[0-9]\{4\}T[0-9]\{4\} up failed" "$OUT")
        PROBE_PROMPTS=$G bash "$QC" "$KIT" "$L/gate" "$@"
        [[ $(grep -ac "$lab-[0-9]\{4\}T[0-9]\{4\} up failed" "$OUT") == "$n0" ]] && return 0
        echo "$(date -u +%T) $lab failed to start (attempt $k)"
    done
    echo "$(date -u +%T) $lab failed twice: queue stopped"; exit 1
}
PROBE_N=200 LONGCTX_SPECS="16000:32" qc "$E/config/site-141-e11-sparse.env"
PROBE_N=600 LONGCTX_SPECS="16000:64" qc "$E/config/site-141-arcopy-hc.env"
echo "$(date -u +%T) q141c done"
