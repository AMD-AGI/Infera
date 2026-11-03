#!/usr/bin/env bash
# 11-03 second queue on crsuse2-m2m-141, after q141_1103.sh was stopped: the bad start does not reproduce on the same-node
# deployment (HiCache off 0 / 400, HiCache on 0 / 387), so the all-reduce copy-in cannot be judged here. E11 garbles on its
# own (140: probe 0.670, 44 garbled), so it can: E11 control (200 prompts), then E11 + capture copy-in for all-gather /
# reduce-scatter and all-reduce (600 prompts + 16k x 64), both with HiCache off.
# Usage (on 141): q141b_1103.sh > $L/q141b-1103.out 2>&1
set -uo pipefail
E=/home/liyingli/bench_agentx/baseline/Infera/llying/car-copyin-141-20261103
QC=/home/liyingli/bench_agentx/baseline/Infera/llying/prefill-opt-20261027/scripts/quick_correct.sh
KIT=/home/liyingli/bench_agentx/baseline/2p1d-kit/llr35-141
L=/home/liyingli/bench_agentx/llr141/logs
G=/home/liyingli/bench_agentx/llr141/data/gsm8k_samples_1319.jsonl
OUT=$L/q141b-1103.out
while pgrep -f "prefill-opt-20261027/scripts/quick_correc[t].sh" >/dev/null; do sleep 30; done
echo "$(date -u +%T) previous deployment finished"

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
PROBE_N=200 LONGCTX_SPECS="" qc "$E/config/site-141-e11.env"
PROBE_N=600 LONGCTX_SPECS="16000:64" qc "$E/config/site-141-e11-copyin.env"
echo "$(date -u +%T) q141b done"
