#!/usr/bin/env bash
# 11-03 on crsuse2-m2m-141 (same-node PD, E10 recipe, real acceptance):
#   1. control (custom all-reduce on, image behaviour), HiCache off; if its bad start is < 3, repeat the control with HiCache on
#      and run step 2 with HiCache on as well;
#   2. decode aiter all-reduce capture copy-in;
#   3. E11 (SGLANG_USE_ROCM700A=0) control, smoke + 200-prompt probe;
#   4. E11 + capture copy-in for all-gather / reduce-scatter (SGLANG_AITER_CAPTURE_COPY_IN=1) and all-reduce.
# Usage (on 141): q141_1103.sh  > $L/q141-1103.out 2>&1
set -uo pipefail
E=/home/liyingli/bench_agentx/baseline/Infera/llying/car-copyin-141-20261103
QC=/home/liyingli/bench_agentx/baseline/Infera/llying/prefill-opt-20261027/scripts/quick_correct.sh
KIT=/home/liyingli/bench_agentx/baseline/2p1d-kit/llr35-141
L=/home/liyingli/bench_agentx/llr141/logs
G=/home/liyingli/bench_agentx/llr141/data/gsm8k_samples_1319.jsonl
OUT=$L/q141-1103.out
mkdir -p "$L/gate"

until grep -q "^rc=0" "$L/image-xfer.out" 2>/dev/null; do
    grep -q "^rc=[1-9]" "$L/image-xfer.out" 2>/dev/null && { echo "$(date -u +%T) image load failed"; exit 1; }
    sleep 30
done
echo "$(date -u +%T) image ready: $(docker image inspect -f '{{.Id}}' infera-sglang:v0522-fdbafb60)"

# one retry when the stack does not come up (seen: Mooncake ibv_reg_mr 'Invalid argument' on one decode buffer, transient on 140)
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
# bad_start count of the latest probe line of a label prefix (label = site basename + -MMDDTHHMM)
bs() { grep -a "$1-[0-9]\{4\}T[0-9]\{4\} probe:" "$OUT" | tail -1 | grep -o 'bad_start= *[0-9]*' | grep -o '[0-9]*$'; }

PROBE_N=400 LONGCTX_SPECS="16000:64" qc "$E/config/site-141-ctl.env"
n=$(bs site-141-ctl)
sfx=
if [[ -z "$n" || "$n" -lt 3 ]]; then
    echo "$(date -u +%T) control (HiCache off) bad_start=${n:-none} < 3: repeat with HiCache on"
    PROBE_N=400 LONGCTX_SPECS="16000:64" qc "$E/config/site-141-ctl-hc.env"
    sfx=-hc
    echo "$(date -u +%T) control (HiCache on) bad_start=$(bs site-141-ctl-hc)"
fi
PROBE_N=600 LONGCTX_SPECS="16000:64" qc "$E/config/site-141-arcopy$sfx.env"
echo "$(date -u +%T) all-reduce copy-in$sfx bad_start=$(bs site-141-arcopy$sfx)"

PROBE_N=200 LONGCTX_SPECS="" qc "$E/config/site-141-e11.env"
PROBE_N=600 LONGCTX_SPECS="16000:64" qc "$E/config/site-141-e11-copyin.env"
echo "$(date -u +%T) q141 done"
