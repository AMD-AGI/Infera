#!/usr/bin/env bash
# 10-31 E11 (decode SGLANG_USE_ROCM700A=0 -> MoE combine via aiter reduce-scatter) on 138 -> 140:
#   1. quick correctness with real acceptance (smoke, 200-prompt probe, long context 16k x 64), packet capture off;
#   2. if no garbling (probe >= 0.97, garbled 0, long-context needle >= 60/64): E11 C104 scoring + analyze_run.sh.
set -uo pipefail
L="$1"
P=/home/liyingli/bench_agentx/baseline/Infera/llying
C=$P/prefill-opt-20261027/config
S=$P/prefill-opt-20261027/scripts
R=$P/prefill-opt-20261027/results
KIT=/home/liyingli/bench_agentx/baseline/2p1d-kit/llr35
bash $S/disk_check.sh crsuse2-m2m-140 95 "before E11 check" || exit 0
(cd "$KIT" && LONGCTX_SPECS="16000:64" bash $S/quick_correct.sh "$KIT" $L/e11-check $C/site-140-e11-check.env)
echo "$(date -u +%T) night29: E11 check done"
touch $L/e11-check/CHECK_DONE
# the gate is decided by the operator (Claude) reading the check log; it writes GO or STOP
until [[ -e $L/e11-check/GO || -e $L/e11-check/STOP ]]; do sleep 30; done
[[ -e $L/e11-check/STOP ]] && { echo "$(date -u +%T) night29: E11 stopped after the check"; echo "$(date -u +%T) night29 done"; exit 0; }
LS=llr37-140-e11
for k in 1 2; do
    out="$(cd "$KIT" && bash $P/prefill-car-20261023/scripts/perf_plan.sh "$KIT" "$L/perf" $C/plan-140-e11.txt 2>&1)"
    echo "$out"
    grep -q "up failed" <<<"$out" || break
    mv "$L/perf/$LS/c104.log" "$L/perf/$LS/c104.log.failed$k"
done
(cd $P/prefill-opt-20261027 && bash scripts/analyze_run.sh $LS 104 > $R/run-$LS-c104.txt 2>&1)
grep -E "^\| C104|^pp[0-3]:" $R/run-$LS-c104.txt
bash $S/disk_check.sh crsuse2-m2m-140 95 "after E11"
echo "$(date -u +%T) night29 done"
