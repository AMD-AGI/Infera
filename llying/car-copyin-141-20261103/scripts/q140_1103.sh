#!/usr/bin/env bash
# Cross-node follow-up on 138 -> 140 (NOT started: 138 / 140 run another session's night35_1103.sh on 11-03; start only
# after it has finished and the user agrees). Capped-pool gates: 200 prompts + two 16k x 128 passes each; the E10 control
# with CAR on had bad start 3-8 / 200 in 6 of 8 gates.
#   1. E10 + all-reduce capture copy-in (CAR on) x 2      -> does the copy-in remove the bad start?
#   2. E10 + decode graphs pinned to sparse (CAR on) x 2  -> is the bad start the per-rank graph mismatch?
#   3. E11 + pinned sparse x 1                             -> E11 correct cross-node under eviction?
#   4. C104 scoring: E10 + pinned sparse (prediction >= 51,900: no per-step .item() sync), then E11 + pinned sparse
#      (prediction >= 52,400); same pair E10 = 51,374.
# Usage (on 138): q140_1103.sh LOG_ROOT
set -uo pipefail
L="$1"
P=/home/liyingli/bench_agentx/baseline/Infera/llying
E=$P/car-copyin-141-20261103
S=$P/prefill-opt-20261027/scripts
R=$P/prefill-opt-20261027/results
KIT=/home/liyingli/bench_agentx/baseline/2p1d-kit/llr35
G=/mnt/m2m_nobackup/liyingli_3p1d/mix-v0522/2p1d/work/runs/correct-v2/gsm8k/samples_gsm8k_2026-10-20T16-13-32.558577.jsonl
while pgrep -f "night35_110[3]" >/dev/null; do sleep 60; done
mkdir -p "$L/gate"
gate() {  # SITE LABEL
    bash $S/disk_check.sh crsuse2-m2m-140 95 "before $2" || exit 0
    local g="$L/gate/full-$2-$(date -u +%m%dT%H%M).out"
    PROBE_N=200 PROBE_PROMPTS=$G LONGCTX_SPECS="16000:128 16000:128" bash $S/quick_correct.sh "$KIT" "$L/gate" "$1" >"$g" 2>&1
    grep -aE "smoke|probe:|longctx" "$g" | cut -c1-250
    echo "$(date -u +%T) $2 verdict $(python3 $S/gate_pass2.py "$g" "$(basename "$1" .env)" --passes 2 2>&1 | tail -1)"
}
score() {  # LABEL SITE
    local LS="$1" k out
    echo "$LS $2 104" > $E/config/plan-$LS.txt
    for k in 1 2; do
        out="$(cd "$KIT" && bash $P/prefill-car-20261023/scripts/perf_plan.sh "$KIT" "$L/perf" $E/config/plan-$LS.txt 2>&1)"
        echo "$out" | tail -3
        grep -q "up failed" <<<"$out" || break
        mv "$L/perf/$LS/c104.log" "$L/perf/$LS/c104.log.failed$k" 2>/dev/null
    done
    (cd $P/prefill-opt-20261027 && bash scripts/analyze_run.sh $LS 104 > $R/run-$LS-c104.txt 2>&1)
    grep -E "^\| C104" $R/run-$LS-c104.txt
}
for k in 1 2; do gate $E/config/site-140-arcopy-gate.env arcopy-gate-r$k; done
for k in 1 2; do gate $E/config/site-140-sparse-gate.env sparse-gate-r$k; done
gate $E/config/site-140-e11-sparse-gate.env e11-sparse-gate
score llr37-140-e10-sparse $E/config/site-140-e10-sparse.env
score llr37-140-e11-sparse $E/config/site-140-e11-sparse.env
echo "$(date -u +%T) q140 done"
