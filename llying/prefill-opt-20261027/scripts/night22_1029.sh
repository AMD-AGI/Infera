#!/usr/bin/env bash
# 10-29 queue on 138 -> 140: E9 (background pre-write-back, on E1 + E10), stopping at the first failed check.
#   1. 20-min short run (C104, DURATION=1200): no crash / divergence assert, and the mechanism -- each PP rank's mean
#      eviction time at least 60 ms below E10's (358 / 195 / 259 / 225 ms) -- else stop;
#   2. capped-pool gate (gate_pass2.py PASS), else stop;
#   3. AgentX C104 scoring, analyze_run.sh, beat_vs_evict.py.
# A start-up that fails (decode Mooncake ibv_reg_mr EINVAL flake, 10-29) is retried once.
# Usage: night22_1029.sh LOG_ROOT
set -uo pipefail
L="$1"
P=/home/liyingli/bench_agentx/baseline/Infera/llying
C=$P/prefill-opt-20261027/config
S=$P/prefill-opt-20261027/scripts
KIT=/home/liyingli/bench_agentx/baseline/2p1d-kit/llr35
G=/mnt/m2m_nobackup/liyingli_3p1d/mix-v0522/2p1d/work/runs/correct-v2/gsm8k/samples_gsm8k_2026-10-20T16-13-32.558577.jsonl
PERF="bash $P/prefill-car-20261023/scripts/perf_plan.sh"
cd "$KIT" || exit 1
perf() {  # LABEL PLAN: one plan line, retried once on a failed start-up
    local label="$1" plan="$2" k
    for k in 1 2; do
        $PERF "$KIT" "$L/perf" "$plan"
        grep -q "up failed" "$L/perf/$label/c104.log" || return 0
        echo "$(date -u +%T) night22: $label start-up failed (attempt $k)"
        mv "$L/perf/$label/c104.log" "$L/perf/$label/c104.log.failed$k"
    done
    return 1
}
LS=llr37-e9-short
echo "$(date -u +%T) night22: short run $LS"
perf $LS $C/plan-e9-short.txt || { echo "$(date -u +%T) night22: short run never started, stop"; exit 0; }
cd $P/prefill-opt-20261027 && bash scripts/analyze_run.sh $LS 104 > results/run-$LS-c104.txt 2>&1
cat results/run-$LS-c104.txt | grep -E "C104|^pp[0-3]:|beat ms"
RUN=$(grep -o 'run=[^ ]*' $L/perf/$LS/c104.log | head -1 | cut -d= -f2)
if grep -aqE "pre-write-back selections diverged|duplicate-reclaim victims diverged|Traceback" $RUN/logs/glm52_2p1d-prefill0.log; then
    echo "$(date -u +%T) night22: assert / traceback in the prefill log, stop"
    grep -anE "diverged|Traceback" $RUN/logs/glm52_2p1d-prefill0.log | head -5
    exit 0
fi
ok=$(python3 - results/run-$LS-c104.txt <<'PY'
import re, sys
ref = {0: 358, 1: 195, 2: 259, 3: 225}
got = {int(m.group(1)): int(m.group(2)) for m in re.finditer(r"^pp(\d): .*? avg (\d+)ms", open(sys.argv[1]).read(), re.M)}
print("PASS" if len(got) == 4 and all(got[r] <= ref[r] - 60 for r in ref) else f"FAIL {got}")
PY
)
echo "$(date -u +%T) night22: mechanism check: $ok"
[[ "$ok" == PASS ]] || { echo "$(date -u +%T) night22: mechanism not as predicted, stop"; exit 0; }
cd "$KIT"
g="$L/gate/full-site-llr35-v3-evict-bq16-prewb-gate-$(date -u +%m%dT%H%M).out"
echo "$(date -u +%T) night22: capped gate E9 ($g)"
PROBE_N=200 PROBE_PROMPTS=$G LONGCTX_SPECS="16000:128 16000:128" bash $S/quick_correct.sh "$KIT" "$L/gate" \
    $C/site-llr35-v3-evict-bq16-prewb-gate.env >"$g" 2>&1
grep -E " smoke: | probe: | longctx " "$g" | cut -c1-300
verdict=$(python3 $S/gate_pass2.py "$g" site-llr35-v3-evict-bq16-prewb-gate --passes 2)
echo "$(date -u +%T) night22: gate verdict: $verdict"
[[ "$verdict" == PASS ]] || { echo "$(date -u +%T) night22: gate FAILED, no scoring"; exit 0; }
LABEL=llr37-v3-evict512k-bq16-prewb
echo "$(date -u +%T) night22: AgentX C104 $LABEL"
perf $LABEL $C/plan-e9.txt || { echo "$(date -u +%T) night22: scoring never started"; exit 0; }
cd $P/prefill-opt-20261027 && bash scripts/analyze_run.sh $LABEL 104 > results/run-$LABEL-c104.txt 2>&1
RUN=$(grep -o 'run=[^ ]*' $L/perf/$LABEL/c104.log | head -1 | cut -d= -f2)
[[ -n "$RUN" ]] && python3 $S/beat_vs_evict.py $RUN/logs/glm52_2p1d-prefill0.log \
    $RUN/agentx/aiperf_artifacts/server_metrics_export.json > results/beat-vs-evict-1029/beat-vs-evict-$(basename $RUN).txt 2>&1
head -5 results/run-$LABEL-c104.txt
echo "$(date -u +%T) night22 queue done"
