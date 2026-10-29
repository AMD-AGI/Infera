#!/usr/bin/env bash
# 10-29 07:30 queue on 138 -> 140: E10 (write_back eviction copy block_quota 2 -> 16, on top of E1).
#   1. capped-pool gate of E1 + E10 (the only test that drives the write_back copy hard; pass 2 loads the copies back);
#   2. only if it passes: AgentX C104 of E1 + E10 (plan-night20-1029.txt), then analyze_run.sh and beat_vs_evict.py.
# Usage: night20_1029.sh LOG_ROOT
set -uo pipefail
L="$1"
P=/home/liyingli/bench_agentx/baseline/Infera/llying
C=$P/prefill-opt-20261027/config
S=$P/prefill-opt-20261027/scripts
KIT=/home/liyingli/bench_agentx/baseline/2p1d-kit/llr35
G=/mnt/m2m_nobackup/liyingli_3p1d/mix-v0522/2p1d/work/runs/correct-v2/gsm8k/samples_gsm8k_2026-10-20T16-13-32.558577.jsonl
LABEL=llr37-v3-evict512k-bq16
cd "$KIT" || exit 1
g="$L/gate/full-site-llr35-v3-evict-bq16-gate-$(date -u +%m%dT%H%M).out"
echo "$(date -u +%T) night20: capped gate E1 + E10 ($g)"
PROBE_N=200 PROBE_PROMPTS=$G LONGCTX_SPECS="16000:128 16000:128" bash $S/quick_correct.sh "$KIT" "$L/gate" \
    $C/site-llr35-v3-evict-bq16-gate.env >"$g" 2>&1
grep -E " smoke: | probe: | longctx " "$g" | cut -c1-300
verdict=$(python3 $S/gate_pass2.py "$g" site-llr35-v3-evict-bq16-gate --passes 2)
echo "$(date -u +%T) night20: gate verdict: $verdict"
if [[ "$verdict" != PASS ]]; then
    echo "$(date -u +%T) night20: gate FAILED, no performance run"
    exit 0
fi
echo "$(date -u +%T) night20: AgentX C104 $LABEL"
bash $P/prefill-car-20261023/scripts/perf_plan.sh "$KIT" "$L/perf" $C/plan-night20-1029.txt
cd $P/prefill-opt-20261027 && bash scripts/analyze_run.sh $LABEL 104 > results/run-$LABEL-c104.txt 2>&1
RUN=$(grep -o 'run=[^ ]*' $L/perf/$LABEL/c104.log | head -1 | cut -d= -f2)
[[ -n "$RUN" ]] && python3 $S/beat_vs_evict.py $RUN/logs/glm52_2p1d-prefill0.log \
    $RUN/agentx/aiperf_artifacts/server_metrics_export.json > results/beat-vs-evict-1029/beat-vs-evict-$(basename $RUN).txt 2>&1
head -5 results/run-$LABEL-c104.txt
echo "$(date -u +%T) night20 queue done"
