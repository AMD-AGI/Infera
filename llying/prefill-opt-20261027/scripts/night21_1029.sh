#!/usr/bin/env bash
# 10-29 12:00 queue on 138 -> 140: E10 C104 scoring only (night20's gate PASSED at 07:20; its C104 failed at
# startup: decode Mooncake "Failed to register memory ... Invalid argument" -> PD probe to dp_rank=3 failed 3x,
# run c104-1029T0726; first occurrence in the last 25 runs, unrelated to E10).
# Usage: night21_1029.sh LOG_ROOT
set -uo pipefail
L="$1"
P=/home/liyingli/bench_agentx/baseline/Infera/llying
C=$P/prefill-opt-20261027/config
S=$P/prefill-opt-20261027/scripts
KIT=/home/liyingli/bench_agentx/baseline/2p1d-kit/llr35
LABEL=llr37-v3-evict512k-bq16
cd "$KIT" || exit 1
[[ -f $L/perf/$LABEL/c104.log ]] && mv $L/perf/$LABEL/c104.log $L/perf/$LABEL/c104.log.failed-1029T0726
echo "$(date -u +%T) night21: AgentX C104 $LABEL"
bash $P/prefill-car-20261023/scripts/perf_plan.sh "$KIT" "$L/perf" $C/plan-night20-1029.txt
cd $P/prefill-opt-20261027 && bash scripts/analyze_run.sh $LABEL 104 > results/run-$LABEL-c104.txt 2>&1
RUN=$(grep -o 'run=[^ ]*' $L/perf/$LABEL/c104.log | head -1 | cut -d= -f2)
[[ -n "$RUN" ]] && python3 $S/beat_vs_evict.py $RUN/logs/glm52_2p1d-prefill0.log \
    $RUN/agentx/aiperf_artifacts/server_metrics_export.json > results/beat-vs-evict-1029/beat-vs-evict-$(basename $RUN).txt 2>&1
head -5 results/run-$LABEL-c104.txt
echo "$(date -u +%T) night21 queue done"
