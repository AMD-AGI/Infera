#!/usr/bin/env bash
# 10-27 second night queue on 138 -> 140: waits for night_1027.sh to print "night queue done", then runs
# config/plan-night2-1027.txt (lines are read one at a time, so later experiments are appended while it runs).
# Usage: night2_1027.sh LOG_ROOT FIRST_QUEUE_LOG
set -uo pipefail
L="$1" first="$2"
P=/home/liyingli/bench_agentx/baseline/Infera/llying
KIT=/home/liyingli/bench_agentx/baseline/2p1d-kit/llr35
until grep -q "night queue done" "$first"; do sleep 60; done
cd "$KIT" || exit 1
echo "$(date -u +%T) night2: plan-night2-1027.txt"
bash $P/prefill-car-20261023/scripts/perf_plan.sh "$KIT" "$L/perf" $P/prefill-opt-20261027/config/plan-night2-1027.txt
echo "$(date -u +%T) night2 queue done"
