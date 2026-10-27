#!/usr/bin/env bash
# 10-27 fourth night queue on 138 -> 140: after night3_1027.sh prints "night3 queue done", waits until
# config/plan-night4-1027.txt has a line (written once E1 / the E2 gate are analysed), then runs it.
# Usage: night4_1027.sh LOG_ROOT NIGHT3_LOG
set -uo pipefail
L="$1" prev="$2"
P=/home/liyingli/bench_agentx/baseline/Infera/llying
KIT=/home/liyingli/bench_agentx/baseline/2p1d-kit/llr35
plan=$P/prefill-opt-20261027/config/plan-night4-1027.txt
until grep -q "night3 queue done" "$prev"; do sleep 60; done
until grep -q '[^[:space:]]' "$plan" 2>/dev/null; do sleep 60; done
cd "$KIT" || exit 1
echo "$(date -u +%T) night4: plan-night4-1027.txt"
bash $P/prefill-car-20261023/scripts/perf_plan.sh "$KIT" "$L/perf" "$plan"
echo "$(date -u +%T) night4 queue done"
