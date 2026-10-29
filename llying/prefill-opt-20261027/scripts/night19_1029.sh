#!/usr/bin/env bash
# 10-29 03:10 queue on 138 -> 140 (replaces night18): only AgentX C104 of E1 + GPU_MAX_HW_QUEUES=8 (one cheap knob, adoptable
# as is). E1 + E7d + HWQ8 was cancelled (E7d is neither faster nor a correctness fix in the full suite) and E8 dropped
# (scripts/evict_vs_idle.py: queued long gaps are no more frequent after evictions than after random times).
# Usage: night19_1029.sh LOG_ROOT
set -uo pipefail
L="$1"
P=/home/liyingli/bench_agentx/baseline/Infera/llying
KIT=/home/liyingli/bench_agentx/baseline/2p1d-kit/llr35
cd "$KIT" || exit 1
bash $P/prefill-car-20261023/scripts/perf_plan.sh "$KIT" "$L/perf" $P/prefill-opt-20261027/config/plan-night18b-1029.txt
echo "$(date -u +%T) night19 queue done"
