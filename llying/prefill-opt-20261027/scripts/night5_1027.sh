#!/usr/bin/env bash
# 10-28 queue on 138 -> 140, after night4_1027.sh prints "night4 queue done":
#   1. prefill trace of llr37 + v3 + E1 (250 GB host tier) under C104, 120 steps per PP rank, 600 s into the scored
#      phase: what the long beats (p75 580 / p90 820 ms vs a 245 ms GPU step) are made of after E1;
#   2. AgentX C104 of E1 with the host tier at 300 GB per rank (InferenceX DRAM budget), config/plan-night5-1027.txt.
# Usage: night5_1027.sh LOG_ROOT NIGHT4_LOG
set -uo pipefail
L="$1" prev="$2"
P=/home/liyingli/bench_agentx/baseline/Infera/llying
C=$P/prefill-opt-20261027/config
KIT=/home/liyingli/bench_agentx/baseline/2p1d-kit/llr35
until grep -q "night4 queue done" "$prev"; do sleep 60; done
cd "$KIT" || exit 1
echo "$(date -u +%T) night5: prefill trace of E1 (120 steps)"
SIM=2.99 bash $P/prefill-car-20261023/scripts/prefill_profile_chain.sh \
    "$KIT" $C/site-llr35-v3-evict512k.env "$L/prof-llr37-v3-evict512k" 104 900 600 120
echo "$(date -u +%T) night5: plan-night5-1027.txt"
bash $P/prefill-car-20261023/scripts/perf_plan.sh "$KIT" "$L/perf" $C/plan-night5-1027.txt
echo "$(date -u +%T) night5 queue done"
