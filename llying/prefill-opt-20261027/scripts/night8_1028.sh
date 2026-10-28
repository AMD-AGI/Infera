#!/usr/bin/env bash
# 10-28 queue on 138 -> 140, after night7_1028.sh (E1 gate) prints "night7 queue done":
#   1. AgentX C104 of E1 + E4 (no early cached-prefix send), config/plan-night8-1028.txt;
#   2. correctness gate of E1 + E4 (300K-token pool, two identical 16k x 128 passes, smoke, 200-prompt probe).
# Usage: night8_1028.sh LOG_ROOT NIGHT7_LOG
set -uo pipefail
L="$1" prev="$2"
P=/home/liyingli/bench_agentx/baseline/Infera/llying
C=$P/prefill-opt-20261027/config
S=$P/prefill-opt-20261027/scripts
KIT=/home/liyingli/bench_agentx/baseline/2p1d-kit/llr35
until grep -q "night7 queue done" "$prev"; do sleep 60; done
cd "$KIT" || exit 1
echo "$(date -u +%T) night8: plan-night8-1028.txt"
bash $P/prefill-car-20261023/scripts/perf_plan.sh "$KIT" "$L/perf" $C/plan-night8-1028.txt
g="$L/gate/gate-e4-$(date -u +%m%dT%H%M).out"
echo "$(date -u +%T) night8: gate site-llr35-v3-evict-noearly-gate ($g)"
PROBE_N=200 LONGCTX_SPECS="16000:128 16000:128" bash $S/quick_correct.sh "$KIT" "$L/gate" $C/site-llr35-v3-evict-noearly-gate.env >"$g" 2>&1
cat "$g"
python3 $S/gate_pass.py "$g" site-llr35-v3-evict-noearly-gate && echo "$(date -u +%T) night8: E4 gate passed" \
    || echo "$(date -u +%T) night8: E4 gate FAILED"
echo "$(date -u +%T) night8 queue done"
