#!/usr/bin/env bash
# 10-28 queue on 138 -> 140, after the four-rank E1 trace (night6 log prints "prof-c104 down"):
#   correctness gate of E1 (eviction headroom) with the prefill pool capped at 300K tokens / rank, two identical
#   16k x 128 long-context passes, smoke and the 200-prompt probe; fail -> the same gate without E1 (E1 or the cap?).
# Usage: night7_1028.sh LOG_ROOT NIGHT6_LOG
set -uo pipefail
L="$1" prev="$2"
P=/home/liyingli/bench_agentx/baseline/Infera/llying
C=$P/prefill-opt-20261027/config
S=$P/prefill-opt-20261027/scripts
KIT=/home/liyingli/bench_agentx/baseline/2p1d-kit/llr35
until grep -q "prof-c104 down" "$prev"; do sleep 60; done
cd "$KIT" || exit 1
mkdir -p "$L/gate"
g="$L/gate/gate-e1-$(date -u +%m%dT%H%M).out"
echo "$(date -u +%T) night7: gate site-llr35-v3-evict-gate ($g)"
PROBE_N=200 LONGCTX_SPECS="16000:128 16000:128" bash $S/quick_correct.sh "$KIT" "$L/gate" $C/site-llr35-v3-evict-gate.env >"$g" 2>&1
cat "$g"
if python3 $S/gate_pass.py "$g" site-llr35-v3-evict-gate; then
    echo "$(date -u +%T) night7: E1 gate passed"
else
    echo "$(date -u +%T) night7: E1 gate FAILED -> control gate without E1"
    PROBE_N=200 LONGCTX_SPECS="16000:128 16000:128" bash $S/quick_correct.sh "$KIT" "$L/gate" $C/site-llr35-v3-cap300k-gate.env >>"$g" 2>&1
    python3 $S/gate_pass.py "$g" site-llr35-v3-cap300k-gate
fi
echo "$(date -u +%T) night7 queue done"
