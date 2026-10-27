#!/usr/bin/env bash
# 10-27 third night queue on 138 -> 140 (after night2_1027.sh prints "night2 queue done"):
#   1. correctness gate of the no-write-fence patch with the prefill pool capped at 300K tokens / rank, two identical
#      16k x 128 long-context passes (evict + write_back backups, then load-back), smoke and the 200-prompt probe;
#   2. pass -> AgentX C104 of the patch (config/plan-night3-1027.txt);
#      fail -> the same gate without the patch (is the failure the patch or the cap?), no performance run.
# Usage: night3_1027.sh LOG_ROOT NIGHT2_LOG
set -uo pipefail
L="$1" prev="$2"
P=/home/liyingli/bench_agentx/baseline/Infera/llying
C=$P/prefill-opt-20261027/config
S=$P/prefill-opt-20261027/scripts
KIT=/home/liyingli/bench_agentx/baseline/2p1d-kit/llr35
until grep -q "night2 queue done" "$prev"; do sleep 60; done
cd "$KIT" || exit 1
mkdir -p "$L/gate"
g="$L/gate/gate-$(date -u +%m%dT%H%M).out"
echo "$(date -u +%T) night3: gate site-llr35-v3-nofence-gate ($g)"
PROBE_N=200 LONGCTX_SPECS="16000:128 16000:128" bash $S/quick_correct.sh "$KIT" "$L/gate" $C/site-llr35-v3-nofence-gate.env >"$g" 2>&1
cat "$g"
if python3 $S/gate_pass.py "$g" site-llr35-v3-nofence-gate; then
    echo "$(date -u +%T) night3: gate passed -> plan-night3-1027.txt"
    bash $P/prefill-car-20261023/scripts/perf_plan.sh "$KIT" "$L/perf" $C/plan-night3-1027.txt
else
    echo "$(date -u +%T) night3: gate FAILED -> control gate without the patch, no performance run"
    PROBE_N=200 LONGCTX_SPECS="16000:128 16000:128" bash $S/quick_correct.sh "$KIT" "$L/gate" $C/site-llr35-v3-cap300k-gate.env >>"$g" 2>&1
    python3 $S/gate_pass.py "$g" site-llr35-v3-cap300k-gate
fi
echo "$(date -u +%T) night3 queue done"
