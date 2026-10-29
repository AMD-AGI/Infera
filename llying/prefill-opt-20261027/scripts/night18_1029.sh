#!/usr/bin/env bash
# 10-29 early-morning queue on 138 -> 140 after night15_1029.sh prints "night15 queue done" (replaces night17): E7d (receive
# stream + keep-alive) looks ~6 % slower mid-run, so the hardware-queue aliasing hypothesis is tested first.
#   1. full correctness of E1 + E7d in the production config;
#   2. AgentX C104 of E1 + E7d + GPU_MAX_HW_QUEUES=8 (plan-night18a);
#   3. AgentX C104 of E1 + GPU_MAX_HW_QUEUES=8 (plan-night18b);
#   4. capped-pool gate of E1 + E7d + E8 (idle-time eviction), then its C104 (plan-night17).
# Usage: night18_1029.sh LOG_ROOT NIGHT15_LOG
set -uo pipefail
L="$1" prev="$2"
P=/home/liyingli/bench_agentx/baseline/Infera/llying
C=$P/prefill-opt-20261027/config
S=$P/prefill-opt-20261027/scripts
KIT=/home/liyingli/bench_agentx/baseline/2p1d-kit/llr35
G=/mnt/m2m_nobackup/liyingli_3p1d/mix-v0522/2p1d/work/runs/correct-v2/gsm8k/samples_gsm8k_2026-10-20T16-13-32.558577.jsonl
until grep -q "night15 queue done" "$prev"; do sleep 60; done
cd "$KIT" || exit 1
full() {  # SITE_ENV [LONGCTX_SPECS] [PASSES] [PROBE_N]: prints the summary lines, returns gate_pass2's verdict
    local g="$L/gate/full-$(basename "$1" .env)-$(date -u +%m%dT%H%M).out"
    echo "$(date -u +%T) night18: correctness $(basename "$1") ($g)"
    PROBE_N="${4:-1319}" PROBE_PROMPTS=$G LONGCTX_SPECS="${2-16000:128 100000:32}" bash $S/quick_correct.sh "$KIT" "$L/gate" "$1" >"$g" 2>&1
    grep -E " smoke: | probe: | longctx " "$g" | cut -c1-300
    python3 $S/gate_pass2.py "$g" "$(basename "$1" .env)" --passes "${3:-2}"
}
(cd $P/prefill-opt-20261027 && bash scripts/analyze_run.sh llr37-v3-evict512k-recvstream-ka 104 > results/run-llr37-v3-evict512k-recvstream-ka-c104.txt 2>&1)
echo "$(date -u +%T) night18: E1 + E7d $(grep -m1 '^| C104' $P/prefill-opt-20261027/results/run-llr37-v3-evict512k-recvstream-ka-c104.txt)"
full $C/site-llr35-v3-evict512k-recvstream-ka.env
for plan in plan-night18a-1029.txt plan-night18b-1029.txt; do
    echo "$(date -u +%T) night18: $plan"
    bash $P/prefill-car-20261023/scripts/perf_plan.sh "$KIT" "$L/perf" $C/$plan
done
if full $C/site-llr35-v3-e1-e7d-e8-gate.env "16000:128 16000:128" 2 200; then
    echo "$(date -u +%T) night18: E8 capped gate passed -> plan-night17-1029.txt"
    bash $P/prefill-car-20261023/scripts/perf_plan.sh "$KIT" "$L/perf" $C/plan-night17-1029.txt
else
    echo "$(date -u +%T) night18: E8 capped gate FAILED, no performance run"
fi
echo "$(date -u +%T) night18 queue done"
