#!/usr/bin/env bash
# 10-28/29 night queue on 138 -> 140 (replaces night12 after its first step, the E1 + E7b capped gate, which is left to
# finish): E7's record_stream is suspected of the 200 ms scheduler stalls in its trace, so E3 is measured on E1 alone and
# the correctness fix is retried as E7c (keep-alive of received tensors, no events).
#   1. capped-pool gate of E1 + E7c;
#   2. capped-pool gate of E1 + E3, full correctness of E1 + E3, then AgentX C104 of E1 + E3 (plan-night13a);
#   3. full correctness of E1 + E7c, then AgentX C104 of it (plan-night13b).
# Usage: night13_1028.sh LOG_ROOT
set -uo pipefail
L="$1"
P=/home/liyingli/bench_agentx/baseline/Infera/llying
C=$P/prefill-opt-20261027/config
S=$P/prefill-opt-20261027/scripts
KIT=/home/liyingli/bench_agentx/baseline/2p1d-kit/llr35
G=/mnt/m2m_nobackup/liyingli_3p1d/mix-v0522/2p1d/work/runs/correct-v2/gsm8k/samples_gsm8k_2026-10-20T16-13-32.558577.jsonl
while pgrep -f "quick_correc[t].sh" >/dev/null; do sleep 30; done
cd "$KIT" || exit 1
full() {  # SITE_ENV [LONGCTX_SPECS] [PASSES] [PROBE_N]: prints the summary lines, returns gate_pass2's verdict
    local g="$L/gate/full-$(basename "$1" .env)-$(date -u +%m%dT%H%M).out"
    echo "$(date -u +%T) night13: correctness $(basename "$1") ($g)"
    PROBE_N="${4:-1319}" PROBE_PROMPTS=$G LONGCTX_SPECS="${2-16000:128 100000:32}" bash $S/quick_correct.sh "$KIT" "$L/gate" "$1" >"$g" 2>&1
    grep -E " smoke: | probe: | longctx " "$g" | cut -c1-300
    python3 $S/gate_pass2.py "$g" "$(basename "$1" .env)" --passes "${3:-2}"
}
full $C/site-llr35-v3-evict-keepalive-gate.env "16000:128 16000:128" 2 200
if full $C/site-llr35-v3-e1-e3-gate.env "16000:128 16000:128" 2 200 && full $C/site-llr35-v3-e1-e3.env; then
    echo "$(date -u +%T) night13: E3 correctness passed -> plan-night13a-1028.txt"
    bash $P/prefill-car-20261023/scripts/perf_plan.sh "$KIT" "$L/perf" $C/plan-night13a-1028.txt
else
    echo "$(date -u +%T) night13: E3 correctness FAILED, no E3 performance run"
fi
if full $C/site-llr35-v3-evict512k-keepalive.env; then
    echo "$(date -u +%T) night13: E7c correctness passed -> plan-night13b-1028.txt"
    bash $P/prefill-car-20261023/scripts/perf_plan.sh "$KIT" "$L/perf" $C/plan-night13b-1028.txt
else
    echo "$(date -u +%T) night13: E7c correctness FAILED, no E7c performance run"
fi
echo "$(date -u +%T) night13 queue done"
