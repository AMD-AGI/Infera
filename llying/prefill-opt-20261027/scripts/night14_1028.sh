#!/usr/bin/env bash
# 10-29 night queue on 138 -> 140 (replaces night13): E7b's capped gate (record_stream only) still had 4 garbled / 8 bad
# starts, so the received-tensor lifetime is not the bad-start cause and E7c (keep-alive, same idea) is dropped.
#   1. capped-pool gate of E1 + E7 (receives on their own stream): does the receive ordering remove the bad starts?
#   2. capped-pool gate of E1 + E3, full correctness of E1 + E3, then AgentX C104 of E1 + E3 (plan-night13a);
#   3. four-rank prefill trace of E1 + E3 (AgentX 2400 s, 120 steps): do the eviction stall waves shrink?
# Usage: night14_1028.sh LOG_ROOT
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
    echo "$(date -u +%T) night14: correctness $(basename "$1") ($g)"
    PROBE_N="${4:-1319}" PROBE_PROMPTS=$G LONGCTX_SPECS="${2-16000:128 100000:32}" bash $S/quick_correct.sh "$KIT" "$L/gate" "$1" >"$g" 2>&1
    grep -E " smoke: | probe: | longctx " "$g" | cut -c1-300
    python3 $S/gate_pass2.py "$g" "$(basename "$1" .env)" --passes "${3:-2}"
}
full $C/site-llr35-v3-evict-recvstream-gate.env "16000:128 16000:128" 2 200
if full $C/site-llr35-v3-e1-e3-gate.env "16000:128 16000:128" 2 200 && full $C/site-llr35-v3-e1-e3.env; then
    echo "$(date -u +%T) night14: E3 correctness passed -> plan-night13a-1028.txt"
    bash $P/prefill-car-20261023/scripts/perf_plan.sh "$KIT" "$L/perf" $C/plan-night13a-1028.txt
    echo "$(date -u +%T) night14: E3 four-rank trace"
    SIM=2.99 bash $P/prefill-car-20261023/scripts/prefill_profile_chain.sh "$KIT" $C/site-llr35-v3-e1-e3.env "$L/prof-llr37-v3-e1-e3-4r" 104 2400 600 120
else
    echo "$(date -u +%T) night14: E3 correctness FAILED, no E3 performance run"
fi
echo "$(date -u +%T) night14 queue done"
