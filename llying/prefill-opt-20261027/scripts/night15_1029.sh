#!/usr/bin/env bash
# 10-29 queue on 138 -> 140 (replaces night14 while its E1 + E3 capped gate runs; that gate is left to finish and judged
# here). E7's capped gate had 0 garbled / 0 bad starts (control 4 / 6), so the receive ordering is the bad-start fix;
# E7d keeps it without record_stream.
#   1. judge the running E1 + E3 capped gate; if it passes: full correctness of E1 + E3, then AgentX C104 (plan-night13a);
#   2. capped-pool gate of E1 + E7d, then AgentX C104 of E1 + E7d (plan-night15);
#   3. four-rank trace of E1 + E3 if E3 ran.
# Usage: night15_1029.sh LOG_ROOT
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
    echo "$(date -u +%T) night15: correctness $(basename "$1") ($g)"
    PROBE_N="${4:-1319}" PROBE_PROMPTS=$G LONGCTX_SPECS="${2-16000:128 100000:32}" bash $S/quick_correct.sh "$KIT" "$L/gate" "$1" >"$g" 2>&1
    grep -E " smoke: | probe: | longctx " "$g" | cut -c1-300
    python3 $S/gate_pass2.py "$g" "$(basename "$1" .env)" --passes "${3:-2}"
}
e3ran=0
g3=$(ls -t $L/gate/full-site-llr35-v3-e1-e3-gate-*.out | head -1)
echo "$(date -u +%T) night15: E1 + E3 capped gate ($g3)"
grep -E " smoke: | probe: | longctx " "$g3" | cut -c1-300
if python3 $S/gate_pass2.py "$g3" site-llr35-v3-e1-e3-gate --passes 2 && full $C/site-llr35-v3-e1-e3.env; then
    echo "$(date -u +%T) night15: E3 correctness passed -> plan-night13a-1028.txt"
    bash $P/prefill-car-20261023/scripts/perf_plan.sh "$KIT" "$L/perf" $C/plan-night13a-1028.txt
    e3ran=1
else
    echo "$(date -u +%T) night15: E3 correctness FAILED, no E3 performance run"
fi
if full $C/site-llr35-v3-evict-recvstream-ka-gate.env "16000:128 16000:128" 2 200; then
    echo "$(date -u +%T) night15: E7d capped gate passed -> plan-night15-1029.txt"
    bash $P/prefill-car-20261023/scripts/perf_plan.sh "$KIT" "$L/perf" $C/plan-night15-1029.txt
else
    echo "$(date -u +%T) night15: E7d capped gate FAILED, no performance run"
fi
if [[ $e3ran == 1 ]]; then
    echo "$(date -u +%T) night15: E3 four-rank trace"
    SIM=2.99 bash $P/prefill-car-20261023/scripts/prefill_profile_chain.sh "$KIT" $C/site-llr35-v3-e1-e3.env "$L/prof-llr37-v3-e1-e3-4r" 104 2400 600 120
fi
echo "$(date -u +%T) night15 queue done"
