#!/usr/bin/env bash
# 10-28 late-night queue on 138 -> 140, after night10_1028.sh prints "night10 queue done" (replaces night11_1028.sh):
#   1. capped-pool gate of E1 + E7 (200 probe + two 16k x 128 passes at 300K tokens / rank): its control (no E1) and E1
#      each had 4 garbled / 6 bad starts; E7's full suite had 0 / 0 against E1's 4 / 5, so this checks that E7 removes them;
#   2. if E7's correctness passed in night10: full correctness of E1 + E7 + F1 (outputs relayed on arrival), and if that
#      passes, AgentX C104 of it (config/plan-night11-1028.txt).
# Usage: night11b_1028.sh LOG_ROOT NIGHT10_LOG
set -uo pipefail
L="$1" prev="$2"
P=/home/liyingli/bench_agentx/baseline/Infera/llying
C=$P/prefill-opt-20261027/config
S=$P/prefill-opt-20261027/scripts
KIT=/home/liyingli/bench_agentx/baseline/2p1d-kit/llr35
G=/mnt/m2m_nobackup/liyingli_3p1d/mix-v0522/2p1d/work/runs/correct-v2/gsm8k/samples_gsm8k_2026-10-20T16-13-32.558577.jsonl
until grep -q "night10 queue done" "$prev"; do sleep 60; done
cd "$KIT" || exit 1
full() {  # SITE_ENV [LONGCTX_SPECS] [PASSES] [PROBE_N]: prints the summary lines, returns gate_pass2's verdict
    local g="$L/gate/full-$(basename "$1" .env)-$(date -u +%m%dT%H%M).out"
    echo "$(date -u +%T) night11b: correctness $(basename "$1") ($g)"
    PROBE_N="${4:-1319}" PROBE_PROMPTS=$G LONGCTX_SPECS="${2-16000:128 100000:32}" bash $S/quick_correct.sh "$KIT" "$L/gate" "$1" >"$g" 2>&1
    grep -E " smoke: | probe: | longctx " "$g" | cut -c1-300
    python3 $S/gate_pass2.py "$g" "$(basename "$1" .env)" --passes "${3:-2}"
}
full $C/site-llr35-v3-evict-recvstream-gate.env "16000:128 16000:128" 2 200
if grep -q "E7 correctness passed" "$prev"; then
    if full $C/site-llr35-v3-evict512k-relay.env; then
        echo "$(date -u +%T) night11b: F1 correctness passed -> plan-night11-1028.txt"
        bash $P/prefill-car-20261023/scripts/perf_plan.sh "$KIT" "$L/perf" $C/plan-night11-1028.txt
    else
        echo "$(date -u +%T) night11b: F1 correctness FAILED, no performance run"
    fi
else
    echo "$(date -u +%T) night11b: E7 did not pass correctness, F1 (built on it) skipped"
fi
echo "$(date -u +%T) night11b queue done"
