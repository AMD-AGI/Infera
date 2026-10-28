#!/usr/bin/env bash
# 10-28 late-night queue on 138 -> 140, after night10_1028.sh prints "night10 queue done":
#   1. diagnostic correctness of E1 with the PD KV checksum on both roles (1319 probe; is "bad start" corrupted KV?);
#   2. if E7's correctness passed in night10: full correctness of E1 + E7 + F1 (outputs relayed on arrival), and if that
#      passes, AgentX C104 of it (config/plan-night11-1028.txt).
# Usage: night11_1028.sh LOG_ROOT NIGHT10_LOG
set -uo pipefail
L="$1" prev="$2"
P=/home/liyingli/bench_agentx/baseline/Infera/llying
C=$P/prefill-opt-20261027/config
S=$P/prefill-opt-20261027/scripts
KIT=/home/liyingli/bench_agentx/baseline/2p1d-kit/llr35
G=/mnt/m2m_nobackup/liyingli_3p1d/mix-v0522/2p1d/work/runs/correct-v2/gsm8k/samples_gsm8k_2026-10-20T16-13-32.558577.jsonl
until grep -q "night10 queue done" "$prev"; do sleep 60; done
cd "$KIT" || exit 1
full() {  # site env -> prints the summary lines, returns gate_pass2's verdict
    local g="$L/gate/full-$(basename "$1" .env)-$(date -u +%m%dT%H%M).out"
    echo "$(date -u +%T) night11: full correctness $(basename "$1") ($g)"
    PROBE_N=1319 PROBE_PROMPTS=$G LONGCTX_SPECS="${2-16000:128 100000:32}" bash $S/quick_correct.sh "$KIT" "$L/gate" "$1" >"$g" 2>&1
    grep -E " smoke: | probe: | longctx " "$g" | cut -c1-300
    python3 $S/gate_pass2.py "$g" "$(basename "$1" .env)" --passes "${3:-2}"
}
full $C/site-llr35-v3-evict512k-kvcsum.env "" 0
R=$(ls -td /mnt/m2m_nobackup/liyingli_3p1d/mix-v0522/2p1d/work/runs/site-llr35-v3-evict512k-kvcsum-* 2>/dev/null | head -1)
echo "$(date -u +%T) night11: KV checksum mismatches in decode log: $(grep -c 'KV checksum mismatch' "$R/logs/glm52_2p1d-decode.log" 2>/dev/null)"
if grep -q "E7 correctness passed" "$prev"; then
    if full $C/site-llr35-v3-evict512k-relay.env; then
        echo "$(date -u +%T) night11: F1 correctness passed -> plan-night11-1028.txt"
        bash $P/prefill-car-20261023/scripts/perf_plan.sh "$KIT" "$L/perf" $C/plan-night11-1028.txt
    else
        echo "$(date -u +%T) night11: F1 correctness FAILED, no performance run"
    fi
else
    echo "$(date -u +%T) night11: E7 did not pass correctness, F1 (built on it) skipped"
fi
echo "$(date -u +%T) night11 queue done"
