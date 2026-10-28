#!/usr/bin/env bash
# 10-28 late-night queue on 138 -> 140 after night10_1028.sh prints "night10 queue done" (replaces night11b: E7's receive
# stream costs throughput, so F1 on top of it is dropped): E7b = only E7's correctness part (record_stream of received PP
# tensors, stock comm timing).
#   1. capped-pool gate of E1 + E7b (200 probe + two 16k x 128 at 300K tokens / rank; control and E1: 4 garbled / 6 bad starts);
#   2. full correctness of E1 + E7b (1319 probe, 16k x 128, 100k x 32; E1 had 4 garbled / 5 bad starts, E1 + E7 0 / 0);
#   3. if that passes, AgentX C104 of E1 + E7b (config/plan-night11c-1028.txt).
# Usage: night11c_1028.sh LOG_ROOT NIGHT10_LOG
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
    echo "$(date -u +%T) night11c: correctness $(basename "$1") ($g)"
    PROBE_N="${4:-1319}" PROBE_PROMPTS=$G LONGCTX_SPECS="${2-16000:128 100000:32}" bash $S/quick_correct.sh "$KIT" "$L/gate" "$1" >"$g" 2>&1
    grep -E " smoke: | probe: | longctx " "$g" | cut -c1-300
    python3 $S/gate_pass2.py "$g" "$(basename "$1" .env)" --passes "${3:-2}"
}
full $C/site-llr35-v3-evict-recvrecord-gate.env "16000:128 16000:128" 2 200
if full $C/site-llr35-v3-evict512k-recvrecord.env; then
    echo "$(date -u +%T) night11c: E7b correctness passed -> plan-night11c-1028.txt"
    bash $P/prefill-car-20261023/scripts/perf_plan.sh "$KIT" "$L/perf" $C/plan-night11c-1028.txt
else
    echo "$(date -u +%T) night11c: E7b correctness FAILED, no performance run"
fi
echo "$(date -u +%T) night11c queue done"
