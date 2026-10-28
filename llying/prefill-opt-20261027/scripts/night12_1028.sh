#!/usr/bin/env bash
# 10-28/29 night queue on 138 -> 140 after night10_1028.sh prints "night10 queue done" (replaces night11c):
#   1. capped-pool gate of E1 + E7b (record_stream of received PP tensors; control and E1: 4 garbled / 6 bad starts / 200);
#   2. capped-pool gate of E1 + E7b + E3 (batched write-back eviction; eviction-heaviest check);
#   3. full correctness of E1 + E7b + E3 (1319 probe, 16k x 128, 100k x 32); pass -> AgentX C104 of it (plan-night12a);
#   4. AgentX C104 of E1 + E7b alone (plan-night12b), to separate E3's effect from E7b's.
# Usage: night12_1028.sh LOG_ROOT NIGHT10_LOG
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
    echo "$(date -u +%T) night12: correctness $(basename "$1") ($g)"
    PROBE_N="${4:-1319}" PROBE_PROMPTS=$G LONGCTX_SPECS="${2-16000:128 100000:32}" bash $S/quick_correct.sh "$KIT" "$L/gate" "$1" >"$g" 2>&1
    grep -E " smoke: | probe: | longctx " "$g" | cut -c1-300
    python3 $S/gate_pass2.py "$g" "$(basename "$1" .env)" --passes "${3:-2}"
}
full $C/site-llr35-v3-evict-recvrecord-gate.env "16000:128 16000:128" 2 200
if full $C/site-llr35-v3-e1-e7b-e3-gate.env "16000:128 16000:128" 2 200 && full $C/site-llr35-v3-e1-e7b-e3.env; then
    echo "$(date -u +%T) night12: E3 correctness passed -> plan-night12a-1028.txt"
    bash $P/prefill-car-20261023/scripts/perf_plan.sh "$KIT" "$L/perf" $C/plan-night12a-1028.txt
else
    echo "$(date -u +%T) night12: E3 correctness FAILED, no E3 performance run"
fi
echo "$(date -u +%T) night12: plan-night12b-1028.txt (E1 + E7b)"
bash $P/prefill-car-20261023/scripts/perf_plan.sh "$KIT" "$L/perf" $C/plan-night12b-1028.txt
echo "$(date -u +%T) night12 queue done"
