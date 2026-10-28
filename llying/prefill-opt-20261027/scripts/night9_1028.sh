#!/usr/bin/env bash
# 10-28 evening queue on 138 -> 140 (unattended):
#   1. full correctness of E1 in the production config (no pool cap), the same suite as the llr35 + v3 reference:
#      smoke, 1319-prompt GSM8K probe, long context 16k x 128 and 100k x 32 (reference: 0.972 / garbled 0 / bad start 0,
#      64/64 / 63/64, 16/16 / 16/16);
#   2. AgentX C104 of E1 + depth 2 (config/plan-night9-1028.txt);
#   3. full correctness of E1 + depth 2 (same suite).
# Usage: night9_1028.sh LOG_ROOT
set -uo pipefail
L="$1"
P=/home/liyingli/bench_agentx/baseline/Infera/llying
C=$P/prefill-opt-20261027/config
S=$P/prefill-opt-20261027/scripts
KIT=/home/liyingli/bench_agentx/baseline/2p1d-kit/llr35
G=/mnt/m2m_nobackup/liyingli_3p1d/mix-v0522/2p1d/work/runs/correct-v2/gsm8k/samples_gsm8k_2026-10-20T16-13-32.558577.jsonl
cd "$KIT" || exit 1
mkdir -p "$L/gate"
full() {  # site env -> driver output file
    local g="$L/gate/full-$(basename "$1" .env)-$(date -u +%m%dT%H%M).out"
    echo "$(date -u +%T) night9: full correctness $(basename "$1") ($g)"
    PROBE_N=1319 PROBE_PROMPTS=$G LONGCTX_SPECS="16000:128 100000:32" bash $S/quick_correct.sh "$KIT" "$L/gate" "$1" >"$g" 2>&1
    grep -E " smoke: | probe: | longctx " "$g" | cut -c1-300
}
full $C/site-llr35-v3-evict512k.env
echo "$(date -u +%T) night9: plan-night9-1028.txt"
bash $P/prefill-car-20261023/scripts/perf_plan.sh "$KIT" "$L/perf" $C/plan-night9-1028.txt
full $C/site-llr35-v3-evict512k-depth2.env
echo "$(date -u +%T) night9 queue done"
