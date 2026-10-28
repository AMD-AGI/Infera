#!/usr/bin/env bash
# 10-28 night queue on 138 -> 140, after night9_1028.sh prints "night9 queue done":
#   1. full correctness of E1 + E7 (PP receives on their own stream): smoke, 1319 probe, 16k x 128, 100k x 32,
#      judged by gate_pass2.py relative to the stack's pre-existing bad-start rate;
#   2. pass -> AgentX C104 of E1 + E7 (config/plan-night10-1028.txt), then a four-rank prefill trace of it
#      (120 steps, AgentX 2400 s so every rank's trace is written before teardown);
#      fail -> stop (no performance run on an incorrect stack).
# Usage: night10_1028.sh LOG_ROOT NIGHT9_LOG
set -uo pipefail
L="$1" prev="$2"
P=/home/liyingli/bench_agentx/baseline/Infera/llying
C=$P/prefill-opt-20261027/config
S=$P/prefill-opt-20261027/scripts
KIT=/home/liyingli/bench_agentx/baseline/2p1d-kit/llr35
G=/mnt/m2m_nobackup/liyingli_3p1d/mix-v0522/2p1d/work/runs/correct-v2/gsm8k/samples_gsm8k_2026-10-20T16-13-32.558577.jsonl
E=$C/site-llr35-v3-evict512k-recvstream.env
until grep -q "night9 queue done" "$prev"; do sleep 60; done
cd "$KIT" || exit 1
g="$L/gate/full-$(basename "$E" .env)-$(date -u +%m%dT%H%M).out"
echo "$(date -u +%T) night10: full correctness $(basename "$E") ($g)"
PROBE_N=1319 PROBE_PROMPTS=$G LONGCTX_SPECS="16000:128 100000:32" bash $S/quick_correct.sh "$KIT" "$L/gate" "$E" >"$g" 2>&1
grep -E " smoke: | probe: | longctx " "$g" | cut -c1-300
if python3 $S/gate_pass2.py "$g" "$(basename "$E" .env)"; then
    echo "$(date -u +%T) night10: E7 correctness passed -> plan-night10-1028.txt"
    bash $P/prefill-car-20261023/scripts/perf_plan.sh "$KIT" "$L/perf" $C/plan-night10-1028.txt
    echo "$(date -u +%T) night10: E7 four-rank trace"
    SIM=2.99 bash $P/prefill-car-20261023/scripts/prefill_profile_chain.sh "$KIT" "$E" "$L/prof-llr37-v3-recvstream-4r" 104 2400 600 120
else
    echo "$(date -u +%T) night10: E7 correctness FAILED, no performance run"
fi
echo "$(date -u +%T) night10 queue done"
