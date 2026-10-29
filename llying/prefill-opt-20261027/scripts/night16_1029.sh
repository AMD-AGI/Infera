#!/usr/bin/env bash
# 10-29 early-morning queue on 138 -> 140 after night15_1029.sh prints "night15 queue done":
#   1. full correctness of E1 + E7d in the production config (1319 probe, 16k x 128, 100k x 32);
#   2. a second AgentX C104 of E1 + E7d if its first run reached the E1 mean (49,719 tok/s/GPU) minus 1 %, otherwise a third
#      E1 run (tighter baseline for the night's comparisons).
# Usage: night16_1029.sh LOG_ROOT NIGHT15_LOG
set -uo pipefail
L="$1" prev="$2"
P=/home/liyingli/bench_agentx/baseline/Infera/llying
C=$P/prefill-opt-20261027/config
S=$P/prefill-opt-20261027/scripts
KIT=/home/liyingli/bench_agentx/baseline/2p1d-kit/llr35
G=/mnt/m2m_nobackup/liyingli_3p1d/mix-v0522/2p1d/work/runs/correct-v2/gsm8k/samples_gsm8k_2026-10-20T16-13-32.558577.jsonl
until grep -q "night15 queue done" "$prev"; do sleep 60; done
cd "$KIT" || exit 1
E=$C/site-llr35-v3-evict512k-recvstream-ka.env
g="$L/gate/full-$(basename "$E" .env)-$(date -u +%m%dT%H%M).out"
echo "$(date -u +%T) night16: full correctness $(basename "$E") ($g)"
PROBE_N=1319 PROBE_PROMPTS=$G LONGCTX_SPECS="16000:128 100000:32" bash $S/quick_correct.sh "$KIT" "$L/gate" "$E" >"$g" 2>&1
grep -E " smoke: | probe: | longctx " "$g" | cut -c1-300
python3 $S/gate_pass2.py "$g" "$(basename "$E" .env)"
(cd $P/prefill-opt-20261027 && bash scripts/analyze_run.sh llr37-v3-evict512k-recvstream-ka 104 > results/run-llr37-v3-evict512k-recvstream-ka-c104.txt 2>&1)
tok=$(grep -m1 "^| C104" $P/prefill-opt-20261027/results/run-llr37-v3-evict512k-recvstream-ka-c104.txt | awk -F'|' '{gsub(/[ ,]/,"",$3); print $3}')
echo "$(date -u +%T) night16: E1 + E7d C104 = ${tok:-?} tok/s/GPU"
if [[ -n "$tok" ]] && (( tok >= 49222 )); then
    echo "llr37-v3-evict512k-recvstream-ka-r2 $E 104" > $C/plan-night16-1029.txt
else
    echo "llr37-v3-evict512k-r3 $C/site-llr35-v3-evict512k.env 104" > $C/plan-night16-1029.txt
fi
echo "$(date -u +%T) night16: $(cat $C/plan-night16-1029.txt | cut -d' ' -f1)"
bash $P/prefill-car-20261023/scripts/perf_plan.sh "$KIT" "$L/perf" $C/plan-night16-1029.txt
echo "$(date -u +%T) night16 queue done"
