#!/usr/bin/env bash
# 10-27 night queue on 138 -> 140, one deployment after another (each step waits for the GPUs itself):
#   1. llr37 + patch v3 at 4+4 C104 (AgentX 3600 s), against yihou's 48,408 on 135/136;
#   2. a prefill trace of the same stack under C104 load (30 steps per PP rank, 420 s after warmup, input shapes);
#   3. llr37 + patch v3 with HiCache 500 GB per PP rank at C104.
# Usage: night_1027.sh LOG_ROOT
set -uo pipefail
L="$1"
P=/home/liyingli/bench_agentx/baseline/Infera/llying
C=$P/prefill-opt-20261027/config
KIT=/home/liyingli/bench_agentx/baseline/2p1d-kit/llr35
cd "$KIT" || exit 1
echo "$(date -u +%T) step 1: llr37-v3 C104"
bash $P/prefill-car-20261023/scripts/perf_plan.sh "$KIT" "$L/perf" $C/plan-llr37-v3.txt
echo "$(date -u +%T) step 2: prefill trace"
SIM=2.99 PROFILE_RECORD_SHAPES=1 bash $P/prefill-car-20261023/scripts/prefill_profile_chain.sh \
    "$KIT" $C/site-llr35-v3.env "$L/prof-llr37-v3" 104 900 420 30
echo "$(date -u +%T) step 3: llr37-v3-hc500 C104"
bash $P/prefill-car-20261023/scripts/perf_plan.sh "$KIT" "$L/perf" $C/plan-llr37-v3-hc500.txt
echo "$(date -u +%T) night queue done"
