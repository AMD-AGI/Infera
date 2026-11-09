#!/usr/bin/env bash
# K2 step 1: aiter's existing MoE prologue variants for the TP8 decode MoE (prologue_bench.py), inside the
# llying-kopt-bench container. Usage: run_prologue.sh INDEX (0 = production, saves the reference outputs; run it first).
# Totals with the default graph settings (as in production), the per-kernel split with DEBUG_CLR_GRAPH_PACKET_CAPTURE=0.
# Output: out/prologue_INDEX.log
B=/perf_apps/liyingli/bench_agentx/kopt-20261109/bench
MS=${MS:-576,704,768,832}
REF=$B/out/prologue_ref.pt
LABELS=(
    "production: Opus multi-phase sort (policy 2) + split quant"
    "FlyDSL sort (policy 0)"
    "Opus auto (policy 0, no FlyDSL sort)"
    "Opus policy 1"
    "CK sort (policy 0)"
    "fused quant + scale sort, Opus policy 2"
    "fused quant + scale sort, FlyDSL sort (policy 0)"
)
ENVS=(
    "AITER_USE_FLYDSL_MOE_SORTING=1"
    "AITER_USE_FLYDSL_MOE_SORTING=1"
    "AITER_USE_FLYDSL_MOE_SORTING=0"
    "AITER_USE_FLYDSL_MOE_SORTING=0"
    "AITER_USE_CK_MOE_SORTING=1"
    "AITER_USE_FLYDSL_MOE_SORTING=1"
    "AITER_USE_FLYDSL_MOE_SORTING=1"
)
ARGS=(
    "--policy 2"
    "--policy 0"
    "--policy 0"
    "--policy 1"
    "--policy 0"
    "--policy 2 --fused-quant"
    "--policy 0 --fused-quant"
)
i=$1
O=$B/out/prologue_$i.log
cd $B
echo "== $i: ${LABELS[$i]}" > $O
if [[ $i == 0 ]]; then io="--save $REF"; else io="--ref $REF"; fi
env ${ENVS[$i]} timeout 1800 python3 prologue_bench.py --ms $MS ${ARGS[$i]} $io 2>&1 | grep -E "^RESULT|Error|error" >> $O
[[ $i == 0 ]] && io=""
env ${ENVS[$i]} DEBUG_CLR_GRAPH_PACKET_CAPTURE=0 TOPN=8 timeout 1800 python3 prologue_bench.py --ms 704 ${ARGS[$i]} $io 2>&1 \
    | grep -E "^   |Error" >> $O
