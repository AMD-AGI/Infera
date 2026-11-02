#!/usr/bin/env bash
# 11-02 E11 disentangling: the first E11 check (ROCM700A=0 + DEBUG_CLR_GRAPH_PACKET_CAPTURE=0, real acceptance) gave garbled
# smoke output (4/6). Two flags changed at once, so test them one at a time with the same quick check:
#   A. ROCM700A=0 only (site-140-e11.env);  B. DEBUG_CLR_GRAPH_PACKET_CAPTURE=0 only (site-140-pc0.env).
# Usage: night30_1102.sh LOG_ROOT
set -uo pipefail
L="$1"
P=/home/liyingli/bench_agentx/baseline/Infera/llying
C=$P/prefill-opt-20261027/config
S=$P/prefill-opt-20261027/scripts
KIT=/home/liyingli/bench_agentx/baseline/2p1d-kit/llr35
for v in site-140-e11 site-140-pc0; do
    bash $S/disk_check.sh crsuse2-m2m-140 95 "before $v" || exit 0
    (cd "$KIT" && PROBE_N=200 LONGCTX_SPECS="16000:32" bash $S/quick_correct.sh "$KIT" $L/e11-check2 $C/$v.env)
    echo "$(date -u +%T) night30: $v done"
done
echo "$(date -u +%T) night30 done"
