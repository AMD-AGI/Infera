#!/usr/bin/env bash
# 10-30 decode profile of E10 under AgentX C104 load with DEBUG_CLR_GRAPH_PACKET_CAPTURE=0 on the decode (collection
# only, not scored), 138 -> 135. Goal: see the target_verify graph's kernels (0 recorded in the 07:04 trace).
# Usage: night28_1030.sh LOG_ROOT
set -uo pipefail
L="$1"
P=/home/liyingli/bench_agentx/baseline/Infera/llying
C=$P/prefill-opt-20261027/config
S=$P/prefill-opt-20261027/scripts
KIT=/home/liyingli/bench_agentx/baseline/2p1d-kit/llr35
DEC=crsuse2-m2m-135
bash $S/disk_check.sh $DEC 95 "before decode profile (packet capture off)" || { echo "$(date -u +%T) night28: disk limit, stop"; exit 0; }
(cd "$KIT" && SIM=2.99 bash $S/decode_profile_chain.sh "$KIT" $C/site-135-e10-dprof-nopc.env $L/dprof-135-e10-nopc-1030 104 1500 600 40 300)
bash $S/disk_check.sh $DEC 95 "after decode profile"
echo "$(date -u +%T) night28 done"
