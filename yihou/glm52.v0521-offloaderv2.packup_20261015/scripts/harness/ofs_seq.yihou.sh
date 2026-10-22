#!/usr/bin/env bash
# Task v0521cap2 (ON the held node, local ssh shim): OffloaderV2 ratio sweep, one node per ratio.
#   correctness round (chain_cx, sim OFF, stack stopped after) -> perf round (chain_cap2 C1 C2 [C3], sim 3.61).
# Usage: ofs_seq.yihou.sh <corr-round> <perf-round> <node-short> C1 C2 [C3]
set -uo pipefail
RC=$1 RP=$2 NS=$3; shift 3
N=/shared_nfs/yihou_glm52/v0521cap2_20261013
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs; H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export PATH=$N/shimbin:$PATH
ts(){ date -u +%FT%TZ; }
gpus_free(){ local i k; for i in $(seq 1 60); do k=$(rocm-smi --showpids 2>/dev/null | grep -cE '^[0-9]+ '); [[ "$k" == 0 ]] && return 0; sleep 30; done; return 1; }
echo "$(ts) -> $RC"
bash $H/chain_cx.yihou.sh $H/env.$RC-$NS.yihou.sh > $R/$RC.chain.log 2>&1 < /dev/null
grep -q "cx DONE" $R/$RC.chain.log || { echo "$(ts) STOP: $RC did not finish"; exit 1; }
gpus_free || { echo "$(ts) STOP: GPUs still busy after $RC"; exit 1; }
echo "$(ts) -> $RP $*"
exec bash $H/chain_cap2.yihou.sh $H/env.$RP-$NS.yihou.sh "$@" > $R/$RP.chain.log 2>&1 < /dev/null
