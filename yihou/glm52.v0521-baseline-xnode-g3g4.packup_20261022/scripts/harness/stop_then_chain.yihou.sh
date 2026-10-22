#!/usr/bin/env bash
# Task v0521cap2: on the node, stop a finished round's stack, wait for no KFD process, then start chain_cap2 for a new round.
# Usage: bash stop_then_chain.yihou.sh <old-env> <new-env> C1 C2 [C3]   (chain log -> runs/<new round>.chain.log)
set -uo pipefail
OLD=$1; NEW=$2; shift 2
N=/shared_nfs/yihou_glm52/v0521cap2_20261013
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs; H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export PATH=$N/shimbin:$PATH
ts(){ date -u +%FT%TZ; }
old_round=$(source "$OLD"; echo "$YIHOU_ROUND"); new_round=$(source "$NEW"; echo "$YIHOU_ROUND"); key=$(source "$OLD"; echo "${YIHOU_NODE##*-}")
echo "$(ts) stop $old_round"
( cd $H && source "$OLD" && bash $H/run.v3.yihou.sh stop ) || echo "$(ts) stop rc!=0 (checking containers)"
left=$(docker ps --format '{{.Names}}' | grep -c "^glm52-v0521-yihou-$key-$old_round-")
[[ "$left" == 0 ]] || { echo "$(ts) STOP: $left $old_round containers still running"; exit 1; }
for i in $(seq 1 60); do k=$(rocm-smi --showpids 2>/dev/null | grep -cE '^[0-9]+ '); [[ "$k" == 0 ]] && break; sleep 30; done
echo "$(ts) $old_round stopped (KFD procs=$k) -> $new_round $*"
exec bash $H/chain_cap2.yihou.sh "$NEW" "$@" > $R/$new_round.chain.log 2>&1 < /dev/null
