#!/usr/bin/env bash
# after_x_then (task g4tune xnode, 2026-10-21): ON the prefill node. Wait until <old round>'s bench <tag> (e.g.
# agentx-c64-A4) is recorded ok/FAIL in its status.tsv, stop the old cross-node stack (run.v3x stop = docker stop),
# require 0 containers of the old round and 0 KFD processes on BOTH nodes, then start chain_x for the new round and,
# if concurrencies are given, conc_after_x on the same new stack.
# Usage: setsid nohup bash after_x_then.yihou.sh <old-env> <bench-tag> <new-env> C1 C2 [next-tag-number C ...] > log 2>&1 &
set -uo pipefail
OLD=$1; TAG=$2; NEW=$3; C1=$4; C2=$5; shift 5
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs; H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
ts(){ date -u +%FT%TZ; }
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10)
old=$(source "$OLD"; echo "$YIHOU_ROUND"); new=$(source "$NEW"; echo "$YIHOU_ROUND")
dn=$(source "$OLD"; echo "$XNODE_DECODE_NODE"); pn=$(source "$OLD"; echo "$YIHOU_NODE")
[[ "$(hostname -s)" == "$pn" ]] || { echo "$(ts) STOP: run on $pn"; exit 1; }
echo "$(ts) waiting for $old $TAG"
until grep -qE "bench	(ok|FAIL).*$TAG" "$R/$old/status.tsv" 2>/dev/null; do sleep 60; done
grep -a "$TAG" "$R/$old/status.tsv" | tail -1
echo "$(ts) stop $old"
( cd $H && source "$OLD" && bash $H/run.v3x.yihou.sh stop ) > "$R/$old.stop.log" 2>&1 || echo "$(ts) stop rc!=0 (checking)"
for n in "$pn" "$dn"; do
  for i in $(seq 1 40); do
    c=$("${SSH[@]}" -n "$n" "docker ps --format '{{.Names}}' | grep -c '^glm52-v0521-yihou-${pn##*-}-$old-'")
    k=$("${SSH[@]}" -n "$n" "rocm-smi --showpids 2>/dev/null | grep -cE '^[0-9]+ '")
    [[ "$c" == 0 && "$k" == 0 ]] && break; sleep 30
  done
  [[ "$c" == 0 && "$k" == 0 ]] || { echo "$(ts) STOP: $n still has containers=$c kfd=$k"; exit 1; }
  echo "$(ts) $n clean"
done
echo "$(ts) -> chain_x $new $C1 $C2"
bash $H/chain_x.yihou.sh "$NEW" "$C1" "$C2" > "$R/$new.chain.log" 2>&1 < /dev/null
grep -q "chain DONE" "$R/$new.chain.log" || { echo "$(ts) STOP: $new chain did not finish"; exit 1; }
if (( $# >= 2 )); then
  echo "$(ts) -> conc_after_x $new $*"
  bash $H/conc_after_x.yihou.sh "$NEW" "$@" > "$R/$new.conc.log" 2>&1 < /dev/null
fi
echo "$(ts) after_x_then DONE"
