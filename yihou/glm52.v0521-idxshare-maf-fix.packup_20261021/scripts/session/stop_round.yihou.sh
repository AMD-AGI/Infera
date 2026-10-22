#!/usr/bin/env bash
# Task g4tune 2026-10-20: stop one of OUR rounds on the node this runs on (docker stop only, never rm).
# Usage (ON the node): stop_round.yihou.sh <node-key> <round>
set -u
K=$1; RD=$2
[[ "$K" =~ ^[0-9]{3}$ && "$RD" =~ ^(g4|dbg)[a-z0-9]+$ ]] || { echo "bad args"; exit 1; }
for pat in chain_resume.yihou.sh chain_cap2.yihou.sh pipeline.v3.yihou.sh warmup_pyspy.yihou.sh a2_after_orphan.yihou.sh agentx_bench.sh; do
    for p in $(pgrep -f "$pat"); do
        grep -qa "$RD" /proc/$p/cmdline /proc/$p/environ 2>/dev/null || continue
        pkill -TERM -P "$p"; kill -TERM "$p" && echo "killed $p $pat"
    done
done
names=$(docker ps --format '{{.Names}}' | grep -E "^glm52-v0521-yihou-$K-$RD-")
echo "stopping: $names"
[[ -n "$names" ]] && docker stop -t 20 $names
docker ps --format '{{.Names}}'
