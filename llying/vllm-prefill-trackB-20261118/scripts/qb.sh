#!/usr/bin/env bash
# Track B: q2.sh with the item run from this work directory (q2.sh's header exports R as the main session's directory).
#   qb.sh [-w] <name> '<command>'
W=/home/liyingli/bench_agentx/vllm-trackB-20261118
w=; [ "${1:-}" = -w ] && { w=-w; shift; }
name=${1:?name}; shift
exec /home/liyingli/bench_agentx/vllm-pd-20261117/q2.sh $w "$name" "cd $W && export R=$W && $*"
