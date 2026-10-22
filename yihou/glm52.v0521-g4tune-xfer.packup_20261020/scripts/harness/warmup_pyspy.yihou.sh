#!/usr/bin/env bash
# Task v0521cap2 (ON the node): if the prefill PD warm-up of <round> started after <since-ISO> has not ended within
# <limit> s, take two py-spy dumps (60 s apart) of every python process in the prefill container. Read-only otherwise.
# Usage: warmup_pyspy.yihou.sh <round> <node-key> <since-ISO> [limit=300]
set -uo pipefail
RD=$1; K=$2; SINCE=$3; LIM=${4:-300}
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs; F=$R/$RD/server-logs/prefill-0.log; C=glm52-v0521-yihou-$K-$RD-prefill-0
OUT=$R/$RD/pyspy; mkdir -p $OUT
ts(){ date -u +%FT%TZ; }
started(){ grep -a "Start of pd disaggregation warmup" $F 2>/dev/null | awk -v s="$SINCE" '$1 >= s' | tail -1; }
ended(){ grep -a "End of disaggregation warmup" $F 2>/dev/null | awk -v s="$SINCE" '$1 >= s' | tail -1; }
echo "$(ts) watching $F since $SINCE (limit $LIM s)"
for i in $(seq 1 720); do [[ -n "$(started)" ]] && break; sleep 10; done
st=$(started); [[ -n "$st" ]] || { echo "$(ts) warm-up never started (2 h)"; exit 0; }
echo "$(ts) warm-up started: ${st:0:40}"
for i in $(seq 1 $((LIM/10))); do [[ -n "$(ended)" ]] && { echo "$(ts) warm-up ended: $(ended | cut -c1-40)"; exit 0; }; sleep 10; done
for n in 1 2; do
  echo "$(ts) warm-up still running after $LIM s: py-spy dump #$n"
  docker exec $C bash -c 'for p in $(pgrep -f python); do echo "===== pid $p: $(tr "\0" " " < /proc/$p/cmdline | cut -c1-200)"; py-spy dump --pid $p 2>&1; done' > $OUT/dump$n.$(date -u +%H%M%S).txt 2>&1
  echo "$(ts) rc=$? $(ls -la $OUT | tail -1)"
  [[ -n "$(ended)" ]] && break; sleep 60
done
echo "$(ts) done; ended=$(ended | cut -c1-40)"
