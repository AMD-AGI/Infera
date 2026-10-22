#!/usr/bin/env bash
# Task g4tune: run ON the node. Waits for the orphaned A1 agentx client container of <round> to exit
# (its chain died on EDQUOT), then does what chain_cap2 run_warm A2 does: flush decode+prefill, bench A2 @C.
# Usage: a2_after_orphan.yihou.sh <env> <C>
set -uo pipefail
ENVF=$1; C=$2
N=/shared_nfs/yihou_glm52/v0521cap2_20261013; H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
CAP=/shared_nfs/yihou_glm52/v0521cap_20261012
export PATH=$N/shimbin:$PATH
ts(){ date -u +%FT%TZ; }
cd $H && source "$ENVF"
[[ "$(hostname -s)" == "$YIHOU_NODE" ]] || { echo "$(ts) STOP: wrong node"; exit 1; }
MD=$N/runs/$YIHOU_ROUND; BASE_OVR="$YIHOU_OVERRIDES"
pat="glm52-v0521-yihou-${YIHOU_NODE##*-}-$YIHOU_ROUND-agentx-client"
while docker ps --format "{{.Names}}" | grep -q "^$pat"; do sleep 30; done
echo "$(ts) A1 client gone"
flush(){ local port=$1 i code
  for i in $(seq 1 60); do
    code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "http://$YIHOU_IP:$port/flush_cache")
    echo "$(ts) flush :$port try $i -> $code"; [[ "$code" == 200 ]] && return 0; sleep 10
  done; return 1; }
flush 29357 || { echo "$(ts) STOP: decode flush failed"; exit 1; }
flush 29101 || { echo "$(ts) STOP: prefill flush failed"; exit 1; }
setsid nohup python3 $CAP/tools/metrics_sampler.yihou.py "$MD/metrics.A2.tsv" 5400 5 \
    decode=http://$YIHOU_IP:29357 prefill=http://$YIHOU_IP:29101 > /dev/null 2>&1 < /dev/null &
echo "$(ts) stage bench A2 C$C"
YIHOU_OVERRIDES="$BASE_OVR CONC=$C" BENCH_TAG=A2 bash $H/run.v3.yihou.sh bench || { echo "$(ts) STOP at bench A2"; exit 1; }
echo "$(ts) A2 C$C DONE"
