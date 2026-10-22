#!/usr/bin/env bash
# e2g A2: same live stack after A1. Flush both legs exactly as before the 21,696 run (R028: POST
# /flush_cache on decode and prefill; prefill answers 400 while requests are pending -> retry until 200),
# then a second C40 bench into runs/e2g/agentx-c40-A2 (BENCH_TAG=A2).
set -uo pipefail
H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
source $H/env.e2g-102.yihou.sh
ts(){ date -u +%FT%TZ; }
flush(){ local port=$1 i code
  for i in $(seq 1 60); do
    code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://$YIHOU_IP:$port/flush_cache")
    echo "$(ts) flush :$port try $i -> $code"; [[ "$code" == 200 ]] && return 0; sleep 10
  done; return 1; }
flush 29357 || { echo "$(ts) STOP: decode flush failed"; exit 1; }
flush 29101 || { echo "$(ts) STOP: prefill flush failed"; exit 1; }
echo "$(ts) stage bench A2"
BENCH_TAG=A2 bash $H/run.v2.yihou.sh bench || { echo "$(ts) STOP at bench A2"; exit 1; }
echo "$(ts) A2 DONE"
