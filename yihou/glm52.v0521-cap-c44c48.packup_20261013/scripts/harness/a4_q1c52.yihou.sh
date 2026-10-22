#!/usr/bin/env bash
# Task v0521cap: C48 as the 3rd run on the warm q1 stack (137). Waits for "q1 chain done", flushes decode then prefill,
# login-host /metrics sampler, bench CONC=48 BENCH_TAG=A4. Does NOT stop the stack.
set -uo pipefail
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs; H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
CAP=/shared_nfs/yihou_glm52/v0521cap_20261012
ts(){ date -u +%FT%TZ; }
until grep -qE "A3 C48 DONE|STOP" $R/q1.A3c48.log 2>/dev/null; do sleep 60; done
grep -q "A3 C48 DONE" $R/q1.A3c48.log || { echo "$(ts) A3 not clean -> no A4"; exit 1; }
cd $H && source $H/env.q1c52-137.yihou.sh
flush(){ local port=$1 i code
  for i in $(seq 1 60); do
    code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://$YIHOU_IP:$port/flush_cache")
    echo "$(ts) flush :$port try $i -> $code"; [[ "$code" == 200 ]] && return 0; sleep 10
  done; return 1; }
flush 29357 || { echo "$(ts) STOP: decode flush failed"; exit 1; }
flush 29101 || { echo "$(ts) STOP: prefill flush failed"; exit 1; }
setsid nohup python3 $CAP/tools/metrics_sampler.yihou.py $CAP/runs/q1/metrics.A4c52.tsv 5400 5 \
  decode=http://$YIHOU_IP:29357 prefill=http://$YIHOU_IP:29101 > /dev/null 2>&1 < /dev/null &
echo "$(ts) stage bench A4 (C52)"
BENCH_TAG=A4 bash $H/run.v2.yihou.sh bench || { echo "$(ts) STOP at bench A4"; exit 1; }
echo "$(ts) A4 C52 DONE"
