#!/usr/bin/env bash
# Flush decode then prefill on 054, then bench_retry TAG on ROUND.
set -uo pipefail
round=$1; TAG=$2; JOB=228279; H=/shared_nfs/yihou_glm52/dcp_pd_20261021/pd/harness
ts(){ date -u +%FT%TZ; }
for port in 29357 29101; do for i in $(seq 1 60); do
  c=$(spur exec $JOB bash -c "curl -s -o /dev/null -w '%{http_code}' -X POST http://10.245.144.125:$port/flush_cache" 2>/dev/null)
  echo "$(ts) flush :$port -> $c"; [[ "$c" == 200 ]] && break; sleep 10; done
  [[ "$c" == 200 ]] || { echo "$(ts) STOP: flush $port failed"; exit 1; }; done
bash $H/bench_retry.yihou.sh 054 "$round" "$TAG"
