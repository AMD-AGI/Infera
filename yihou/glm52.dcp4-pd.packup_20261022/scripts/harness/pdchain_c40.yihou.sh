#!/usr/bin/env bash
# dcp_pd warm-stack C40 chain (pdrun), protocol of the 22k pack-up (REPRODUCE s3-4):
#   [optional stop of an old round] -> launch_pd (up + checks) -> A1 AgentX (warm-up, not scored)
#   -> /flush_cache decode then prefill -> A2 AgentX (scored).
# Login host:  setsid nohup bash pdchain_c40.yihou.sh NK JOB VARIANT ATTEMPT [OLD_ROUND] > log 2>&1 < /dev/null &
# Env passed through: PD_EXTRA, PD_CONC (default 40). Robust to a lost bench driver (spur exec transport error):
# after each bench it waits for the client container to exit and checks the JSON.
set -uo pipefail
NK=$1; JOB=$2; V=$3; ATT=$4; OLD=${5:-}
PD=/shared_nfs/yihou_glm52/dcp_pd_20261021/pd; H=$PD/harness
export PD_JOB=$JOB PD_CONC=${PD_CONC:-40}
round=pd$V-$ATT; RD=$PD/runs/$round
ts(){ date -u +%FT%TZ; }
nx(){ spur exec "$JOB" bash -c "cd /tmp && $1"; }
if [[ -n "$OLD" ]]; then
  echo "$(ts) stop old round $OLD"
  (source "$H/env.$OLD-$NK.yihou.sh" && cd "$H" && bash run.v2.yihou.sh stop) > "$PD/runs/$OLD/stop.txt" 2>&1
  echo "$(ts) stop rc=$?"
fi
bash "$H/launch_pd.yihou.sh" "$NK" "$JOB" "$V" "$ATT" || { echo "$(ts) STOP: launch failed"; exit 1; }
source "$H/env.$round-$NK.yihou.sh"
pfx=yihou-dcp-pd-$NK-$round
bench(){ local tag=$1 out="$RD/agentx-c$PD_CONC-$1" i
  echo "$(ts) bench $tag"
  (cd "$H" && BENCH_TAG=$tag bash run.v2.yihou.sh bench); echo "$(ts) bench driver rc=$?"
  for i in $(seq 1 120); do   # up to 2 h for the client
    nx "docker ps --format '{{.Names}}' | grep -q '^$pfx-agentx-client-'" || break
    sleep 60
  done
  ls "$out"/agentx_conc*.json >/dev/null 2>&1 && { echo "$(ts) bench $tag JSON present"; return 0; }
  echo "$(ts) bench $tag: no JSON in $out"; return 1; }
flush(){ local port=$1 i code
  for i in $(seq 1 60); do
    code=$(nx "curl -s -o /dev/null -w '%{http_code}' -X POST http://$YIHOU_IP:$port/flush_cache")
    echo "$(ts) flush :$port try $i -> $code"; [[ "$code" == 200 ]] && return 0; sleep 10
  done; return 1; }
bench A1 || { echo "$(ts) STOP at A1"; exit 1; }
flush 29357 || { echo "$(ts) STOP: decode flush failed"; exit 1; }
flush 29101 || { echo "$(ts) STOP: prefill flush failed"; exit 1; }
bench A2 || { echo "$(ts) STOP at A2"; exit 1; }
echo "$(ts) CHAIN DONE round=$round (stack left up)"
