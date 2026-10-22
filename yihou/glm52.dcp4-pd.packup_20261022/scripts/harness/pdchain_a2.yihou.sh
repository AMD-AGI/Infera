#!/usr/bin/env bash
# Continue a dcp_pd warm-stack chain on a LIVE stack: wait for the running A1 client, flush decode+prefill, run A2.
# Login host: setsid nohup bash pdchain_a2.yihou.sh NK JOB ROUND [TAG_PREV=A1] [TAG_NEXT=A2] > log 2>&1 < /dev/null &
# Container polling tolerates spur exec transport errors (retries; only a clean "not running" counts).
set -uo pipefail
NK=$1; JOB=$2; round=$3; PREV=${4:-A1}; NEXT=${5:-A2}
PD=/shared_nfs/yihou_glm52/dcp_pd_20261021/pd; H=$PD/harness; RD=$PD/runs/$round
export PD_JOB=$JOB PD_CONC=${PD_CONC:-40}
source "$H/env.$round-$NK.yihou.sh"
pfx=yihou-dcp-pd-$NK-$round
ts(){ date -u +%FT%TZ; }
nx(){ spur exec "$JOB" bash -c "cd /tmp && $1"; }
client_running(){ local i o   # 0 running, 1 not running; retries on transport errors
  for i in 1 2 3 4 5 6 7 8 9 10; do
    o=$(nx "docker ps --format '{{.Names}}' | grep -c '^$pfx-agentx-client-'; echo RC=\$?" 2>/dev/null)
    [[ "$o" == *RC=* ]] && { [[ "${o%%$'\n'*}" -gt 0 ]] && return 0 || return 1; }
    sleep 20
  done; echo "$(ts) WARN: cannot query node, assume running"; return 0; }
wait_client(){ local i; for i in $(seq 1 150); do client_running || return 0; sleep 60; done; return 1; }
flush(){ local port=$1 i code
  for i in $(seq 1 60); do
    code=$(nx "curl -s -o /dev/null -w '%{http_code}' -X POST http://$YIHOU_IP:$port/flush_cache" 2>/dev/null)
    echo "$(ts) flush :$port try $i -> $code"; [[ "$code" == 200 ]] && return 0; sleep 10
  done; return 1; }
echo "$(ts) waiting for $PREV client"; wait_client || { echo "$(ts) STOP: $PREV client still running after 150 min"; exit 1; }
ls "$RD/agentx-c$PD_CONC-$PREV"/agentx_conc*.json >/dev/null 2>&1 || { echo "$(ts) STOP: no $PREV JSON"; exit 1; }
echo "$(ts) $PREV JSON present"
flush 29357 || { echo "$(ts) STOP: decode flush failed"; exit 1; }
flush 29101 || { echo "$(ts) STOP: prefill flush failed"; exit 1; }
echo "$(ts) bench $NEXT"; (cd "$H" && BENCH_TAG=$NEXT bash run.v2.yihou.sh bench); echo "$(ts) bench driver rc=$?"
sleep 120; wait_client || { echo "$(ts) STOP: $NEXT client still running after 150 min"; exit 1; }
ls "$RD/agentx-c$PD_CONC-$NEXT"/agentx_conc*.json >/dev/null 2>&1 && echo "$(ts) $NEXT JSON present" || echo "$(ts) $NEXT: no JSON"
echo "$(ts) CHAIN-A2 DONE round=$round (stack left up)"
