#!/usr/bin/env bash
# Start an AgentX bench on a live round and make sure a client container exists (retries the driver when a
# spur-exec transport error kills it before the client starts), then poll the client to the end.
# Usage: PD_EXTRA=... bash bench_retry.yihou.sh NK ROUND TAG
set -uo pipefail
NK=$1; round=$2; TAG=$3
PD=/shared_nfs/yihou_glm52/dcp_pd_20261021/pd; H=$PD/harness; JOB=228279; RD=$PD/runs/$round
export PD_JOB=$JOB PD_CONC=${PD_CONC:-40}
pfx=yihou-dcp-pd-$NK-$round
ts(){ date -u +%FT%TZ; }
nx(){ spur exec "$JOB" bash -c "cd /tmp && $1"; }
nclients(){ local i o; for i in $(seq 1 10); do
    o=$(nx "docker ps --format '{{.Names}}' | grep -c '^$pfx-agentx-client-'; echo RC=\$?" 2>/dev/null)
    [[ "$o" == *RC=* ]] && { echo "${o%%$'\n'*}"; return; }; sleep 20; done; echo unknown; }
for try in 1 2 3 4; do
  t=$TAG; [[ $try -gt 1 ]] && t=$TAG-t$try
  echo "$(ts) bench $t try $try"
  (cd "$H" && source env.$round-$NK.yihou.sh && BENCH_TAG=$t bash run.v2.yihou.sh bench); echo "$(ts) driver rc=$? (attach errors expected)"
  sleep 60; n=$(nclients); echo "$(ts) clients running: $n"
  [[ "$n" != 0 ]] && break
done
for i in $(seq 1 150); do n=$(nclients); [[ "$n" == 0 ]] && break; sleep 60; done
echo "$(ts) client done; JSON: $(ls $RD/agentx-c$PD_CONC-$TAG*/agentx_conc*.json 2>/dev/null | tr '\n' ' ')"
