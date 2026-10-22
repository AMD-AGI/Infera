#!/usr/bin/env bash
# r4: stop the r2 stack, fresh full stack = r2 except decode custom AR ON + decode NCCL debug off (sim 3.61),
# then ONE fresh AgentX C40 3600 s (tag A1). Client polled by container state (spur exec retries).
# Login host: setsid nohup bash pd_r4.yihou.sh > runs/pd_r4.log 2>&1 < /dev/null &
set -uo pipefail
PD=/shared_nfs/yihou_glm52/dcp_pd_20261021/pd; H=$PD/harness; JOB=228279; NK=054
OLD=pdBeperf-r2; round=pdBeperf-r4; RD=$PD/runs/$round
export PD_JOB=$JOB PD_CONC=40
ts(){ date -u +%FT%TZ; }
nx(){ spur exec "$JOB" bash -c "cd /tmp && $1"; }
echo "$(ts) stop $OLD (all containers, docker stop)"
(export PD_EXTRA="PREFILL_HICACHE_SIZE=450"; cd "$H" && source env.$OLD-$NK.yihou.sh && bash run.v2.yihou.sh stop) > "$PD/runs/$OLD/stop.txt" 2>&1
echo "$(ts) stop rc=$? ; remaining:"; nx "docker ps --format '{{.Names}}' | grep '^yihou-dcp-pd-054-$OLD-' || echo none"
export PD_EXTRA="PREFILL_HICACHE_SIZE=450 DECODE_CUSTOM_AR=1 DECODE_EXTRA_ENV_BASE=MC_LOG_LEVEL=INFO"
bash "$H/launch_pd.yihou.sh" $NK $JOB Beperf r4 || { echo "$(ts) STOP: launch failed"; exit 1; }
pfx=yihou-dcp-pd-$NK-$round
client_running(){ local i o
  for i in $(seq 1 10); do
    o=$(nx "docker ps --format '{{.Names}}' | grep -c '^$pfx-agentx-client-'; echo RC=\$?" 2>/dev/null)
    [[ "$o" == *RC=* ]] && { [[ "${o%%$'\n'*}" -gt 0 ]] && return 0 || return 1; }
    sleep 20; done; return 0; }
wait_client(){ local i; for i in $(seq 1 150); do client_running || return 0; sleep 60; done; return 1; }
echo "$(ts) bench A1 (fresh, C40 3600 s)"
(cd "$H" && source env.$round-$NK.yihou.sh && BENCH_TAG=A1 bash run.v2.yihou.sh bench)
echo "$(ts) bench driver returned rc=$? (attach lost is expected; client keeps running) -- polling client"
sleep 120; wait_client
ls "$RD"/agentx-c40-A1/agentx_conc40.json >/dev/null 2>&1 && echo "$(ts) A1 JSON present" || echo "$(ts) A1: no JSON"
echo "$(ts) R4 DONE (stack left up)"
