#!/usr/bin/env bash
# e2_perf.mixrun.sh ROUND [PD_EXTRA...]: E2 perf round on v3-070: fresh stack with sim 3.61 (e2_up.mixrun.sh), then ONE
# AgentX C40 bench (BENCH_TAG=A1, AGENTX duration from the config = 3600 s). Login host, detached.
set -uo pipefail
PD=/shared_nfs/yihou_glm52/dcp_pd_20261021/pd; H=$PD/harness; JOB=230266; NK=v3-070
round=$1; shift; extra="$*"
ts(){ date -u +%FT%TZ; }
nx(){ spur exec "$JOB" bash -c "cd /tmp && $1"; }
export PD_CONC=40
bash "$PD/tools/e2_up.mixrun.sh" "$round" 3.61 $extra || { echo "$(ts) STOP: up failed"; exit 1; }
pfx=yihou-dcp-pd-070-$round
client_running(){ local i o
  for i in $(seq 1 10); do
    o=$(nx "docker ps --format '{{.Names}}' | grep -c '^$pfx-agentx-client-'; echo RC=\$?" 2>/dev/null)
    [[ "$o" == *RC=* ]] && { [[ "${o%%$'\n'*}" -gt 0 ]] && return 0 || return 1; }
    sleep 20; done; return 0; }
wait_client(){ local i; for i in $(seq 1 150); do client_running || return 0; sleep 60; done; return 1; }
echo "$(ts) bench A1 (fresh, C40)"
(export PD_JOB=$JOB PD_EXTRA="$extra"; cd "$H" && source env.$round-$NK.yihou.sh && BENCH_TAG=A1 bash run.v2.yihou.sh bench)
echo "$(ts) bench driver returned rc=$? -- polling client"
sleep 120; wait_client
ls "$PD/runs/$round"/agentx-c40-A1/agentx_conc40.json >/dev/null 2>&1 && echo "$(ts) A1 JSON present" || echo "$(ts) A1: no JSON"
echo "$(ts) E2 PERF DONE (stack left up)"
