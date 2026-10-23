#!/usr/bin/env bash
# orch16 (task ppdc 2026-10-23, ON 135): launch <round> with chain C1 C2, wait for A1 ok, stop.
# containers) A1 only -> stop -> dpax1 (DPA control) A1 only -> stop. Also aborts on a chain STOP line.
set -uo pipefail
W=/shared_nfs/yihou_glm52/g4tune_20261019; H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness; R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs
ts(){ date -u +%FT%TZ; }
stop_round(){ # $1 = round
  local r=$1 pids c5 c6 k5 k6
  pids=$(pgrep -f "^bash $H/chain_x.yihou.sh $W/env.$r-135x136.yihou.sh"); echo "$(ts) stop $r chain pids: $pids"
  for p in $pids; do pkill -TERM -P "$p"; kill -TERM "$p"; done; sleep 3
  for c in $(docker ps --format '{{.Names}}' | grep "^glm52-v0521-yihou-135-$r-agentx-client"); do docker stop -t 10 "$c"; done
  ( cd $H && source $W/env.$r-135x136.yihou.sh && bash $H/run.v3x.yihou.sh stop ) > /dev/null 2>&1
  for i in $(seq 1 30); do
    c5=$(docker ps -q --filter name=yihou | wc -l); c6=$(ssh -o BatchMode=yes crsuse2-m2m-136 "docker ps -q --filter name=yihou | wc -l")
    k5=$(rocm-smi --showpids 2>/dev/null | grep -c -E "^[0-9]+ "); k6=$(ssh -o BatchMode=yes crsuse2-m2m-136 'rocm-smi --showpids 2>/dev/null | grep -c -E "^[0-9]+ "')
    [[ "$c5" == 0 && "$c6" == 0 && "$k5" == 0 && "$k6" == 0 ]] && { echo "$(ts) $r stopped, both nodes clean"; return 0; }; sleep 10
  done
  echo "$(ts) $r NOT CLEAN c5=$c5 c6=$c6 k5=$k5 k6=$k6"; return 1
}
launch(){ # $1 = round
  df -h / | tail -1; ssh -o BatchMode=yes crsuse2-m2m-136 "df -h / | tail -1"
  cd /; setsid nohup bash $H/chain_x.yihou.sh $W/env.$1-135x136.yihou.sh 56 56 > $R/$1.chain.log 2>&1 < /dev/null &
  echo "$(ts) launched $1"
}
wait_status(){ # $1 = round, $2 = ok pattern in status.tsv, $3 = timeout s; returns 1 early on a fail line
  local i; for i in $(seq 1 $(( $3 / 30 ))); do grep -q -E "$2" $R/$1/status.tsv 2>/dev/null && return 0
    grep -q -E "[[:space:]](fail|FAIL)" $R/$1/status.tsv 2>/dev/null && return 1
    grep -q "STOP" $R/$1.chain.log 2>/dev/null && return 1; sleep 30; done; return 1; }
wait_profiling(){ # $1 = round, $2 = minutes after profiling start, $3 = timeout s
  local i; for i in $(seq 1 $(( $3 / 30 ))); do
    grep -q "Phase profiling (profiling) started" $R/$1/agentx-c56-A1.log 2>/dev/null && { sleep $(( $2 * 60 )); return 0; }
    grep -q -E "fail|FAIL" $R/$1/status.tsv 2>/dev/null && return 1; sleep 30; done; return 1; }
r=$1; c1=$2; c2=$3
echo "$(ts) orch16 start: $r C$c1 C$c2"
df -h / | tail -1; ssh -n -o BatchMode=yes crsuse2-m2m-136 "df -h / | tail -1"
cd /; setsid nohup bash $H/chain_x.yihou.sh $W/env.$r-135x136.yihou.sh $c1 $c2 > $R/$r.chain.log 2>&1 < /dev/null &
echo "$(ts) launched $r"
if wait_status $r "bench.*ok.*A1" 14400; then grep -a "bench.*ok" $R/$r/status.tsv | cut -c1-200
else echo "$(ts) $r did not finish A1 ok:"; tail -3 $R/$r/status.tsv | cut -c1-200; fi
stop_round $r
echo "$(ts) orch16 done"
