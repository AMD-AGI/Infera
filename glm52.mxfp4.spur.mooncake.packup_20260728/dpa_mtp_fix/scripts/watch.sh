#!/bin/bash
# Poll the decode leg every 3 min until it reaches a terminal state, then exit.
# Replaces the notification/callback pattern: run this in the background and read
# watch.log whenever you want status, instead of hand-issuing check commands.
#
# Usage:
#   bash watch.sh [JOB] [IP] [MAX_MIN]
# Terminal states: READY | CRASH | HANG | TIMEOUT
set -u
JOB="${1:-9006}"
IP="${2:-10.245.146.21}"
MAX_MIN="${3:-30}"
PERIOD=180
LOG=/home/yihou/glm52_fix/watch.log
export DOCKER_CONFIG=/tmp/dockercfg

say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

probe() {
  # NOTE: outer double quotes so $IP expands locally; everything the remote shell
  # must expand itself is escaped. `grep -c` on a missing file prints "0" AND
  # returns non-zero, so the `|| echo 0` fallback can emit a second line --
  # tr -d '\n' collapses each value to a single token before assembly.
  timeout 110 spur exec "$JOB" bash -c "docker exec dbg2 bash -c '
    L=/tmp/pd_decode_30000.log
    S=\$(grep -cE \"Load weight end\" \$L 2>/dev/null | head -1)
    R=\$(grep -cE \"ready to roll\" \$L 2>/dev/null | head -1)
    E=\$(grep -cE \"RuntimeError|Scheduler hit an exception\" \$L 2>/dev/null | head -1)
    W=\$(grep -cE \"End of disaggregation warmup\" \$L 2>/dev/null | head -1)
    H=\$(curl -s -o /dev/null -w \"%{http_code}\" --max-time 5 http://$IP:30000/health 2>/dev/null | head -1)
    echo \"shards=\${S:-0} ready=\${R:-0} err=\${E:-0} warmup=\${W:-0} health=\${H:-000}\"
  '" 2>/dev/null | grep -oE 'shards=[0-9]+ ready=[0-9]+ err=[0-9]+ warmup=[0-9]+ health=[0-9]+' | tail -1
}

say "=== watch start: job=$JOB ip=$IP period=${PERIOD}s max=${MAX_MIN}m ==="
elapsed=0; last_shards=-1; stall_ct=0

while [ "$elapsed" -lt "$((MAX_MIN * 60))" ]; do
  out=$(probe)
  say "${out:-<probe failed>}"

  err=$(echo "$out"    | grep -oE 'err=[0-9]+'    | cut -d= -f2)
  ready=$(echo "$out"  | grep -oE 'ready=[0-9]+'  | cut -d= -f2)
  shards=$(echo "$out" | grep -oE 'shards=[0-9]+' | cut -d= -f2)

  [ "${err:-0}" -gt 0 ]   && { say "RESULT=CRASH  see /tmp/pd_decode_30000.log"; exit 2; }
  [ "${ready:-0}" -gt 0 ] && { say "RESULT=READY  server is up";                 exit 0; }

  # boot-stall detector: shard count frozen across 3 polls (9 min), still not ready
  if [ -n "${shards:-}" ] && [ "${shards}" = "$last_shards" ]; then
    stall_ct=$((stall_ct + 1))
  else
    stall_ct=0; last_shards="${shards:-0}"
  fi
  [ "$stall_ct" -ge 3 ] && { say "RESULT=HANG  shards frozen at $last_shards for 9m -- py-spy the ranks"; exit 3; }

  sleep "$PERIOD"; elapsed=$((elapsed + PERIOD))
done

say "RESULT=TIMEOUT after ${MAX_MIN}m"
exit 4
