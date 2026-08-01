#!/usr/bin/env bash
# Wait until a leg is genuinely serving. Runs ON a node.
#
# WHY NOT A LOG GREP
# ------------------
# The obvious check -- grep the leg log for "ready to roll" -- is wrong when the
# log is APPENDED to across runs, which restart_replay.sh does. The grep matches
# a "ready to roll" from an EARLIER run within seconds and the caller proceeds
# against an engine still loading weights. Seen on the built-image run: "ready
# after 10s" for something that takes minutes.
#
# Poll the HTTP endpoint instead. It can only answer once the engine is up.
#
# Cold start is 3-9 min here (weight load + CUDA-graph capture). Silence is NOT
# a hang.
#
#   PORT=30000 IP=10.2.122.10 bash wait_ready.sh          # an engine leg
#   PORT=8100  IP=10.2.122.10 PATH_=/health bash wait_ready.sh   # the router
set -u
CTR="${CTR:-merged_run}"
IP="${IP:?}"
PORT="${PORT:-30000}"
PATH_="${PATH_:-/health}"
TRIES="${TRIES:-90}"

for i in $(seq 1 "$TRIES"); do
  if docker exec "$CTR" curl -sf -m5 "http://$IP:$PORT$PATH_" >/dev/null 2>&1; then
    echo "  $IP:$PORT serving after $((i * 10))s"
    exit 0
  fi
  sleep 10
done
echo "  $IP:$PORT did NOT come up in $((TRIES * 10))s" >&2
exit 1
