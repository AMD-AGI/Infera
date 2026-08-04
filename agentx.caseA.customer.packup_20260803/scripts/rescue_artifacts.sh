#!/bin/bash
# The customer's replay_caseA.sh mounts only $HERE into the aiperf container, so
# an --output-artifact-dir outside $HERE (our OUT=/root/agentx_20260803/results)
# lands INSIDE the container namespace and dies with the container.
#
# Rather than modify the customer script, poll for the running aiperf container
# and `docker cp` its artifact tree out to the host on every tick. Last copy
# before exit is the complete one.
set -uo pipefail
DEST="${DEST:-/root/agentx_20260803/rescue}"
SRC_ROOT="${SRC_ROOT:-/root/agentx_20260803/results}"
INTERVAL="${INTERVAL:-20}"
mkdir -p "$DEST"

while true; do
  C=$(docker ps -q --filter ancestor=aiperf-agentx:v1.0 | head -1)
  if [ -n "$C" ]; then
    for d in $(docker exec "$C" sh -c "ls -d $SRC_ROOT/c*/art 2>/dev/null"); do
      conc=$(echo "$d" | sed -E 's#.*/(c[0-9]+)/art#\1#')
      docker cp "$C:$d" "$DEST/${conc}_art.tmp" >/dev/null 2>&1 \
        && rm -rf "$DEST/${conc}_art" && mv "$DEST/${conc}_art.tmp" "$DEST/${conc}_art"
    done
    echo "$(date -u +%FT%TZ) synced from $C"
  else
    echo "$(date -u +%FT%TZ) no aiperf container"
  fi
  sleep "$INTERVAL"
done
