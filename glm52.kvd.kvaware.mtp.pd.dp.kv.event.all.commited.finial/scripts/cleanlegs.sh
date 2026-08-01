#!/usr/bin/env bash
# Kill the engine AND the sglang.launch_server child it spawns.
#
# start_leg.sh only pkills 'infera.engine.sglang'; the wrapper exits but its
# launch_server child keeps the DP kv-event port block bound, and the next leg
# dies with "port_base at N is not available". Kill the tree, then WAIT for the
# ports to be released -- the check is the point, not the kill.
set -u
docker exec merged_run bash -c '
  pkill -9 -f sglang.launch_server 2>/dev/null
  pkill -9 -f infera.engine.sglang 2>/dev/null
  for i in $(seq 1 20); do
    n=$(ps aux | grep -E "launch_server|infera.engine" | grep -v grep | wc -l)
    [ "$n" -eq 0 ] && { echo "  engines gone after $((i*2))s"; exit 0; }
    sleep 2
  done
  echo "  ENGINES STILL PRESENT"; exit 1'
