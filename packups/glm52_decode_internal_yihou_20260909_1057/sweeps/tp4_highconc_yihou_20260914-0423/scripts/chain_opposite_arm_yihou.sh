#!/usr/bin/env bash
# Wait for a running driver pid, then run the opposite EP arm on the SAME node,
# so each node ends up with both arms measured under identical conditions.
set -euo pipefail
WAIT_PID=$1; shift
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
exec bash "$(dirname "$0")/run_highconc_yihou.sh" "$@"
