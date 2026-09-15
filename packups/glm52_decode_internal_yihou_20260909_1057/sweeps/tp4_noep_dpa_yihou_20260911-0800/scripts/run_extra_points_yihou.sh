#!/usr/bin/env bash
# Wait for the in-flight EP4 control driver, then extend the no-EP DPA sweep to C=32/40/48.
set -euo pipefail
WAIT_PID=$1; shift
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
exec bash "$(dirname "$0")/run_tp4_noep_dpa_sweep_yihou.sh" "$@"
