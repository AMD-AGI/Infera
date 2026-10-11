#!/usr/bin/env bash
# Wait for one or more pids to exit, then run $RUNNER with the given points.
# A separate file from chain_opposite_arm_yihou.sh on purpose: bash reads scripts
# incrementally, so editing a script that other waiting shells are still executing
# can corrupt them.
set -euo pipefail
: "${RUNNER:?Set RUNNER to a script name in this directory}"
WAIT_PIDS=$1; shift
for pid in ${WAIT_PIDS//,/ }; do
    while kill -0 "$pid" 2>/dev/null; do sleep 20; done
done
exec bash "$(dirname "$0")/$RUNNER" "$@"
