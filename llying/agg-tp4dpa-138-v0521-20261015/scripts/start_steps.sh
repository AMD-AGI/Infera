#!/usr/bin/env bash
# Detach run_steps.sh on the worker node, so a driver does not depend on the login session (on
# 10-15 ~19:05 the login-side driver, sampler and agentx_bench of a3 C64 all vanished with it).
# No-op if a run_steps.sh for the same steps file is already running there.
# Usage: start_steps.sh STEPS_FILE [NODE=crsuse2-m2m-138]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
steps="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
node="${2:-crsuse2-m2m-138}"
name="$(basename "$steps" .steps)"
running="$(ssh -o BatchMode=yes -o ConnectTimeout=10 -n "$node" \
    "ps -u \$(id -u) -o pid=,args= | grep '[r]un_steps.sh $steps' | awk '{print \$1}'")"
if [[ -n "$running" ]]; then
    echo "run_steps.sh $name already running on $node: $running"
    exit 0
fi
ssh -o BatchMode=yes -o ConnectTimeout=10 -n "$node" \
    "setsid nohup bash $HERE/run_steps.sh $steps > /dev/null 2>&1 < /dev/null &"
echo "started run_steps.sh $name on $node"
