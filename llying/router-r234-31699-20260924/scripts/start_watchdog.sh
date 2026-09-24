#!/usr/bin/env bash
set -Eeuo pipefail
root=/perf_apps/liyingli/bench_agentx/router-r234-31699-20260924
source "$root/config/common.sh"
role=$1
node=$2
nohup env SLURM_CONF="$SLURM_CONF" PATH="$PATH" python3 "$root/scripts/allocation_watchdog.py" --job 31699 --prefix llying-adaptive-31699 --role "$role" --node "$node" --allocation-start 2026-09-24T06:29:58 --root "$root/events" --active-run-file "$root/active-run.txt" > "$root/events/watchdog-$role.log" 2>&1 < /dev/null &
echo $! > "$root/events/watchdog-$role.pid"
