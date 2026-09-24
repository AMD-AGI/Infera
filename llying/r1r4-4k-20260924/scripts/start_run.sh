#!/usr/bin/env bash
set -Eeuo pipefail
root=/perf_apps/liyingli/bench_agentx/r1r4-4k-20260924
export CONFIG="${CONFIG:?choose smoke.sh or performance.sh}"
set -a; source "$CONFIG"; set +a
python3 "$root/scripts/require_performance_approval.py"
python3 "$root/scripts/validate_allocation.py"
python3 "$root/scripts/verify_guards.py"
[[ ! -e "$RUN/STATUS" ]]
nohup env CONFIG="$CONFIG" bash "$root/scripts/run_fresh_case.sh" > "$root/events/driver.log" 2>&1 < /dev/null &
echo $! > "$root/events/driver.pid"
printf '%s STARTING_R1_R4\n' "$(date -u --iso-8601=seconds)" > "$root/STATUS"
cat "$root/events/driver.pid"
