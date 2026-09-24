#!/usr/bin/env bash
set -Eeuo pipefail
root=/perf_apps/liyingli/bench_agentx/r2-4k-31705-31706-20260924
export CONFIG="$root/config/run.sh"
set -a; source "$CONFIG"; set +a
python3 "$root/scripts/require_performance_approval.py"
python3 "$root/scripts/validate_allocation.py"
python3 "$root/scripts/verify_guards.py"
[[ ! -e "$RUN/STATUS" ]]
nohup env CONFIG="$CONFIG" bash "$root/scripts/run_fresh_case.sh" > "$root/events/driver.log" 2>&1 < /dev/null &
echo $! > "$root/events/driver.pid"
printf '%s STARTING_R2_ON_C80_4K\n' "$(date -u --iso-8601=seconds)" > "$root/STATUS"
cat "$root/events/driver.pid"
