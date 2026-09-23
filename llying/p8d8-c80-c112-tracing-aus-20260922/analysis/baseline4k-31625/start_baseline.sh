#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/perf_apps/liyingli/bench_agentx/p8d8-baseline4k-31625-20260923
set -a
source "$ROOT/config/config.sh"
set +a
mkdir -p "$RUN/snapshot"
python3 "$ROOT/scripts/validate_allocation.py"
python3 "$ROOT/scripts/wait_baseline_ready.py" --previous-prefix llying-aus-8k-31625 --output "$RUN/snapshot/resource-release.jsonl"
python3 "$ROOT/scripts/validate_allocation.py"
date -u --iso-8601=seconds > "$RUN/snapshot/resources-ready.txt"
exec bash "$ROOT/scripts/run_baseline_c80.sh"
