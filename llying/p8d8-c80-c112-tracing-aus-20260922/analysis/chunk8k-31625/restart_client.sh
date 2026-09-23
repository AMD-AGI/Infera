#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/perf_apps/liyingli/bench_agentx/p8d8-chunk8k-31625-20260923
set -a
source "$ROOT/config/config.sh"
set +a
kill -TERM 246122 246123 246124
for i in $(seq 1 30); do
    if ! kill -0 246122 2>/dev/null && ! kill -0 246123 2>/dev/null; then break; fi
    sleep 1
done
! kill -0 246122 2>/dev/null
! kill -0 246123 2>/dev/null
archive="$RUN/recovery/client-offline-failure"
mkdir -p "$archive"
mv "$RUN/c80" "$RUN/c80-started.txt" "$archive/"
cp "$RUN/STATUS" "$archive/STATUS"
cp "$ROOT/config/config.sh" "$ROOT/scripts/pin_chunk8k_dataset.py" "$ROOT/scripts/bench-harness/agentx_bench.sh" "$RUN/snapshot/"
python3 "$ROOT/scripts/validate_allocation.py"
python3 "$ROOT/scripts/validate_chunk8k.py"
exec bash "$ROOT/scripts/run_chunk8k.sh" --resume-after-smoke
