#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/perf_apps/liyingli/bench_agentx/p8d8-chunk8k-31625-20260923
set -a
source "$ROOT/config/config.sh"
set +a
mkdir -p "$RUN/analysis"
trap 'echo "$(date -u --iso-8601=seconds) ANALYSIS_FAILED line=$LINENO" > "$RUN/analysis/STATUS"' ERR
for i in $(seq 1 720); do
    status=$(cat "$RUN/STATUS")
    if [[ "$status" == *FAILED* ]]; then
        echo "$(date -u --iso-8601=seconds) BENCHMARK_FAILED_NO_COMPARISON" > "$RUN/analysis/STATUS"
        exit 1
    fi
    if [[ "$status" == *BENCHMARKS_COMPLETED_ANALYSIS_PENDING* ]]; then
        sleep 30
        bash "$ROOT/scripts/capture_live.sh"
        python3 "$ROOT/scripts/analyze.py" "$RUN"
        python3 "$ROOT/scripts/analyze_runtime.py" "$RUN"
        python3 "$ROOT/scripts/analyze_spans.py" "$RUN"
        python3 "$ROOT/scripts/compare_chunk8k.py" "$BASELINE_RUN" "$RUN"
        echo "$(date -u --iso-8601=seconds) AUTOMATIC_COMPARISON_COMPLETE_REVIEW_PENDING" > "$RUN/analysis/STATUS"
        exit 0
    fi
    sleep 30
done
echo "$(date -u --iso-8601=seconds) WAIT_TIMEOUT" > "$RUN/analysis/STATUS"
exit 1
