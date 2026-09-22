#!/usr/bin/env bash
# Detached report generation survives the interactive session.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/config/config.sh"
mkdir -p "$RUN/analysis"
while true; do
    if [[ -d "$RUN/c80" ]]; then
        python3 "$ROOT/scripts/analyze.py" "$RUN" > "$RUN/logs/analyze.log" 2>&1 || true
    fi
    status=$(cat "$RUN/STATUS" 2>/dev/null || true)
    if [[ "$status" == *BENCHMARKS_COMPLETED_ANALYSIS_PENDING* ]]; then
        python3 "$ROOT/scripts/analyze_spans.py" "$RUN" > "$RUN/logs/analyze-spans.log" 2>&1
        exit $?
    fi
    [[ "$status" != *FAILED* ]] || exit 1
    sleep 60
done
