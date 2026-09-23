#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/perf_apps/liyingli/bench_agentx/p8d8-baseline4k-31625-20260923
set -a
source "$ROOT/config/config.sh"
set +a
cp "$RUN/analysis/STATUS" "$RUN/analysis/STATUS.before-script-recovery"
trap 'echo "$(date -u --iso-8601=seconds) ANALYSIS_FAILED line=$LINENO" > "$RUN/analysis/STATUS"' ERR
python3 "$ROOT/scripts/analyze_chunk_evidence.py" "$RUN" "$RUN/analysis"
python3 "$ROOT/scripts/collect_baseline_manifest.py" "$RUN"
python3 "$ROOT/scripts/compare_c80_runs.py" "$RUN" "$BASELINE_RUN" --reference-label current-nodes-4K --candidate-label current-nodes-8K --output-dir "$RUN/analysis/8k-vs-4k"
python3 "$ROOT/scripts/compare_c80_runs.py" /perf_apps/liyingli/bench_agentx/p8d8-tracing-aus-20260922/runs/main-20260922 "$RUN" --reference-label historical-4K --candidate-label current-nodes-4K --output-dir "$RUN/analysis/4k-baseline-drift"
echo "$(date -u --iso-8601=seconds) BASELINE_COLLECTION_AND_COMPARISON_COMPLETE_REVIEW_PENDING" > "$RUN/analysis/STATUS"
