#!/usr/bin/env bash
# Remove a run's containers once it ends, so an unattended run does not hold the
# GPUs: right after MEASUREMENT_COMPLETE, or after GRACE seconds (default 3600)
# for inspection when it failed or its orchestrator died.
# Usage: B4_RUN_ID=<id> B4_PREFILL_NODE=<p> B4_DECODE_NODE=<d> stop_after.sh <orchestrator-pid>
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
status="$HERE/../runtime/runs/$B4_RUN_ID/STATUS"
while ! grep -qE 'MEASUREMENT_COMPLETE|SERVICES_PRESERVED' "$status" 2>/dev/null && kill -0 "$1" 2>/dev/null; do
    sleep 60
done
echo "$(date -u +%FT%TZ) run ended: $(cat "$status" 2>/dev/null)"
grep -q MEASUREMENT_COMPLETE "$status" 2>/dev/null || sleep "${GRACE:-3600}"
bash "$HERE/run_b4.sh" stop
