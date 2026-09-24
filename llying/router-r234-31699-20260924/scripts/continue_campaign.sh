#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/perf_apps/liyingli/bench_agentx/router-r234-31699-20260924
for stage in control r2 r3 r1-smoke r1-r4-smoke; do
    CONFIG="$ROOT/config/$stage.sh"
    export CONFIG
    set -a; source "$CONFIG"; set +a
    printf '%s %s\n' "$(date -u --iso-8601=seconds)" "WAITING_$stage" > "$ROOT/CAMPAIGN_STATUS"
    if [[ "$stage" != control ]]; then
        bash "$ROOT/scripts/prepare_case.sh"
        bash "$ROOT/scripts/run_fresh_case.sh" > "$ROOT/events/$stage-driver.log" 2>&1
    fi
    while true; do
        phase=$(cat "$RUN/STATUS" 2>/dev/null || true)
        [[ ! -e "$RUN/INVALID" ]] || { echo "INVALID: $RUN"; exit 1; }
        [[ "$phase" != *FAILED* ]] || { echo "$phase"; exit 1; }
        if [[ "$phase" == *COMPLETE_REVIEW_PENDING* || "$phase" == *SMOKE_COMPLETE* ]]; then break; fi
        sleep 30
    done
    if [[ "$SMOKE_ONLY" == 0 ]]; then
        python3 "$ROOT/scripts/analyze_guard_lifecycle.py" "$RUN" > "$RUN/logs/guard-analysis.log"
        python3 "$ROOT/scripts/analyze_router_picks.py" "$RUN" > "$RUN/logs/router-analysis.log"
        python3 "$ROOT/scripts/audit_case.py" "$RUN" > "$RUN/logs/audit.log"
    fi
    printf '%s %s\n' "$(date -u --iso-8601=seconds)" "RETIRING_$stage" > "$ROOT/CAMPAIGN_STATUS"
    python3 "$ROOT/scripts/retire_case.py"
done
printf '%s ALL_STAGES_COMPLETE\n' "$(date -u --iso-8601=seconds)" > "$ROOT/CAMPAIGN_STATUS"
