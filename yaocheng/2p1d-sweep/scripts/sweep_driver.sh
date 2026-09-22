#!/usr/bin/env bash
# Run every point against the recorded deployment; execute on CONTROL_NODE.
source "$(dirname "$0")/common.sh"
[[ -s "$RUN_DIR/deployment.json" ]] || { echo "missing deployment snapshot; use run_full_sweep.sh first" >&2; exit 1; }
if [[ "${SWEEP_LOCK_HELD:-0}" != 1 ]]; then
    mkdir -p "$SWEEP_TMP_DIR"
    exec 9> "$SWEEP_TMP_DIR/.deployment.lock"
    flock -n 9 || { echo "another sweep is running" >&2; exit 1; }
fi
exec > >(tee -a "$RUN_DIR/driver.log") 2>&1
point_dir=""
point_state() {
    python3 - "$point_dir/status.json" "$1" "$DURATION" <<'PY'
import datetime, json, pathlib, sys
pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "state": sys.argv[2], "requested_duration_s": int(sys.argv[3]),
    "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}, indent=2) + "\n")
PY
}
archive_point() {
    # Keep full request JSONL in .tmp/raw; copy compact summaries to .tmp/results.
    local name
    for name in "agentx_conc$conc.json" runtime.env runner.log benchmark.log benchmark_command.txt; do
        [[ ! -f "$raw/$name" ]] || cp "$raw/$name" "$point_dir/"
    done
    [[ ! -d "$raw/service" ]] || cp -a "$raw/service" "$point_dir/"
    python3 - "$raw" "$point_dir" <<'PY'
from pathlib import Path
import shutil, sys
raw, target = map(Path, sys.argv[1:])
for source in raw.rglob('profile_export_aiperf.csv'):
    dest = target / 'aiperf-summary' / source.relative_to(raw)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)
PY
}
finish() {
    local rc=$?
    trap - EXIT
    if (( rc != 0 )) && [[ -n "$point_dir" ]]; then
        point_state failed
    fi
    python3 "$SCRIPT_DIR/summarize.py" "$RUN_DIR" || rc=1
    python3 "$SCRIPT_DIR/publish_results.py" "$RUN_DIR" "$RESULTS_DIR" || rc=1
    exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for conc in "${points[@]}"; do
    tag="$(printf 'c%03d' "$conc")"
    target="$RUN_DIR/$tag"
    raw="$RAW_DIR/$tag"
    # Never interpret an existing/partial directory as a completed measurement.
    [[ ! -e "$target" && ! -e "$raw" ]] || {
        echo "$tag already exists; select only unstarted points or use a new RUN_ID" >&2; exit 1;
    }
    point_dir="$target"
    mkdir -p "$point_dir"
    point_state running
    printf '%s\n' "$CONTROL_NODE:$raw" > "$point_dir/REMOTE_ARTIFACTS.txt"
    python3 "$SCRIPT_DIR/audit.py" "$point_dir/before.json" --baseline "$RUN_DIR/deployment.json"
    say "START $tag duration=$DURATION warmup=$AGENTX_WARMUP_REQUESTS_PER_LANE/lane"
    rc=0
    timeout --signal=TERM --kill-after=60s "$POINT_TIMEOUT" \
        bash "$HARNESS_DIR/agentx_bench.sh" "CONFIG=$CONFIG" "TOPOLOGY=$TOPOLOGY" \
        "CONC=$conc" "DURATION=$DURATION" "OUT_DIR=$raw" \
        > "$point_dir/console.log" 2>&1 || rc=$?
    archive_point
    audit_rc=0
    python3 "$SCRIPT_DIR/audit.py" "$point_dir/after.json" \
        --baseline "$RUN_DIR/deployment.json" --before "$point_dir/before.json" || audit_rc=$?
    printf 'benchmark_exit=%s\naudit_exit=%s\n' "$rc" "$audit_rc" > "$point_dir/exit-codes.txt"
    (( rc == 0 )) || { say "FAIL $tag benchmark exit=$rc"; exit "$rc"; }
    (( audit_rc == 0 )) || { say "FAIL $tag service health/fault check"; exit "$audit_rc"; }
    python3 "$SCRIPT_DIR/summarize.py" --validate "$point_dir/agentx_conc$conc.json" \
        --conc "$conc" --duration "$DURATION" > "$point_dir/metrics.json"
    point_state complete
    point_dir=""
    python3 "$SCRIPT_DIR/summarize.py" "$RUN_DIR"
    python3 "$SCRIPT_DIR/publish_results.py" "$RUN_DIR" "$RESULTS_DIR"
    say "COMPLETE $tag"
done
