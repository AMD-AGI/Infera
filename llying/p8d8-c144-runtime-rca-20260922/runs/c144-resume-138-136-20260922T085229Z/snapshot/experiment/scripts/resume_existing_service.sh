#!/usr/bin/env bash
# Run a fresh full C144 benchmark against the retained, already-warm service.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="$ROOT/config/config.c144.p8d8.sh"
TOPOLOGY="$ROOT/config/topology.136-138.tsv"
source "$CONFIG" || exit 1

RUN_ID="${RUN_ID:-c144-resume-138-136-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN="$ROOT/runs/$RUN_ID"
BENCH="$RUN/bench"
LOGS="$RUN/logs"
CACHE="${AGENTX_CACHE_DIR:-$ROOT/cache/agentx}"
SOURCE_RUN="${SOURCE_RUN:-$ROOT/runs/c144-138-136-20260922T071057Z}"

if [[ -e "$RUN" ]]; then
    echo "run path already exists: $RUN" >&2
    exit 1
fi
mkdir -p "$RUN" "$LOGS"
printf '%s\n' "$RUN_ID" >"$RUN/run-id.txt"
printf '%s\n' "$SOURCE_RUN" >"$RUN/reused-service-from.txt"
date -u --iso-8601=ns >"$RUN/started-at.txt"

python3 "$ROOT/scripts/assert_live_config.py" \
    >"$RUN/live-config.json" 2>"$LOGS/live-config.stderr" || exit 1
"$ROOT/scripts/snapshot_experiment.sh" "$RUN" \
    >"$LOGS/snapshot.log" 2>&1 || exit 1

DURATION=0 ENGINE_INTERVAL="${ENGINE_INTERVAL:-2}" NODE_INTERVAL="${NODE_INTERVAL:-5}" \
    bash "$ROOT/scripts/run_sampler.sh" "$RUN" \
    >"$LOGS/sampler-console.log" 2>&1 &
sampler_pid=$!
printf '%s\n' "$sampler_pid" >"$RUN/sampler-wrapper.pid"

stop_sampler() {
    if kill -0 "$sampler_pid" >/dev/null 2>&1; then
        kill -TERM "$sampler_pid" >/dev/null 2>&1 || true
    fi
    wait "$sampler_pid" >/dev/null 2>&1 || true
}
trap 'stop_sampler; exit 130' INT
trap 'stop_sampler; exit 143' TERM

deadline=$(( $(date +%s) + 120 ))
while [[ ! -s "$RUN/sampling/live/preflight.json" ]]; do
    if ! kill -0 "$sampler_pid" >/dev/null 2>&1; then
        wait "$sampler_pid"
        echo "sampler exited before preflight: rc=$?" >&2
        exit 1
    fi
    if (( $(date +%s) > deadline )); then
        echo "sampler preflight timed out" >&2
        stop_sampler
        exit 1
    fi
    sleep 1
done

echo "running replacement AgentX C144 warmup/lane=10 duration=3600"
bash "$BENCH_DIR/agentx_bench.sh" \
    "CONFIG=$CONFIG" "TOPOLOGY=$TOPOLOGY" \
    "CONC=144" "DURATION=3600" \
    "AGENTX_WARMUP_REQUESTS_PER_LANE=10" \
    "OUT_DIR=$BENCH" "AGENTX_CACHE_DIR=$CACHE" \
    >"$LOGS/bench-console.log" 2>&1
bench_rc=$?
echo "$bench_rc" >"$RUN/bench-exit-code.txt"

stop_sampler
trap - INT TERM
date -u --iso-8601=ns >"$RUN/completed-at.txt"

if (( bench_rc != 0 )); then
    echo "FAILED: preserve live service and inspect before cleanup" >"$RUN/STATUS"
else
    echo "COMPLETED: live service retained for post-run inspection" >"$RUN/STATUS"
fi
exit "$bench_rc"
