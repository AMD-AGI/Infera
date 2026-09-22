#!/usr/bin/env bash
# Launch the fixed service, start fail-closed sampling, and run AgentX C144.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="$ROOT/config/config.c144.p8d8.sh"
TOPOLOGY="$ROOT/config/topology.136-138.tsv"
source "$CONFIG" || exit 1

RUN_ID="${RUN_ID:-c144-138-136-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN="$ROOT/runs/$RUN_ID"
LAUNCH="$RUN/launch"
BENCH="$RUN/bench"
LOGS="$RUN/logs"
KERNEL="$RUN/kernel"
CACHE="${AGENTX_CACHE_DIR:-$ROOT/cache/agentx}"

if [[ -e "$RUN" ]]; then
    echo "run path already exists: $RUN" >&2
    exit 1
fi
mkdir -p "$RUN" "$LOGS" "$KERNEL"
printf '%s\n' "$RUN_ID" >"$RUN/run-id.txt"
date -u --iso-8601=ns >"$RUN/started-at.txt"

python3 "$ROOT/scripts/assert_nodes_free.py" \
    >"$RUN/nodes-free.json" 2>"$LOGS/nodes-free.stderr" || {
        cat "$RUN/nodes-free.json" >&2
        exit 1
    }
"$ROOT/scripts/snapshot_experiment.sh" "$RUN" \
    >"$LOGS/snapshot.log" 2>&1 || exit 1

for node in "$PREFILL_NODE" "$DECODE_NODE"; do
    ssh -o BatchMode=yes "$node" date -u --iso-8601=ns \
        >"$KERNEL/$node-start.txt"
done

echo "launching P8D8 C144 diagnostic service: $RUN_ID"
ssh -o BatchMode=yes "$PREFILL_NODE" \
    bash "$BENCH_DIR/launch.sh" \
    "CONFIG=$CONFIG" "TOPOLOGY=$TOPOLOGY" "OUT_DIR=$LAUNCH" \
    2>&1 | tee "$LOGS/launch-console.log"
launch_rc=${PIPESTATUS[0]}
echo "$launch_rc" >"$RUN/launch-exit-code.txt"
if (( launch_rc != 0 )); then
    exit "$launch_rc"
fi

python3 "$ROOT/scripts/assert_live_config.py" \
    >"$RUN/live-config.json" 2>"$LOGS/live-config.stderr" || exit 1

# Keep router logs alongside launch worker logs. Worker logs are already
# persisted by launch.sh through SERVER_LOG.
ssh -o BatchMode=yes "$PREFILL_NODE" \
    "nohup docker logs --timestamps -f ${CONTAINER_PREFIX}-router >'$LAUNCH/server-logs/router.log' 2>&1 </dev/null &" \
    >/dev/null

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
        sampler_rc=$?
        echo "sampler exited before preflight: rc=$sampler_rc" >&2
        exit 1
    fi
    if (( $(date +%s) > deadline )); then
        echo "sampler preflight timed out" >&2
        stop_sampler
        exit 1
    fi
    sleep 1
done
python3 - "$RUN/sampling/live/preflight.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
if not payload.get("passed"):
    raise SystemExit(f"sampler preflight failed: {payload}")
PY

echo "running AgentX C144 warmup/lane=10 duration=3600"
bash "$BENCH_DIR/agentx_bench.sh" \
    "CONFIG=$CONFIG" "TOPOLOGY=$TOPOLOGY" \
    "CONC=144" "DURATION=3600" \
    "AGENTX_WARMUP_REQUESTS_PER_LANE=10" \
    "OUT_DIR=$BENCH" "AGENTX_CACHE_DIR=$CACHE" \
    2>&1 | tee "$LOGS/bench-console.log"
bench_rc=${PIPESTATUS[0]}
echo "$bench_rc" >"$RUN/bench-exit-code.txt"

stop_sampler
trap - INT TERM

ssh -o BatchMode=yes "$PREFILL_NODE" \
    docker logs --timestamps "${CONTAINER_PREFIX}-prefill-0" \
    >"$LOGS/prefill-final.log" 2>&1 || true
ssh -o BatchMode=yes "$DECODE_NODE" \
    docker logs --timestamps "${CONTAINER_PREFIX}-decode-0" \
    >"$LOGS/decode-final.log" 2>&1 || true
ssh -o BatchMode=yes "$PREFILL_NODE" \
    docker logs --timestamps "${CONTAINER_PREFIX}-router" \
    >"$LOGS/router-final.log" 2>&1 || true

for node in "$PREFILL_NODE" "$DECODE_NODE"; do
    since="$(cat "$KERNEL/$node-start.txt")"
    ssh -o BatchMode=yes "$node" \
        "journalctl -k --no-pager --since '$since' 2>&1 || dmesg 2>&1" \
        >"$KERNEL/$node.log" || true
    ssh -o BatchMode=yes "$node" rocm-smi --showuse --showmemuse --showpids \
        >"$RUN/$node-gpu-after.txt" 2>&1 || true
done

date -u --iso-8601=ns >"$RUN/completed-at.txt"
if (( bench_rc != 0 )); then
    printf '%s\n' "FAILED: preserve live service and inspect before cleanup" \
        >"$RUN/STATUS"
else
    printf '%s\n' "COMPLETED: live service retained for post-run inspection" \
        >"$RUN/STATUS"
fi
echo "C144 diagnostic captured at $RUN (bench exit $bench_rc)"
echo "service remains live; use scripts/stop_experiment.sh after inspection"
exit "$bench_rc"
