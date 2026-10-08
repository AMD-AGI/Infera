#!/usr/bin/env bash
# Launch and run one immutable P8D8/C80 cross-rank reproduction window.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${CONFIG:-$ROOT/config/config.rca.p8d8.sh}"
source "$CONFIG" || exit 1
export CONFIG
TOPOLOGY="$RCA_TOPOLOGY"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-${EXPECTED_IMAGE_ID:7:8}-c080}"
RUN="$ROOT/runs/$RUN_ID"
LAUNCH="$RUN/launch"
BENCH="$RUN/bench"
COUNTERS="$RUN/counters"
KERNEL="$RUN/kernel"
LOGS="$RUN/logs"
CACHE="${AGENTX_CACHE_DIR:-$ROOT/cache/agentx}"

if [[ -e "$RUN" ]]; then
    echo "run path already exists: $RUN" >&2
    exit 1
fi
mkdir -p "$RUN" "$COUNTERS" "$KERNEL" "$LOGS"
printf '%s\n' "$RUN_ID" >"$RUN/run-id.txt"
date -u --iso-8601=ns >"$RUN/started-at.txt"

for node in "$PREFILL_NODE" "$DECODE_NODE"; do
    ssh -o BatchMode=yes "$node" date -u --iso-8601=seconds \
        >"$KERNEL/$node-start.txt"
done

bash "$ROOT/scripts/capture_provenance.sh" "$RUN/provenance" \
    >"$LOGS/provenance.log" 2>&1 || exit 1
python3 "$ROOT/scripts/capture_hca_counters.py" "$COUNTERS/before.json" \
    >"$LOGS/counters-before.log" 2>&1 || exit 1

echo "launching RCA service: $RUN_ID"
ssh -o BatchMode=yes "$PREFILL_NODE" \
    bash "$BENCH_DIR/launch.sh" "RCA_PREFILL_NODE=$PREFILL_NODE" "RCA_DECODE_NODE=$DECODE_NODE" \
    "CONFIG=$CONFIG" "TOPOLOGY=$TOPOLOGY" "OUT_DIR=$LAUNCH" \
    2>&1 | tee "$LOGS/launch-console.log"
launch_rc=${PIPESTATUS[0]}
if (( launch_rc != 0 )); then
    echo "$launch_rc" >"$RUN/launch-exit-code.txt"
    exit "$launch_rc"
fi
echo 0 >"$RUN/launch-exit-code.txt"

ssh -o BatchMode=yes "$PREFILL_NODE" \
    "nohup docker logs --timestamps -f ${CONTAINER_PREFIX}-router >'$LAUNCH/server-logs/router.log' 2>&1 </dev/null &" \
    >/dev/null

python3 "$ROOT/scripts/assert_live_config.py" \
    >"$RUN/live-config.json" 2>"$LOGS/live-config.stderr" || exit 1

for target in \
    "router http://$PREFILL_IP:28000/metrics" \
    "prefill http://$PREFILL_IP:29001/metrics" \
    "decode http://$DECODE_IP:29002/metrics"; do
    read -r name url <<<"$target"
    curl -fsS --max-time 30 "$url" >"$RUN/${name}-metrics-before.prom" || true
done

python3 "$ROOT/scripts/monitor_hca_counters.py" \
    "$COUNTERS/timeseries.jsonl" --interval "${RCA_COUNTER_INTERVAL:-10}" --duration 7200 \
    >"$LOGS/counter-monitor.log" 2>&1 &
counter_pid=$!

echo "running AgentX C80 warmup/lane=1 duration=1200"
bash "$BENCH_DIR/agentx_bench.sh" \
    "CONFIG=$CONFIG" "TOPOLOGY=$TOPOLOGY" \
    "CONC=80" "DURATION=1200" \
    "AGENTX_WARMUP_REQUESTS_PER_LANE=1" \
    "OUT_DIR=$BENCH" "AGENTX_CACHE_DIR=$CACHE" \
    2>&1 | tee "$LOGS/bench-console.log"
bench_rc=${PIPESTATUS[0]}
echo "$bench_rc" >"$RUN/bench-exit-code.txt"

kill "$counter_pid" >/dev/null 2>&1 || true
wait "$counter_pid" >/dev/null 2>&1 || true
python3 "$ROOT/scripts/capture_hca_counters.py" "$COUNTERS/after.json" \
    >"$LOGS/counters-after.log" 2>&1 || true

for target in \
    "router http://$PREFILL_IP:28000/metrics" \
    "prefill http://$PREFILL_IP:29001/metrics" \
    "decode http://$DECODE_IP:29002/metrics"; do
    read -r name url <<<"$target"
    curl -fsS --max-time 30 "$url" >"$RUN/${name}-metrics-after.prom" || true
done

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
    ssh -o BatchMode=yes "$node" rocm-smi --showmemuse --showpids \
        >"$RUN/$node-gpu-after.txt" 2>&1 || true
done

date -u --iso-8601=ns >"$RUN/completed-at.txt"
echo "RCA run captured at $RUN (bench exit $bench_rc)"
exit "$bench_rc"
