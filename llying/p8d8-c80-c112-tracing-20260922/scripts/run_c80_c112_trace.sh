#!/usr/bin/env bash
# Fresh launch, then traced C80 -> C112 on one deployment. Never runs implicitly.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="$ROOT/config/config.trace.p8d8.sh"
TOPOLOGY="$ROOT/config/topology.trace.tsv"
PACKUP="$ROOT/../../../../yihou/glm52.p8d8.agentx-sweep.packup_20260920"
C144_ROOT="$ROOT/../p8d8-c144-runtime-rca-20260922"
source "$CONFIG"

[[ "${1:-}" == "--confirm-exclusive" ]] || {
    echo "usage: $0 --confirm-exclusive" >&2
    exit 2
}
[[ -s "$ROOT/image-id.txt" ]] || {
    echo "build_trace_image.sh must run first" >&2
    exit 1
}
RUN_ID="${RUN_ID:-c80-c112-trace-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN="$ROOT/runs/$RUN_ID"
[[ ! -e "$RUN" ]] || { echo "run exists: $RUN" >&2; exit 1; }
mkdir -p "$RUN/snapshot" "$RUN/logs"
date -u --iso-8601=ns >"$RUN/started-at.txt"

python3 "$C144_ROOT/scripts/assert_nodes_free.py" \
    >"$RUN/nodes-free.json" 2>"$RUN/logs/nodes-free.stderr"
expected="$(<"$ROOT/image-id.txt")"
for node in "$PREFILL_NODE" "$DECODE_NODE"; do
    actual="$(
        ssh -o BatchMode=yes "$node" \
            docker image inspect "$IMAGE" --format '{{.Id}}'
    )"
    [[ "$actual" == "$expected" ]] || {
        echo "$node trace image=$actual expected=$expected" >&2
        exit 1
    }
done

cp -a "$ROOT/config" "$ROOT/docker" "$ROOT/scripts" "$RUN/snapshot/"
sha256sum \
    "$BENCH_DIR/launch.sh" "$BENCH_DIR/engine.sh" "$BENCH_DIR/agentx_bench.sh" \
    "$CONFIG" "$TOPOLOGY" >"$RUN/snapshot/harness-sha256.txt"
git -C /home/liyingli/bench_agentx/baseline/Infera rev-parse HEAD \
    >"$RUN/snapshot/git-head.txt"

cleanup() {
    bash "$ROOT/scripts/collector_ctl.sh" stop >/dev/null 2>&1 || true
    bash "$BENCH_DIR/stop.sh" "CONFIG=$CONFIG" "TOPOLOGY=$TOPOLOGY" \
        >"$RUN/logs/stop.log" 2>&1 || true
}
trap cleanup EXIT INT TERM

# Collector must be reachable before traced workers initialize.
bash "$ROOT/scripts/collector_ctl.sh" start "$RUN/startup-traces" \
    >"$RUN/logs/collector-startup.log" 2>&1
ssh -o BatchMode=yes "$PREFILL_NODE" \
    bash "$BENCH_DIR/launch.sh" \
    "CONFIG=$CONFIG" "TOPOLOGY=$TOPOLOGY" "OUT_DIR=$RUN/launch" \
    >"$RUN/logs/launch.log" 2>&1
python3 "$ROOT/scripts/assert_live_trace_config.py" \
    >"$RUN/live-config.json"
bash "$ROOT/scripts/collector_ctl.sh" stop \
    >"$RUN/logs/collector-startup-stop.log" 2>&1

bash "$ROOT/scripts/run_point.sh" 80 "$RUN/c080"
bash "$ROOT/scripts/run_point.sh" 112 "$RUN/c112"

cleanup
trap - EXIT INT TERM
TIMEOUT_S=5400 INTERVAL_S=30 THRESHOLD_PCT=1 \
    bash "$PACKUP/scripts/wait_gpus_free.yihou.sh" \
    "$PREFILL_NODE" "$DECODE_NODE" >"$RUN/logs/wait-gpus-free.log" 2>&1
date -u --iso-8601=ns >"$RUN/completed-at.txt"
echo "COMPLETED" >"$RUN/STATUS"
echo "$RUN"
