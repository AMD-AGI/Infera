#!/usr/bin/env bash
# Start/stop the persistent OTLP JSONL collector on the Prefill control node.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/config/config.trace.p8d8.sh"
ACTION="${1:?usage: $0 start OUTPUT_DIR | stop}"
CONTAINER="${CONTAINER_PREFIX}-otel-collector"

case "$ACTION" in
    start)
        OUT="${2:?usage: $0 start OUTPUT_DIR}"
        mkdir -p "$OUT"
        rm -f "$OUT/ready.json"
        ssh -o BatchMode=yes "$PREFILL_NODE" \
            docker run -d --init --name "$CONTAINER" --network host \
            -v "$ROOT:$ROOT" \
            "$IMAGE" \
            python3 "$ROOT/scripts/otlp_jsonl_collector.py" \
            --port 4317 \
            --output "$OUT/spans.jsonl" \
            --ready-file "$OUT/ready.json" \
            >"$OUT/container-id.txt"
        deadline=$(( $(date +%s) + 60 ))
        while [[ ! -s "$OUT/ready.json" ]]; do
            ssh -o BatchMode=yes "$PREFILL_NODE" \
                docker inspect "$CONTAINER" --format '{{.State.Running}}' |
                awk '$1 == "true" {ok=1} END {exit !ok}'
            (( $(date +%s) <= deadline )) || {
                echo "OTLP collector readiness timeout" >&2
                exit 1
            }
            sleep 1
        done
        ;;
    stop)
        ssh -o BatchMode=yes "$PREFILL_NODE" \
            docker stop --time 30 "$CONTAINER" >/dev/null 2>&1 || true
        ssh -o BatchMode=yes "$PREFILL_NODE" \
            docker rm "$CONTAINER" >/dev/null 2>&1 || true
        ;;
    *)
        echo "unknown action: $ACTION" >&2
        exit 2
        ;;
esac
