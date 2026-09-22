#!/usr/bin/env bash
# Purpose: Send one configurable long-context request through the router.
# Usage: ./eval/long_context.sh [TOKENS=N] [OUT_DIR=PATH] [KEY=VALUE ...]
# Artifacts: runner.log, full response.json, and timing/token summary.json.
# Artifact paths: OUT_DIR defaults to results/<UTC>-long-context.
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"

for assignment in "$@"; do
    [[ "$assignment" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] ||
        { echo "expected KEY=VALUE, got '$assignment'" >&2; exit 2; }
    export "$assignment"
done
CONFIG="${CONFIG:-$DIR/config.sh}"
[[ -r "$CONFIG" ]] || { echo "config is not readable: $CONFIG" >&2; exit 1; }
set -a
source "$CONFIG"
set +a
TOPOLOGY="${TOPOLOGY:-$DIR/topology.tsv}"
control_ip="$(python3 "$DIR/tools/topology.py" node-ip "$TOPOLOGY" "$CONTROL_NODE")"
OUT_DIR="${OUT_DIR:-$DIR/results/$(date -u +%Y%m%dT%H%M%SZ)-long-context}"
[[ "$OUT_DIR" == /* ]] || OUT_DIR="$DIR/$OUT_DIR"
[[ ! -e "$OUT_DIR" ]] || { echo "output path already exists: $OUT_DIR" >&2; exit 1; }
mkdir -p "$OUT_DIR"

python3 "$DIR/eval/long_context.py" \
    --url "http://$control_ip:$ROUTER_PORT" --model "$SERVED_MODEL" \
    --output-dir "$OUT_DIR" --tokens "${TOKENS:-250000}" \
    --timeout "${LONG_CONTEXT_TIMEOUT:-900}" 2>&1 | tee "$OUT_DIR/runner.log"
