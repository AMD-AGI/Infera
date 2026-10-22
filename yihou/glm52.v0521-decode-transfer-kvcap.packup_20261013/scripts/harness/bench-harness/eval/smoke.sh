#!/usr/bin/env bash
# Purpose: Run router health, discovery, chat, tool-call, and burst smoke checks.
# Usage: ./eval/smoke.sh [OUT_DIR=PATH] [SMOKE_TIMEOUT=SECONDS] [KEY=VALUE ...]
# Artifacts: runner.log plus one JSON file for each smoke-check result.
# Artifact paths: OUT_DIR defaults to results/<UTC>-smoke.
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
prefills="$(python3 "$DIR/tools/topology.py" count "$TOPOLOGY" prefill)"
decodes="$(python3 "$DIR/tools/topology.py" count "$TOPOLOGY" decode)"
OUT_DIR="${OUT_DIR:-$DIR/results/$(date -u +%Y%m%dT%H%M%SZ)-smoke}"
[[ "$OUT_DIR" == /* ]] || OUT_DIR="$DIR/$OUT_DIR"
[[ ! -e "$OUT_DIR" ]] || { echo "output path already exists: $OUT_DIR" >&2; exit 1; }
mkdir -p "$OUT_DIR"

python3 "$DIR/eval/smoke.py" \
    --url "http://$control_ip:$ROUTER_PORT" --model "$SERVED_MODEL" \
    --prefill-count "$prefills" --decode-count "$decodes" \
    --output-dir "$OUT_DIR" --timeout "${SMOKE_TIMEOUT:-180}" \
    2>&1 | tee "$OUT_DIR/runner.log"
