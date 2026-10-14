#!/usr/bin/env bash
# Purpose: Aggregate AgentX point directories and render their Pareto plot.
# Usage: ./analyze_agentx.sh RESULT_DIR=PATH [KEY=VALUE ...]
# Artifacts: one aggregate CSV and one PNG plot.
# Artifact paths: RESULT_DIR/results.csv and RESULT_DIR/pareto.png.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

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
: "${RESULT_DIR:?set RESULT_DIR}"
[[ "$RESULT_DIR" == /* ]] || RESULT_DIR="$DIR/$RESULT_DIR"
RESULT_DIR="$(cd "$RESULT_DIR" && pwd)"
python3 "$DIR/tools/plot_agentx.py" "$RESULT_DIR"
echo "CSV: $RESULT_DIR/results.csv"
echo "plot: $RESULT_DIR/pareto.png"
