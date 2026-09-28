#!/usr/bin/env bash
set -euo pipefail
: "${CONFIG:?set CONFIG}"
export CONFIG
set -a; source "$CONFIG"; set +a
python3 "$TRACE_RUNTIME/scripts/transition_two_node.py"
python3 "$TRACE_RUNTIME/scripts/performance_preflight.py"
bash "$TRACE_RUNTIME/scripts/run-performance.sh"
python3 "$TRACE_RUNTIME/scripts/analyze_decode_prefix.py" "$RUN" > "$RUN/logs/decode-prefix-analysis.log"
