#!/usr/bin/env bash
# Send one configurable long-context request directly to the router.
set -euo pipefail
COMPONENT=long-context
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$DIR/lib/common.sh"

load_config "$@"
OUT_DIR="${OUT_DIR:-$DIR/results/$(date -u +%Y%m%dT%H%M%SZ)-long-context}"
[[ "$OUT_DIR" == /* ]] || OUT_DIR="$DIR/$OUT_DIR"
if [[ -d "$OUT_DIR" && -n "$(ls -A "$OUT_DIR")" ]]; then
    die "output directory is not empty: $OUT_DIR"
fi
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
RUN_LOG="$OUT_DIR/runner.log"
export OUT_DIR RUN_LOG
start_log

python3 "$DIR/eval/long_context.py" \
    --url "$(router_url)" \
    --model "$SERVED_MODEL" \
    --output-dir "$OUT_DIR" \
    --tokens "${TOKENS:-250000}" \
    --timeout "${LONG_CONTEXT_TIMEOUT:-900}"
