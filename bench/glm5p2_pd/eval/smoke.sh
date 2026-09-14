#!/usr/bin/env bash
# Run direct health, discovery, chat, tool-call, and burst checks.
set -euo pipefail
COMPONENT=smoke
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$DIR/lib/common.sh"

load_config "$@"
OUT_DIR="${OUT_DIR:-$DIR/results/$(date -u +%Y%m%dT%H%M%SZ)-smoke}"
[[ "$OUT_DIR" == /* ]] || OUT_DIR="$DIR/$OUT_DIR"
if [[ -d "$OUT_DIR" && -n "$(ls -A "$OUT_DIR")" ]]; then
    die "output directory is not empty: $OUT_DIR"
fi
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
RUN_LOG="$OUT_DIR/runner.log"
export OUT_DIR RUN_LOG
start_log

python3 "$DIR/eval/smoke.py" \
    --url "$(router_url)" \
    --model "$SERVED_MODEL" \
    --prefill-count "$(topology_count prefill)" \
    --decode-count "$(topology_count decode)" \
    --output-dir "$OUT_DIR" \
    --timeout "${SMOKE_TIMEOUT:-180}"
