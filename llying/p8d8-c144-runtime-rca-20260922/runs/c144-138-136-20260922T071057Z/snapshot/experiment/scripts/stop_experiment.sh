#!/usr/bin/env bash
# Stop only this experiment's containers, then wait for HiCache VRAM release.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/config/config.c144.p8d8.sh"
OUT="${1:-$ROOT/operations/stop-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT"

for node in "$PREFILL_NODE" "$DECODE_NODE"; do
    ssh -o BatchMode=yes "$node" docker ps -a --no-trunc \
        >"$OUT/$node-before.txt"
    mapfile -t names < <(
        ssh -o BatchMode=yes "$node" docker ps -a --format '{{.Names}}' |
            awk -v prefix="$CONTAINER_PREFIX" 'index($0, prefix) == 1'
    )
    if (( ${#names[@]} )); then
        printf 'stopping on %s: %s\n' "$node" "${names[*]}"
        ssh -o BatchMode=yes "$node" docker stop --time 300 "${names[@]}" \
            >"$OUT/$node-stop.txt" 2>&1 || true
        ssh -o BatchMode=yes "$node" docker rm "${names[@]}" \
            >"$OUT/$node-remove.txt" 2>&1 || true
    else
        echo "no matching containers" >"$OUT/$node-stop.txt"
    fi
    ssh -o BatchMode=yes "$node" docker ps -a --no-trunc \
        >"$OUT/$node-after.txt"
done

TIMEOUT_S=5400 INTERVAL_S=30 THRESHOLD_PCT=1 \
    bash "$PACKUP_ROOT/scripts/wait_gpus_free.yihou.sh" \
    "$PREFILL_NODE" "$DECODE_NODE" | tee "$OUT/wait-gpus-free.log"
date -u --iso-8601=ns >"$OUT/completed-at.txt"
echo "experiment containers stopped; evidence at $OUT"
