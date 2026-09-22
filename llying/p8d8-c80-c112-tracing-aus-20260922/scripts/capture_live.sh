#!/usr/bin/env bash
# Copy diagnostics from node-local disks; preserve remote data if copying fails.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/config/config.sh"
read -r -a opts <<< "$SSH_OPTS"
mkdir -p "$RUN/diagnostics/prefill" "$RUN/diagnostics/decode" "$RUN/server-logs"
for role in prefill decode; do
    if [[ "$role" == prefill ]]; then node="$PREFILL_NODE"; else node="$DECODE_NODE"; fi
    ssh "${opts[@]}" "$node" "tar -C /tmp/aus-diag-$RUN_ID/$role -cf - ." |
        tar -xf - -C "$RUN/diagnostics/$role"
    ssh "${opts[@]}" "$node" docker logs --timestamps "$CONTAINER_PREFIX-$role-0" \
        > "$RUN/server-logs/$role.log" 2>&1
done
date -u --iso-8601=ns > "$RUN/last-capture.txt"
