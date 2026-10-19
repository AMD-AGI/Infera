#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Save each kit container's log and `docker inspect` into the current run's
# logs/, remove the containers, then WAIT for VRAM to drain. Relaunching
# before the driver has reclaimed the engines' memory fails the next start.
#
# Usage: bash engine/down.sh
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "$KIT/engine/common.sh"

out="$(cat "$WORK_DIR/current_run" 2>/dev/null || echo "$WORK_DIR")/logs"
mkdir -p "$out"
failed=""
# Prefills on other nodes (REMOTE_PREFILLS) go down in parallel.
remote=""
if [[ -n "$REMOTE_PREFILLS" ]]; then
    bash "$REMOTE_PREFILLS" down &
    remote=$!
fi
for c in "${CONTAINERS[@]}"; do
    docker inspect --type container "$c" >"$out/$c.inspect.json" 2>/dev/null || { rm -f "$out/$c.inspect.json"; continue; }
    # Let ATOM release its workers and registered KV pools before removal.
    docker stop --time 60 "$c" >/dev/null || echo "graceful stop of $c failed; removing" >&2
    # --rm clients are gone after the stop; their output is already tee'd into the run dir.
    docker logs --timestamps "$c" >"$out/$c.log" 2>&1 || rm -f "$out/$c.log"
    # An engine holding pinned LMCache memory can outlast one removal attempt.
    for _ in 1 2 3 4 5 6; do
        docker rm -f "$c" >/dev/null 2>&1 && break
        docker inspect --type container "$c" >/dev/null 2>&1 || break
        sleep 10
    done
    if docker inspect --type container "$c" >/dev/null 2>&1; then failed+=" $c"; continue; fi
    if [[ -f "$out/$c.log" ]]; then echo "removed $c (log: $out/$c.log)"; else echo "removed $c"; fi
done
[[ -z "$remote" ]] || wait "$remote" || failed+=" (remote prefills)"
[[ -z "$failed" ]] || die "failed to remove:$failed"

echo "waiting for VRAM to drain"
wait_vram_free
echo "down"
