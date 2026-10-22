#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Save each kit container's log and `docker inspect` into the current run's
# logs/, then remove the containers (by name; never by pattern on a shared
# node): this node's first, then the decode on the decode node.
#
# Usage: bash engine/down.sh
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "$KIT/engine/common.sh"

out="$(cat "$WORK_DIR/current_run" 2>/dev/null || echo "$WORK_DIR")/logs"
mkdir -p "$out"
for c in "${PREFILL_CONTAINERS[@]}"; do
    docker inspect --type container "$c" >"$out/$c.inspect.json" 2>/dev/null || continue
    docker stop --time 60 "$c" >/dev/null || true
    docker logs --timestamps "$c" >"$out/$c.log" 2>&1 || true
    echo "stopped $c (log: $out/$c.log); container kept (yihou rule: docker stop, never rm)"
done
c="$DECODE_CONTAINER"
if on_decode "docker inspect --type container $c" >"$out/$c.inspect.json" 2>/dev/null; then
    on_decode "docker stop --time 60 $c" >/dev/null || true
    on_decode "docker logs --timestamps $c" >"$out/$c.log" 2>&1 || true
    echo "stopped $c on $DECODE_SSH (log: $out/$c.log); container kept"
fi
echo "The driver frees the engines' VRAM 10-40 minutes after removal; check rocm-smi on both nodes before up.sh."
