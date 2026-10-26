#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Save each kit container's log and `docker inspect` into the current run's
# logs/, then remove the containers (by name; never by pattern on a shared node).
#
# Usage: bash engine/down.sh
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "$KIT/engine/common.sh"

out="$(cat "$WORK_DIR/current_run" 2>/dev/null || echo "$WORK_DIR")/logs"
mkdir -p "$out"
for c in "${CONTAINERS[@]}"; do
    docker inspect --type container "$c" >"$out/$c.inspect.json" 2>/dev/null || { rm -f "$out/$c.inspect.json"; continue; }
    docker stop --time 60 "$c" >/dev/null || true
    docker logs --timestamps "$c" >"$out/$c.log" 2>&1 || true
    docker rm -f "$c" >/dev/null || { echo "could not remove $c yet; run down.sh again"; continue; }
    echo "removed $c (log: $out/$c.log)"
done
echo "The driver frees the engine's VRAM 10-40 minutes after removal; check rocm-smi before up.sh."
