#!/usr/bin/env bash
# Block until the shared volume holding the run (SPACE_PATH, default /perf_apps on AUS) has at least MIN_GB free. The volume is shared by every user
# and sits at 97-100%; when it is momentarily full every write fails with EDQUOT (2026-10-14 21:59,
# 10-15 05:03 and 07:57, when the whole orchestration died), so steps wait here instead of starting.
# Usage: wait_home_space.sh [MIN_GB=20] [MAX_WAIT_S=10800]   -> exit 0 when there is room, 1 on timeout.
min_gb="${1:-20}"
max_wait="${2:-10800}"
waited=0
while true; do
    free_gb="$(df -BG --output=avail "${SPACE_PATH:-/perf_apps}" 2>/dev/null | tail -1 | tr -dc '0-9')"
    (( ${free_gb:-0} >= min_gb )) && exit 0
    (( waited >= max_wait )) && { echo "/home still has ${free_gb:-?} GB free after ${waited}s" >&2; exit 1; }
    (( waited % 600 == 0 )) && echo "$(date -u +%FT%TZ) waiting for /home space: ${free_gb:-?} GB free, need ${min_gb}" >&2
    sleep 60
    waited=$((waited + 60))
done
