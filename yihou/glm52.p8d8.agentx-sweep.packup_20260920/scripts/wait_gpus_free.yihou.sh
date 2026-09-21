#!/usr/bin/env bash
# Purpose: block until EVERY GPU on EVERY listed node reports <=1 % VRAM (THRESHOLD_PCT default 1; the idle floor is ~284 MiB of driver reserve, never literally 0), then exit 0.
#   This is the pre-launch gate. It exists because of a real incident on 2026-09-18:
#   a deployment was relaunched two minutes after a teardown, while the previous
#   deployment's GPU memory was still being released. Several ranks could not get
#   their allocation, startup jammed with three GPUs still near-empty, and the
#   wreckage was then misdiagnosed as a driver-level node fault that "needed an
#   admin reset". The node was fine; it recovered on its own ~20 minutes later.
#
#   The specific trap: after `stop.sh` the containers disappear and HOST memory
#   returns quickly, so teardown *looks* finished. GPU VRAM keeps draining for
#   many more minutes, one device at a time. Checking `docker ps` and `free` is
#   not checking the thing that matters.
#
# Usage: ./wait_gpus_free.yihou.sh NODE [NODE ...]
#   Env: TIMEOUT_S (default 1800), INTERVAL_S (default 20), THRESHOLD_PCT (default 1)
# Artifacts: none; prints a per-poll line and exits 0 (all free) or 1 (timed out).
set -uo pipefail

(( $# > 0 )) || { echo "usage: $0 NODE [NODE ...]" >&2; exit 2; }
TIMEOUT_S="${TIMEOUT_S:-1800}"
INTERVAL_S="${INTERVAL_S:-20}"
THRESHOLD_PCT="${THRESHOLD_PCT:-1}"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=10}"

start=$(date +%s)
while :; do
    all_free=1
    line="$(date -u +%H:%M:%S)"
    for node in "$@"; do
        # shellcheck disable=SC2086
        vals="$(timeout 30 ssh $SSH_OPTS "$node" \
            'rocm-smi --showmemuse 2>/dev/null | grep -oE "VRAM%\): [0-9]+" | grep -oE "[0-9]+$"' \
            2>/dev/null | tr '\n' ' ')"
        if [[ -z "$vals" ]]; then
            # An unreachable node or an unparseable rocm-smi is NOT "free".
            # Reporting free on missing data is exactly how the original mistake
            # was made, so this counts as not-free and keeps waiting.
            line+="  $node=UNREADABLE"
            all_free=0
            continue
        fi
        busy=0
        for v in $vals; do (( v > THRESHOLD_PCT )) && busy=1; done
        line+="  $node=[$vals]"
        (( busy )) && all_free=0
    done
    echo "$line"
    (( all_free )) && { echo "ALL GPUS FREE on: $*"; exit 0; }
    now=$(date +%s)
    if (( now - start > TIMEOUT_S )); then
        echo "TIMED OUT after ${TIMEOUT_S}s — GPUs still busy. Do NOT launch." >&2
        exit 1
    fi
    sleep "$INTERVAL_S"
done
