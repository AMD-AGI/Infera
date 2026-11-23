#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: passive capture of a DP-lockstep hang in one leg (the kit's engine/hang_watch.sh, generalized to prefill).
#       Optionally waits for WAIT_FILE, then for a new "No available shared memory broadcast block" line in the leg's
#       log (vLLM prints it once a worker has not returned a step for 60 s) and dumps every Python process under the
#       leg's serve process with py-spy (Python + native frames) twice, 60 s apart, plus GPU state. Never signals or
#       restarts anything.
# why : both DP2 legs occasionally stop with one rank in a real step and the other in a dummy batch (decode: AUS 1 of
#       4, crusoe 2; prefill with MegaMoE: crusoe 1); the stacks say where each rank is waiting.
# how : hang_watch2.sh LOG_DIR CTR PORT [WAIT_FILE] [LEG]
#       WAIT_FILE: file under LOG_DIR to wait for (smoke-golden.json), or `none`; LEG: decode | prefill (decode).
#       Dumps go to LOG_DIR/hang-dumps-LEG/.
set -uo pipefail
LOG_DIR=${1:?} CTR=${2:?} PORT=${3:?} WAIT_FILE=${4:-smoke-golden.json} LEG=${5:-decode}
LEG_LOG=$LOG_DIR/$LEG.log
OUT=$LOG_DIR/hang-dumps-$LEG
ts() { date -u +%FT%TZ; }
[ "$WAIT_FILE" = none ] || until [ -s "$LOG_DIR/$WAIT_FILE" ]; do sleep 20; done
PID=$(docker exec "$CTR" pgrep -f "cli.main serve .*--port $PORT( |$)" | head -1)
[ -n "$PID" ] || { echo "$(ts) no $LEG serve process on :$PORT; exiting"; exit 0; }
docker exec "$CTR" bash -c "command -v py-spy >/dev/null || timeout 300 python3 -m pip install -q --no-warn-script-location py-spy" > /dev/null 2>&1
docker exec "$CTR" bash -c "command -v py-spy" > /dev/null || echo "$(ts) py-spy unavailable; dumps will hold GPU state only"
base=$(grep -ac "shm_broadcast.py:878" "$LEG_LOG")
echo "$(ts) watching $LEG_LOG (baseline $base stall lines), $LEG serve pid $PID"
dump() {
    local tag=$1-$(date -u +%H%M%S)
    mkdir -p "$OUT"
    docker exec "$CTR" bash -c "
        kids() { for c in \$(pgrep -P \$1); do echo \$c; kids \$c; done; }
        for p in \$(kids $PID); do
            [ \"\$(cat /proc/\$p/comm 2>/dev/null)\" = python3 ] || continue
            echo \"===== pid \$p: \$(tr '\0' ' ' < /proc/\$p/cmdline | cut -c1-200)\"
            timeout 90 py-spy dump --native --pid \$p 2>&1
        done" > "$OUT/pyspy-$tag.txt" 2>&1
    docker exec "$CTR" bash -c "rocm-smi --showuse --showmemuse --showpower 2>&1; amd-smi metric 2>&1 | head -400" > "$OUT/gpu-$tag.txt" 2>&1
    echo "$(ts) dumped $tag: $(wc -l < "$OUT/pyspy-$tag.txt") lines"
}
while sleep 10; do
    docker exec "$CTR" kill -0 "$PID" 2>/dev/null || { echo "$(ts) $LEG serve pid gone; exiting"; exit 0; }
    n=$(grep -ac "shm_broadcast.py:878" "$LEG_LOG")
    if ((n > base)); then
        echo "$(ts) stall line seen ($n > $base): $(grep -a 'shm_broadcast.py:878' "$LEG_LOG" | tail -1 | cut -c1-120)"
        dump first
        sleep 60
        dump second
        exit 0
    fi
done
