#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: passive capture of a decode DP-lockstep hang. Waits for decode's golden restart (smoke-golden.json), then for a
#       new "No available shared memory broadcast block" line in decode.log (vLLM prints it once a worker has not
#       returned a step for 60 s) and dumps every Python process under the decode serve process with py-spy (Python +
#       native frames) twice, 60 s apart, plus GPU state. Never signals or restarts anything.
# why : the decode DP2 engine occasionally stops with both workers inside a step (AUS 1 of 4 formal runs, crusoe
#       1 of 2; see llying/vllm-1p1d-dsv41-20261120/DECODE_HANG.zh-CN.md); the stacks say which collective each rank
#       is waiting in.
# how : hang_watch.sh LOG_DIR CTR DECODE_PORT [WAIT_FILE]   (the crusoe wrapper starts it in the background for
#       `agentx`). WAIT_FILE defaults to smoke-golden.json under LOG_DIR; `none` starts watching the running decode now.
set -uo pipefail
LOG_DIR=${1:?} CTR=${2:?} PORT=${3:?} WAIT_FILE=${4:-smoke-golden.json}
OUT=$LOG_DIR/hang-dumps
ts() { date -u +%FT%TZ; }
[ "$WAIT_FILE" = none ] || until [ -s "$LOG_DIR/$WAIT_FILE" ]; do sleep 20; done
PID=$(docker exec "$CTR" pgrep -f "cli.main serve .*--port $PORT( |$)" | head -1)
[ -n "$PID" ] || { echo "$(ts) no decode serve process on :$PORT; exiting"; exit 0; }
docker exec "$CTR" bash -c "command -v py-spy >/dev/null || timeout 300 python3 -m pip install -q --no-warn-script-location py-spy" > /dev/null 2>&1
docker exec "$CTR" bash -c "command -v py-spy" > /dev/null || echo "$(ts) py-spy unavailable; dumps will hold GPU state only"
base=$(grep -ac "shm_broadcast.py:878" "$LOG_DIR/decode.log")
echo "$(ts) watching $LOG_DIR/decode.log (baseline $base stall lines), decode serve pid $PID"
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
    docker exec "$CTR" kill -0 "$PID" 2>/dev/null || { echo "$(ts) decode serve pid gone; exiting"; exit 0; }
    n=$(grep -ac "shm_broadcast.py:878" "$LOG_DIR/decode.log")
    if ((n > base)); then
        echo "$(ts) stall line seen ($n > $base): $(grep -a 'shm_broadcast.py:878' "$LOG_DIR/decode.log" | tail -1 | cut -c1-120)"
        dump first
        sleep 60
        dump second
        exit 0
    fi
done
