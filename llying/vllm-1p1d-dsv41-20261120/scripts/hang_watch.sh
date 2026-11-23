#!/usr/bin/env bash
# Passive decode-hang capture (runs on the node). Waits for a new "No available shared memory broadcast block"
# line in the decode log -- vLLM prints it once a worker has not taken a step for 60 s -- then dumps every
# process under the decode serve PID with py-spy (Python + native frames) twice, 60 s apart, plus GPU state.
# Never signals or restarts anything.
#   hang_watch.sh LOG_DIR DECODE_SERVE_PID|auto [CTR]
# auto: wait for the golden smoke (smoke-golden.json), i.e. the decode's last restart before AgentX, then find the
# decode serve process and install py-spy in the container.
set -uo pipefail
LOG_DIR=${1:?} PID=${2:?} CTR=${3:-llying-v1p1d-pd}
OUT=$LOG_DIR/hang-dumps
ts() { date -u +%FT%TZ; }
if [ "$PID" = auto ]; then
    until [ -s "$LOG_DIR/smoke-golden.json" ]; do sleep 20; done
    PID=$(docker exec "$CTR" pgrep -f "cli.main serve .*--port 30001" | head -1)
    [ -n "$PID" ] || { echo "$(ts) no decode serve process found; exiting"; exit 1; }
    docker exec "$CTR" bash -c "command -v py-spy >/dev/null || timeout 300 python3 -m pip install -q --no-warn-script-location py-spy" > /dev/null 2>&1
    docker exec "$CTR" bash -c "command -v py-spy" > /dev/null || echo "$(ts) py-spy unavailable; dumps will be empty"
fi
base=$(grep -ac "shm_broadcast.py:878" "$LOG_DIR/decode.log")
echo "$(ts) watching $LOG_DIR/decode.log (baseline $base stall lines), decode serve pid $PID"
dump() {
    local tag=$1
    mkdir -p "$OUT"
    docker exec "$CTR" bash -c "
        kids() { for c in \$(pgrep -P \$1); do echo \$c; kids \$c; done; }
        for p in \$(kids $PID); do
            [ \"\$(cat /proc/\$p/comm 2>/dev/null)\" = python3 ] || continue
            echo \"===== pid \$p: \$(tr '\0' ' ' < /proc/\$p/cmdline | cut -c1-200)\"
            timeout 90 py-spy dump --native --pid \$p 2>&1
        done" > "$OUT/pyspy-$tag.txt" 2>&1
    docker exec "$CTR" bash -c "amd-smi metric -g 4 5 2>&1 || rocm-smi 2>&1" > "$OUT/gpu-$tag.txt" 2>&1
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
