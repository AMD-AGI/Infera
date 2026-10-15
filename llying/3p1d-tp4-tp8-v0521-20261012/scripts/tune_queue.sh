#!/usr/bin/env bash
# Run the tuning queue in order. The queue file is re-read before every step, so steps can
# be appended while it runs; when it is exhausted the driver waits for more. Line forms
# (blank lines and # comments are skipped; a step's position never changes):
#   round VARIANT CONC [INFERA_X=v ...]   -> tune_round.sh
#   run SCRIPT [KEY=VALUE ...]            -> bash scripts/SCRIPT (engine restarts etc.)
#   stop
# Stops at the first failing step. Progress (steps done) survives restarts of the driver.
# Run detached on the control node:
#   ROUND=r2 QUEUE=<file> setsid nohup bash tune_queue.sh > <log> 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
R="$(cd "$HERE/.." && pwd)"
: "${QUEUE:?}" "${ROUND:?}"
export ROUND
TUNE="$R/runtime/runs/$ROUND/tune"
mkdir -p "$TUNE"
echo $$ > "$TUNE/queue.pid"
done_file="$TUNE/queue.done"
n="$(cat "$done_file" 2>/dev/null || echo 0)"
log() { echo "$(date -u +%FT%TZ) queue: $*"; }
log "start pid=$$ queue=$QUEUE done=$n"
while true; do
    mapfile -t steps < <(grep -vE '^[[:space:]]*(#|$)' "$QUEUE")
    if (( n >= ${#steps[@]} )); then
        sleep 30
        continue
    fi
    read -r -a f <<< "${steps[$n]}"
    # Steps write to the shared /home (logs, results, restart state); never start one while it is full.
    bash "$HERE/wait_home_space.sh" 20 10800 || { log "/home has no room; stopping before step $((n + 1))"; exit 1; }
    log "step $((n + 1)): ${steps[$n]}"
    case "${f[0]}" in
        round)
            bash "$HERE/tune_round.sh" "VARIANT=${f[1]}" "CONC=${f[2]}" "ROUTER_ENV=${f[*]:3}" ||
                { log "step $((n + 1)) failed rc=$?; stopping"; exit 1; } ;;
        run)
            bash "$HERE/${f[1]}" "${f[@]:2}" ||
                { log "step $((n + 1)) failed rc=$?; stopping"; exit 1; } ;;
        stop)
            log "stop"; until echo $((n + 1)) > "$done_file"; do sleep 60; done; exit 0 ;;
        *)
            log "unknown step '${f[0]}'; stopping"; exit 1 ;;
    esac
    n=$((n + 1))
    until echo "$n" > "$done_file"; do sleep 60; done
done
