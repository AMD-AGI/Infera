#!/usr/bin/env bash
# Patrol on the login host: every PERIOD seconds look at the holder jobs, the holder log, verify.out, ref.out and the
# optimisation queue log; print one AGENT_LOOP_WAKE_vfix line (picked up by the agent's output watch) when something
# changed that needs a look (new result / failure line, job state change, holder item finished, no progress for STALL
# seconds), and an AGENT_LOOP_TICK_vfix heartbeat every HEARTBEAT seconds otherwise; every SYNC seconds changed results are
# committed through sync_records.sh. Status lines go to $STATE/patrol.log.
set -uo pipefail
PERIOD=${PERIOD:-600} HEARTBEAT=${HEARTBEAT:-1800} STALL=${STALL:-3600}
L=/perf_apps/liyingli/vfix-logs
H=/apps/liyingli/holder
STATE=${STATE:-$L/patrol}
mkdir -p "$STATE"
files=("$L/verify.out" "$L/ref/ref.out" "$L/opt/opt.out" "$H/holder-35060.out" "$H/holder-35103.out")
size() { stat -c %s "$1" 2>/dev/null || echo 0; }
declare -A last
for f in "${files[@]}"; do last[$f]=$(size "$f"); done
last_jobs="$(squeue -h -u "$USER" -o '%i %T %N' 2>/dev/null | sort)"
last_change=$SECONDS last_beat=$SECONDS last_sync=$SECONDS synced_change=$SECONDS
while sleep "$PERIOD"; do
    why=()
    jobs="$(squeue -h -u "$USER" -o '%i %T %N %L' 2>/dev/null | sort)"
    [[ "$(cut -d' ' -f1-3 <<<"$jobs")" == "$last_jobs" ]] || why+=("jobs: $(tr '\n' ';' <<<"$jobs")")
    last_jobs="$(cut -d' ' -f1-3 <<<"$jobs")"
    for f in "${files[@]}"; do
        s=$(size "$f")
        if ((s != ${last[$f]})); then
            new="$(tail -c +$((${last[$f]} + 1)) "$f" 2>/dev/null)"
            last[$f]=$s last_change=$SECONDS
            echo "$(date -u +%T) ${f##*/}: $(grep -c '' <<<"$new") new lines" >> "$STATE/patrol.log"
            bad="$(grep -aiE 'fail|stopping|error|did not come up|traceback|killed|cancel|preempt|timeout|requeue' <<<"$new" | grep -v 'errors=0' | grep -v ' up: ' | tail -3)"
            [[ -z "$bad" ]] || why+=("${f##*/}: ${bad//$'\n'/ | }")
            res="$(grep -aE ' (probe|down|score|lm-eval|audit|done|run |up failed)|holder .*(run|done)|judge|FAIL lines' <<<"$new" | tail -4)"
            [[ -z "$res" ]] || why+=("${f##*/}: ${res//$'\n'/ | }")
        fi
    done
    ((SECONDS - last_change > STALL)) && { why+=("no log progress for $(((SECONDS - last_change) / 60)) min"); last_change=$SECONDS; }
    # Hourly: results that changed go into the Infera checkout and are committed (records survive a missed wake).
    if ((SECONDS - last_sync >= ${SYNC:-3600} && last_change != synced_change)); then
        COMMIT="llying: AUS vfix / opt auto-sync of results $(date -u +%m%dT%H%M)" bash "$(dirname "$0")/sync_records.sh" >> "$STATE/patrol.log" 2>&1
        last_sync=$SECONDS synced_change=$last_change
    fi
    if ((${#why[@]})); then
        msg="$(printf '%s || ' "${why[@]}" | cut -c1-900 | tr -d '"')"
        echo "$(date -u +%T) WAKE $msg" >> "$STATE/patrol.log"
        echo "AGENT_LOOP_WAKE_vfix {\"prompt\":\"patrol: $msg\"}"
        last_beat=$SECONDS
    elif ((SECONDS - last_beat >= HEARTBEAT)); then
        echo "AGENT_LOOP_TICK_vfix {\"prompt\":\"patrol heartbeat: check progress and advance the plan\"}"
        last_beat=$SECONDS
    fi
done
