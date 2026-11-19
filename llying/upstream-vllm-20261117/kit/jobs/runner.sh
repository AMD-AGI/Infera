#!/usr/bin/env bash
# Plan runner for one run-and-exit job (started by run_plan.sbatch from a copy in /tmp, so this file can be edited).
#
# The plan (plans/$PLAN.plan, re-read before every step, so steps can be appended while the job works) has one step
# per line:   ID  NEEDS  TIMEOUT_S  ON_FAIL      (NEEDS: comma-separated step ids or "-"; ON_FAIL: continue | stop)
# Step ID runs steps/ID.sh (sourced helpers: jobs/lib.sh) with OUT=results/ID and the job's env.
#
#   - Resume: a step whose results/ID/rc is 0 and whose results/ID/key equals the current key is skipped. The key
#     hashes the step script, its plan line and the output of every "# key: <command>" line in the script (e.g. the
#     commit of the branch it tests), so changing the code or the step reruns it and nothing else.
#   - A step whose needs did not all pass is marked skipped (rc "skip:<id>") and not run.
#   - On a non-zero rc jobs/diag.sh captures the node state into results/ID/diag/; ON_FAIL=stop ends the plan.
#   - When nothing is runnable the runner waits up to DRAIN_S (default 600 s) for appended steps, then exits. It also
#     stops on queue file plans/$PLAN.STOP, or when less than MIN_LEFT_S (default 600 s) of the job is left.
# Always exits 0 (spur ends the job on a non-zero step).
U=/home/liyingli/bench_agentx/upstream-20261119
P=$U/plans/${PLAN:?PLAN}.plan
R=$U/results
END=${END:?END}
DRAIN_S=${DRAIN_S:-600}
MIN_LEFT_S=${MIN_LEFT_S:-600}
mkdir -p "$R" "$U/logs"
LOG=$U/logs/plan-$PLAN-$JOB.log
log() { echo "$(date -u '+%F %T') $*" | tee -a "$LOG"; }
( while true; do date +%s > "$U/plans/$PLAN.hb"; sleep 20; done ) &
hb=$!

plan_lines() { grep -vE '^\s*(#|$)' "$P" 2>/dev/null; }

step_key() {
    local id=$1 line=$2 f=$U/steps/$1.sh
    {
        cat "$f"
        echo "$line"
        grep -E '^# key: ' "$f" | sed 's/^# key: //' | while read -r c; do bash -c "$c" 2>&1; done
    } | sha256sum | cut -c1-16
}

status() {  # pass | fail | skip | pending
    local d=$R/$1 rc
    [[ -e "$d/rc" && -e "$d/job" && "$(cat "$d/job")" == "$JOB" ]] || { echo pending; return; }
    rc=$(cat "$d/rc")
    case "$rc" in 0) echo pass ;; skip:*) echo skip ;; *) echo fail ;; esac
}

log "runner up on $NODE, job $JOB, plan $PLAN, ends $(date -u -d @"$END" '+%T')"
idle_since=""
while true; do
    [[ -e "$U/plans/$PLAN.STOP" ]] && { log "STOP file"; break; }
    left=$(( END - $(date +%s) ))
    (( left < MIN_LEFT_S )) && { log "only $left s left; stopping"; break; }
    ran=0
    while read -r id needs timeout on_fail; do
        line="$id $needs $timeout $on_fail"
        d=$R/$id
        s=$(status "$id")
        mkdir -p "$d"
        key=$(step_key "$id" "$line")
        # A step that passed in this job reruns only if its code or step changed since.
        if [[ "$s" == pass ]]; then
            [[ "$(cat "$d/key" 2>/dev/null)" == "$key" ]] && continue
            s=fail
        fi
        # In this job: rerun a failed step once its code or step changed, and a skipped one once its needs pass.
        if [[ "$s" == fail && "$(cat "$d/key" 2>/dev/null)" == "$key" ]]; then continue; fi
        if [[ "$s" == skip ]]; then
            ok=1
            for n in ${needs//,/ }; do [[ "$(status "$n")" == pass ]] || ok=0; done
            (( ok )) || continue
        fi
        if [[ "$s" == pending && -z "${FORCE:-}" && -e "$d/rc" && "$(cat "$d/rc")" == 0 \
              && "$(cat "$d/key" 2>/dev/null)" == "$key" ]]; then
            echo "$JOB" > "$d/job"
            log "skip $id (passed earlier with the same key $key)"
            ran=1; break
        fi
        blocked=""
        if [[ "$needs" != - ]]; then
            for n in ${needs//,/ }; do
                s=$(status "$n")
                [[ "$s" == pending ]] && { blocked=wait; break; }
                [[ "$s" != pass ]] && { blocked=$n; break; }
            done
        fi
        [[ "$blocked" == wait ]] && continue
        if [[ -n "$blocked" ]]; then
            [[ "$s" == skip ]] && continue
            echo "skip:$blocked" > "$d/rc"; echo "$key" > "$d/key"; echo "$JOB" > "$d/job"
            log "skip $id (needs $blocked, which did not pass)"
            ran=1; break
        fi
        limit=$(( timeout < left - 300 ? timeout : left - 300 ))
        log "start $id (timeout $limit s, key $key)"
        rm -rf "$d/diag"
        t0=$(date +%s)
        OUT=$d STEP=$id timeout -k 30 "$limit" bash "$U/steps/$id.sh" > "$d/log" 2>&1
        rc=$?
        echo "$rc" > "$d/rc"; echo "$key" > "$d/key"; echo "$JOB" > "$d/job"
        echo "$(( $(date +%s) - t0 ))" > "$d/seconds"
        log "end $id rc=$rc after $(( $(date +%s) - t0 )) s"
        if (( rc != 0 )); then
            bash "$U/jobs/diag.sh" "$d/diag" > /dev/null 2>&1
            [[ "$on_fail" == stop ]] && { log "$id failed with ON_FAIL=stop; ending the plan"; touch "$U/plans/$PLAN.STOP.auto"; }
        fi
        docker ps -a --format '{{.Names}}' | grep -E '^up-' | xargs -r docker rm -f > /dev/null 2>&1
        ran=1; break
    done < <(plan_lines)
    [[ -e "$U/plans/$PLAN.STOP.auto" ]] && { rm -f "$U/plans/$PLAN.STOP.auto"; break; }
    if (( ran )); then idle_since=""; continue; fi
    [[ -z "$idle_since" ]] && { idle_since=$(date +%s); log "nothing runnable; waiting up to $DRAIN_S s for appended steps"; }
    (( $(date +%s) - idle_since >= DRAIN_S )) && { log "drain window over"; break; }
    sleep 15
done
kill "$hb" 2> /dev/null
log "runner done"
exit 0
