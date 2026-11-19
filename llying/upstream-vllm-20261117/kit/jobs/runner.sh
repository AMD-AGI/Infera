#!/usr/bin/env bash
# Plan runner for one run-and-exit job (started by run_plan.sbatch from a copy in /tmp, so this file can be edited).
#
# The plan (plans/$PLAN.plan, re-read whenever the runner schedules, so steps can be appended while the job works) has
# one step per line:   ID  NEEDS  TIMEOUT_S  ON_FAIL  [GPUS]
#   NEEDS    comma-separated step ids or "-";  ON_FAIL  continue | stop
#   GPUS     how many of the job's GPUs the step uses (default 1). Steps run in parallel as long as GPUs are free;
#            0 = CPU only (at most CPU_LANES of those at once); "all" (or the job's GPU count) = the whole job, alone
#            (also for steps that need most of the host memory). The job's GPUs are the render nodes it can see (a
#            non-exclusive allocation shows only its own); a step gets indices into them as GPU_LIST (gpu_args maps
#            only those into its containers, GPU_LIST=none for CPU-only steps) and a share of the job's CPUs as
#            STEP_CPUS. A step that asks for more GPUs than the job has fails at once.
# Step ID runs a snapshot of steps/ID.sh (results/ID/step.sh) with OUT=results/ID; helpers in jobs/lib.sh.
#
#   - Resume: a step whose results/ID/rc is 0 and whose results/ID/key equals the current key is skipped. The key
#     hashes the step script, its plan line and the output of every "# key: <command>" line in the script, so
#     changing the code under test or the step reruns it and nothing else (also within a job).
#   - A step whose needs did not all pass is marked skipped (rc "skip:<id>"); it is reconsidered if they pass later.
#   - On a non-zero rc jobs/diag.sh captures the node state into results/ID/diag/; ON_FAIL=stop starts no new step.
#   - When nothing runs and nothing is runnable the runner waits DRAIN_S (default 0: exit at once) for appended
#     steps. It stops scheduling on plans/$PLAN.STOP, or when less than MIN_LEFT_S (default 600 s) of the job is left.
# Always exits 0 (spur ends the job on a non-zero step).
U=/home/liyingli/bench_agentx/upstream-20261119
P=$U/plans/${PLAN:?PLAN}.plan
R=$U/results
END=${END:?END}
DRAIN_S=${DRAIN_S:-0}
MIN_LEFT_S=${MIN_LEFT_S:-600}
CPU_LANES=${CPU_LANES:-4}
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

status() {  # pass | fail | skip | pending (in this job)
    local d=$R/$1 rc
    [[ -e "$d/rc" && -e "$d/job" && "$(cat "$d/job")" == "$JOB" ]] || { echo pending; return; }
    rc=$(cat "$d/rc")
    case "$rc" in 0) echo pass ;; skip:*) echo skip ;; *) echo fail ;; esac
}

declare -A RUN_PID RUN_GPUS RUN_T0 RUN_KEY RUN_ONFAIL RUN_EXCL
NGPU=$(ls /dev/dri/renderD* 2> /dev/null | wc -l)
FREE=$(seq -s ' ' 0 $((NGPU - 1)) 2> /dev/null)
(( NGPU > 0 )) || FREE=""
JOB_CPUS=${JOB_CPUS:-${SLURM_CPUS_ON_NODE:-$(nproc)}}
CPU_LANES=${CPU_LANES:-$(( JOB_CPUS >= 32 ? 4 : JOB_CPUS >= 8 ? 2 : 1 ))}
export JOB_CPUS
cpu_running=0 exclusive=0 stopping=0

take_gpus() {  # N: move N free GPUs from FREE to TAKEN (same shell, no subshell); fails if fewer are free
    local n=$1 rest="" g k=0
    TAKEN=""
    for g in $FREE; do if (( k < n )); then TAKEN+="$g "; k=$((k + 1)); else rest+="$g "; fi; done
    (( k == n )) || { TAKEN=""; return 1; }
    FREE=$rest
}

done_pid() {  # PID: has the step's process exited (reaped by bash, or a zombie)?
    ! kill -0 "$1" 2> /dev/null || [[ "$(ps -o stat= -p "$1" 2> /dev/null)" == Z* ]]
}

finish() {  # ID RC
    local id=$1 rc=$2 d=$R/$1
    echo "$rc" > "$d/rc"; echo "${RUN_KEY[$id]}" > "$d/key"; echo "$JOB" > "$d/job"
    echo "$(( $(date +%s) - ${RUN_T0[$id]} ))" > "$d/seconds"
    log "end $id rc=$rc after $(( $(date +%s) - ${RUN_T0[$id]} )) s"
    if (( rc != 0 )); then
        bash "$U/jobs/diag.sh" "$d/diag" > /dev/null 2>&1
        [[ "${RUN_ONFAIL[$id]}" == stop ]] && { log "$id failed with ON_FAIL=stop; no new steps"; stopping=1; }
    fi
    local g=${RUN_GPUS[$id]}
    if [[ "$g" == cpu ]]; then cpu_running=$((cpu_running - 1)); else FREE+=" $g"; fi
    [[ -n "${RUN_EXCL[$id]:-}" ]] && exclusive=0
    docker ps -a --format '{{.Names}}' | grep -E "^up-$id(-|$)" | xargs -r docker rm -f > /dev/null 2>&1
    unset "RUN_PID[$id]" "RUN_GPUS[$id]" "RUN_T0[$id]" "RUN_KEY[$id]" "RUN_ONFAIL[$id]" "RUN_EXCL[$id]"
}

log "runner up on $NODE, job $JOB, plan $PLAN, ends $(date -u -d @"$END" '+%T'), drain ${DRAIN_S} s, $NGPU GPUs, $JOB_CPUS CPUs, $CPU_LANES CPU lanes"
idle_since=""
while true; do
    for id in "${!RUN_PID[@]}"; do
        done_pid "${RUN_PID[$id]}" || continue
        wait "${RUN_PID[$id]}"; finish "$id" $?
    done
    [[ -e "$U/plans/$PLAN.STOP" ]] && stopping=1
    left=$(( END - $(date +%s) ))
    (( left < MIN_LEFT_S )) && stopping=1
    started=0
    if (( ! stopping && ! exclusive )); then
        while read -r id needs timeout on_fail gpus; do
            gpus=${gpus:-1}
            [[ -n "${RUN_PID[$id]:-}" ]] && continue
            line="$id $needs $timeout $on_fail $gpus"
            d=$R/$id
            s=$(status "$id")
            mkdir -p "$d"
            key=$(step_key "$id" "$line")
            [[ "$s" == pass && "$(cat "$d/key" 2>/dev/null)" == "$key" ]] && continue
            [[ "$s" == pass ]] && s=fail
            [[ "$s" == fail && "$(cat "$d/key" 2>/dev/null)" == "$key" ]] && continue
            blocked=""
            if [[ "$needs" != - ]]; then
                for n in ${needs//,/ }; do
                    ns=$(status "$n")
                    [[ "$ns" == pending || -n "${RUN_PID[$n]:-}" ]] && { blocked=wait; break; }
                    [[ "$ns" != pass ]] && { blocked=$n; break; }
                done
            fi
            [[ "$blocked" == wait ]] && continue
            if [[ -n "$blocked" ]]; then
                [[ "$s" == skip ]] && continue
                echo "skip:$blocked" > "$d/rc"; echo "$key" > "$d/key"; echo "$JOB" > "$d/job"
                log "skip $id (needs $blocked, which did not pass)"
                continue
            fi
            if [[ "$s" == pending && -z "${FORCE:-}" && "$(cat "$d/rc" 2>/dev/null)" == 0 \
                  && "$(cat "$d/key" 2>/dev/null)" == "$key" ]]; then
                echo "$JOB" > "$d/job"
                log "skip $id (passed earlier with the same key $key)"
                continue
            fi
            # Resources: 8 = alone; 0 = a CPU lane; else that many free GPUs.
            excl=""
            [[ "$gpus" == all ]] && gpus=$NGPU
            if (( gpus > NGPU )); then
                echo 125 > "$d/rc"; echo "$key" > "$d/key"; echo "$JOB" > "$d/job"
                echo "needs $gpus GPUs, the job has $NGPU" > "$d/log"
                log "fail $id: needs $gpus GPUs, the job has $NGPU"
                continue
            fi
            if (( gpus > 0 && gpus == NGPU )); then
                # Needs all of the job's GPUs: start nothing else until it has run.
                (( ${#RUN_PID[@]} == 0 )) || break
                take_gpus "$gpus" || break
                list=$TAKEN excl=1 exclusive=1
                cpus=$JOB_CPUS
            elif (( gpus == 0 )); then
                (( cpu_running < CPU_LANES )) || continue
                list=cpu
                cpu_running=$((cpu_running + 1))
                cpus=$(( JOB_CPUS / (CPU_LANES + (NGPU > 0 ? 1 : 0)) ))
            else
                take_gpus "$gpus" || continue
                list=$TAKEN
                cpus=$(( JOB_CPUS * gpus / NGPU ))
            fi
            (( cpus >= 2 )) || cpus=2
            limit=$(( timeout < left - 300 ? timeout : left - 300 ))
            rm -rf "$d/diag"
            cp "$U/steps/$id.sh" "$d/step.sh"
            gl=none
            [[ "$list" != cpu ]] && gl=$(echo $list | tr ' ' ',')
            log "start $id (gpus $gl, cpus $cpus, timeout $limit s, key $key)"
            (
                export OUT=$d STEP=$id GPU_LIST=$gl STEP_CPUS=$cpus
                exec timeout -k 30 "$limit" bash "$d/step.sh" > "$d/log" 2>&1
            ) &
            RUN_PID[$id]=$!; RUN_GPUS[$id]=$list; RUN_T0[$id]=$(date +%s); RUN_KEY[$id]=$key; RUN_ONFAIL[$id]=$on_fail
            RUN_EXCL[$id]=$excl
            started=1
            (( exclusive )) && break
        done < <(plan_lines)
    fi
    if (( ${#RUN_PID[@]} == 0 )); then
        (( stopping )) && { log "stopping (STOP, ON_FAIL=stop or time)"; break; }
        if (( ! started )); then
            [[ -z "$idle_since" ]] && { idle_since=$(date +%s); log "nothing runnable; waiting up to $DRAIN_S s for appended steps"; }
            (( $(date +%s) - idle_since >= DRAIN_S )) && { log "drain window over"; break; }
        fi
    else
        idle_since=""
    fi
    sleep 5
done
kill "$hb" 2> /dev/null
rm -f "$U/plans/$PLAN.hb"
log "runner done"
exit 0
