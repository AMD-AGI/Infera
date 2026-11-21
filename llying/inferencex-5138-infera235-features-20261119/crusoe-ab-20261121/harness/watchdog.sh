#!/usr/bin/env bash
# Login-node watchdog, so a stalled point does not hold a node idle: every 60 s it reads each running point's logs
# under runs/ (on /home). If a point's AgentX client has completed nothing for STALL s (AIPerf's warmup "returned=" or
# profiling "done=" count unchanged), it stops that client; if a point's workers are not all registered BOOT s after
# they started, it stops them. Either way point.sh then tears down and the job moves on or ends. Actions go through a
# short overlap step that exits 0 (spur ends a job when one of its steps exits non-zero), so this process can die
# without touching the jobs. It exits once no campaign job is left. DRY=1 only prints what it would do.
# Start detached: setsid nohup bash watchdog.sh >>logs/watchdog.log 2>&1 </dev/null &
R=/home/liyingli/bench_agentx/ix5138-crusoe-20261120
STALL=${STALL:-1200}
BOOT=${BOOT:-2700}
declare -A value since acted

campaign_jobs() {  # "id state nodes" of this campaign's benchmark jobs (not the reap jobs)
    squeue --me -h -o "%i %j %T %N" 2>/dev/null | awk '$2 ~ /^lyl-ix5138-(q|diag2n|dbg)$/ {print $1, $3, $4}'
}

job_of() {  # host -> id of the running campaign job on it
    local id state nodes
    while read -r id state nodes; do
        [[ "$state" == RUNNING ]] || continue
        scontrol show hostnames "$nodes" 2>/dev/null | grep -qx "$1" && { echo "$id"; return; }
    done < <(campaign_jobs)
}

act() {  # job host command
    echo "$(date -u +%T) act on $2 (job $1): $3"
    [[ -n "${DRY:-}" ]] && return
    srun --jobid "$1" --overlap -N1 -n1 -w "$2" -D /tmp --gres=gpu:8 /bin/bash -c "$3; exit 0" </dev/null 2>&1 |
        tr -d '\r' | grep -v '^srun: dispatched'
}

progress() {  # AIPerf log -> its latest completion count (warmup "returned=N/M", profiling "done=N"; N may hold commas)
    tr '\r' '\n' <"$1" 2>/dev/null | grep -aoE 'returned=[0-9,]+/|done=[0-9,]+ ok=' | tail -1
}

record() {  # point dir, message: the point's notes, unless dry
    echo "$(date -u +%T) $(basename "$1"): $2"
    [[ -n "${DRY:-}" ]] || echo "watchdog: $2" >>"$1/notes.txt"
}

echo "$(date -u +%T) watchdog start: STALL=$STALL BOOT=$BOOT DRY=${DRY:-}"
while :; do
    if [[ -z "$(campaign_jobs)" ]]; then
        echo "$(date -u +%T) no campaign job left; exiting"
        exit 0
    fi
    now=$(date +%s)
    for d in "$R"/runs/*/; do
        label=$(basename "$d")
        p=$d/point.log
        [[ -f "$p" ]] && ! grep -q "point $label done" "$p" || continue
        read -ra hosts <<<"$(sed -n 's/.*, nodes \(.*\)$/\1/p' "$p" | head -1)"
        ((${#hosts[@]})) || continue
        job=$(job_of "${hosts[0]}")
        [[ -n "$job" && -z "${acted[$label]:-}" ]] || continue
        if grep -q 'AgentX C' "$p" && ! grep -q 'AgentX exited' "$p"; then
            v=$(progress "$d/agentx.log")
            if [[ "${value[$label]-unset}" != "$v" ]]; then
                value[$label]=$v since[$label]=$now
            elif ((now - since[$label] > STALL)); then
                act "$job" "${hosts[0]}" "docker stop lyl-ix5138-$label-agentx >/dev/null"
                record "$d" "AgentX completed nothing for $STALL s (last ${v:-none}); client stopped"
                acted[$label]=1
            fi
        elif grep -q 'start etcd and workers' "$p" && ! grep -qE 'router: .* active|FAIL' "$p"; then
            t=$(date -d "$(sed -n 's/^== \(.*\) start etcd and workers$/\1/p' "$p" | head -1)" +%s 2>/dev/null) || continue
            if ((now - t > BOOT)); then
                while IFS=$'\t' read -r name node kind container _; do
                    [[ "$kind" == prefill || "$kind" == decode ]] &&
                        act "$job" "${hosts[$node]}" "docker stop $container >/dev/null"
                done <"$d/manifest.tsv"
                record "$d" "workers not registered $BOOT s after start; workers stopped"
                acted[$label]=1
            fi
        fi
    done
    [[ -n "${ONCE:-}" ]] && exit 0
    sleep 60
done
