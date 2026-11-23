#!/usr/bin/env bash
# Login-node monitor for this work dir's spur jobs (lyl-dsv41-*): prints every new summary.log line and job .out
# line, "JOB ENDED <id>" when a job leaves the queue, and "NO PROGRESS <id>" when a running job's logs have not
# changed for STALL_MIN minutes. Light work only (squeue, tail, stat); runs until killed.
W=/home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123
STALL_MIN=${STALL_MIN:-45}
cd "$W/logs" || exit 1
declare -A seen_lines last_change known
sl=$(wc -l < summary.log 2>/dev/null || echo 0)
while :; do
    n=$(wc -l < summary.log 2>/dev/null || echo 0)
    if (( n > sl )); then tail -n $((n - sl)) summary.log | cut -c1-600 | sed 's/^/SUMMARY /'; sl=$n; fi
    now=$(date +%s)
    jobs=$(squeue -u liyingli -h -o '%i %j %T' 2>/dev/null | awk '$2 ~ /^lyl-dsv41/ {print $1":"$3}')
    for j in $jobs; do
        id=${j%%:*}; st=${j#*:}; known[$id]=$st
        f=job-$id.out
        if [[ -f $f ]]; then
            c=$(wc -l < "$f"); p=${seen_lines[$id]:-0}
            (( c > p )) && { tail -n $((c - p)) "$f" | sed "s/^/JOB $id /"; seen_lines[$id]=$c; }
        fi
        if [[ $st == RUNNING ]]; then
            latest=$(find . -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1)
            if (( now - ${latest:-$now} > STALL_MIN * 60 )); then
                (( now - ${last_change[$id]:-0} > 1800 )) && { echo "NO PROGRESS $id: no log written for $(( (now - latest) / 60 )) min"; last_change[$id]=$now; }
            fi
        fi
    done
    for id in "${!known[@]}"; do
        grep -q "^$id:" <<<"$(tr ' ' '\n' <<<"$jobs")" || { echo "JOB ENDED $id (was ${known[$id]})"; [[ -f job-$id.out ]] && tail -n 3 "job-$id.out" | sed "s/^/JOB $id /"; unset "known[$id]"; }
    done
    sleep 60
done
