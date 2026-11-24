#!/usr/bin/env bash
# Login-node monitor (monitor.sh plus hang-repro detail): prints every new summary.log line and job .out line,
# new lines of the running hang runs' repro-N.log and hang-watch logs, "JOB ENDED <id>" when a job leaves the queue
# and "NO PROGRESS <id>" when a running job's logs have not changed for STALL_MIN minutes. Lines worth acting on are
# prefixed "ALERT#<n>" with n counting up from 1, so a waiter can block on the next number instead of matching old
# output. Light work only (squeue, tail, stat); runs until killed.
W=/home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123
STALL_MIN=${STALL_MIN:-45}
ALERT_RE=${ALERT_RE:-'JOB ENDED|NO PROGRESS|FAILED|engine died|stall line|AgentX c[0-9]+ rc|RESULT|hang [0-9]+ in|repro segment|ENGINE DIED|single ok=False|NOT STARTED|redeploy'}
cd "$W/logs" || exit 1
declare -A seen_lines last_change known file_lines
na=0
emit() {  # emit < <(lines): number the lines worth acting on (not in a pipe: na must survive)
    local l
    while IFS= read -r l; do
        if grep -qE "$ALERT_RE" <<<"$l"; then na=$((na + 1)); echo "ALERT#$na $(date -u +%H:%M) $l"; else echo "$l"; fi
    done
}
follow() {  # follow FILE PREFIX: print lines added since the last call
    local f=$1 c p
    c=$(wc -l < "$f" 2>/dev/null || echo 0); p=${file_lines[$f]:-0}
    (( c > p )) && { emit < <(tail -n $((c - p)) "$f" | cut -c1-600 | sed "s|^|$2 |"); file_lines[$f]=$c; }
}
sl=$(wc -l < summary.log 2>/dev/null || echo 0)
echo "monitor2 started $(date -u +%FT%TZ), summary.log at line $sl"
while :; do
    n=$(wc -l < summary.log 2>/dev/null || echo 0)
    if (( n > sl )); then emit < <(tail -n $((n - sl)) summary.log | cut -c1-600 | sed 's/^/SUMMARY /'); sl=$n; fi
    now=$(date +%s)
    jobs=$(squeue -u liyingli -h -o '%i %j %T' 2>/dev/null | awk '$2 ~ /^lyl-dsv41/ {print $1":"$3}')
    for j in $jobs; do
        id=${j%%:*}; st=${j#*:}; known[$id]=$st
        f=job-$id.out
        if [[ -f $f ]]; then
            c=$(wc -l < "$f"); p=${seen_lines[$id]:-0}
            (( c > p )) && { emit < <(tail -n $((c - p)) "$f" | sed "s/^/JOB $id /"); seen_lines[$id]=$c; }
        fi
        if [[ $st == RUNNING ]]; then
            run=$(grep -a "start on .* job $id," summary.log | tail -1 | awk '{print $2}' | tr -d :)
            if [[ $run == hang* && -d $run ]]; then
                for r in "$run"/repro-*.log; do [[ -f $r ]] && follow "$r" "REPRO $run/$(basename "$r" .log)"; done
                # hang-watch-<leg>-rep<N>.log: a new name per segment; logs moved into hangN/ were already reported.
                for h in "$run"/hang-watch-*.log; do
                    [[ -f $h ]] || continue
                    c=$(grep -c 'stall line seen' "$h"); p=${file_lines[$h]:-0}
                    (( c > p )) && { emit < <(grep 'stall line seen' "$h" | tail -n $((c - p)) | sed "s|^|WATCH $h |"); file_lines[$h]=$c; }
                done
            fi
            latest=$(find "$f" "job-$id-pipeline.log" ${run:+"$run"} -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1)
            if (( now - ${latest:-$now} > STALL_MIN * 60 )); then
                (( now - ${last_change[$id]:-0} > 1800 )) && { emit <<<"NO PROGRESS $id: no log written for $(( (now - latest) / 60 )) min"; last_change[$id]=$now; }
            fi
        fi
    done
    for id in "${!known[@]}"; do
        grep -q "^$id:" <<<"$(tr ' ' '\n' <<<"$jobs")" || {
            emit <<<"JOB ENDED $id (was ${known[$id]})"
            [[ -f job-$id.out ]] && tail -n 3 "job-$id.out" | sed "s/^/JOB $id /"
            unset "known[$id]"
        }
    done
    sleep 60
done
