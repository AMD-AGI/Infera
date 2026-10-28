#!/usr/bin/env bash
# Patrol loop for the night queue (login node): every INTERVAL s prints one status line, and an ALERT line for
# a guard / crash / failed start in the current point's log, a kill / die event of a container on 140 since the
# previous check, a new line in the queue log, or the queue finishing.
# Alerts are numbered (ALERT#1, ALERT#2, ...) so a waiter can block on the next one.
# Follows the queue logs night*-*.out (the most recently written one); done when no night queue runs.
# Usage: watch_night6.sh LOG_ROOT [INTERVAL=300]
L="$1" interval="${2:-300}"
last="$(date -u +%FT%TZ)" seen_queue="" seen_bad="" n=0
alert() { while IFS= read -r line; do [[ -n "$line" ]] || continue; n=$((n + 1)); echo "$(date -u +%T) ALERT#$n $line"; done; }
while true; do
    now="$(date -u +%FT%TZ)"
    s138="$(timeout 90 ssh -o BatchMode=yes crsuse2-m2m-138 "
        tail -n 1 \$(ls -t $L/night*-*.out | head -1)
        f=\$(ls -t $L/perf/*/c*.log $L/prof-*/prof-*.log $L/gate/*.log 2>/dev/null | head -1)
        echo \"LOG \$f\"
        tr '\r' '\n' <\"\$f\" | grep -aE 'guard|crashed|Traceback|up failed' | tail -n 2 | sed 's/^/BAD /'
        tr '\r' '\n' <\"\$f\" | grep -aE 'progress|Warmup complete|profile requested| rps=.*done=' | tail -n 1 | cut -c1-160
        pgrep -f 'scripts/night[2-5]?_[1]02[78].sh' >/dev/null && echo RUNNING || echo QUEUE_DONE" </dev/null 2>&1)"
    s140="$(timeout 60 ssh -o BatchMode=yes crsuse2-m2m-140 "timeout 10 docker events --since '$last' --until '$now' \
        --filter event=kill --filter event=die --format '{{.Actor.Attributes.name}} {{.Action}} exit={{.Actor.Attributes.exitCode}}'" </dev/null 2>&1)"
    last="$now"
    qline="$(head -n 1 <<<"$s138")"
    echo "$(date -u +%T) | $(grep -v '^BAD\|^LOG\|RUNNING\|QUEUE_DONE' <<<"$s138" | tail -n 1)"
    bad="$(grep '^BAD' <<<"$s138")"
    [[ -z "$bad" || "$bad" == "$seen_bad" ]] || { alert <<<"$bad"; seen_bad="$bad"; }
    [[ -z "$s140" ]] || alert < <(sed 's/^/140 event: /' <<<"$s140")
    [[ "$qline" == "$seen_queue" ]] || { echo "queue: $qline" | alert; seen_queue="$qline"; }
    if grep -q QUEUE_DONE <<<"$s138"; then echo "queue finished" | alert; break; fi
    sleep "$interval"
done
