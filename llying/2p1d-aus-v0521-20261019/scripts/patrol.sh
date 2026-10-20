#!/usr/bin/env bash
# Patrol for unattended runs (login host, detached). Every INTERVAL seconds it checks, and appends one line per new
# problem to runtime/patrol/ALERT.log (the operator's wake-up signal); a one-line summary per pass goes to patrol.log.
#   - Slurm job state (gone / preempted);  prepare_alloc.sh FAIL lines;
#   - queue: failed step, driver not running, queue exhausted for > 10 min (nothing queued = idle stack);
#   - engines + router /health (skipped while a bring-up / restart step is running);
#   - fatal patterns in the server logs since the last pass (GPU fault, segfault, OOM, HSA error, watchdog);
#     decode transfer failures > 20 per pass;
#   - an AgentX round whose agentx.log has not grown for 15 min while its client container runs.
# Usage: JOB=<id> ROUND=<r> [INTERVAL=120] patrol.sh
set -uo pipefail
: "${JOB:?}" "${ROUND:?}"
HERE="$(cd "$(dirname "$0")" && pwd)"
R="$(cd "$HERE/.." && pwd)"
P="$R/runtime/patrol"; mkdir -p "$P"
ALERT="$P/ALERT.log"; LOG="$P/patrol.log"; STATE="$P/state"; mkdir -p "$STATE"
touch "$ALERT"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10 -n)
FATAL='Memory access fault|Segfault encountered|Fatal Python error|out of memory|OutOfMemory|HSA_STATUS_ERROR|Scheduler watchdog timeout|scheduler died|exited with code'
alert() {  # alert <key> <message>: once per key until the key is cleared by a healthy pass
    local key="$1"; shift
    [[ -e "$STATE/alert.$key" ]] && return 0
    touch "$STATE/alert.$key"
    echo "$(date -u +%FT%TZ) [$key] $*" | tee -a "$ALERT" >> "$LOG"
}
clear_alert() { rm -f "$STATE/alert.$1"; }

while true; do
    now="$(date -u +%FT%TZ)"; summary=()
    [[ -s "$R/operations/current_round" ]] && ROUND="$(cat "$R/operations/current_round")"
    summary+=("round=$ROUND")
    runs="$R/runtime/runs/$ROUND"; Q="$runs/tune/queue.log"
    st="$(squeue -h -j "$JOB" -o %T 2>/dev/null)"
    if [[ -z "$st" ]]; then
        alert job "job $JOB not in queue: $(sacct -n -X -j "$JOB" -o State,End 2>/dev/null | xargs)"
        summary+=("job=gone")
    else
        summary+=("job=$st"); clear_alert job
    fi
    grep -q 'prepare: FAIL' "$R/logs/prepare.$JOB.log" 2>/dev/null && alert prepare "$(grep 'prepare: FAIL' "$R/logs/prepare.$JOB.log" | tail -1)"

    if [[ "$st" == RUNNING && -s "$R/config/topology.2p1d.aus.tsv" && -e "$Q" ]]; then
        set +u; set -a; source "$R/config/config.3p1d.sh" >/dev/null 2>&1; set +a; set -u
        last_step="$(grep 'queue: step' "$Q" | tail -1)"
        step_no="$(sed -nE 's/.*queue: step ([0-9]+):.*/\1/p' <<< "$last_step")"
        summary+=("step=${step_no:-0}")
        if tail -2 "$Q" | grep -qE 'failed|stopping|no room'; then
            alert "queue-failed-${step_no:-0}" "$(tail -1 "$Q")"
        fi
        drv="$(ps -u "$(id -u)" -o args= | grep -c '^bash tune_queue.sh')"   # driver runs on the login host (job 34439)
        summary+=("driver=$drv")
        if [[ "$drv" == 0 ]] && ! tail -1 "$Q" | grep -qE 'failed|stop'; then
            alert "driver-${step_no:-0}" "queue driver not running on the login host (last: $(tail -1 "$Q"))"
        fi
        nsteps="$(grep -cvE '^[[:space:]]*(#|$)' "$R/operations/tune_queue.tsv")"
        done_n="$(cat "$runs/tune/queue.done" 2>/dev/null || echo 0)"
        summary+=("done=$done_n/$nsteps")
        if (( done_n >= nsteps )); then
            [[ -e "$STATE/empty_since" ]] || date +%s > "$STATE/empty_since"
            (( $(date +%s) - $(cat "$STATE/empty_since") > 600 )) && alert "queue-empty-$done_n" "queue exhausted ($done_n steps done); stack idle"
        else
            rm -f "$STATE/empty_since"
        fi
        restarting=0
        [[ "$last_step" =~ bringup|restart_ ]] && (( done_n < step_no )) && restarting=1
        if (( restarting == 0 )) && grep -q $'\tall\tdone' "$runs/status.tsv" 2>/dev/null; then
            down=()
            while IFS=$'\t' read -r index inst role node ip; do
                port=$((ENGINE_PORT_BASE + index * ENGINE_PORT_STRIDE))
                code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "http://$ip:$port/health")"
                [[ "$code" == 200 ]] || down+=("$inst:$code")
            done < <(python3 "$BENCH_DIR/tools/topology.py" rows "$TOPOLOGY")
            cip="$(python3 "$BENCH_DIR/tools/topology.py" node-ip "$TOPOLOGY" "$CONTROL_NODE")"
            code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "http://$cip:$ROUTER_PORT/health")"
            [[ "$code" == 200 ]] || down+=("router:$code")
            if (( ${#down[@]} )); then
                # Rounds restart the router for a few seconds: alert only when the same set is down on two passes.
                [[ "$(cat "$STATE/down_prev" 2>/dev/null)" == "${down[*]}" ]] &&
                    alert "down-$(tr ' ' '_' <<< "${down[*]}")" "unhealthy on two passes: ${down[*]} (step ${step_no:-0})"
                echo "${down[*]}" > "$STATE/down_prev"
                summary+=("down=${down[*]}")
            else
                summary+=("health=ok"); rm -f "$STATE/down_prev"
                for k in "$STATE"/alert.down-*; do [[ -e "$k" ]] && rm -f "$k"; done
            fi
        else
            summary+=("health=skip")
        fi
        # Fatal patterns and decode transfer failures since the last pass (byte offsets per log).
        for f in "$runs"/launch/server-logs/*.log; do
            [[ -e "$f" ]] || continue
            key="$(basename "$f")"; off="$(cat "$STATE/off.$key" 2>/dev/null || echo 0)"; size="$(stat -c %s "$f")"
            (( size < off )) && off=0
            if (( size > off )); then
                chunk="$(tail -c +$((off + 1)) "$f" | head -c $((size - off)))"
                hit="$(grep -E "$FATAL" <<< "$chunk" | grep -v 'GET /' | head -2 | cut -c1-220)"
                [[ -n "$hit" ]] && alert "fatal-$key-$size" "$key: $hit"
                tf="$(grep -c 'Decode transfer failed' <<< "$chunk")"
                (( tf > 20 )) && alert "xfer-$key-$size" "$key: $tf decode transfer failures in one pass"
                echo "$size" > "$STATE/off.$key"
            fi
        done
        # AgentX progress during a round.
        cur="$(ls -td "$runs"/tune/*-c[0-9]* 2>/dev/null | head -1)"
        if [[ -n "$cur" && -e "$cur/agentx.log" && ! -e "$cur/published" ]]; then
            client="$("${SSH[@]}" "$CONTROL_NODE" "docker ps --format '{{.Names}}' | grep -c '^$CONTAINER_PREFIX-agentx-client-'" 2>/dev/null || echo "?")"
            age=$(( $(date +%s) - $(stat -c %Y "$cur/agentx.log") ))
            summary+=("round=$(basename "$cur") client=$client log_age=${age}s")
            [[ "$client" == 1 ]] && (( age > 900 )) && alert "stall-$(basename "$cur")" "$(basename "$cur"): agentx.log silent for ${age}s"
        fi
        hw="$(tail -1 "$runs/tune/hang_watch.log" 2>/dev/null | cut -c1-120)"
        [[ -n "$hw" ]] && summary+=("hang_watch: ${hw#* }")
    fi
    echo "$now ${summary[*]}" >> "$LOG"
    sleep "${INTERVAL:-120}"
done
