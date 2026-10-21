#!/usr/bin/env bash
# Patrol for the PR 210 kit runs (login host, detached), next to scripts/patrol.sh (which still watches the job and
# the queue driver). Every INTERVAL seconds, for the run in pr210/current_run: the kit's containers on both nodes,
# router / prefill / decode /health, fatal lines appearing in the followed logs, and AgentX progress (its runner log
# must keep moving while the client container runs). New problems go to runtime/patrol/ALERT.log as "[kit] ...";
# one summary line per pass goes to runtime/patrol/kit_patrol.log.
# Usage: INTERVAL=120 kit_patrol.sh
set -uo pipefail
source "$(dirname "$0")/kit_env.sh"
INTERVAL="${INTERVAL:-120}"
ALERT="$R/runtime/patrol/ALERT.log"; LOG="$R/runtime/patrol/kit_patrol.log"
mkdir -p "$(dirname "$ALERT")"
FATAL='Traceback|Memory access fault|HSA_STATUS_ERROR|NCCL error|RCCL error|out of memory|Segmentation fault|core dumped|Watchdog timeout|KVTransferError|Failed to register memory'
declare -A seen offset fails
alert() {  # alert KEY MESSAGE: once per key until the condition clears
    [[ -n "${seen[$1]:-}" ]] && return
    seen[$1]=1
    echo "$(date -u +%FT%TZ) [kit] $2" >> "$ALERT"
}
clear_key() { unset "seen[$1]"; }
last_runner="" ; runner_since=$(date +%s)
while true; do
    sleep "$INTERVAL"
    state="$(squeue -h -j "$SHIM_JOB" -o %T 2>/dev/null)"
    [[ "$state" == RUNNING ]] || { echo "$(date -u +%FT%TZ) job=${state:-gone}" >> "$LOG"; continue; }
    run="$(cat "$R/pr210/current_run" 2>/dev/null)"
    [[ -n "$run" ]] || { echo "$(date -u +%FT%TZ) no run" >> "$LOG"; continue; }
    source "$(dirname "$0")/kit_env.sh"   # node names change with the allocation
    p_ctrs="$(ssh "$P_NODE" "docker ps --format '{{.Names}}'" 2>/dev/null | grep "^$CTR-" | tr '\n' ' ')"
    d_ctrs="$(ssh "$D_NODE" "docker ps --format '{{.Names}}'" 2>/dev/null | grep "^$CTR-" | tr '\n' ' ')"
    summary=("run=$run" "P=[${p_ctrs% }]" "D=[${d_ctrs% }]")
    if [[ -z "$p_ctrs$d_ctrs" ]]; then
        echo "$(date -u +%FT%TZ) ${summary[*]} (not deployed)" >> "$LOG"; continue
    fi
    if [[ -e "$PR_RUNS/$run/.starting" ]]; then services=(); summary+=("starting"); else
        services=("router $ROUTER_URL" "prefill $PREFILL_URL" "decode http://$D_IP:27301"); fi
    for svc in "${services[@]}"; do
        read -r n url <<< "$svc"
        if curl -sf -m 20 -o /dev/null "$url/health"; then
            fails[$n]=0; clear_key "down-$n"; summary+=("$n=ok")
        else
            fails[$n]=$(( ${fails[$n]:-0} + 1 )); summary+=("$n=DOWN")
            (( fails[$n] >= 2 )) && alert "down-$n" "$n /health not answering on ${fails[$n]} passes ($url)"
        fi
    done
    for f in "$PR_RUNS/$run"/logs/*.log; do
        [[ -f "$f" ]] || continue
        size=$(stat -c %s "$f"); from=${offset[$f]:-$size}
        (( size < from )) && from=0
        if (( size > from )); then
            hit="$(tail -c +$((from + 1)) "$f" | grep -E -m1 "$FATAL" | cut -c1-220)"
            [[ -n "$hit" ]] && alert "fatal-$f-$size" "$(basename "$f"): $hit"
        fi
        offset[$f]=$size
    done
    if [[ "$p_ctrs" == *"$CTR-agentx"* ]]; then
        line="$(ssh "$P_NODE" "tail -n 1 /scratch/liyingli_2p1d/pr210/work/runs/$run/agentx/runner.log" 2>/dev/null | cut -c1-160)"
        now=$(date +%s)
        if [[ "$line" != "$last_runner" ]]; then last_runner="$line"; runner_since=$now; clear_key agentx-stall
        elif (( now - runner_since > 600 )); then alert agentx-stall "AgentX runner log unchanged for $(( (now - runner_since) / 60 )) min: $line"; fi
        summary+=("agentx: ${line:0:90}")
    fi
    echo "$(date -u +%FT%TZ) ${summary[*]}" >> "$LOG"
done
