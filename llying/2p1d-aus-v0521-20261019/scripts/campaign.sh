#!/usr/bin/env bash
# Unattended driver across preemptions (login host, detached). Job JOB is QOS batch with Requeue=1: it was preempted at
# 15:23 (34389) and again at ~19:50 (34439, requeued). On every (re)start of the job:
#   1. ROUND = a<Slurm restart count> (a fresh run dir and fresh variant names per allocation);
#   2. prepare_alloc.sh (node names, image, root-owned cache dirs, dataset cache, 16-GPU stream check);
#   3. config/round.<ROUND>.sh <- the file named in operations/current_stage (the config the plan is at);
#   4. operations/tune_queue.tsv <- operations/queue.next.tsv with {R} replaced by ROUND; container guard step
#      (guard_containers.sh: removes our containers when Slurm signals the steps at preemption) and queue driver started;
#   5. while the job runs: nothing (the queue, patrol and operator do the work);
#   6. job no longer running: stop this round's login-host processes, log it, raise an alert, back to 1.
# operations/current_round holds the live ROUND (read by patrol.sh and wait_event.sh).
# The operator edits queue.next.tsv / current_stage as experiments finish, so a restart resumes where the plan is.
# Usage: JOB=<id> [RESUME_ROUND=<live round, when restarting this script during an allocation>] campaign.sh
set -uo pipefail
: "${JOB:?}"
HERE="$(cd "$(dirname "$0")" && pwd)"
R="$(cd "$HERE/.." && pwd)"
O="$R/operations"
export SHIM_JOB="$JOB" PATH="$R/tools/sshshim:$PATH"   # see tools/sshshim/ssh
log() { echo "$(date -u +%FT%TZ) campaign: $*"; }
# SIGNALING shows while Slurm signals a step of a running job (seen 10-22 12:20 when kit containers were removed):
# it is not the job leaving its nodes.
running() { [[ "$(squeue -h -j "$JOB" -o %T 2>/dev/null)" =~ ^(RUNNING|SIGNALING)$ ]]; }
alert() { mkdir -p "$R/runtime/patrol"; echo "$(date -u +%FT%TZ) [campaign] $*" >> "$R/runtime/patrol/ALERT.log"; log "$*"; }
# Watchdog: a queue that stopped on a failed step leaves the allocation idle until the next preemption, which can be
# hours away (a10: 7 h). After STALL_MIN minutes of that, give the nodes back by requeueing the job.
STALL_MIN="${STALL_MIN:-30}"
watch_round() {
    local qlog="$R/runtime/runs/$ROUND/tune/queue.log" last age
    while running; do
        sleep 60
        last="$(grep -E 'queue: step [0-9]+' "$qlog" 2>/dev/null | tail -n 1)"
        [[ "$last" == *"failed rc="* ]] || continue
        age=$(( $(date +%s) - $(date -d "$(cut -c1-20 <<< "$last")" +%s) ))
        if (( age > STALL_MIN * 60 )); then
            alert "queue of $ROUND stopped on a failed step ${age}s ago ($last); requeueing job $JOB to release the nodes"
            scontrol requeue "$JOB" || alert "scontrol requeue $JOB failed"
            sleep 120
        fi
    done
    alert "job $JOB left RUNNING during ROUND=$ROUND (state $(squeue -h -j "$JOB" -o %T 2>/dev/null); restarts now $(scontrol show job "$JOB" | sed -nE 's/.*Restarts=([0-9]+).*/\1/p'))"
    for pat in '^bash tune_queue.sh' 'sampler.py' 'hang_watch.py' 'agentx_bench.sh' 'tune_round.sh' 'tune_post.sh'; do
        pkill -u "$(id -u)" -f "$pat" 2>/dev/null
    done
    log "stopped login-host processes of $ROUND"
}
if [[ -n "${RESUME_ROUND:-}" ]]; then
    ROUND="$RESUME_ROUND"
    log "resuming ROUND=$ROUND of the running allocation"
    watch_round
fi
while true; do
    while ! running; do
        [[ -z "$(squeue -h -j "$JOB" -o %T 2>/dev/null)" ]] && { alert "job $JOB is gone ($(sacct -n -X -j "$JOB" -o State | xargs)); stopping"; exit 1; }
        sleep 30
    done
    restarts="$(scontrol show job "$JOB" | sed -nE 's/.*Restarts=([0-9]+).*/\1/p')"
    ROUND="a${restarts:-0}"
    if [[ -e "$R/runtime/runs/$ROUND" ]]; then ROUND="${ROUND}x$(date +%H%M)"; fi
    log "job $JOB running (restart $restarts) -> ROUND=$ROUND"
    echo "$ROUND" > "$O/current_round"
    JOB="$JOB" ROUND="$ROUND" START_QUEUE=0 bash "$HERE/prepare_alloc.sh" > "$R/logs/prepare.$JOB.$ROUND.log" 2>&1 ||
        alert "prepare failed for $ROUND (logs/prepare.$JOB.$ROUND.log); starting the queue anyway (its steps check what they need)"
    stage="$(cat "$O/current_stage")"
    cp "$R/$stage" "$R/config/round.$ROUND.sh"
    sed "s/{R}/$ROUND/g" "$O/queue.next.tsv" > "$O/tune_queue.tsv"
    cp "$O/tune_queue.tsv" "$O/tune_queue.$ROUND.tsv"
    log "stage $stage installed as config/round.$ROUND.sh; queue from queue.next.tsv"
    setsid nohup srun --jobid="$JOB" --overlap -N "$(squeue -h -j "$JOB" -o %D)" --ntasks-per-node=1 --export=NONE \
        /bin/bash "$HERE/guard_containers.sh" glm52-llying-2p1d- >> "$R/logs/guard.$JOB.log" 2>&1 < /dev/null &
    bash "$HERE/start_queue.sh" "ROUND=$ROUND"
    watch_round
done
