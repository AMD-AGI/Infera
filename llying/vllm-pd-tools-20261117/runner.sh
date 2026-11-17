#!/usr/bin/env bash
# Runs on the node inside the job's one exec-into-job session: executes every queue/<id>.sh the login node drops, in
# the background, writing <id>.log and then <id>.rc. A step's processes end with the step, so this loop must keep
# running; containers (dockerd's children) do not. Stop it with: touch queue/STOP. Always exits 0 (spur ends the job
# on a non-zero step exit). Heartbeat: queue/RUNNER.hb is touched every loop.
R=/home/liyingli/bench_agentx/vllm-pd-20261117
Q=$R/queue
trap 'echo "runner signalled $(date -u "+%F %T")" >> "$Q/RUNNER"; exit 0' TERM INT HUP
mkdir -p "$Q"
rm -f "$Q/STOP"
echo "runner on $(hostname -s) pid $$ since $(date -u '+%F %T')" > "$Q/RUNNER"
while [ ! -e "$Q/STOP" ]; do
    for f in "$Q"/*.sh; do
        [ -e "$f" ] || continue
        id=${f%.sh}
        mv "$f" "$id.run" || continue
        ( cd "$R" && bash "$id.run" > "$id.log" 2>&1; echo $? > "$id.rc" ) &
    done
    touch "$Q/RUNNER.hb"
    sleep 2
    ls "$Q" > /dev/null  # refresh the NFS directory cache
done
echo "runner stopped $(date -u '+%F %T')" >> "$Q/RUNNER"
wait
exit 0
