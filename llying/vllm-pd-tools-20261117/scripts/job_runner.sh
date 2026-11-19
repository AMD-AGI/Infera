#!/usr/bin/env bash
# Queue runner for one job's second node (the job's only step there): executes every Q/<id>.sh in the background,
# writing <id>.log and then <id>.rc, until Q/STOP appears; then waits for running items, removes our containers on
# this node and exits 0 (spur ends the job on a non-zero step exit). Also removes them when the step is signalled.
#   job_runner.sh QUEUE_DIR
R=/home/liyingli/bench_agentx/vllm-pd-20261117
Q=${1:?queue dir}
rm_ours() { docker ps -a --format '{{.Names}}' | grep -E '^(vpd-|mcx-)' | xargs -r docker rm -f > /dev/null 2>&1; }
trap 'echo "runner signalled $(date -u "+%F %T")" >> "$Q/RUNNER"; rm_ours; exit 0' TERM INT HUP
mkdir -p "$Q"
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
    ls "$Q" > /dev/null
done
echo "runner stopped $(date -u '+%F %T')" >> "$Q/RUNNER"
wait
rm_ours
exit 0
