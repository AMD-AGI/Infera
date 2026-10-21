#!/bin/bash
# Runs as a Slurm step on every node of the job (scripts/campaign.sh starts it after prepare). Preemption or the end of the
# job sends SIGTERM to all steps before SIGKILL; this then force-removes every container of ours on the node, so nothing
# keeps GPUs or pinned host memory on a node that the scheduler has given to someone else (a1's decode containers were
# still running on n02-33 12 hours after 34439 was preempted on 10-20 11:57).
# Usage (srun task): guard_containers.sh <container name prefix>
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
prefix="${1:?container name prefix}"
[[ "$prefix" == *llying* ]] || { echo "refusing prefix $prefix" >&2; exit 2; }
cleanup() {
    names="$(docker ps -a --format '{{.Names}}' | grep "^$prefix" || true)"
    echo "$(date -u +%FT%TZ) $(hostname -s) signal: removing ${names:-nothing}"
    for n in $names; do docker rm -f "$n" >/dev/null 2>&1 & done   # in parallel: SIGKILL follows SIGTERM after KillWait=30 s
    wait
    exit 0
}
trap cleanup TERM INT HUP
echo "$(date -u +%FT%TZ) $(hostname -s) guarding containers ${prefix}*"
while true; do sleep 3600 & wait $!; done
