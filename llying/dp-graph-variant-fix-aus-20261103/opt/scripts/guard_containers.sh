#!/bin/bash
# Runs as a Slurm step (srun --overlap) on each node of a holder job while an opt item uses it. Preemption or the end of
# the job sends SIGTERM to every step before SIGKILL; this then force-removes our containers on the node, so nothing
# keeps the GPUs of the job the scheduler gives the node to next (35103 preempted 11-04 11:46: prefill and decode
# containers kept ~270 / ~307 GB on all 8 GPUs of n06-25 / n06-33 under zirzhang's 35141; ssh had been refused since
# 11:42, so the item's own down.sh could not reach the decode).
# Copy of 2p1d-aus-v0521-20261019/scripts/guard_containers.sh for the kit's container names ($CTR-etcd/-prefillN/
# -router/-decode, CTR=glm52_2p1d, which analyze_pd_logs.py expects).
# Usage (srun task): guard_containers.sh <container name prefix ending in "-">
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
prefix="${1:?container name prefix}"
[[ "$prefix" == *llying* || "$prefix" == glm52_2p1d- || "$prefix" == glm52_agentx- ]] || { echo "refusing prefix $prefix" >&2; exit 2; }
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
