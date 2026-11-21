#!/usr/bin/env bash
# Overlap step for a burst job: when the job is preempted or ends, spur signals its steps; this one then removes the
# campaign's containers on the node, which would otherwise keep the GPUs after the job is gone. Exits 0 either way
# (spur ends a job when one of its steps exits non-zero).
clean() {
    ids=$(docker ps -aq --filter name=lyl-ix5138-)
    echo "$(date -u +%T) $(hostname -s): signalled; removing $(wc -w <<<"$ids") container(s)"
    [ -n "$ids" ] && docker rm -f $ids >/dev/null 2>&1
    exit 0
}
trap clean TERM INT HUP
echo "$(date -u +%T) $(hostname -s): guard up"
while :; do sleep 5 & wait $!; done
