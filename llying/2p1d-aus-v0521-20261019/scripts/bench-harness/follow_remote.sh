#!/usr/bin/env bash
# Login-host variant of follow_log.sh for job 34439 (no ssh to the nodes; the ssh in PATH is tools/sshshim/ssh, which
# runs each command as an srun step): streams the container's docker logs over that step and appends them to FILE for
# as long as the container runs, resuming after the last timestamp in FILE. "GET /metrics" access lines are dropped.
# Usage (login host, detached, PATH with the shim and SHIM_JOB set): follow_remote.sh FILE NODE CONTAINER
set -uo pipefail
file="$1"; node="$2"; container="$3"
id="$(ssh -n "$node" "docker inspect -f '{{.Id}}' $container" 2>/dev/null)" || exit 0
[[ -n "$id" ]] || exit 0
while [[ "$(ssh -n "$node" "docker inspect -f '{{.Id}} {{.State.Running}}' $container" 2>/dev/null)" == "$id true" ]]; do
    since=""
    if [[ -s "$file" ]]; then
        last="$(tail -c 65536 "$file" | grep -a -oE '^[0-9]{4}-[0-9-]{5}T[0-9:]{8}\.[0-9]+Z' | tail -1)"
        [[ -z "$last" ]] || since="$(python3 -c '
import sys
base, frac = sys.argv[1].rstrip("Z").split(".")
print("%s.%09dZ" % (base, int(frac.ljust(9, "0")[:9]) + 1))' "$last")"
    fi
    ssh -n "$node" "docker logs --timestamps -f ${since:+--since $since} $container 2>&1" |
        grep -a --line-buffered -v '"GET /metrics HTTP' >> "$file"
    sleep 30
done
