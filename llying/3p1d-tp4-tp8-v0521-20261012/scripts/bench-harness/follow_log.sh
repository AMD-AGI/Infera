#!/usr/bin/env bash
# Append a container's docker logs (with timestamps) to FILE for as long as the container runs.
# docker logs -f exits when a write fails -- the shared /home volume returns "disk quota
# exceeded" whenever it is momentarily full (2026-10-14 21:59, 10-15 05:03) -- so after it exits
# this waits and resumes from just after the last timestamp already in FILE: no gap, no duplicates.
# Usage (on the container's node, detached): follow_log.sh FILE CONTAINER
set -uo pipefail
file="$1"
container="$2"
# A restart reuses the container name and starts its own follower; stop at the first id change.
id="$(docker inspect -f '{{.Id}}' "$container" 2>/dev/null)" || exit 0
while [[ "$(docker inspect -f '{{.Id}} {{.State.Running}}' "$container" 2>/dev/null)" == "$id true" ]]; do
    since=""
    if [[ -s "$file" ]]; then
        last="$(tail -c 65536 "$file" | grep -a -oE '^[0-9]{4}-[0-9-]{5}T[0-9:]{8}\.[0-9]+Z' | tail -1)"
        [[ -z "$last" ]] || since="$(python3 -c '
import sys
base, frac = sys.argv[1].rstrip("Z").split(".")
print("%s.%09dZ" % (base, int(frac.ljust(9, "0")[:9]) + 1))' "$last")"
    fi
    docker logs --timestamps -f ${since:+--since "$since"} "$container" >> "$file" 2>/dev/null
    sleep 30
done
