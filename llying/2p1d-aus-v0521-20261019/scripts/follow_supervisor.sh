#!/usr/bin/env bash
# Keeps one login-host follower (bench-harness/follow_remote.sh) per server log while FOLLOW_ON_NODE=0. engine.sh drops
# "<server log>.follow" (node <TAB> container) next to each log it starts; for a container name only the newest marker
# counts (a restart reuses the name with a new log file). Checks every 20 s; a follower that ended (container gone or
# stream broken) is started again and resumes after the last timestamp in its file.
# Usage (login host, detached, PATH with the shim and SHIM_JOB set): follow_supervisor.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
R="$(cd "$HERE/.." && pwd)"
while true; do
    declare -A newest=()
    while IFS= read -r m; do
        IFS=$'\t' read -r node container < "$m" || continue
        newest[$container]="$m"
    done < <(ls -tr "$R"/runtime/runs/*/launch/server-logs/*.follow 2>/dev/null)
    for container in "${!newest[@]}"; do
        m="${newest[$container]}"; file="${m%.follow}"
        IFS=$'\t' read -r node c < "$m"
        want="bash $HERE/bench-harness/follow_remote.sh $file $node $c"
        if ! ps -u "$(id -u)" -o args= | awk -v w="$want" '$0 == w {f = 1} END {exit !f}'; then
            setsid nohup bash "$HERE/bench-harness/follow_remote.sh" "$file" "$node" "$c" > /dev/null 2>&1 < /dev/null &
            echo "$(date -u +%FT%TZ) follower started: $(basename "$file") <- $node/$c"
        fi
    done
    unset newest
    sleep 20
done
