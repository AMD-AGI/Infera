#!/usr/bin/env bash
# py-spy record (non-blocking, SECONDS s, 50 Hz, raw) of every Python process in the decode container plus a thread CPU
# snapshot, into the container's /steplog (the run directory's steplog/ on the host). Always exits 0: it runs as an
# srun step inside a scoring job, and a non-zero step ends a spur job.
#   srun --jobid=JOB --overlap -N1 -n1 -D /tmp bash pyspy_decode.sh TAG [SECONDS=20] [CONTAINER=vpd-decode]
tag=${1:-snap} secs=${2:-20} ctr=${3:-vpd-decode}
# shellcheck disable=SC2016
docker exec -e TAG="$tag" -e SECS="$secs" "$ctr" bash -c 'pip install -q py-spy > /dev/null 2>&1
    for d in /proc/[0-9]*; do
        pid=${d#/proc/}
        case "$(readlink "$d/exe" 2>/dev/null)" in *python*) ;; *) continue ;; esac
        name=$(tr -c "A-Za-z0-9_\n" "_" < "$d/comm")
        timeout $((SECS + 40)) py-spy record --nonblocking -d "$SECS" -r 50 -f raw -o "/steplog/pyspy-$TAG-$pid-$name.raw" -p "$pid" > /dev/null 2>&1 &
    done
    top -b -H -n 2 -d 5 -w 200 > "/steplog/top-$TAG.txt" 2>&1
    wait' > /dev/null 2>&1
echo "$(date -u +%T) py-spy $tag done on $(hostname -s)"
exit 0
