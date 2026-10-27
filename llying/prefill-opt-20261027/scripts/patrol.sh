#!/usr/bin/env bash
# One patrol of the night queue, run from the login node: queue log, the current point's guard / crash / progress
# lines, the containers on both nodes and the decode node's container kills since START (docker events).
# Usage: patrol.sh LOG_ROOT [START=1 hour ago, RFC 3339]
L="$1" start="${2:-$(date -u -d '-1 hour' +%FT%TZ)}"
date -u +%T
timeout 60 ssh -o BatchMode=yes crsuse2-m2m-138 "
    tail -n 6 \$(ls -t $L/night-*.out | head -1)
    f=\$(ls -t $L/perf/*/c*.log $L/prof-*/prof-*.log 2>/dev/null | head -1); echo \"== \$f\"
    tr '\r' '\n' <\"\$f\" | grep -aE 'guard|crashed|Traceback|up failed|agentx exited|progress|Warmup complete|profile requested' | tail -n 3 | cut -c1-220
    docker ps --format '138 {{.Names}} {{.Status}}'" </dev/null
timeout 60 ssh -o BatchMode=yes crsuse2-m2m-140 "
    docker ps --format '140 {{.Names}} {{.Status}}'
    timeout 10 docker events --since '$start' --until \"\$(date -u +%FT%TZ)\" --filter event=kill --filter event=die \
        --format '140 event {{.Action}} {{.Actor.Attributes.name}} exit={{.Actor.Attributes.exitCode}}' | tail -n 5" </dev/null
