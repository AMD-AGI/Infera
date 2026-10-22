#!/usr/bin/env bash
# Task idxshare debug 2026-10-20: run ON a held node. At <UTC HH:MM> (before the job's 24 h MaxWall), docker stop
# (never rm) every container of OURS on this node (name prefix glm52-v0521-yihou-<key>-), so nothing outlives the job.
# Usage: stop_before_maxwall.yihou.sh <key> <HH:MM UTC>
set -u
K=$1; AT=$2
[[ "$K" =~ ^[0-9]{3}$ && "$AT" =~ ^[0-9]{2}:[0-9]{2}$ ]] || { echo "bad args"; exit 1; }
target=$(date -u -d "today $AT" +%s); now=$(date -u +%s)
(( target <= now )) && target=$(date -u -d "tomorrow $AT" +%s)
echo "$(date -u +%FT%TZ) armed: stop glm52-v0521-yihou-$K-* at $(date -u -d @$target +%FT%TZ)"
sleep $(( target - now ))
names=$(docker ps --format '{{.Names}}' | grep -E "^glm52-v0521-yihou-$K-")
echo "$(date -u +%FT%TZ) stopping: $names"
[[ -n "$names" ]] && docker stop -t 30 $names
docker ps --format '{{.Names}}'
