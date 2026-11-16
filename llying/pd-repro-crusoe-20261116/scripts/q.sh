#!/usr/bin/env bash
# From the login node: run a command on NODE0 of the hold job through its queue runner (hold2.sbatch).
#   q.sh [-w] <name> <command...>
#   -w  wait for it, then print its log and exit code (otherwise print the item id and return)
# Inside the item: NODE0 / NODE1 / JOB are set, `ssh $NODE1 cmd` goes through tools/sshshim (srun --overlap).
set -uo pipefail
Q=/home/liyingli/bench_agentx/pd-repro-20261116/queue
wait=0
[ "${1:-}" = -w ] && { wait=1; shift; }
name=${1:?name}; shift
id="$Q/$(date -u +%m%d-%H%M%S)-$name"
{
    echo "export HOME=$HOME USER=$USER"
    echo "$*"
} > "$id.tmp" && mv "$id.tmp" "$id.sh"
echo "$id"
[ "$wait" = 1 ] || exit 0
while [ ! -e "$id.rc" ]; do sleep 3; ls "$Q" > /dev/null; done
cat "$id.log"
rc=$(cat "$id.rc")
echo "[rc=$rc]"
exit "$rc"
