#!/usr/bin/env bash
# From the login node: run a command on the node of hold_n6.sbatch (its runner).
#   q.sh [-w] <name> <command...>
#   -w  wait for it and print its log and exit code (otherwise print the id and return)
# The item runs from the work directory R with NODE0 set to the node name.
set -uo pipefail
R=/home/liyingli/bench_agentx/vllm-pd-20261117
Q=$R/queue-n6
wait=0
[ "${1:-}" = -w ] && { wait=1; shift; }
name=${1:?name}; shift
id="$Q/$(date -u +%m%d-%H%M%S)-$name"
{
    echo "export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=$HOME USER=$USER R=$R NODE0=\$(hostname -s)"
    echo "$*"
} > "$id.tmp" && mv "$id.tmp" "$id.sh"
echo "$id"
[ "$wait" = 1 ] || exit 0
while [ ! -e "$id.rc" ]; do sleep 3; ls "$Q" > /dev/null; done
cat "$id.log"
rc=$(cat "$id.rc")
echo "[rc=$rc]"
exit "$rc"
