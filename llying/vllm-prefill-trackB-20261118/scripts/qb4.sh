#!/usr/bin/env bash
# Track B: run a command on the track-B node (hold_b.sbatch runner, queue-b/), from this work directory.
#   qb4.sh [-w] <name> '<command>'
W=/home/liyingli/bench_agentx/vllm-trackB-20261118
Q=$W/queue-b
wait=0; [ "${1:-}" = -w ] && { wait=1; shift; }
name=${1:?name}; shift
mkdir -p "$Q"
id="$Q/$(date -u +%m%d-%H%M%S)-$name"
{
    echo "export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=$HOME USER=$USER R=$W NODE0=\$(hostname -s)"
    echo "cd $W && $*"
} > "$id.tmp" && mv "$id.tmp" "$id.sh"
echo "$id"
[ "$wait" = 1 ] || exit 0
while [ ! -e "$id.rc" ]; do sleep 3; ls "$Q" > /dev/null; done
cat "$id.log"
echo "[rc=$(cat "$id.rc")]"
