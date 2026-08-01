#!/usr/bin/env bash
# Launch one PD leg inside the merged-image container. Runs ON the node.
#
# Identical to the merge kit's start_leg.sh except for the default container
# name, with ONE addition that matters: it kills the sglang.launch_server CHILD
# as well as the infera wrapper. The original pkills only 'infera.engine.sglang';
# the wrapper exits but its child keeps the DP kv-event port block bound, and the
# next leg dies with
#
#     ValueError: port_base at 30234 is not available in 30 seconds
#
# which reads as a port-allocation bug rather than as leftover state. Cost a full
# cycle on the built-image run.
#
#   ROLE=prefill|decode  MY_IP=<rail ip>  ETCD_IP=<prefill ip>  [MTP=0|1] [TAG=g0]
set -u
ROLE="${ROLE:?}"
MY_IP="${MY_IP:?}"
ETCD_IP="${ETCD_IP:?}"
MTP="${MTP:-0}"
TAG="${TAG:-g0}"
CTR="${CTR:-merged_run}"
MODEL="${MODEL:-/mnt/vast/xiaobo/models/GLM-5.2-MXFP4}"
KIT="${KIT:-/mnt/vast/c_huggingface/merge_20260731}"
LOG="$KIT/logs/${TAG}_${ROLE}.log"

mkdir -p "$KIT/logs"

# Kill the tree, then WAIT for it to actually be gone -- the wait is the point.
docker exec "$CTR" bash -c '
  pkill -9 -f sglang.launch_server 2>/dev/null
  pkill -9 -f infera.engine.sglang 2>/dev/null
  for i in $(seq 1 20); do
    n=$(ps aux | grep -E "launch_server|infera.engine" | grep -v grep | wc -l)
    [ "$n" -eq 0 ] && exit 0
    sleep 2
  done
  echo "  WARNING: engines still present after 40s" >&2' || true

docker exec -d "$CTR" env \
  ROLE="$ROLE" MY_IP="$MY_IP" ETCD_IP="$ETCD_IP" MODEL="$MODEL" \
  SERVED=glm5.2-mxfp4 PORT=30000 KVAWARE=1 KVD=1 HICACHE_GB=16 \
  KVD_SOCK=/tmp/kvd/kvd.sock MTP="$MTP" \
  LOG="$LOG" bash /glm52_leg.sh

echo "[$TAG] $ROLE launched on $(hostname) mtp=$MTP -> $LOG"
