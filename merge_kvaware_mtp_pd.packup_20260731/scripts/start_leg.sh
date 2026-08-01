#!/usr/bin/env bash
# Launch one PD leg inside the already-patched container. Runs ON the node.
#   ROLE=prefill|decode  MY_IP=<rail ip>  ETCD_IP=<prefill ip>  [MTP=0|1]  [TAG=g0]
set -u
ROLE="${ROLE:?}"
MY_IP="${MY_IP:?}"
ETCD_IP="${ETCD_IP:?}"
MTP="${MTP:-0}"
TAG="${TAG:-g0}"
CTR="${CTR:-merge_g0}"
MODEL="${MODEL:-/mnt/vast/xiaobo/models/GLM-5.2-MXFP4}"
KIT=/mnt/vast/c_huggingface/merge_20260731
LOG="$KIT/logs/${TAG}_${ROLE}.log"

mkdir -p "$KIT/logs"
# Any previous engine in this container must be gone before a new one binds.
docker exec "$CTR" bash -c "pkill -9 -f 'infera.engine.sglang' 2>/dev/null; sleep 6" || true

docker exec -d "$CTR" env \
  ROLE="$ROLE" MY_IP="$MY_IP" ETCD_IP="$ETCD_IP" MODEL="$MODEL" \
  SERVED=glm5.2-mxfp4 PORT=30000 KVAWARE=1 KVD=1 HICACHE_GB=16 \
  KVD_SOCK=/tmp/kvd/kvd.sock MTP="$MTP" \
  LOG="$LOG" bash /glm52_leg.sh

echo "[$TAG] $ROLE launched on $(hostname) mtp=$MTP -> $LOG"
