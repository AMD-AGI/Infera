#!/bin/bash
# Boot one PD leg. Writes an env file to shared storage and runs it detached
# inside the container -- NEVER backgrounds a long docker client inside
# `spur exec`, which the exec namespace teardown kills.
#
# Usage: boot.sh <prefill|decode> [ctx] [kvd]
#   kvd=1 (default) wires the infera-kvd HiCacheStorage backend on this leg.
#   kvd=0 leaves kv-aware routing ON but no hierarchical cache -- see notes.md
#   for why the decode leg runs kvd=0 here.
set -eu
ROLE="${1:?prefill|decode}"
CTX="${2:-131072}"
KVD="${3:-1}"
export DOCKER_CONFIG=/tmp/dockercfg
W=/shared_nfs/yihou_agentbench

PJOB=19254; PIP=10.245.153.38
DJOB=19255; DIP=10.245.151.183

if [ "$ROLE" = prefill ]; then JOB=$PJOB; MY=$PIP; PORT=30000
else JOB=$DJOB; MY=$DIP; PORT=30001; fi

ENVF=$W/env_${ROLE}.sh
LOG=$W/${ROLE}.log

{
  echo "export ROLE=$ROLE MY_IP=$MY P_IP=$PIP ETCD_IP=$PIP PORT=$PORT"
  echo "export CTX=$CTX DPA=1 KVAWARE=1 KVD=$KVD"
  echo "export LOG=$LOG"
  echo "export TORCHINDUCTOR_COMPILE_THREADS=1"
  echo "export TORCHINDUCTOR_CACHE_DIR=$W/inductor_cache"
  echo "export TRITON_CACHE_DIR=$W/triton_cache"
  echo "bash $W/scripts/glm52_leg_spur.sh"
} > "$ENVF"

echo "booting role=$ROLE job=$JOB ip=$MY port=$PORT ctx=$CTX kvd=$KVD"
sed 's/^/    /' "$ENVF"
spur exec "$JOB" bash -c \
  "export DOCKER_CONFIG=/tmp/dockercfg; docker exec -d agbench bash $ENVF" \
  2>&1 | grep -v libtinfow
echo "  -> log: $LOG"
