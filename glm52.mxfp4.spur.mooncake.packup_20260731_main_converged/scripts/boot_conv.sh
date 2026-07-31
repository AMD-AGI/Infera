#!/bin/bash
# Boot one PD leg for the FINAL validation run.
#
# Differences from the debug-round boot.sh:
#   * container is `final`, from the image built by Dockerfile.sglang.dmabuf;
#   * NO in-container patching -- the patches are baked into the image;
#   * graph-usage instrumentation is NOT applied either. Criterion "the draft
#     graph is actually used" is checked from the server's own startup log and
#     from the accept-length, not from a probe, because the deliverable image
#     must be the thing under test.
#
# Usage: boot_final.sh <prefill|decode>
set -eu
ROLE="${1:?prefill|decode}"
export DOCKER_CONFIG=/tmp/dockercfg
W=/shared_nfs/yihou_exp3way
F=$W/conv

PJOB=17443; PIP=10.245.152.84
DJOB=17444; DIP=10.245.151.183

if [ "$ROLE" = prefill ]; then JOB=$PJOB; MY=$PIP; PORT=30000
else JOB=$DJOB; MY=$DIP; PORT=30001; fi

ENVF=$F/env_${ROLE}.sh
LOG=$F/${ROLE}.log

{
  echo "export MY_IP=$MY P_IP=$PIP ROLE=$ROLE PORT=$PORT DPA=1 MTP=1"
  echo "export LOG=$LOG"
  echo "export TORCHINDUCTOR_COMPILE_THREADS=1"
  echo "export TORCHINDUCTOR_CACHE_DIR=$W/inductor_cache"
  echo "export TRITON_CACHE_DIR=$W/triton_cache"
  echo "export PREFILL_MTP=0"
  echo "bash $W/pd_leg_exp.sh"
} > "$ENVF"

echo "booting FINAL role=$ROLE job=$JOB ip=$MY port=$PORT"
sed 's/^/    /' "$ENVF"
spur exec "$JOB" bash -c \
  "export DOCKER_CONFIG=/tmp/dockercfg; docker exec -d conv bash $ENVF" \
  2>&1 | grep -v libtinfow
echo "  -> log: $LOG"
