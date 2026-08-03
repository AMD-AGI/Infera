#!/bin/bash
# Boot one PD leg. Writes an env file to shared storage and runs it detached
# inside the container -- NEVER backgrounds a long docker client inside
# `spur exec`, which the exec namespace teardown kills.
#
# Also kills the sglang.launch_server CHILD, not just the infera wrapper: the
# wrapper exits but its child keeps the DP kv-event port block bound, and the
# next leg dies with "port_base at N is not available in 30 seconds", which reads
# as a port-allocation bug rather than as leftover state.
#
# Usage: boot.sh <prefill|decode> [ctx] [kvd] [mtp] [tag]
set -eu
ROLE="${1:?prefill|decode}"
CTX="${2:-262144}"
KVD="${3:-1}"
MTP="${4:-0}"
TAG="${5:-run}"
DPA="${DPA:-1}"          # 6th knob, env-only: 0 disables DP-attention on THIS leg
CHUNK="${CHUNK:-65536}"  # global prefill token budget; pinned across both arms
# GMU is forwarded because the noDPA arm needs a LOWER value than the DPA arm:
# without dp-attention one rank carries the whole chunk's attention activations
# rather than its 1/8 slice, and 0.80 (tuned for dp8) leaves too little outside
# the static reservation. Empty by default so the leg script's per-role default
# still applies when the caller says nothing.
GMU="${GMU:-}"
W=/shared_nfs/yihou_agbench_mtp
CTR="${CTR:-agbench_mtp}"

PJOB="${PJOB:-28490}"; PIP="${PIP:-10.245.150.172}"
DJOB="${DJOB:-28485}"; DIP="${DIP:-10.245.152.249}"

if [ "$ROLE" = prefill ]; then JOB=$PJOB; MY=$PIP; PORT=30000
else JOB=$DJOB; MY=$DIP; PORT=30001; fi

ENVF=$W/env_${ROLE}.sh
LOG=$W/logs/${TAG}_${ROLE}.log
mkdir -p "$W/logs"

{
  echo "export ROLE=$ROLE MY_IP=$MY P_IP=$PIP ETCD_IP=$PIP PORT=$PORT"
  echo "export CTX=$CTX DPA=$DPA KVAWARE=1 KVD=$KVD MTP=$MTP"
  # CHUNK is pinned so it is IDENTICAL on both arms -- see the long comment
  # in glm52_leg_spur_mtp.sh. DPA=0 used to silently cut it 8x.
  echo "export CHUNK=$CHUNK"
  [ -n "$GMU" ] && echo "export GMU=$GMU"
  echo "export LOG=$LOG"
  # TORCHINDUCTOR_COMPILE_THREADS=4 is NOT sufficient on a cold Inductor cache --
  # the boot-time deadlock recurred with it set. =1 plus a warm shared cache dir
  # is the real fix.
  echo "export TORCHINDUCTOR_COMPILE_THREADS=1"
  echo "export TORCHINDUCTOR_CACHE_DIR=$W/inductor_cache"
  echo "export TRITON_CACHE_DIR=$W/triton_cache"
  echo "bash $W/scripts/glm52_leg_spur_mtp.sh"
} > "$ENVF"

echo "booting role=$ROLE job=$JOB ip=$MY port=$PORT ctx=$CTX kvd=$KVD mtp=$MTP tag=$TAG"
sed 's/^/    /' "$ENVF"

# Patterns are BRACKETED ([s]glang) on purpose. A bare
# `pkill -9 -f infera.engine.sglang` also matches the `bash -c '...'` command
# string that CONTAINS that text -- i.e. this very shell -- so pkill kills
# itself. The `|| true` then hides it and THE WAIT LOOP BELOW NEVER RUNS, which
# is the entire point of this step: without the wait, the next leg can start
# while the old one still holds the DP kv-event port block and dies with
# "port_base at N is not available in 30 seconds".
spur exec "$JOB" bash -c "export DOCKER_CONFIG=/tmp/dockercfg
docker exec $CTR bash -c '
  pkill -9 -f \"[s]glang.launch_server\" 2>/dev/null
  pkill -9 -f \"[i]nfera.engine.sglang\" 2>/dev/null
  for i in \$(seq 1 20); do
    n=\$(ps aux | grep -cE \"[l]aunch_server|[i]nfera[.]engine\")
    [ \"\$n\" -eq 0 ] && { echo \"  engine tree gone after \$((i*2))s\"; exit 0; }
    sleep 2
  done
  echo \"  WARNING: engines still present after 40s\" >&2' || true
docker exec -d $CTR bash $ENVF" 2>&1 | grep -v libtinfow
echo "  -> log: $LOG"
