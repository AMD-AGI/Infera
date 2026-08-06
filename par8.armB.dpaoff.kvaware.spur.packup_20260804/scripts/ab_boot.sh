#!/bin/bash
# Boot one PD leg for the A/B par8 pair. Generalises boot.sh, whose node IDs and
# IPs were hardcoded to the (now dead) 28485/28490 allocation, and which had no
# way to pass DPA / CHUNK / GMU through -- both of which arm B needs.
#
# Same detach discipline as boot.sh: write an env file to shared storage and run
# it with `docker exec -d`. NEVER background a long docker client inside
# `spur exec` -- the exec namespace teardown kills it.
#
# Usage: JOB=<job> MY=<ip> PIP=<prefill ip> ab_boot.sh <prefill|decode> <tag>
set -eu
ROLE="${1:?prefill|decode}"
TAG="${2:?tag}"
W=/shared_nfs/yihou_final_pr
CTR="${CTR:-agbench_mtp}"

JOB="${JOB:?JOB=spur job id}"
MY="${MY:?MY=this node ens3 ip}"
PIP="${PIP:?PIP=prefill node ens3 ip (etcd + bootstrap host)}"

CTX="${CTX:-262144}"
KVD="${KVD:-1}"
MTP="${MTP:-0}"
DPA="${DPA:-1}"
CHUNK="${CHUNK:-}"          # empty => glm52_leg_spur_mtp.sh derives it
GMU="${GMU:-}"              # empty => leg script's per-role default

[ "$ROLE" = prefill ] && PORT=30000 || PORT=30001

ENVF=$W/env_${TAG}_${ROLE}.sh
LOG=$W/logs/${TAG}_${ROLE}.log
mkdir -p "$W/logs"

{
  echo "export ROLE=$ROLE MY_IP=$MY P_IP=$PIP ETCD_IP=$PIP PORT=$PORT"
  echo "export CTX=$CTX DPA=$DPA KVAWARE=1 KVD=$KVD MTP=$MTP"
  [ -n "$CHUNK" ] && echo "export CHUNK=$CHUNK"
  [ -n "$GMU" ]   && echo "export GMU=$GMU"
  echo "export LOG=$LOG"
  # TORCHINDUCTOR_COMPILE_THREADS=4 is NOT sufficient on a cold Inductor cache --
  # the boot-time deadlock recurred with it set. =1 plus a warm shared cache dir
  # is the real fix.
  echo "export TORCHINDUCTOR_COMPILE_THREADS=1"
  echo "export TORCHINDUCTOR_CACHE_DIR=$W/inductor_cache"
  echo "export TRITON_CACHE_DIR=$W/triton_cache"
  echo "bash $W/scripts/glm52_leg_spur_mtp.sh"
} > "$ENVF"

echo "booting tag=$TAG role=$ROLE job=$JOB ip=$MY port=$PORT dpa=$DPA mtp=$MTP kvd=$KVD chunk=${CHUNK:-<derived>} gmu=${GMU:-<default>}"
sed 's/^/    /' "$ENVF"

# Patterns are BRACKETED ([s]glang) on purpose. A bare
# `pkill -9 -f infera.engine.sglang` also matches the `bash -c '...'` command
# string that CONTAINS that text -- i.e. this very shell -- so pkill kills
# itself, `|| true` hides it, and THE WAIT LOOP NEVER RUNS. Without the wait the
# next leg can start while the old one still holds the DP kv-event port block and
# dies with "port_base at N is not available in 30 seconds".
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
