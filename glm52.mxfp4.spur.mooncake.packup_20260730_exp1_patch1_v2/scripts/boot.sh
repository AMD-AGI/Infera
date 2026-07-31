#!/bin/bash
# Boot one PD leg for one experiment arm. Run from the HOST; it dispatches into
# the node's dbg2 container with `docker exec -d` (never `spur exec ... &` --
# the exec namespace teardown kills a backgrounded client, CLAUDE.md).
#
# Usage: boot.sh <arm e1|e2|e3> <role prefill|decode>
#
# The per-leg environment is written to a file and sourced inside the
# container, rather than interpolated through the host -> spur -> docker -> sh
# quoting chain. E2 passes a JSON argument containing double quotes and braces;
# putting it in a file is the only way to be sure of what the server receives.
set -eu
ARM="${1:?e1|e2|e3}"
ROLE="${2:?prefill|decode}"
export DOCKER_CONFIG=/tmp/dockercfg
W=/shared_nfs/yihou_exp3way

# --- node table (see common/NODES.md) ---------------------------------------
case "$ARM" in
e1) PJOB=14315; PIP=10.245.159.138; DJOB=14316; DIP=10.245.157.171 ;;
e2) PJOB=14317; PIP=10.245.152.243; DJOB=14318; DIP=10.245.155.111 ;;
e3) PJOB=14320; PIP=10.245.154.156; DJOB=14321; DIP=10.245.144.119 ;;
*)  echo "unknown arm $ARM" >&2; exit 1 ;;
esac

if [ "$ROLE" = prefill ]; then
  JOB=$PJOB; MY=$PIP; PORT=30000
else
  JOB=$DJOB; MY=$DIP; PORT=30001
fi

mkdir -p "$W/$ARM"
ENVF=$W/$ARM/env_${ROLE}.sh
LOG=$W/$ARM/${ROLE}.log

# --- per-arm launcher configuration -----------------------------------------
# E2 is the only arm that turns IndexShare off and enables MTP on the prefill
# leg; the two go together (llying's recipe). E1/E3 keep the configuration all
# prior measurements were taken under, so the patch set is the only variable.
{
  echo "export MY_IP=$MY P_IP=$PIP ROLE=$ROLE PORT=$PORT DPA=1 MTP=1"
  echo "export LOG=$LOG"
  echo "export TORCHINDUCTOR_COMPILE_THREADS=1"
  # The JIT caches used to live under /home, which is now 100% full and
  # read-only in practice; a failed cache write would be silent and only show
  # up as a much longer boot.
  echo "export TORCHINDUCTOR_CACHE_DIR=$W/inductor_cache"
  echo "export TRITON_CACHE_DIR=$W/triton_cache"
  case "$ARM" in
  e2)
    echo "export PREFILL_MTP=1"
    echo "export EXTRA_ARGS='--json-model-override-args {\"index_share_for_mtp_iteration\":false}'"
    ;;
  *)
    echo "export PREFILL_MTP=0"
    ;;
  esac
  echo "bash $W/pd_leg_exp.sh"
} > "$ENVF"

echo "booting arm=$ARM role=$ROLE job=$JOB ip=$MY port=$PORT"
sed 's/^/    /' "$ENVF"

spur exec "$JOB" bash -c \
  "export DOCKER_CONFIG=/tmp/dockercfg; docker exec -d dbg2 bash $ENVF" \
  2>&1 | grep -v libtinfow
echo "  -> log: $LOG"
