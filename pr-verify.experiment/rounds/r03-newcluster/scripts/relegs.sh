#!/usr/bin/env bash
# Restart BOTH PD legs in a chosen arm, without tearing down containers or etcd.
#
#   ARM=stock   bash relegs.sh
#   ARM=patched bash relegs.sh
#
# Order matters and is the whole point:
#   1. stop the running engines
#   2. set the arm (git checkout / leave), and VERIFY it with arm.sh
#   3. only then relaunch
# The arm must be set while nothing is importing the module -- flipping files under a
# live engine changes nothing, because the code is already loaded.
#
# arm.sh's wait_event guard (stock=0, patched=9) is what stops a mislabelled A/B from
# being recorded as a result. If it refuses, this script stops; do not "fix" that by
# loosening the guard.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM="${ARM:?ARM=stock|patched}"
export SSH_CMD="${SSH_CMD:-$HERE/spur_ssh.sh}"

PREFILL_NODE="${PREFILL_NODE:-crsuse2-m2m-237}"
DECODE_NODE="${DECODE_NODE:-crsuse2-m2m-106}"
PREFILL_IP="${PREFILL_IP:-10.245.154.191}"
DECODE_IP="${DECODE_IP:-10.245.159.121}"
CTR="${CTR:-glm52_pd}"
KIT="${KIT:-/home/yihou/dev/git/infera.upstream.pr.verify/examples/sglang_1p1d_glm5.2}"
MODEL="${MODEL:-/mnt/m2m_nobackup/models/zai-org__GLM-5.2-FP8}"
LEG_TRIES="${LEG_TRIES:-140}"

say(){ echo -e "\033[0;32m[relegs]\033[0m $*"; }

# ---- 1. stop the engines --------------------------------------------------------------
say "stopping engines in both containers"
# Kill by GPU ownership, not by command string.
#
# `pgrep -f infera.engine.sglang` is NOT enough and cost a whole bring-up: the
# launcher execs sglang.launch_server, which forks one TP worker per GPU, and those
# workers are what actually hold VRAM. Killing the launcher left three orphans holding
# ~224 GB each, so the next legs died with
#   torch.OutOfMemoryError: HIP out of memory ... 96.00 MiB is free
# after the ranks had already connected -- minutes in, and nowhere near the real cause.
#
# So: kill every python3 in the container, then WAIT for the KFD process table to
# actually empty. VRAM is released asynchronously; a prompt `docker exec` can still
# see the old allocations.
for node in "$PREFILL_NODE" "$DECODE_NODE"; do
  $SSH_CMD "$node" "docker exec $CTR bash -c 'pkill -9 -f python3 2>/dev/null; true'"
done
for node in "$PREFILL_NODE" "$DECODE_NODE"; do
  for i in $(seq 1 30); do
    n=$($SSH_CMD "$node" "rocm-smi --showpids 2>/dev/null | grep -c UNKNOWN" 2>/dev/null | tr -d '\r\n ')
    [ "${n:-1}" = "0" ] && { say "  $node GPUs released"; break; }
    sleep 4
  done
done

# Wait for /health to actually STOP answering before going further. Without this the
# old engine is still serving when the new legs are launched, and wait_leg below
# returns 200 off the CORPSE within seconds -- "serving after 10s" for something whose
# cold start is minutes. That is the same class of stale-state trap common.sh warns
# about for log greps, and it silently produces an A/B against the wrong binary.
for spec in "$PREFILL_NODE:$PREFILL_IP:${PREFILL_PORT:-30000}:prefill" \
            "$DECODE_NODE:$DECODE_IP:${DECODE_PORT:-30001}:decode"; do
  IFS=: read -r _n _ip _p _r <<< "$spec"
  for i in $(seq 1 30); do
    code=$($SSH_CMD "$_n" "docker exec $CTR bash -c \"curl -s -o /dev/null -w '%{http_code}' -m 2 http://$_ip:$_p/health\"" 2>/dev/null | tr -d '\r\n ')
    [ "$code" != "200" ] && { say "  $_r stopped answering"; break; }
    sleep 2
  done
done

# ---- 2. set + verify the arm ----------------------------------------------------------
say "setting arm=$ARM"
ARM="$ARM" CTR="$CTR" PREFILL_NODE="$PREFILL_NODE" DECODE_NODE="$DECODE_NODE" \
  bash "$HERE/arm.sh" || { echo "[relegs] arm guard refused -- not launching" >&2; exit 1; }

# ---- 3. relaunch ----------------------------------------------------------------------
COMMON_ENV="CTR=$CTR INFERA_IMAGE=${INFERA_IMAGE:-infera-local:sglang-prverify-20260824} \
MODEL=$MODEL MODEL_MOUNT=${MODEL_MOUNT:-/mnt/m2m_nobackup/models} \
SERVED=${SERVED:-glm5.2-fp8} TOKENIZER=$MODEL \
ETCD_IP=$PREFILL_IP ETCD_PORT=${ETCD_PORT:-2379} ROUTER_PORT=${ROUTER_PORT:-8100} \
PREFILL_PORT=${PREFILL_PORT:-30000} DECODE_PORT=${DECODE_PORT:-30001} \
BOOTSTRAP_PORT=${BOOTSTRAP_PORT:-8998} KVD_SOCK=${KVD_SOCK:-/tmp/kvd/kvd.sock} \
TP=${TP:-8} CTX=${CTX:-262144} CHUNK=${CHUNK:-131072} KVAWARE=${KVAWARE:-0} \
CUDA_GRAPH_BS=${CUDA_GRAPH_BS:-128} \
PREFILL_MTP=0 RDMA_IB_DEVICES=${RDMA_IB_DEVICES:-mlx5_0} MC_GID_INDEX=${MC_GID_INDEX:-3} \
MOONCAKE_DISABLE_HIP_DMABUF=${MOONCAKE_DISABLE_HIP_DMABUF:-0} \
MC_MS_AUTO_DISC=${MC_MS_AUTO_DISC:-0} MC_MS_FILTERS=${MC_MS_FILTERS:-mlx5_0} \
RDMAV_FORK_SAFE=1 GMU_PREFILL=${GMU_PREFILL:-0.70} GMU_DECODE=${GMU_DECODE:-0.85}"

say "launching legs"
$SSH_CMD "$PREFILL_NODE" "$COMMON_ENV ROLE=prefill MY_IP=$PREFILL_IP PORT=${PREFILL_PORT:-30000} \
  DPA=${PREFILL_DPA:-0} MTP=0 KVD=0 bash $KIT/engine/leg.sh"
$SSH_CMD "$DECODE_NODE" "$COMMON_ENV ROLE=decode MY_IP=$DECODE_IP PORT=${DECODE_PORT:-30001} \
  DPA=${DECODE_DPA:-1} MTP=${DECODE_MTP:-1} KVD=0 bash $KIT/engine/leg.sh"

# ---- 4. wait -------------------------------------------------------------------------
# Poll from inside each node's container. Never grep the log for a ready line: logs are
# appended to across restarts, so a grep matches the PREVIOUS run within seconds.
wait_leg(){
  local node="$1" ip="$2" port="$3" role="$4"
  for i in $(seq 1 "$LEG_TRIES"); do
    code=$($SSH_CMD "$node" "docker exec $CTR bash -c \"curl -s -o /dev/null -w '%{http_code}' -m 3 http://$ip:$port/health\"" 2>/dev/null | tr -d '\r\n ')
    [ "$code" = "200" ] && { say "  $role serving after $((i*10))s"; return 0; }
    sleep 10
  done
  echo "[relegs] $role never became ready" >&2; return 1
}
say "waiting for both legs (cold start is minutes -- silence is not a hang)"
wait_leg "$PREFILL_NODE" "$PREFILL_IP" "${PREFILL_PORT:-30000}" prefill || exit 1
wait_leg "$DECODE_NODE"  "$DECODE_IP"  "${DECODE_PORT:-30001}" decode  || exit 1

say "restarting router so it re-pairs the fresh legs"
bash "$HERE/router.sh" || exit 1
say "UP, arm=$ARM"
