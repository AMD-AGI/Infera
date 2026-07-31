#!/usr/bin/env bash
# GLM-5.2-MXFP4 PD leg (prefill|decode) over MoRI-IO RDMA on rocm/infera:sglang-v0.1.0-rc6.
# Self-rewritten from jiejing's launch_glm52_mori.sh (READ-ONLY reference). Runs ON a node
# (called via ssh). Starts a persistent container (libionic mount for RDMA visibility) then
# launches infera.engine.sglang with GLM-5.2 DSA ROCm envs + mori PD flags.
# Required env: ROLE=prefill|decode  DATA_PLANE_IP  ETCD_EP  NATS_SERVER
set -euo pipefail
: "${ROLE:?prefill|decode}"; : "${DATA_PLANE_IP:?data-plane IP}"; : "${ETCD_EP:?etcd host:port}"; : "${NATS_SERVER:?nats url}"
IMAGE="${IMAGE:-rocm/infera:sglang-v0.1.0-rc6}"
MODEL="${MODEL:-/mnt/vast/xiaobo/models/GLM-5.2-MXFP4}"
SERVED="${SERVED:-glm5.2-mxfp4}"
CONC="${CONC:-64}"
NAME=glm52-pdmori-$ROLE
LOGDIR=/mnt/vast/c_huggingface/glm52_p3
mkdir -p "$LOGDIR"
[[ "$ROLE" == prefill ]] && MEMFRAC="${MEMFRAC:-0.85}" || MEMFRAC="${MEMFRAC:-0.70}"

SRC=$(readlink -f /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so)
docker rm -f "$NAME" >/dev/null 2>&1 || true
# persistent container: privileged + infiniband + memlock for MoRI RDMA; host libionic mounted
# over the image's older one (else libibverbs enumerates 0 devices -> MoRI sees no RDMA).
docker run -d --name "$NAME" --network=host --privileged --ipc=host --shm-size 64g \
  --ulimit memlock=-1 --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
  --group-add video --group-add render -v /mnt/vast:/mnt/vast \
  -v "$SRC":/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:ro \
  -v "$SRC":/usr/lib/x86_64-linux-gnu/libionic.so.1:ro \
  "$IMAGE" sleep infinity >/dev/null

# all ACTIVE ionic NICs (MoRI pairs NIC<->NIC by GID subnet -> needs them ALL)
IB=$(docker exec "$NAME" bash -c 'for d in /sys/class/infiniband/*; do n=$(basename "$d"); s=$(cat "$d/ports/1/state" 2>/dev/null); dr=$(basename "$(readlink -f "$d/device/driver" 2>/dev/null)"); [[ "$s" == *ACTIVE* && "$dr" == ionic ]] && echo "$n"; done | sort -V | paste -sd,')
[[ -z "$IB" ]] && { echo "no active ionic NICs in $NAME"; exit 1; }
echo "[$ROLE] data_ip=$DATA_PLANE_IP mori NICs=$IB gid=AUTO(-1) memfrac=$MEMFRAC"

ROLE_FLAGS="--disaggregation-mode $ROLE"
[[ "$ROLE" == prefill ]] && ROLE_FLAGS+=" --disaggregation-bootstrap-port 8998"

docker exec -d "$NAME" bash -lc "
  exec >$LOGDIR/pdmori_${ROLE}.log 2>&1
  export SGLANG_HOST_IP=$DATA_PLANE_IP HOST_IP=$DATA_PLANE_IP
  # MoRI per-NIC GID auto-select. MUST be -1 EXPLICITLY (infera rocm_rdma_env does
  # setdefault('MORI_IB_GID_INDEX','1') -> forcing idx1 => ENODATA on NICs where idx1 empty
  # => QP RTR crash => TP collapse code=-9). -1 => MoRI AutoSelectGidIndex per NIC.
  export MORI_IB_GID_INDEX=-1
  export SGLANG_USE_AITER=1 SGLANG_ROCM_FUSED_DECODE_MLA=0
  export SGLANG_OPT_USE_TILELANG_INDEXER=1 SGLANG_OPT_USE_TOPK_V2=0 SGLANG_OPT_USE_JIT_NORM=0
  export HSA_NO_SCRATCH_RECLAIM=1 SAFETENSORS_FAST_GPU=1 HIP_FORCE_DEV_KERNARG=1
  export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1800 SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1800
  python3 -m infera.engine.sglang \
    --discovery-backend etcd --etcd-endpoint $ETCD_EP --advertise-host $DATA_PLANE_IP \
    --request-transport nats --nats-server $NATS_SERVER --kv-events off \
    --model-path $MODEL --served-model-name $SERVED --host 0.0.0.0 --port 30000 \
    --tp-size 8 --trust-remote-code \
    --nsa-prefill-backend tilelang --nsa-decode-backend tilelang \
    --kv-cache-dtype fp8_e4m3 --mem-fraction-static $MEMFRAC --context-length 400000 \
    --chunked-prefill-size 8192 --cuda-graph-max-bs 64 --max-running-requests $CONC \
    $ROLE_FLAGS \
    --disaggregation-transfer-backend mori --disaggregation-ib-device $IB
"
echo "[$ROLE] launched -> $LOGDIR/pdmori_${ROLE}.log"
