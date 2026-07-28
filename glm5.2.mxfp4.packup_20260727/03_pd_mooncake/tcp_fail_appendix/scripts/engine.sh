#!/usr/bin/env bash
# GLM-5.2-MXFP4 PD leg (prefill|decode) over MOONCAKE on rocm/infera:sglang-v0.1.0-rc6.
# Self-rewritten from jiejing's launch_pd_{prefill,decode}{,_tcp}.sh (READ-ONLY reference).
# MODE=tcp -> MC_FORCE_TCP=1 (jiejing's proven correct-output path; RDMA is a kernel/driver
# dead-end on 6.8.0: no CONFIG_PCI_P2PDMA/DMABUF_MOVE_NOTIFY, peermem hard-faults on real xfer).
# MODE=rdma -> attempt RDMA via jiejing's protocol-priority-patched engine.so + peermem
# (MOONCAKE_DISABLE_HIP_DMABUF=1). Expected to hit the segfault wall; used only to confirm.
# Required env: ROLE=prefill|decode  DATA_PLANE_IP  ETCD_EP  NATS_SERVER  [MODE=tcp|rdma]
set -euo pipefail
: "${ROLE:?prefill|decode}"; : "${DATA_PLANE_IP:?data-plane IP}"; : "${ETCD_EP:?etcd host:port}"; : "${NATS_SERVER:?nats url}"
IMAGE="${IMAGE:-rocm/infera:sglang-v0.1.0-rc6}"
MODEL="${MODEL:-/mnt/vast/xiaobo/models/GLM-5.2-MXFP4}"
SERVED="${SERVED:-glm5.2-mxfp4}"
CONC="${CONC:-64}"; MODE="${MODE:-tcp}"
NAME=glm52-pdmc-$ROLE
LOGDIR=/mnt/vast/c_huggingface/glm52_p2
RDMAPRIO_SO=/mnt/vast/jiejing/crusoe_glm_52/patched/mooncake_engine_rdmaprio.so  # read-only ref .so
mkdir -p "$LOGDIR"
[[ "$ROLE" == prefill ]] && MEMFRAC="${MEMFRAC:-0.85}" || MEMFRAC="${MEMFRAC:-0.70}"

SRC=$(readlink -f /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so)
# transport env by mode
if [[ "$MODE" == tcp ]]; then
  MC_ENV="MC_FORCE_TCP=1 MC_DISABLE_HIP_TRANSPORT=1 MC_GID_INDEX=1 MC_IB_GID_INDEX=1 MOONCAKE_DISABLE_HIP_DMABUF=1"
  SO_MOUNT=""
else
  MC_ENV="MC_DISABLE_HIP_TRANSPORT=1 MC_GID_INDEX=1 MC_IB_GID_INDEX=1 MOONCAKE_DISABLE_HIP_DMABUF=1"
  SO_MOUNT="-v $RDMAPRIO_SO:/opt/venv/lib/python3.10/site-packages/mooncake/engine.cpython-310-x86_64-linux-gnu.so:ro"
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --network=host --privileged --ipc=host --shm-size 64g \
  --ulimit memlock=-1 --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
  --group-add video --group-add render --cap-add IPC_LOCK --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined -v /mnt/vast:/mnt/vast \
  $SO_MOUNT \
  -v "$SRC":/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:ro \
  -v "$SRC":/usr/lib/x86_64-linux-gnu/libionic.so.1:ro \
  "$IMAGE" sleep infinity >/dev/null

IB=$(docker exec "$NAME" bash -c 'for d in /sys/class/infiniband/*; do n=$(basename "$d"); s=$(cat "$d/ports/1/state" 2>/dev/null); dr=$(basename "$(readlink -f "$d/device/driver" 2>/dev/null)"); [[ "$s" == *ACTIVE* && "$dr" == ionic ]] && echo "$n"; done | sort -V | paste -sd,')
echo "[$ROLE/$MODE] data_ip=$DATA_PLANE_IP mooncake NICs=$IB memfrac=$MEMFRAC"

ROLE_FLAGS="--disaggregation-mode $ROLE"
[[ "$ROLE" == prefill ]] && ROLE_FLAGS+=" --disaggregation-bootstrap-port 8998"

docker exec -d "$NAME" bash -lc "
  exec >$LOGDIR/pdmc_${ROLE}.log 2>&1
  export SGLANG_HOST_IP=$DATA_PLANE_IP HOST_IP=$DATA_PLANE_IP
  export $MC_ENV
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
    --disaggregation-transfer-backend mooncake --disaggregation-ib-device $IB
"
echo "[$ROLE/$MODE] launched -> $LOGDIR/pdmc_${ROLE}.log"
