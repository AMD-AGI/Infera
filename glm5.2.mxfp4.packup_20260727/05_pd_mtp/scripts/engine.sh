#!/usr/bin/env bash
# GLM-5.2-MXFP4 PD leg (prefill|decode) over MoRI-IO RDMA, with optional MTP on the DECODE leg.
# Extends the Phase-3 mori engine with spec-dec (EAGLE) on decode only. Self-rewritten (jiejing = ref).
# Required env: ROLE=prefill|decode  DATA_PLANE_IP  ETCD_EP  NATS_SERVER  [MTP=1 for decode]
set -euo pipefail
: "${ROLE:?prefill|decode}"; : "${DATA_PLANE_IP:?data-plane IP}"; : "${ETCD_EP:?etcd host:port}"; : "${NATS_SERVER:?nats url}"
IMAGE="${IMAGE:-rocm/infera:sglang-v0.1.0-rc6}"
MODEL="${MODEL:-/mnt/vast/xiaobo/models/GLM-5.2-MXFP4}"
SERVED="${SERVED:-glm5.2-mxfp4}"
CONC="${CONC:-64}"; MTP="${MTP:-0}"
NAME=glm52-pdmtp-$ROLE
LOGDIR=/mnt/vast/c_huggingface/glm52_p3b
NEXTN_PATCH=/mnt/vast/c_huggingface/glm52_nextn_patch/deepseek_nextn.py   # MY rc6 1-line fix
mkdir -p "$LOGDIR"
# decode mem-frac 0.80 (jiejing: MTP draft-extend needs KV headroom; 0.70 OOMs at conc). prefill 0.85.
[[ "$ROLE" == prefill ]] && MEMFRAC="${MEMFRAC:-0.85}" || MEMFRAC="${MEMFRAC:-0.80}"

SRC=$(readlink -f /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so)
# mount MY nextn patch only when MTP on (harmless otherwise, but keep it scoped to the MTP leg)
MTP_MOUNT=""
[[ "$MTP" == 1 ]] && MTP_MOUNT="-v $NEXTN_PATCH:/sgl-workspace/sglang/python/sglang/srt/models/deepseek_nextn.py:ro"

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --network=host --privileged --ipc=host --shm-size 64g \
  --ulimit memlock=-1 --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
  --group-add video --group-add render -v /mnt/vast:/mnt/vast \
  $MTP_MOUNT \
  -v "$SRC":/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:ro \
  -v "$SRC":/usr/lib/x86_64-linux-gnu/libionic.so.1:ro \
  "$IMAGE" sleep infinity >/dev/null

IB=$(docker exec "$NAME" bash -c 'for d in /sys/class/infiniband/*; do n=$(basename "$d"); s=$(cat "$d/ports/1/state" 2>/dev/null); dr=$(basename "$(readlink -f "$d/device/driver" 2>/dev/null)"); [[ "$s" == *ACTIVE* && "$dr" == ionic ]] && echo "$n"; done | sort -V | paste -sd,')
[[ -z "$IB" ]] && { echo "no active ionic NICs in $NAME"; exit 1; }
echo "[$ROLE] data_ip=$DATA_PLANE_IP mori NICs=$IB memfrac=$MEMFRAC MTP=$MTP"

ROLE_FLAGS="--disaggregation-mode $ROLE"
[[ "$ROLE" == prefill ]] && ROLE_FLAGS+=" --disaggregation-bootstrap-port 8998"

# MTP flags on DECODE leg only. Draft steps 3 (jiejing: 5 OOMs decode KV pool under PD concurrency;
# 3 + reserved 256 stable through N=64). EAGLE over the model's own nextn head.
MTP_FLAGS=""; MTP_ENV=""
if [[ "$MTP" == 1 && "$ROLE" == decode ]]; then
  MTP_FLAGS="--speculative-algorithm EAGLE --speculative-num-steps ${SPEC_STEPS:-3} --speculative-eagle-topk 1 --speculative-num-draft-tokens ${SPEC_DRAFT:-4} --num-reserved-decode-tokens ${RESERVED_TOK:-256}"
  # SGLANG_DSA_ENABLE_MTP_PRECOMPUTE_METADATA=0: skip gfx950-incompatible CUDA fused_metadata_copy JIT.
  MTP_ENV="SGLANG_DSA_ENABLE_MTP_PRECOMPUTE_METADATA=0"
fi

docker exec -d "$NAME" bash -lc "
  exec >$LOGDIR/pdmtp_${ROLE}.log 2>&1
  export SGLANG_HOST_IP=$DATA_PLANE_IP HOST_IP=$DATA_PLANE_IP
  export MORI_IB_GID_INDEX=1   # chi2832/chi2879 all NICs have routable GID at idx1
  export SGLANG_USE_AITER=1 SGLANG_ROCM_FUSED_DECODE_MLA=0
  export SGLANG_OPT_USE_TILELANG_INDEXER=1 SGLANG_OPT_USE_TOPK_V2=0 SGLANG_OPT_USE_JIT_NORM=0
  export HSA_NO_SCRATCH_RECLAIM=1 SAFETENSORS_FAST_GPU=1 HIP_FORCE_DEV_KERNARG=1
  export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1800 SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1800
  $MTP_ENV \
  python3 -m infera.engine.sglang \
    --discovery-backend etcd --etcd-endpoint $ETCD_EP --advertise-host $DATA_PLANE_IP \
    --request-transport nats --nats-server $NATS_SERVER --kv-events off \
    --model-path $MODEL --served-model-name $SERVED --host 0.0.0.0 --port 30000 \
    --tp-size 8 --trust-remote-code \
    --nsa-prefill-backend tilelang --nsa-decode-backend tilelang \
    --kv-cache-dtype fp8_e4m3 --mem-fraction-static $MEMFRAC --context-length 400000 \
    --chunked-prefill-size 8192 --cuda-graph-max-bs 64 --max-running-requests $CONC \
    $ROLE_FLAGS $MTP_FLAGS \
    --disaggregation-transfer-backend mori --disaggregation-ib-device $IB
"
echo "[$ROLE] launched -> $LOGDIR/pdmtp_${ROLE}.log"
