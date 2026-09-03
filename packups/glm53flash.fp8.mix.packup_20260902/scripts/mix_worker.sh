#!/bin/bash
# GLM-5.3-Flash **FP8** (original, NOT MXFP4) MIX worker, TP4 on GPUs 4-7 of a
# SHARED node. Runs INSIDE the container.
#
# Adapted from /apps/yihou/packups/glm53flash.mix.packup_20260830/scripts/mix_worker.sh
# (TP8 FP8 bring-up, AMD correctness recipe #36608). Deltas:
#   * TP 8 -> 4, GPUS 0-7 -> 4-7. Half the node belongs to a colleague.
#   * GMU is passed in by mix_up.sh from a live free-VRAM measurement, because
#     --mem-fraction-static is a fraction of TOTAL, not free, memory.
#   * CUDA_GRAPH defaults OFF for round 0 (fewer unknowns), unlike the packup
#     which ships graphs on.
#
# Recipe notes specific to the FP8 checkpoint:
#   * NO --quantization flag: config.json already carries
#     quantization_config{fmt:e4m3, activation_scheme:dynamic}. The MXFP4 lane
#     needs `--quantization quark`; this one must not have it.
#   * MoE runner triton (the aiter FP4 path is for the MXFP4 checkpoint).
#   * KV dtype bfloat16.
#   * NO --disable-shared-experts-fusion. That flag is mandatory for MXFP4
#     because #36607 opened the gfx950 fusion path without a quantization
#     guard; here the precondition does not hold. Counted in
#     model.safetensors.index.json: 129 mlp.shared_experts.*.weight vs 129
#     matching .weight_scale_inv -- a clean 1:1 block-FP8 pairing, so fusion is
#     legitimate. Set DISABLE_SEF=1 if the load dies in _load_w2/_load_w13 with
#     "size of tensor a (N) must match tensor b (2N)".
set -u
MY_IP="${MY_IP:?MY_IP=node IP}"
ETCD_IP="${ETCD_IP:-$MY_IP}"
MODEL="${MODEL:?MODEL=weights dir inside container}"
SERVED="${SERVED:-glm5.3-flash}"
PORT="${PORT:-31400}"
ETCD_PORT="${ETCD_PORT:-23795}"
TP="${TP:-4}"
GPUS="${GPUS:-4,5,6,7}"
CTX="${CTX:-262144}"
CHUNK="${CHUNK:-8192}"
GMU="${GMU:?GMU=mem-fraction-static, set from a live rocm-smi measurement}"

CUDA_GRAPH="${CUDA_GRAPH:-0}"
GRAPH_BS="${GRAPH_BS:-1 2 4 8 16 24 32 48 64 96 128}"
KV_DTYPE="${KV_DTYPE:-bfloat16}"
MOE_RUNNER="${MOE_RUNNER:-triton}"
EP_SIZE="${EP_SIZE:-}"
DISABLE_SEF="${DISABLE_SEF:-0}"

KVAWARE="${KVAWARE:-1}"
KV_PUB_PORT="${KV_PUB_PORT:-15570}"
KV_SNAP_PORT="${KV_SNAP_PORT:-18801}"
LOG="${LOG:-/tmp/glm53_f8_mix.log}"
mkdir -p "$(dirname "$LOG")"

# Gates #36607's AITER mHC pre/post dispatch and the fused k-pool top-k on
# gfx950. Without it the server still starts, still answers correctly, and is
# 4.3-5.4x slower with nothing in any log saying so.
export SGLANG_USE_AITER=1
export SAFETENSORS_FAST_GPU=1 HIP_FORCE_DEV_KERNARG=1 HSA_NO_SCRATCH_RECLAIM=1
export NCCL_IGNORE_CPU_AFFINITY=1
export PYTHONHASHSEED=0
NIC=$(ip -o -4 addr show | awk -v ip="$MY_IP" '$4 ~ ("^" ip "/") {print $2; exit}')
[ -n "$NIC" ] && export SGLANG_LOCAL_IP_NIC="$NIC" GLOO_SOCKET_IFNAME="$NIC"
export SGLANG_HOST_IP="$MY_IP" HOST_IP="$MY_IP"
export INFERA_SGLANG_READY_TIMEOUT="${READY_TIMEOUT:-3600}"

INFERA_ARGS=(--advertise-host "$MY_IP" --etcd-endpoint "$ETCD_IP:$ETCD_PORT"
             --discovery-backend etcd --request-transport http --kv-event-transport zmq)
if [ "$KVAWARE" = "1" ]; then
  INFERA_ARGS+=(--kv-events-bind "tcp://0.0.0.0:$KV_PUB_PORT" --kv-snapshot-port "$KV_SNAP_PORT")
else
  INFERA_ARGS+=(--no-enable-kv-events)
fi

GRAPH_ARGS=()
if [ "$CUDA_GRAPH" = "1" ]; then
  GRAPH_ARGS=(--cuda-graph-backend-decode full --cuda-graph-backend-prefill disabled
              --cuda-graph-bs-decode $GRAPH_BS)
else
  GRAPH_ARGS=(--cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled)
fi
EP_ARGS=(); [ -n "$EP_SIZE" ] && EP_ARGS=(--ep-size "$EP_SIZE")
SEF_ARGS=(); [ "$DISABLE_SEF" = "1" ] && SEF_ARGS=(--disable-shared-experts-fusion)

echo "[glm53-f8-mix] ip=$MY_IP:$PORT nic=${NIC:-?} tp=$TP gpus=$GPUS gmu=$GMU chunk=$CHUNK ctx=$CTX graph=$CUDA_GRAPH kv=$KV_DTYPE moe=$MOE_RUNNER sef_disabled=$DISABLE_SEF -> $LOG"
HIP_VISIBLE_DEVICES="$GPUS" python3 -m infera.engine.sglang \
  --model-path "$MODEL" --served-model-name "$SERVED" --tp-size "$TP" --trust-remote-code \
  --host "$MY_IP" --port "$PORT" \
  --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
  --kv-cache-dtype "$KV_DTYPE" --moe-runner-backend "$MOE_RUNNER" \
  --mem-fraction-static "$GMU" --context-length "$CTX" \
  --chunked-prefill-size "$CHUNK" --watchdog-timeout 3600 \
  --reasoning-parser glm45 --tool-call-parser glm47 \
  "${GRAPH_ARGS[@]}" "${EP_ARGS[@]}" "${SEF_ARGS[@]}" \
  "${INFERA_ARGS[@]}" > "$LOG" 2>&1
