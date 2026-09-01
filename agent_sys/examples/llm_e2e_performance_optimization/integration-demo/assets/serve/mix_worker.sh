#!/bin/bash
# GLM-5.3-Flash MIX (aggregated) worker. RUNS INSIDE THE ENGINE CONTAINER.
#
# Taken from examples/glm53flash-demo/scripts/mix_worker.sh. The engine recipe is
# the AMD-validated one (cookbook #36608) and is NOT to be tuned here:
#   * no MTP/EAGLE — speculative decoding is not validated for this model on ROCm
#   * KV dtype BF16, not fp8_e4m3 (that belongs to the optimized recipe #36732)
#   * DSA flags are --dsa-*-backend, not GLM-5.2's --nsa-*-backend
#   * MoE runner pinned to triton
#   * kvd/hicache OFF: the image builds with APPLY_SGLANG_ROCM_PATCHES=0, so the
#     hicache host-allocator fix is absent and turning hicache on invites the
#     gfx950 host-VA memory-access fault
#
# The only thing this file adds over the demo's copy is that CUDA_GRAPH is the
# axis the two profiling rounds turn: graphs on for the baseline number, off for
# the trace, because a captured graph gives the profiler one launch instead of
# the kernels inside it.
set -u
MY_IP="${MY_IP:?MY_IP=node IP}"
ETCD_IP="${ETCD_IP:-$MY_IP}"
MODEL="${MODEL:?MODEL=weights dir inside container}"
SERVED="${SERVED:-glm5.3-flash}"
PORT="${PORT:-30000}"
ETCD_PORT="${ETCD_PORT:-12379}"   # not 2379: k8s owns that on these nodes
TP="${TP:-8}"
GPUS="${GPUS:-$(seq -s, 0 $((TP - 1)))}"
CTX="${CTX:-262144}"
CHUNK="${CHUNK:-8192}"
GMU="${GMU:-0.85}"

CUDA_GRAPH="${CUDA_GRAPH:-1}"
GRAPH_BS="${GRAPH_BS:-1 2 4 8 16 24 32 48 64 96 128}"
KV_DTYPE="${KV_DTYPE:-bfloat16}"
MOE_RUNNER="${MOE_RUNNER:-triton}"
EP_SIZE="${EP_SIZE:-}"

KVAWARE="${KVAWARE:-1}"
KV_PUB_PORT="${KV_PUB_PORT:-5557}"
KV_SNAP_PORT="${KV_SNAP_PORT:-8801}"
LOG="${LOG:-/tmp/glm53_mix.log}"
mkdir -p "$(dirname "$LOG")"

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

echo "[glm53-mix] ip=$MY_IP:$PORT nic=${NIC:-?} tp=$TP gpus=$GPUS gmu=$GMU chunk=$CHUNK ctx=$CTX graph=$CUDA_GRAPH bs='$GRAPH_BS' kv=$KV_DTYPE moe=$MOE_RUNNER -> $LOG"
HIP_VISIBLE_DEVICES="$GPUS" python3 -m infera.engine.sglang \
  --model-path "$MODEL" --served-model-name "$SERVED" --tp-size "$TP" --trust-remote-code \
  --host "$MY_IP" --port "$PORT" \
  --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
  --kv-cache-dtype "$KV_DTYPE" --moe-runner-backend "$MOE_RUNNER" \
  --mem-fraction-static "$GMU" --context-length "$CTX" \
  --chunked-prefill-size "$CHUNK" --watchdog-timeout 3600 \
  --reasoning-parser glm45 --tool-call-parser glm47 \
  "${GRAPH_ARGS[@]}" "${EP_ARGS[@]}" \
  "${INFERA_ARGS[@]}" > "$LOG" 2>&1
