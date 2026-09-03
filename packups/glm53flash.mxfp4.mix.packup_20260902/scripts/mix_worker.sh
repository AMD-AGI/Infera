#!/bin/bash
# GLM-5.3-Flash MIX (aggregated) worker — prefill+decode colocated. No PD, no
# mooncake, no RDMA. Runs through the infera wrapper
# (`python3 -m infera.engine.sglang`) so infera discovery + kv-aware routing work.
#
# Defaults reproduce the OneNexus GLM-5.3-Flash-MXFP4 recipe VERBATIM (the model
# card shipped in the checkpoint), because round 1 changes nothing the vendor did
# not validate. Every axis we intend to move later is its own variable, so a
# round changes ONE thing and a passing run tells you which thing did it.
#
# THREE THINGS DIFFER FROM THE FP8-FLASH RECIPE, and all three are required:
#   * --quantization quark        the checkpoint is Quark MXFP4 (fp4 E2M1, 1x32
#                                 block scales). The card passes this explicitly
#                                 for Flash even though the big GLM-5.3-MXFP4
#                                 card says the loader auto-detects it.
#   * --moe-runner-backend aiter  native AITER FP4 MoE kernels
#                                 (torch.float4_e2m1fn_x2). With `triton` the
#                                 checkpoint is dequantised to BF16 GEMMs, which
#                                 still serves and is much slower.
#   * SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
#                                 vendor-set for this checkpoint. Not present in
#                                 the FP8 recipe; do not drop it as noise.
#
# CARRIED OVER, and each load-bearing:
#   * SGLANG_USE_AITER=1 gates #36607's AITER mHC pre/post dispatch, the fused
#     k-pool top-k and shared-expert fusion on gfx950. Without it the server
#     starts, answers correctly, and is 4.3-5.4x slower with nothing in any log
#     saying so. Grep the log for the two mHC lines -- that is the real health
#     check, not the absence of errors.
#   * DSA flags are --dsa-*-backend. GLM-5.2 used --nsa-*; copying that recipe
#     forward gets unknown-flag errors.
#   * MTP/speculative decoding is NOT validated for this model on ROCm. The AMD
#     lane disables it. Do not add --speculative-*.
#   * kvd/hicache OFF. Only patch_hicache_rocm_host_alloc.py is in this image;
#     the staged-write-back gate was deliberately excluded, so hicache/kvd is
#     unverified here. Re-derive both gates before turning either on.
#
# Runs INSIDE the container.
set -u
MY_IP="${MY_IP:?MY_IP=node IP (router+clients reach this)}"
ETCD_IP="${ETCD_IP:-$MY_IP}"
MODEL="${MODEL:?MODEL=weights dir inside container}"
SERVED="${SERVED:-glm5.3-flash-mxfp4}"
SHARED_EXPERT_FUSION="${SHARED_EXPERT_FUSION:-0}"
PORT="${PORT:-30000}"
ETCD_PORT="${ETCD_PORT:-2379}"

# TP4 is what the vendor validated for this checkpoint (4x MI350). It also
# leaves 4 GPUs free so a second arm can run concurrently on this node.
TP="${TP:-4}"
GPUS="${GPUS:-$(seq -s, 0 $((TP - 1)))}"

# --- vendor recipe values ----------------------------------------------------
QUANT="${QUANT:-quark}"
MOE_RUNNER="${MOE_RUNNER:-aiter}"
KV_DTYPE="${KV_DTYPE:-bfloat16}"
CTX="${CTX:-65536}"
GMU="${GMU:-0.80}"
MAX_RUNNING="${MAX_RUNNING:-32}"
CHUNK="${CHUNK:-4096}"
MAX_PREFILL="${MAX_PREFILL:-16384}"

# --- axes we intend to move in later rounds ----------------------------------
# Graphs OFF in round 1 because that is what the card validated. The FP8-Flash
# bring-up measured decode graphs at 7x (15.3 -> 106.9 tok/s at conc 1) for 33 s
# of capture and 1.4 GB, so this is expected to move -- but as its own round.
# The bs list is graph COVERAGE, not a concurrency limit: a decode batch is
# padded up to the next captured size and anything larger runs eager.
CUDA_GRAPH="${CUDA_GRAPH:-0}"
GRAPH_BS="${GRAPH_BS:-1 2 4 8 16 24 32 48 64 96 128}"
EP_SIZE="${EP_SIZE:-}"

KVAWARE="${KVAWARE:-1}"
KV_PUB_PORT="${KV_PUB_PORT:-5557}"
KV_SNAP_PORT="${KV_SNAP_PORT:-8801}"
LOG="${LOG:-/tmp/glm53_mix.log}"
mkdir -p "$(dirname "$LOG")"

export SGLANG_USE_AITER=1
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export SAFETENSORS_FAST_GPU=1 HIP_FORCE_DEV_KERNARG=1 HSA_NO_SCRATCH_RECLAIM=1
export NCCL_IGNORE_CPU_AFFINITY=1
# Stable block hashes -> stable kv-aware keys across restarts.
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

# --disable-cuda-graph is a DEPRECATED alias in this engine; pass the real flags
# so the engine cannot silently reinterpret intent. Prefill graphs stay off
# either way: that is what upstream validated on gfx950, and prefill is where
# the DSA/KDA shape variance lives.
if [ "$CUDA_GRAPH" = "1" ]; then
  GRAPH_ARGS=(--cuda-graph-backend-decode full --cuda-graph-backend-prefill disabled
              --cuda-graph-bs-decode $GRAPH_BS)
else
  GRAPH_ARGS=(--cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled)
fi
EP_ARGS=(); [ -n "$EP_SIZE" ] && EP_ARGS=(--ep-size "$EP_SIZE")
SEF_ARGS=(); [ "$SHARED_EXPERT_FUSION" = "0" ] && SEF_ARGS=(--disable-shared-experts-fusion)

echo "[glm53-mxfp4-mix] ip=$MY_IP:$PORT nic=${NIC:-?} tp=$TP gpus=$GPUS quant=$QUANT moe=$MOE_RUNNER kv=$KV_DTYPE graph=$CUDA_GRAPH gmu=$GMU ctx=$CTX chunk=$CHUNK maxrun=$MAX_RUNNING kvaware=$KVAWARE -> $LOG"
HIP_VISIBLE_DEVICES="$GPUS" python3 -m infera.engine.sglang \
  --model-path "$MODEL" --served-model-name "$SERVED" --tp-size "$TP" --trust-remote-code \
  --host "$MY_IP" --port "$PORT" \
  --quantization "$QUANT" \
  --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
  --kv-cache-dtype "$KV_DTYPE" --moe-runner-backend "$MOE_RUNNER" \
  --mem-fraction-static "$GMU" --context-length "$CTX" \
  --max-running-requests "$MAX_RUNNING" \
  --chunked-prefill-size "$CHUNK" --max-prefill-tokens "$MAX_PREFILL" \
  --watchdog-timeout 3600 \
  --reasoning-parser glm45 --tool-call-parser glm47 \
  --mm-feature-transport cpu \
  "${GRAPH_ARGS[@]}" "${EP_ARGS[@]}" "${SEF_ARGS[@]}" \
  "${INFERA_ARGS[@]}" > "$LOG" 2>&1
