#!/bin/bash
# GLM-5.2-MXFP4 MIX (aggregated) worker — prefill+decode colocated on ONE 8-GPU node.
# No PD, no mooncake, no RDMA. Runs through the infera wrapper so kvaware + kvd work.
#
# Derived from the validated PD launcher (par8 glm52_leg.sh) by REMOVING every
# disaggregation/mooncake/RDMA arg and env. The tuned recipe (DSA-ROCm envs, MTP,
# kvd, kv-cache-dtype, ctx, reasoning-parser) is carried over byte-for-byte.
#
# Runs INSIDE the container. Env in from the caller; sensible crsuse defaults.
set -u
MY_IP="${MY_IP:?MY_IP=node data-plane IP (router+clients reach this)}"
ETCD_IP="${ETCD_IP:-$MY_IP}"
MODEL="${MODEL:?MODEL=weights dir inside container}"
SERVED="${SERVED:-glm5.2-mxfp4}"
PORT="${PORT:-30000}"
ETCD_PORT="${ETCD_PORT:-2379}"
TP="${TP:-8}"
GPUS="${GPUS:-$(seq -s, 0 $((TP - 1)))}"
CTX="${CTX:-262144}"
MAX_RUNNING="${MAX_RUNNING:-256}"
CUDA_GRAPH_BS="${CUDA_GRAPH_BS:-32}"
# mix is nodp (DPA off) for conc<=128 — mix guide. chunked-prefill-size is a global
# budget; 8192 is the nodp value (README note 2).
CHUNK="${CHUNK:-8192}"
# Single aggregated worker: no role asymmetry. 0.85 is the recipe's mix value
# (aggregated-kvd manifest). Lower on a prefill activation OOM (README note 3).
GMU="${GMU:-0.85}"

# --- features (full-feature mix baseline: MTP on, kvd on, kvaware on) ---
MTP="${MTP:-1}"
# aiter custom all-reduce deadlocks on gfx950 in EAGLE verify — disable with MTP.
CUSTOM_AR="${CUSTOM_AR:-$([ "$MTP" = "1" ] && echo 0 || echo 1)}"
KVAWARE="${KVAWARE:-1}"
KVD="${KVD:-1}"
KVD_SOCK="${KVD_SOCK:-/tmp/kvd/kvd.sock}"
HICACHE_GB="${HICACHE_GB:-16}"
KV_PUB_PORT="${KV_PUB_PORT:-5557}"
KV_SNAP_PORT="${KV_SNAP_PORT:-8801}"
LOG="${LOG:-/tmp/glm52_mix.log}"
mkdir -p "$(dirname "$LOG")"

# GLM-5.2 DSA-ROCm recipe (mandatory on gfx950) — without these the model serves
# 200s of GARBAGE (README note 5).
export SGLANG_USE_AITER=1 SGLANG_ROCM_FUSED_DECODE_MLA=0
export SGLANG_OPT_USE_TILELANG_INDEXER=1 SGLANG_OPT_USE_TOPK_V2=0 SGLANG_OPT_USE_JIT_NORM=0
export SAFETENSORS_FAST_GPU=1 HIP_FORCE_DEV_KERNARG=1 HSA_NO_SCRATCH_RECLAIM=1
export NCCL_IGNORE_CPU_AFFINITY=1
# Stable block hashes -> stable kvd keys across restarts.
export PYTHONHASHSEED=0
NIC=$(ip -o -4 addr show | awk -v ip="$MY_IP" '$4 ~ ("^" ip "/") {print $2; exit}')
[ -n "$NIC" ] && export SGLANG_LOCAL_IP_NIC="$NIC" GLOO_SOCKET_IFNAME="$NIC"
export SGLANG_HOST_IP="$MY_IP" HOST_IP="$MY_IP"
export INFERA_SGLANG_READY_TIMEOUT="${READY_TIMEOUT:-3600}"

# --ep-size unconditional (expert parallelism != attention parallelism; README note 1).
DP_ARGS=(--ep-size "$TP")   # DPA off: no --dp-size / --enable-dp-attention

INFERA_ARGS=(--advertise-host "$MY_IP" --etcd-endpoint "$ETCD_IP:$ETCD_PORT"
             --discovery-backend etcd --request-transport http --kv-event-transport zmq)
if [ "$KVAWARE" = "1" ]; then
  INFERA_ARGS+=(--kv-events-bind "tcp://0.0.0.0:$KV_PUB_PORT" --kv-snapshot-port "$KV_SNAP_PORT")
else
  INFERA_ARGS+=(--no-enable-kv-events)
fi
[ "$KVD" = "1" ] && INFERA_ARGS+=(--infera-kvd-socket "$KVD_SOCK" --hicache-size "$HICACHE_GB")

MTP_ARGS=()
if [ "$MTP" = "1" ]; then
  MTP_ARGS=(--speculative-algorithm EAGLE --speculative-num-steps "${SPEC_STEPS:-3}" \
            --speculative-eagle-topk 1 --speculative-num-draft-tokens "${SPEC_DRAFT:-4}" \
            --num-reserved-decode-tokens "${RESERVED_TOK:-256}")
fi
CAR_ARGS=(); [ "$CUSTOM_AR" = "1" ] || CAR_ARGS+=(--disable-custom-all-reduce)
EXTRA_ARGS=(--enable-cache-report)

echo "[glm52-mix] ip=$MY_IP:$PORT nic=${NIC:-?} tp=$TP gpus=$GPUS mtp=$MTP car=$CUSTOM_AR kvaware=$KVAWARE kvd=$KVD gmu=$GMU chunk=$CHUNK ctx=$CTX -> $LOG"
HIP_VISIBLE_DEVICES="$GPUS" python3 -m infera.engine.sglang \
  --model-path "$MODEL" --served-model-name "$SERVED" --tp-size "$TP" --trust-remote-code \
  --host "$MY_IP" --port "$PORT" \
  --nsa-prefill-backend tilelang --nsa-decode-backend tilelang \
  --kv-cache-dtype fp8_e4m3 --mem-fraction-static "$GMU" --context-length "$CTX" \
  --chunked-prefill-size "$CHUNK" --cuda-graph-max-bs "$CUDA_GRAPH_BS" \
  --max-running-requests "$MAX_RUNNING" --watchdog-timeout 3600 \
  --reasoning-parser glm45 \
  "${INFERA_ARGS[@]}" "${DP_ARGS[@]}" "${MTP_ARGS[@]}" "${CAR_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" > "$LOG" 2>&1
