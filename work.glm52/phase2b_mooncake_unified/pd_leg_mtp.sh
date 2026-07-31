#!/bin/bash
# GLM-5.2-MXFP4 cross-node PD leg over mooncake on infera/engine-sglang:pd-unified, WITH optional
# MTP on the decode leg. Extends pd_leg.sh. Runs INSIDE the pd_uni container.
# MTP note (pd-unified): needs ONLY the nextn 1-line patch (eh_proj quark-exclude match). The
# gfx950 CUDA fused_metadata_copy kernel is NOT hit here: pd-unified's dsa_backend try/excepts it
# (steps>3) AND uses a plain loop for steps<=3 — so no SGLANG_DSA_ENABLE_MTP_PRECOMPUTE_METADATA
# env needed (it doesn't exist in this image anyway). Use EAGLE steps=3 for PD.
set -u
ROLE="${ROLE:?ROLE=prefill|decode}"
MY_IP="${MY_IP:?MY_IP=data-plane rail IP}"
MODEL="${MODEL:-/mnt/vast/xiaobo/models/GLM-5.2-MXFP4}"
SERVED="${SERVED:-glm5.2-mxfp4}"
PORT="${PORT:-30000}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
TP="${TP:-8}"
CTX="${CTX:-400000}"
CHUNK="${CHUNK:-8192}"
MAX_RUNNING="${MAX_RUNNING:-64}"
DMABUF="${DMABUF:-0}"
MTP="${MTP:-0}"
if [ "$ROLE" = "prefill" ]; then GMU="${GMU:-0.85}"; else GMU="${GMU:-0.80}"; fi
LOG="${LOG:-/mnt/vast/c_huggingface/glm52_p2b/pd_${ROLE}_${PORT}.log}"
mkdir -p "$(dirname "$LOG")"

IB_DEVICES=$(for d in /sys/class/infiniband/*; do
    [ -d "$d" ] || continue; n=$(basename "$d")
    s=$(cat "$d/ports/1/state" 2>/dev/null || echo "")
    drv=$(basename "$(readlink -f "$d/device/driver" 2>/dev/null || echo x)")
    [[ "$s" == *ACTIVE* && "$drv" == ionic ]] && echo "$n"
  done | sort -V | paste -sd,)
[ -z "$IB_DEVICES" ] && { echo "no active ionic NICs" >&2; exit 1; }

if [ "$DMABUF" = "1" ]; then export MOONCAKE_DISABLE_HIP_DMABUF=0; else export MOONCAKE_DISABLE_HIP_DMABUF=1; fi
export MC_GID_INDEX=1 MC_DISABLE_HIP_TRANSPORT=1
unset MC_ENABLE_HIP_TRANSPORT
export RDMAV_FORK_SAFE=1
export NCCL_IB_DISABLE=1 NCCL_IGNORE_CPU_AFFINITY=1 HSA_NO_SCRATCH_RECLAIM=1
export SGLANG_HOST_IP="$MY_IP" HOST_IP="$MY_IP"
NIC=$(ip -o -4 addr show | awk -v ip="$MY_IP" '$4 ~ ("^" ip "/") {print $2; exit}')
export SGLANG_LOCAL_IP_NIC="$NIC" GLOO_SOCKET_IFNAME="$NIC"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1800 SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1800
export SGLANG_USE_AITER=1 SGLANG_ROCM_FUSED_DECODE_MLA=0
export SGLANG_OPT_USE_TILELANG_INDEXER=1 SGLANG_OPT_USE_TOPK_V2=0 SGLANG_OPT_USE_JIT_NORM=0
export SAFETENSORS_FAST_GPU=1 HIP_FORCE_DEV_KERNARG=1

ROLE_ARGS=(--disaggregation-mode "$ROLE" --disaggregation-transfer-backend mooncake \
           --disaggregation-ib-device "$IB_DEVICES")
[ "$ROLE" = "prefill" ] && ROLE_ARGS+=(--disaggregation-bootstrap-port "$BOOTSTRAP_PORT")

# MTP (decode leg only). EAGLE over the model's nextn head. steps=3 for PD KV-pool stability.
MTP_ARGS=()
if [ "$MTP" = "1" ] && [ "$ROLE" = "decode" ]; then
  MTP_ARGS=(--speculative-algorithm EAGLE --speculative-num-steps "${SPEC_STEPS:-3}" \
            --speculative-eagle-topk 1 --speculative-num-draft-tokens "${SPEC_DRAFT:-4}" \
            --num-reserved-decode-tokens "${RESERVED_TOK:-256}")
fi

echo "[glm52-mc] role=$ROLE ip=$MY_IP nic=$NIC gmu=$GMU dmabuf=$DMABUF mtp=$MTP ib=$IB_DEVICES -> $LOG"
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m sglang.launch_server \
  --model-path "$MODEL" --served-model-name "$SERVED" --tp-size "$TP" --trust-remote-code \
  --host "$MY_IP" --port "$PORT" \
  --nsa-prefill-backend tilelang --nsa-decode-backend tilelang \
  --kv-cache-dtype fp8_e4m3 --mem-fraction-static "$GMU" --context-length "$CTX" \
  --chunked-prefill-size "$CHUNK" --cuda-graph-max-bs 64 --max-running-requests "$MAX_RUNNING" \
  --watchdog-timeout 3600 \
  "${MTP_ARGS[@]}" "${ROLE_ARGS[@]}" > "$LOG" 2>&1
