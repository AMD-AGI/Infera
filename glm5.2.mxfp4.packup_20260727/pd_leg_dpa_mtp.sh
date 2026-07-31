#!/bin/bash
# GLM-5.2-MXFP4 cross-node PD leg over mooncake, DP-attention (DPA) + MTP (EAGLE) COMBINED.
# This is the never-before-run combination (exp07=DPA-only, exp06=MTP-only). Merge = exp07's
# pd_leg_dpa.sh skeleton (DPA args + mooncake RDMA env + GLM DSA recipe + capacity tuning)
# with exp06's MTP block (decode-leg EAGLE flags). Runs INSIDE the pd-unified container.
#
# Interaction risks being probed (why this needs its own MVP):
#  - DP-attention shards the running batch per-rank; EAGLE reserves --num-reserved-decode-tokens
#    per running req. reserved*dp_batch vs the DP KV-shard pool has no prior validation.
#  - MTP draft/verify path under DP gatherv (SGLANG_DP_USE_GATHERV=1) untested.
set -u
ROLE="${ROLE:?ROLE=prefill|decode}"
MY_IP="${MY_IP:?MY_IP=data-plane rail IP}"
MODEL="${MODEL:-/mnt/vast/xiaobo/models/GLM-5.2-MXFP4}"
SERVED="${SERVED:-glm5.2-mxfp4}"
PORT="${PORT:-30000}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
TP="${TP:-8}"
DPA="${DPA:-1}"                       # 1 = DP-attention on (both legs symmetric)
MTP="${MTP:-1}"                       # 1 = EAGLE spec-dec on decode leg
CTX="${CTX:-32768}"                   # MVP small ctx; 1k/1k needs <=2k/req
MAX_RUNNING="${MAX_RUNNING:-2048}"
CUDA_GRAPH_BS="${CUDA_GRAPH_BS:-128}"
DELAYER="${DELAYER:-1}"
PREFILL_DELAY_MS="${PREFILL_DELAY_MS:-5000}"
DMABUF="${DMABUF:-0}"
MAX_TOTAL="${MAX_TOTAL:-}"
# MTP knobs (exp06 PD-stable values): steps=3, draft=4, reserved=256
SPEC_STEPS="${SPEC_STEPS:-3}"
SPEC_DRAFT="${SPEC_DRAFT:-4}"
RESERVED_TOK="${RESERVED_TOK:-256}"
# memfrac: DP prefill wants LOWER (HSA-OOM guard); MTP decode wants LOWER (draft KV headroom).
# Combined decode (DPA+MTP) is the tightest -> default 0.80.
if [ "$ROLE" = "prefill" ]; then GMU="${GMU:-0.88}"; else GMU="${GMU:-0.80}"; fi
LOG="${LOG:-/mnt/vast/c_huggingface/glm52_dpa_mtp/pd_${ROLE}_${PORT}_dpamtp.log}"
mkdir -p "$(dirname "$LOG")"

ISL="${ISL:-8192}"
if [ "$DPA" = "1" ]; then CHUNK="${CHUNK:-$((ISL * TP))}"; else CHUNK="${CHUNK:-8192}"; fi

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

# GLM-5.2 DSA-ROCm recipe
export SGLANG_USE_AITER=1 SGLANG_ROCM_FUSED_DECODE_MLA=0
export SGLANG_OPT_USE_TILELANG_INDEXER=1 SGLANG_OPT_USE_TOPK_V2=0 SGLANG_OPT_USE_JIT_NORM=0
export SAFETENSORS_FAST_GPU=1 HIP_FORCE_DEV_KERNARG=1

# DP-attention extra env
[ "$DPA" = "1" ] && export SGLANG_DP_USE_GATHERV=1

ROLE_ARGS=(--disaggregation-mode "$ROLE" --disaggregation-transfer-backend mooncake \
           --disaggregation-ib-device "$IB_DEVICES")
[ "$ROLE" = "prefill" ] && ROLE_ARGS+=(--disaggregation-bootstrap-port "$BOOTSTRAP_PORT")

# DP-attention args (symmetric on both legs)
DP_ARGS=()
if [ "$DPA" = "1" ]; then
  DP_ARGS+=(--dp-size "$TP" --enable-dp-attention --ep-size "$TP")
  if [ "$ROLE" = "prefill" ] && [ "$DELAYER" = "1" ]; then
    DP_ARGS+=(--enable-prefill-delayer --prefill-delayer-max-delay-ms "$PREFILL_DELAY_MS")
  fi
fi

# MTP args (decode leg only; prefill doesn't gen tokens)
MTP_ARGS=()
if [ "$MTP" = "1" ] && [ "$ROLE" = "decode" ]; then
  MTP_ARGS=(--speculative-algorithm EAGLE --speculative-num-steps "$SPEC_STEPS" \
            --speculative-eagle-topk 1 --speculative-num-draft-tokens "$SPEC_DRAFT" \
            --num-reserved-decode-tokens "$RESERVED_TOK")
fi

EXTRA_ARGS=()
[ -n "$MAX_TOTAL" ] && EXTRA_ARGS+=(--max-total-tokens "$MAX_TOTAL")

echo "[glm52-dpamtp] role=$ROLE ip=$MY_IP nic=$NIC dpa=$DPA mtp=$MTP gmu=$GMU chunk=$CHUNK ctx=$CTX maxrun=$MAX_RUNNING ib=$IB_DEVICES -> $LOG"
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m sglang.launch_server \
  --model-path "$MODEL" --served-model-name "$SERVED" --tp-size "$TP" --trust-remote-code \
  --host "$MY_IP" --port "$PORT" \
  --nsa-prefill-backend tilelang --nsa-decode-backend tilelang \
  --kv-cache-dtype fp8_e4m3 --mem-fraction-static "$GMU" --context-length "$CTX" \
  --chunked-prefill-size "$CHUNK" --cuda-graph-max-bs "$CUDA_GRAPH_BS" \
  --max-running-requests "$MAX_RUNNING" --watchdog-timeout 3600 \
  "${DP_ARGS[@]}" "${MTP_ARGS[@]}" "${ROLE_ARGS[@]}" "${EXTRA_ARGS[@]}" > "$LOG" 2>&1
