#!/bin/bash
# GLM-5.2-MXFP4 cross-node PD leg over mooncake, WITH DP-attention (DPA), on
# infera/engine-sglang:pd-unified (PR#19). Extends pd_leg.sh with the DSv4 R4 high-conc
# DP recipe (--dp8 --enable-dp-attention --ep-size 8 + GATHERV + prefill-delayer), adapted
# to GLM-5.2 DSA. Both legs MUST be symmetric-DPA so the KV layout (dp_size x tp_size shard)
# matches across the mooncake transfer. Runs INSIDE the pd_uni container. No MTP.
set -u
ROLE="${ROLE:?ROLE=prefill|decode}"
MY_IP="${MY_IP:?MY_IP=data-plane rail IP}"
MODEL="${MODEL:-/mnt/vast/xiaobo/models/GLM-5.2-MXFP4}"
SERVED="${SERVED:-glm5.2-mxfp4}"
PORT="${PORT:-30000}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
TP="${TP:-8}"
DPA="${DPA:-1}"                       # 1 = DP-attention on (this script's whole point)
CTX="${CTX:-32768}"                   # lowered from 400000: 1k/1k stress needs <=2k/req; frees KV headroom
MAX_RUNNING="${MAX_RUNNING:-2048}"    # server-side scheduler cap; client --max-concurrency drives actual load
CUDA_GRAPH_BS="${CUDA_GRAPH_BS:-128}" # capture up to 128; larger batches replay eager
DELAYER="${DELAYER:-1}"               # prefill-delayer (R4 high-conc lever); only meaningful on prefill leg
PREFILL_DELAY_MS="${PREFILL_DELAY_MS:-5000}"
DMABUF="${DMABUF:-0}"                 # 0 = bare ibv_reg_mr+peermem (proven cross-node); 1 = dmabuf
MAX_TOTAL="${MAX_TOTAL:-}"            # optional explicit KV-pool token cap; empty = sglang auto
if [ "$ROLE" = "prefill" ]; then GMU="${GMU:-0.88}"; else GMU="${GMU:-0.85}"; fi
# DP-attn prefill high-conc HSA OOM guard (memory note): DP prefill wants LOWER memfrac than colocated.
LOG="${LOG:-/mnt/vast/c_huggingface/glm52_p2b/pd_${ROLE}_${PORT}_dpa.log}"
mkdir -p "$(dirname "$LOG")"

# DP-attn chunked-prefill = ISL * TP (R4: 8192*8=65536). no-DP falls back to 8192.
ISL="${ISL:-8192}"
if [ "$DPA" = "1" ]; then CHUNK="${CHUNK:-$((ISL * TP))}"; else CHUNK="${CHUNK:-8192}"; fi

# all active ionic NICs (mooncake pairs by GID subnet across nodes -> needs them ALL)
IB_DEVICES=$(for d in /sys/class/infiniband/*; do
    [ -d "$d" ] || continue; n=$(basename "$d")
    s=$(cat "$d/ports/1/state" 2>/dev/null || echo "")
    drv=$(basename "$(readlink -f "$d/device/driver" 2>/dev/null || echo x)")
    [[ "$s" == *ACTIVE* && "$drv" == ionic ]] && echo "$n"
  done | sort -V | paste -sd,)
[ -z "$IB_DEVICES" ] && { echo "no active ionic NICs" >&2; exit 1; }

# ---- runtime dmabuf switch (0 = bare ibv_reg_mr+peermem = the proven cross-node path) ----
if [ "$DMABUF" = "1" ]; then export MOONCAKE_DISABLE_HIP_DMABUF=0; else export MOONCAKE_DISABLE_HIP_DMABUF=1; fi

# ---- mooncake RDMA env (from the pd-unified packup; NOT MC_FORCE_TCP) ----
export MC_GID_INDEX=1 MC_DISABLE_HIP_TRANSPORT=1
unset MC_ENABLE_HIP_TRANSPORT
export RDMAV_FORK_SAFE=1
export NCCL_IB_DISABLE=1 NCCL_IGNORE_CPU_AFFINITY=1 HSA_NO_SCRATCH_RECLAIM=1
export SGLANG_HOST_IP="$MY_IP" HOST_IP="$MY_IP"
NIC=$(ip -o -4 addr show | awk -v ip="$MY_IP" '$4 ~ ("^" ip "/") {print $2; exit}')
export SGLANG_LOCAL_IP_NIC="$NIC" GLOO_SOCKET_IFNAME="$NIC"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1800 SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1800

# ---- GLM-5.2 DSA-ROCm recipe (my proven Phase 1/3 envs; NOT the DSv4 R4 attn block) ----
export SGLANG_USE_AITER=1 SGLANG_ROCM_FUSED_DECODE_MLA=0
export SGLANG_OPT_USE_TILELANG_INDEXER=1 SGLANG_OPT_USE_TOPK_V2=0 SGLANG_OPT_USE_JIT_NORM=0
export SAFETENSORS_FAST_GPU=1 HIP_FORCE_DEV_KERNARG=1

# ---- DP-attention extra env (R4 DP variant) ----
[ "$DPA" = "1" ] && export SGLANG_DP_USE_GATHERV=1

ROLE_ARGS=(--disaggregation-mode "$ROLE" --disaggregation-transfer-backend mooncake \
           --disaggregation-ib-device "$IB_DEVICES")
[ "$ROLE" = "prefill" ] && ROLE_ARGS+=(--disaggregation-bootstrap-port "$BOOTSTRAP_PORT")

# ---- DP-attention args (symmetric on both legs so KV shard layout matches) ----
DP_ARGS=()
if [ "$DPA" = "1" ]; then
  DP_ARGS+=(--dp-size "$TP" --enable-dp-attention --ep-size "$TP")
  # prefill-delayer batches incoming prefills (R4 high-conc lever); decode leg generates,
  # so delayer only on prefill.
  if [ "$ROLE" = "prefill" ] && [ "$DELAYER" = "1" ]; then
    DP_ARGS+=(--enable-prefill-delayer --prefill-delayer-max-delay-ms "$PREFILL_DELAY_MS")
  fi
fi

EXTRA_ARGS=()
[ -n "$MAX_TOTAL" ] && EXTRA_ARGS+=(--max-total-tokens "$MAX_TOTAL")

echo "[glm52-mc-dpa] role=$ROLE ip=$MY_IP nic=$NIC dpa=$DPA gmu=$GMU chunk=$CHUNK ctx=$CTX maxrun=$MAX_RUNNING dmabuf=$DMABUF ib=$IB_DEVICES -> $LOG"
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m sglang.launch_server \
  --model-path "$MODEL" --served-model-name "$SERVED" --tp-size "$TP" --trust-remote-code \
  --host "$MY_IP" --port "$PORT" \
  --nsa-prefill-backend tilelang --nsa-decode-backend tilelang \
  --kv-cache-dtype fp8_e4m3 --mem-fraction-static "$GMU" --context-length "$CTX" \
  --chunked-prefill-size "$CHUNK" --cuda-graph-max-bs "$CUDA_GRAPH_BS" \
  --max-running-requests "$MAX_RUNNING" --watchdog-timeout 3600 \
  "${DP_ARGS[@]}" "${ROLE_ARGS[@]}" "${EXTRA_ARGS[@]}" > "$LOG" 2>&1
