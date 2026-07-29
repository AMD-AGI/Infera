#!/bin/bash
# GLM-5.2-MXFP4 cross-node PD leg over mooncake on SPUR (mlx5 + dmabuf), with optional
# DP-attention (DPA) and optional MTP (decode leg). Runs INSIDE the pd container.
#
# FUSION of two proven kits:
#   * recipe (GLM-5.2 DSA + DPA + MTP) from glm5.2.mxfp4.packup_20260727/{06,07}  (vultr, ionic)
#   * transport (mlx5 + dmabuf, no peermem) from crsuse dsv4 dmabuf kit          (spur)
# Spur has NO peermem -> the ONLY GPUDirect path is dma-buf via mlx5 (has ODP -> no pin/double).
set -u
ROLE="${ROLE:?ROLE=prefill|decode}"
MY_IP="${MY_IP:?MY_IP=this node mlx5(ens3) IP}"
P_IP="${P_IP:?P_IP=prefill mlx5 IP (bootstrap host)}"
MODEL="${MODEL:-/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4}"
SERVED="${SERVED:-glm5.2-mxfp4}"
PORT="${PORT:-30000}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
TP="${TP:-8}"
DPA="${DPA:-1}"                       # 1 = DP-attention on (07 recipe)
MTP="${MTP:-0}"                       # 1 = EAGLE spec-dec (decode leg only; 06 recipe)
CTX="${CTX:-32768}"                   # 07: lowered from 400000 for 1k/1k high-conc
MAX_RUNNING="${MAX_RUNNING:-2048}"
CUDA_GRAPH_BS="${CUDA_GRAPH_BS:-128}"
DELAYER="${DELAYER:-1}"
PREFILL_DELAY_MS="${PREFILL_DELAY_MS:-5000}"
NIC="${NIC:-ens3}"                    # the single mlx5 netdev with an IP on spur
IBDEV="${IBDEV:-mlx5_0}"             # spur KV NIC = mlx5 (has ODP)
GID="${GID:-3}"                       # mlx5 RoCEv2 routable GID index on spur
DMABUF="${DMABUF:-1}"                 # spur: dmabuf ON (no peermem). ionic path would be 0.
if [ "$ROLE" = "prefill" ]; then GMU="${GMU:-0.88}"; else GMU="${GMU:-0.85}"; fi
LOG="${LOG:-/tmp/pd_${ROLE}_${PORT}.log}"
mkdir -p "$(dirname "$LOG")"

ISL="${ISL:-8192}"
if [ "$DPA" = "1" ]; then CHUNK="${CHUNK:-$((ISL * TP))}"; else CHUNK="${CHUNK:-8192}"; fi

# ---- dmabuf switch: spur has no peermem, dmabuf via mlx5(ODP) is the GPUDirect path ----
if [ "$DMABUF" = "1" ]; then export MOONCAKE_DISABLE_HIP_DMABUF=0; else export MOONCAKE_DISABLE_HIP_DMABUF=1; fi

# ---- mooncake RDMA env, FORCED onto the single mlx5 NIC (not the 8 ionic rails) ----
export MC_GID_INDEX="$GID" MC_DISABLE_HIP_TRANSPORT=1
export MC_MS_AUTO_DISC=0 MC_MS_FILTERS="$IBDEV"
unset MC_ENABLE_HIP_TRANSPORT
export RDMAV_FORK_SAFE=1
export NCCL_IB_DISABLE=1 NCCL_IGNORE_CPU_AFFINITY=1 HSA_NO_SCRATCH_RECLAIM=1
export SGLANG_HOST_IP="$MY_IP" HOST_IP="$MY_IP"
export SGLANG_LOCAL_IP_NIC="$NIC" GLOO_SOCKET_IFNAME="$NIC"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1800 SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1800

# ---- GLM-5.2 DSA-ROCm recipe (proven Phase1/3; NOT the DSv4 attn block) ----
export SGLANG_USE_AITER=1 SGLANG_ROCM_FUSED_DECODE_MLA=0
export SGLANG_OPT_USE_TILELANG_INDEXER=1 SGLANG_OPT_USE_TOPK_V2=0 SGLANG_OPT_USE_JIT_NORM=0
export SAFETENSORS_FAST_GPU=1 HIP_FORCE_DEV_KERNARG=1

# ---- DP-attention extra env ----
[ "$DPA" = "1" ] && export SGLANG_DP_USE_GATHERV=1

ROLE_ARGS=(--disaggregation-mode "$ROLE" --disaggregation-transfer-backend mooncake \
           --disaggregation-ib-device "$IBDEV")
[ "$ROLE" = "prefill" ] && ROLE_ARGS+=(--disaggregation-bootstrap-port "$BOOTSTRAP_PORT")

# ---- DP-attention args (symmetric on both legs so KV shard layout matches) ----
DP_ARGS=()
if [ "$DPA" = "1" ]; then
  DP_ARGS+=(--dp-size "$TP" --enable-dp-attention --ep-size "$TP")
  if [ "$ROLE" = "prefill" ] && [ "$DELAYER" = "1" ]; then
    DP_ARGS+=(--enable-prefill-delayer --prefill-delayer-max-delay-ms "$PREFILL_DELAY_MS")
  fi
fi

# ---- MTP (decode leg only): EAGLE steps=3 for PD KV-pool stability (06 recipe) ----
MTP_ARGS=()
if [ "$MTP" = "1" ] && [ "$ROLE" = "decode" ]; then
  MTP_ARGS=(--speculative-algorithm EAGLE --speculative-num-steps "${SPEC_STEPS:-3}" \
            --speculative-eagle-topk 1 --speculative-num-draft-tokens "${SPEC_DRAFT:-4}" \
            --num-reserved-decode-tokens "${RESERVED_TOK:-256}")
fi

# ---- custom all-reduce: independent of MTP ----
# The aiter custom all-reduce kernel deadlocks on gfx942/gfx950 during EAGLE
# verify at high concurrency (sglang GLM-5.1 cookbook; #28815/#31071/PR #31478).
# This used to live INSIDE the MTP block, which meant MTP=0 silently re-enabled
# a known-broken kernel -- turning "MTP on vs off" into a two-variable
# comparison. Keep it out here so the two arms differ only in MTP.
CAR_ARGS=()
[ "${CUSTOM_AR:-0}" = "1" ] || CAR_ARGS+=(--disable-custom-all-reduce)

echo "[glm52-spur] role=$ROLE ip=$MY_IP nic=$NIC ibdev=$IBDEV gid=$GID dpa=$DPA mtp=$MTP dmabuf=$DMABUF gmu=$GMU chunk=$CHUNK ctx=$CTX -> $LOG"
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m sglang.launch_server \
  --model-path "$MODEL" --served-model-name "$SERVED" --tp-size "$TP" --trust-remote-code \
  --host "$MY_IP" --port "$PORT" \
  --nsa-prefill-backend tilelang --nsa-decode-backend tilelang \
  --kv-cache-dtype fp8_e4m3 --mem-fraction-static "$GMU" --context-length "$CTX" \
  --chunked-prefill-size "$CHUNK" --cuda-graph-max-bs "$CUDA_GRAPH_BS" \
  --max-running-requests "$MAX_RUNNING" --watchdog-timeout 3600 \
  "${DP_ARGS[@]}" "${MTP_ARGS[@]}" "${CAR_ARGS[@]}" "${ROLE_ARGS[@]}" ${EXTRA_ARGS:-} > "$LOG" 2>&1
