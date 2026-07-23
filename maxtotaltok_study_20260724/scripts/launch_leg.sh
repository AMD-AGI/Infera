#!/bin/bash
# Launch one bare-sglang 0.5.13 PD leg (prefill OR decode) for DeepSeek-V4-Pro,
# KV over Mooncake RDMA on ionic. Run INSIDE the mtt_pd container.
#
# Env:
#   ROLE            prefill | decode   (required)
#   MY_IP           this node's data-plane IP (RDMA rail)   (required)
#   MAX_TOTAL_TOKENS  optional int -> --max-total-tokens (the knob under study)
#   GMU             mem-fraction-static (default: prefill 0.85 / decode 0.90)
#   IB_DEVICES      ionic list (default: auto-detect all 8 active)
#   BOOTSTRAP_PORT  prefill only (default 8998)
#   PORT            sglang port (default 30000)
#   LOG             log file path (default under $OUT)
set -u
ROLE="${ROLE:?ROLE=prefill|decode}"
MY_IP="${MY_IP:?MY_IP=data-plane IP}"
MODEL="${MODEL:-/mnt/vast/d_huggingface/models/DeepSeek-V4-Pro}"
PORT="${PORT:-30000}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
TP="${TP:-8}"
CTX="${CTX:-9472}"
CHUNK="${CHUNK:-163840}"
# graceful-degradation knobs (this round): gate admission so retract never fires.
MAX_RUNNING="${MAX_RUNNING:-512}"                 # global; sglang divides by dp internally
NUM_RESERVED_DECODE_TOKENS="${NUM_RESERVED_DECODE_TOKENS:-}"  # per-active-req decode reservation
OUT="${OUT:-/mnt/vast/c_huggingface/mtt_study}"
mkdir -p "$OUT"
if [ "$ROLE" = "prefill" ]; then GMU="${GMU:-0.85}"; else GMU="${GMU:-0.90}"; fi
LOG="${LOG:-$OUT/${ROLE}_$(hostname -s)_$(date +%H%M%S).log}"

# auto-detect all active ionic NICs (mooncake pairs by GID subnet)
if [ -z "${IB_DEVICES:-}" ]; then
  IB_DEVICES=$(for d in /sys/class/infiniband/*; do
      [ -d "$d" ] || continue; n=$(basename "$d")
      s=$(cat "$d/ports/1/state" 2>/dev/null || echo "")
      drv=$(basename "$(readlink -f "$d/device/driver" 2>/dev/null || echo x)")
      [[ "$s" == *ACTIVE* && "$drv" == ionic ]] && echo "$n"
    done | sort -V | paste -sd,)
fi
[ -z "$IB_DEVICES" ] && { echo "no active ionic NICs" >&2; exit 1; }

# R4 perf env (verbatim from kv_gmu_sweep.sh) + mooncake RDMA env
export SGLANG_USE_AITER=1 AITER_BF16_FP8_MOE_BOUND=0
export SGLANG_OPT_FP8_WO_A_GEMM=0 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 SGLANG_OPT_USE_AITER_INDEXER=1
export SGLANG_OPT_USE_TOPK_V2=0 SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1 SGLANG_OPT_USE_FUSED_PAGED_COMPRESS=1
export SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=false SGLANG_ROCM_USE_MULTI_STREAM=false
export SGLANG_OPT_USE_FUSED_COMPRESS=true SGLANG_OPT_USE_FUSED_COMPRESS_TRITON=true
export SGLANG_EAGER_INPUT_NO_COPY=true SGLANG_USE_ROCM700A=0
export SGLANG_OPT_USE_JIT_INDEXER_METADATA=false
export SGLANG_OPT_USE_TILELANG_INDEXER=false SGLANG_OPT_USE_TILELANG_MHC_PRE=false SGLANG_OPT_USE_TILELANG_MHC_POST=false
export SGLANG_DP_USE_GATHERV=1 HSA_NO_SCRATCH_RECLAIM=1
export NCCL_IB_DISABLE=1 NCCL_IGNORE_CPU_AFFINITY=1
# mooncake RDMA on ionic: GID index 1 (ULA RoCEv2), disable HIP transport (cross-node),
# pin host IP to the data-plane rail.
export MC_GID_INDEX=1 MC_DISABLE_HIP_TRANSPORT=1
export SGLANG_HOST_IP="$MY_IP" HOST_IP="$MY_IP"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1800 SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1800

ROLE_ARGS=(--disaggregation-mode "$ROLE" --disaggregation-transfer-backend mooncake \
           --disaggregation-ib-device "$IB_DEVICES")
if [ "$ROLE" = "prefill" ]; then ROLE_ARGS+=(--disaggregation-bootstrap-port "$BOOTSTRAP_PORT"); fi
MTT_ARGS=(); [ -n "${MAX_TOTAL_TOKENS:-}" ] && MTT_ARGS+=(--max-total-tokens "$MAX_TOTAL_TOKENS")
NRDT_ARGS=(); [ -n "${NUM_RESERVED_DECODE_TOKENS:-}" ] && NRDT_ARGS+=(--num-reserved-decode-tokens "$NUM_RESERVED_DECODE_TOKENS")

echo "[launch_leg] role=$ROLE ip=$MY_IP gmu=$GMU mtt=${MAX_TOTAL_TOKENS:-none} max_run=$MAX_RUNNING nrdt=${NUM_RESERVED_DECODE_TOKENS:-default} ctx=$CTX ib=$IB_DEVICES log=$LOG"
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m sglang.launch_server \
  --model-path "$MODEL" --tp-size "$TP" --trust-remote-code \
  --host "$MY_IP" --port "$PORT" --attention-backend dsv4 \
  --cuda-graph-max-bs 512 --disable-radix-cache --page-size 256 \
  --swa-full-tokens-ratio 0.15 --disable-shared-experts-fusion \
  --mem-fraction-static "$GMU" --context-length "$CTX" \
  --kv-cache-dtype fp8_e4m3 --max-running-requests "$MAX_RUNNING" \
  --dp "$TP" --enable-dp-attention --ep-size "$TP" \
  --enable-prefill-delayer --prefill-delayer-max-delay-ms 5000 \
  --chunked-prefill-size "$CHUNK" --max-prefill-tokens "$CHUNK" \
  --watchdog-timeout 3600 \
  "${MTT_ARGS[@]}" "${NRDT_ARGS[@]}" "${ROLE_ARGS[@]}" > "$LOG" 2>&1 &
echo $! > "$OUT/${ROLE}.pid"
echo "[launch_leg] pid=$(cat $OUT/${ROLE}.pid) -> tail -f $LOG"
