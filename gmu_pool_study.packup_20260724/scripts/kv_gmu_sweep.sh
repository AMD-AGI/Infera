#!/bin/bash
# ============================================================================
# KV cache pool vs gmu (--mem-fraction-static) sweep — DSv4-Pro sglang.
# Launches a single leg (decode OR prefill), reads the KV-pool allocation
# figures printed at startup ("Memory pool end. avail mem=", "max_total_num_tokens="),
# snapshots real VRAM, then kills and moves to the next gmu. Pure launch-read;
# does NOT serve requests. Run INSIDE the dsv4_pd_sgl container on chi2879.
#
# Usage:
#   ROLE=decode GMUS="0.80 0.85 0.88 0.90 0.92" bash kv_gmu_sweep.sh
#   ROLE=prefill GMUS="0.85 0.90" bash kv_gmu_sweep.sh
# ============================================================================
set -u
ROLE="${ROLE:-decode}"
GMUS="${GMUS:-0.80 0.85 0.88 0.90 0.92}"
MODEL="${MODEL:-/mnt/vast/d_huggingface/models/DeepSeek-V4-Pro}"
MY_IP="${MY_IP:-10.2.122.10}"           # chi2879 data-plane
PORT="${PORT:-30000}"
TP="${TP:-8}"
CTX="${CTX:-9472}"
CHUNK="${CHUNK:-163840}"
OUT="${OUT:-/mnt/vast/c_huggingface/kvcache_gmu_study_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
SUMMARY="$OUT/${ROLE}_summary.csv"
# DSv4 (sglang 0.5.15) KV pool is multi-pool. The authoritative KV byte budget is
# the per-rank "DSV4 memory calculation: ... available_bytes=X GB ... full_token=N"
# line + "DSV4 pool sizes: full/swa/c4/c128/...". avail_mem is post-pool headroom.
echo "gmu,kv_available_bytes_gb,bytes_per_full_token,full_token,swa,c4,c128,c4_state,c128_state_fixed_gb,avail_mem_gb,vram_used_gb_min,vram_used_gb_max,reached" > "$SUMMARY"

# R4 perf env (verbatim from reference run pd_server.sh)
export SGLANG_USE_AITER=1 AITER_BF16_FP8_MOE_BOUND=0
export SGLANG_OPT_FP8_WO_A_GEMM=0 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 SGLANG_OPT_USE_AITER_INDEXER=1
export SGLANG_OPT_USE_TOPK_V2=0 SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1 SGLANG_OPT_USE_FUSED_PAGED_COMPRESS=1
export SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=false SGLANG_ROCM_USE_MULTI_STREAM=false
export SGLANG_OPT_USE_FUSED_COMPRESS=true SGLANG_OPT_USE_FUSED_COMPRESS_TRITON=true
export SGLANG_EAGER_INPUT_NO_COPY=true SGLANG_USE_ROCM700A=0
export SGLANG_OPT_USE_JIT_INDEXER_METADATA=false
export SGLANG_OPT_USE_TILELANG_INDEXER=false SGLANG_OPT_USE_TILELANG_MHC_PRE=false SGLANG_OPT_USE_TILELANG_MHC_POST=false
export SGLANG_DP_USE_GATHERV=1
export HSA_NO_SCRATCH_RECLAIM=1 NCCL_IB_DISABLE=1 NCCL_IGNORE_CPU_AFFINITY=1

vram_used_range() {  # prints "min max" GB across 8 cards
  rocm-smi --showmeminfo vram --csv 2>/dev/null | awk -F, 'NR>1 && $3 ~ /[0-9]/ {v=$3/1073741824; if(mn==""||v<mn)mn=v; if(v>mx)mx=v} END{printf "%.2f %.2f", mn, mx}'
}

for GMU in $GMUS; do
  LOG="$OUT/${ROLE}_gmu${GMU}.log"
  echo "======== ROLE=$ROLE GMU=$GMU -> $LOG ========"
  # decode leg forced chunk-cache; SWA model needs --no-enable-kv-events on decode
  ROLE_ARGS=(--disaggregation-mode "$ROLE" --disaggregation-transfer-backend mooncake)
  if [ "$ROLE" = "prefill" ]; then ROLE_ARGS+=(--disaggregation-bootstrap-port 8998); fi
  # NOTE: native sglang decode leg does NOT auto-append decode-radix-cache
  # (that's an infera-wrapper behavior). --kv-events-config defaults off, so no
  # SWA conflict here. The infera-only --no-enable-kv-events flag is omitted.

  HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m sglang.launch_server \
    --model-path "$MODEL" --tp-size "$TP" --trust-remote-code \
    --host "$MY_IP" --port "$PORT" --attention-backend dsv4 \
    --cuda-graph-max-bs 512 --disable-radix-cache --page-size 256 \
    --swa-full-tokens-ratio 0.15 --disable-shared-experts-fusion \
    --mem-fraction-static "$GMU" --context-length "$CTX" \
    --max-running-requests 512 \
    --dp "$TP" --enable-dp-attention --ep-size "$TP" \
    --enable-prefill-delayer --prefill-delayer-max-delay-ms 5000 \
    --chunked-prefill-size "$CHUNK" --max-prefill-tokens "$CHUNK" \
    --watchdog-timeout 3600 \
    "${ROLE_ARGS[@]}" > "$LOG" 2>&1 &
  PID=$!

  # wait until "Memory pool end" appears (KV pool sized) OR failure OR timeout
  REACHED=no
  for i in $(seq 1 900); do   # up to 900s cold NFS
    if grep -q "Memory pool end" "$LOG" 2>/dev/null; then REACHED=yes; sleep 3; break; fi
    if grep -qiE "Traceback|Error|Aborted|CUDA out of memory|HSA_STATUS_ERROR" "$LOG" 2>/dev/null; then REACHED=err; break; fi
    if ! kill -0 $PID 2>/dev/null; then REACHED=died; break; fi
    sleep 1
  done

  # snapshot VRAM while process still alive (or right at reach)
  VRAM=$(vram_used_range)

  # parse figures — DP0 rank as canonical (all ranks identical within rounding)
  MEMCALC=$(grep "DSV4 memory calculation" "$LOG" | head -1)
  POOLSZ=$(grep "DSV4 pool sizes" "$LOG" | head -1)
  KVBYTES=$(echo "$MEMCALC" | grep -oE "available_bytes=[0-9.]+" | grep -oE "[0-9.]+")
  BPFT=$(echo "$MEMCALC"    | grep -oE "bytes_per_full_token=[0-9.]+" | grep -oE "[0-9.]+")
  FULLTOK=$(echo "$MEMCALC" | grep -oE "full_token=[0-9]+" | grep -oE "[0-9]+")
  C128FIX=$(echo "$MEMCALC" | grep -oE "c128_state_fixed=[0-9.]+" | grep -oE "[0-9.]+")
  SWA=$(echo "$POOLSZ" | grep -oE "swa=[0-9]+" | grep -oE "[0-9]+")
  C4=$(echo "$POOLSZ"  | grep -oE "c4=[0-9]+" | grep -oE "[0-9]+")
  C128=$(echo "$POOLSZ" | grep -oE "c128=[0-9]+" | grep -oE "[0-9]+")
  C4ST=$(echo "$POOLSZ" | grep -oE "c4_state=[0-9]+" | grep -oE "[0-9]+")
  AVAILMEM=$(grep "Memory pool end" "$LOG" | grep -oE "avail mem=[0-9.]+" | grep -oE "[0-9.]+" | sort -n | head -1)
  echo "${GMU},${KVBYTES:-NA},${BPFT:-NA},${FULLTOK:-NA},${SWA:-NA},${C4:-NA},${C128:-NA},${C4ST:-NA},${C128FIX:-NA},${AVAILMEM:-NA},${VRAM/ /,},${REACHED}" >> "$SUMMARY"
  echo "  -> reached=$REACHED kv_avail=${KVBYTES}GB bpft=${BPFT} full_tok=${FULLTOK} swa=${SWA} c4=${C4} c128=${C128} vram=[$VRAM]GB"

  # teardown + wait VRAM drop
  kill -9 $PID 2>/dev/null
  pkill -9 -f "sglang.launch_server" 2>/dev/null
  for i in $(seq 1 60); do
    read MN MX < <(vram_used_range)
    # MX below 5GB means pool freed
    if awk "BEGIN{exit !($MX < 5)}"; then break; fi
    sleep 2
  done
  sleep 3
done
echo "==== DONE. summary: $SUMMARY ===="
cat "$SUMMARY"
