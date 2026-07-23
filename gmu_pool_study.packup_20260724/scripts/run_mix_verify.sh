#!/bin/bash
# Step 3 verify: single-node mix DSv4 @gmu=0.90, send real 8k/1k requests, prove
# runtime KV usage (#token) stays within the startup-fixed pool (full_token) and
# #retracted-req == 0. Same KV-pool formula as the PD legs (pool sizing is
# role-independent, proven in Step 2). Run INSIDE dsv4_pd_sgl on chi2879.
set -u
MODEL=/mnt/vast/d_huggingface/models/DeepSeek-V4-Pro
MY_IP=10.2.122.10
PORT=30000
OUT=/mnt/vast/c_huggingface/kvcache_gmu_study_20260724_084928
LOG=$OUT/mix_verify_gmu0.90.log
BENCHLOG=$OUT/mix_verify_bench.log
mkdir -p "$OUT"

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

pkill -9 -f sglang.launch_server 2>/dev/null; sleep 6

# mix (non-disagg) mode, same recipe, gmu 0.90, DP-attn on
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m sglang.launch_server \
  --model-path "$MODEL" --tp-size 8 --trust-remote-code \
  --host "$MY_IP" --port "$PORT" --attention-backend dsv4 \
  --cuda-graph-max-bs 512 --disable-radix-cache --page-size 256 \
  --swa-full-tokens-ratio 0.15 --disable-shared-experts-fusion \
  --mem-fraction-static 0.90 --context-length 9472 \
  --max-running-requests 512 \
  --dp 8 --enable-dp-attention --ep-size 8 \
  --enable-prefill-delayer --prefill-delayer-max-delay-ms 5000 \
  --chunked-prefill-size 163840 --max-prefill-tokens 163840 \
  --watchdog-timeout 3600 > "$LOG" 2>&1 &
SPID=$!

# wait for server ready
for i in $(seq 1 900); do
  if grep -q "The server is fired up and ready to roll" "$LOG" 2>/dev/null; then echo "READY at ${i}s"; break; fi
  if ! kill -0 $SPID 2>/dev/null; then echo "SERVER DIED"; tail -20 "$LOG"; exit 1; fi
  sleep 1
done

# record the startup pool capacity
grep -hE "DSV4 pool sizes|DSV4 memory calculation" "$LOG" | grep "DP0 " | head -2

# send a real 8k/1k load: 128 prompts @ conc 64 (enough to fill running slots)
python3 -m sglang.bench_serving --backend sglang-oai --base-url http://$MY_IP:$PORT \
  --model "$MODEL" --tokenizer "$MODEL" \
  --dataset-name random --random-input-len 8192 --random-output-len 1024 --random-range-ratio 1.0 \
  --max-concurrency 64 --num-prompts 128 --warmup-requests 8 \
  --request-rate inf > "$BENCHLOG" 2>&1
echo "BENCH DONE"

# extract runtime peak token usage + retracted from scheduler stats
echo "==== runtime scheduler peaks ===="
grep -oE "#token: [0-9]+" "$LOG" | grep -oE "[0-9]+" | sort -n | tail -1 | sed 's/^/peak_#token(per-rank)=/'
grep -oE "#retracted-req: [0-9]+" "$LOG" | grep -oE "[0-9]+" | sort -n | tail -1 | sed 's/^/peak_#retracted=/'
grep -oE "token usage: [0-9.]+" "$LOG" | grep -oE "[0-9.]+" | sort -n | tail -1 | sed 's/^/peak_token_usage_frac=/'
grep -oE "#running-req: [0-9]+" "$LOG" | grep -oE "[0-9]+" | sort -n | tail -1 | sed 's/^/peak_#running=/'
tail -3 "$BENCHLOG"
pkill -9 -f sglang.launch_server 2>/dev/null
echo "==== DONE mix verify ===="
