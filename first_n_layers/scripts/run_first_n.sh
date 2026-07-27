#!/usr/bin/env bash
# One-shot: launch an N-layer DSv4 server (eager, dsv4 backend, R4 env) in a
# throwaway container, wait for ready, send 2 probe requests, print output.
#
# Prereq: make_first_n_config.sh already produced the N-layer model dir.
# Run ON the node (chi2879). GPUs must be free (rocm-smi --showpids).
#
# Usage: bash run_first_n.sh [N] [PORT] [CONTAINER]
set -u
N="${1:-4}"
PORT="${2:-30000}"
CTR="${3:-dsv4_${N}l}"
IMAGE="infera/engine-sglang:pd-mcgate"
MODEL="/mnt/vast/d_huggingface/models/DeepSeek-V4-Pro-${N}L"
LOG="/work/dsv4_${N}l_server.log"

echo "=== (re)create container $CTR ==="
docker rm -f "$CTR" 2>/dev/null
docker run -d --name "$CTR" --network host --ipc host \
  --device /dev/kfd --device /dev/dri --group-add video \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -v /mnt/vast:/mnt/vast -v /tmp:/work "$IMAGE" sleep infinity

echo "=== launch N=$N server on :$PORT ==="
docker exec -d "$CTR" bash -lc "
set -x
# ---- R4 env (matches sglang_single_r4 manifest) ----
export SGLANG_USE_AITER=1 AITER_BF16_FP8_MOE_BOUND=0
export SGLANG_OPT_FP8_WO_A_GEMM=0 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 SGLANG_OPT_USE_AITER_INDEXER=1
export SGLANG_OPT_USE_TOPK_V2=0 SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1 SGLANG_OPT_USE_FUSED_PAGED_COMPRESS=1
export SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=false SGLANG_ROCM_USE_MULTI_STREAM=false
export SGLANG_OPT_USE_FUSED_COMPRESS=true SGLANG_OPT_USE_FUSED_COMPRESS_TRITON=true
export SGLANG_EAGER_INPUT_NO_COPY=true SGLANG_USE_ROCM700A=0
export SGLANG_OPT_USE_JIT_INDEXER_METADATA=false
export SGLANG_OPT_USE_TILELANG_INDEXER=false SGLANG_OPT_USE_TILELANG_MHC_PRE=false SGLANG_OPT_USE_TILELANG_MHC_POST=false
exec python3 -m sglang.launch_server \
  --model-path $MODEL --tp-size 8 --trust-remote-code \
  --host 127.0.0.1 --port $PORT --attention-backend dsv4 \
  --page-size 256 --swa-full-tokens-ratio 0.15 \
  --disable-shared-experts-fusion --disable-radix-cache \
  --mem-fraction-static 0.85 --cuda-graph-max-bs 8 \
  --disable-cuda-graph --chunked-prefill-size 8192 > $LOG 2>&1
"

echo "=== wait for ready (up to ~5min) ==="
for i in $(seq 1 30); do
  sleep 10
  if docker exec "$CTR" grep -qE "Application startup complete|Uvicorn running" "$LOG" 2>/dev/null; then
    echo "READY after ${i}0s"; break; fi
  if docker exec "$CTR" grep -qiE "Traceback|out of memory|AssertionError" "$LOG" 2>/dev/null; then
    echo "ERROR after ${i}0s"; docker exec "$CTR" grep -A30 -iE "Traceback" "$LOG" | head -35; exit 1; fi
done

echo "=== weight-load evidence (N layers => small mem usage/GPU) ==="
docker exec "$CTR" grep -E "Load weight end|max_total_num_tokens" "$LOG" | head -2

echo "=== probe (temperature=0; output is gibberish by design for small N) ==="
for P in "The capital of France is" "def add(a, b):\n    return"; do
  docker exec "$CTR" curl -s "http://127.0.0.1:${PORT}/generate" \
    -H "Content-Type: application/json" \
    -d "{\"text\": \"$P\", \"sampling_params\": {\"max_new_tokens\": 12, \"temperature\": 0}}" \
    | python3 -c "import sys,json; r=json.load(sys.stdin); print('  prompt:', repr('$P')); print('  text  :', repr(r['text'])); print('  tokens:', r['meta_info']['completion_tokens'])"
done
echo "=== done. server still running in $CTR on :$PORT (kill: docker rm -f $CTR) ==="
