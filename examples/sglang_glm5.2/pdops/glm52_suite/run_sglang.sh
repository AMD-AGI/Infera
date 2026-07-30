#!/usr/bin/env bash
# GLM-5.2-FP8 on MI325X (gfx942) — background launch.
#
# Usage (inside container):
#   ./run_sglang.sh
#   tail -f tmp/glm52_server.log
#   kill "$(cat tmp/glm52_server.pid)"
#
# Defaults to TP=8 over all 8 GPUs: the FP8 checkpoint is ~704 GiB, so TP=8
# leaves ~130 GiB/GPU for KV cache on a 256 GiB card. TP=4 would leave ~40 GiB.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${SCRIPT_DIR}/tmp"

MODEL_PATH="${MODEL_PATH:-/wekafs/models/GLM-5.2-FP8}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-8}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
ENABLE_MIXED_CHUNK="${ENABLE_MIXED_CHUNK:-0}"
# Off by default: with this on, next_token_logits went fully NaN on ~8% of long-context
# cold prefills, which sampling turned into token id 0. See KNOWN_ISSUES.md. Upstream
# SGLang has itself commented out the auto-enable for this model family
# (server_args.py: "TODO (Hubert): Put this back later"), so the MI355X cookbook's
# --enable-aiter-allreduce-fusion was re-enabling something upstream had backed out.
ENABLE_AITER_ALLREDUCE_FUSION="${ENABLE_AITER_ALLREDUCE_FUSION:-0}"
LOG="${LOG:-${SCRIPT_DIR}/tmp/glm52_server.log}"
PID_FILE="${PID_FILE:-${SCRIPT_DIR}/tmp/glm52_server.pid}"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "ERROR: no model at MODEL_PATH=${MODEL_PATH} (config.json missing)" >&2
  exit 1
fi

MIXED_CHUNK_ARGS=()
if [[ "${ENABLE_MIXED_CHUNK}" == "1" ]]; then
  MIXED_CHUNK_ARGS+=(--enable-mixed-chunk)
fi

AITER_AR_ARGS=()
if [[ "${ENABLE_AITER_ALLREDUCE_FUSION}" == "1" ]]; then
  AITER_AR_ARGS+=(--enable-aiter-allreduce-fusion)
fi

if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  if ss -tln 2>/dev/null | grep -q ":${PORT} "; then
    echo "Already running: pid=$(cat "${PID_FILE}") port=${PORT} log=${LOG}"
    exit 0
  fi
  rm -f "${PID_FILE}"
fi
if ss -tln 2>/dev/null | grep -q ":${PORT} "; then
  echo "ERROR: port ${PORT} in use but pid file missing/stale; stop the orphan server first" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HIP_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
export SGLANG_DSA_TRITON_PREFILL=1
# Inert in this config: the log confirms AiterCustomAllreduce is used, so QuickAllReduce
# never runs. Kept only to stay close to the MI355X cookbook; don't read it as active.
export ROCM_QUICK_REDUCE_QUANTIZATION="${ROCM_QUICK_REDUCE_QUANTIZATION:-INT4}"
export SAFETENSORS_FAST_GPU=1

nohup python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tp "${TP_SIZE}" \
  --trust-remote-code \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --dsa-prefill-backend tilelang \
  --dsa-decode-backend tilelang \
  --kv-cache-dtype fp8_e4m3 \
  --chunked-prefill-size 131072 \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  "${MIXED_CHUNK_ARGS[@]}" \
  --watchdog-timeout 1200 \
  "${AITER_AR_ARGS[@]}" \
  --weight-loader-prefetch-checkpoints \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 32}' \
  > "${LOG}" 2>&1 &

echo $! > "${PID_FILE}"
echo "Started GLM-5.2-FP8: pid=$(cat "${PID_FILE}") port=${PORT}"
echo "  model:       ${MODEL_PATH}"
echo "  tp:          ${TP_SIZE} (gpus=${CUDA_VISIBLE_DEVICES})"
echo "  mixed-chunk: ${ENABLE_MIXED_CHUNK}"
echo "  aiter-ar-fusion: ${ENABLE_AITER_ALLREDUCE_FUSION}"
echo "  log:         ${LOG}"
echo "  ready check: curl -s http://127.0.0.1:${PORT}/health"
