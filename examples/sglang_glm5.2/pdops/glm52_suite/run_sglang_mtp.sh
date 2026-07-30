#!/usr/bin/env bash
# GLM-5.2-FP8 + MTP/EAGLE on MI325X (gfx942).
#
# MTP_PROFILE=safe (default) is verified working on this node: accept len ~1.9-2.4
# of 4 draft tokens, and basic/determinism/needle/code-retrieval/deep-api all pass.
# The other two profiles are untested here; num-steps>3 may fail to build on AMD.
#
# Startup takes roughly 2x the baseline (~20 min): SGLang loads the draft model by
# re-reading the whole 704 GiB checkpoint a second time to pull out the nextn layer.
#
# Defaults to TP=8 like the baseline, so this cannot run at the same time as
# run_sglang.sh. Stop the baseline first, or set TP_SIZE=4 plus
# CUDA_VISIBLE_DEVICES on both to split the node.
#
# ENABLE_DP_ATTENTION=1 puts attention on data parallel and keeps the FFN on tensor
# parallel. SGLang requires dp_size == tp_size, and it auto-divides
# chunked-prefill-size by dp_size. For an MLA model this is the interesting knob:
# under plain TP the KV cache is replicated on every rank, while under DP attention
# each rank owns only its own requests' KV, so aggregate KV capacity scales with
# dp_size. Costs extra weight memory, since attention is no longer sharded.
#
# Usage:
#   ./run_sglang_mtp.sh
#   MTP_PROFILE=balanced ./run_sglang_mtp.sh
#   ENABLE_DP_ATTENTION=1 ./run_sglang_mtp.sh
#   ENABLE_DP_ATTENTION=1 ENABLE_DP_LM_HEAD=1 ./run_sglang_mtp.sh
#   tail -f tmp/mtp/server_safe.log
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MTP_DIR="${SCRIPT_DIR}/tmp/mtp"
mkdir -p "${MTP_DIR}"

MODEL_PATH="${MODEL_PATH:-/wekafs/models/GLM-5.2-FP8}"
PORT="${PORT:-30001}"
TP_SIZE="${TP_SIZE:-8}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
MTP_PROFILE="${MTP_PROFILE:-safe}"
ENABLE_DP_ATTENTION="${ENABLE_DP_ATTENTION:-0}"
ENABLE_DP_LM_HEAD="${ENABLE_DP_LM_HEAD:-0}"
DP_SIZE="${DP_SIZE:-${TP_SIZE}}"

LOG_TAG="${MTP_PROFILE}"
DP_ARGS=()
if [[ "${ENABLE_DP_ATTENTION}" == "1" ]]; then
  if (( TP_SIZE % DP_SIZE != 0 )); then
    echo "ERROR: TP_SIZE=${TP_SIZE} must be divisible by DP_SIZE=${DP_SIZE}" >&2
    exit 1
  fi
  DP_ARGS+=(--enable-dp-attention --dp-size "${DP_SIZE}")
  LOG_TAG="${LOG_TAG}_dp${DP_SIZE}"
  if [[ "${ENABLE_DP_LM_HEAD}" == "1" ]]; then
    DP_ARGS+=(--enable-dp-lm-head)
    LOG_TAG="${LOG_TAG}_dplmhead"
  fi
elif [[ "${ENABLE_DP_LM_HEAD}" == "1" ]]; then
  echo "ERROR: ENABLE_DP_LM_HEAD=1 requires ENABLE_DP_ATTENTION=1" >&2
  exit 1
fi

LOG="${LOG:-${MTP_DIR}/server_${LOG_TAG}.log}"
PID_FILE="${PID_FILE:-${MTP_DIR}/server.pid}"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "ERROR: no model at MODEL_PATH=${MODEL_PATH} (config.json missing)" >&2
  exit 1
fi

case "${MTP_PROFILE}" in
  safe)
    # steps<=3 for AMD build limit
    MTP_STEPS=3
    MTP_DRAFT_TOKENS=4
    ;;
  balanced)
    # cookbook mtp-112
    MTP_STEPS=1
    MTP_DRAFT_TOKENS=2
    ;;
  low-latency)
    # cookbook mtp-516 — likely fails on AMD (steps>3 build issue)
    MTP_STEPS=5
    MTP_DRAFT_TOKENS=6
    ;;
  *)
    echo "ERROR: unknown MTP_PROFILE=${MTP_PROFILE} (use safe|balanced|low-latency)" >&2
    exit 1
    ;;
esac

if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  if ss -tln 2>/dev/null | grep -q ":${PORT} "; then
    echo "Already running: pid=$(cat "${PID_FILE}") port=${PORT} log=${LOG}"
    exit 0
  fi
  rm -f "${PID_FILE}"
fi
if ss -tln 2>/dev/null | grep -q ":${PORT} "; then
  echo "ERROR: port ${PORT} in use; stop orphan MTP server first" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HIP_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
export SGLANG_DSA_TRITON_PREFILL=1
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
  --watchdog-timeout 1200 \
  --disable-custom-all-reduce \
  --speculative-algorithm EAGLE \
  --speculative-num-steps "${MTP_STEPS}" \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens "${MTP_DRAFT_TOKENS}" \
  "${DP_ARGS[@]}" \
  --weight-loader-prefetch-checkpoints \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 32}' \
  > "${LOG}" 2>&1 &

echo $! > "${PID_FILE}"
echo "Started GLM-5.2-FP8 + MTP (experimental): pid=$(cat "${PID_FILE}")"
echo "  model:   ${MODEL_PATH}"
echo "  port:    ${PORT}"
echo "  tp:      ${TP_SIZE} (gpus=${CUDA_VISIBLE_DEVICES})"
echo "  profile: ${MTP_PROFILE} (steps=${MTP_STEPS}, draft_tokens=${MTP_DRAFT_TOKENS})"
if [[ "${ENABLE_DP_ATTENTION}" == "1" ]]; then
  echo "  dp-attn: on (dp_size=${DP_SIZE}, dp_lm_head=${ENABLE_DP_LM_HEAD})"
else
  echo "  dp-attn: off"
fi
echo "  log:     ${LOG}"
echo "  note:    startup ~20 min (draft model re-reads the full checkpoint)"
echo "  ready check: curl -s http://127.0.0.1:${PORT}/health"
