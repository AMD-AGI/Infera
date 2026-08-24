#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Launch the SGLang decode leg on the decode node.
# Run INSIDE the engine container:  bash launch/launch_decode.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
source "$HERE/env.sh"
require_ips

LOG="${LOG:-$LOG_DIR/decode.log}"
HOST_IP="${HOST_IP:-$DECODE_IP}"
PORT="${PORT:-$DECODE_PORT}"

# kv events stay off on this leg. Prefill-side prefix locality is what the router
# steers by, and decode events can make SGLang reject the speculative disagg flags.
KV_EVENT_ARGS=(--no-enable-kv-events --kv-events off)

# There is no kvd block here on purpose, even at KVD=1: SGLang issues storage
# prefetch on its aggregated and prefill branches only, so an offload tier on this
# leg would be write-only.

MTP_ARGS=()
if [[ "$MTP" != "0" ]]; then
  MTP_ARGS=(--speculative-algorithm EAGLE --speculative-num-steps "$MTP_STEPS"
            --speculative-eagle-topk "$MTP_TOPK" --speculative-num-draft-tokens "$MTP_DRAFT_TOKENS"
            --json-model-override-args '{"index_share_for_mtp_iteration":false}')
fi

# This is the leg that matters for MTP: speculative verification happens here, so
# `accept len` is in THIS log and sglang:spec_accept_length on THIS /metrics.
METRICS_ARGS=()
if [[ "$ENGINE_METRICS" == "1" ]]; then
  METRICS_ARGS=(--enable-metrics --enable-metrics-for-all-schedulers)
fi

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
export SGLANG_HOST_IP="$HOST_IP" HOST_IP
export SGLANG_DSA_TRITON_PREFILL=1 SAFETENSORS_FAST_GPU=1
export HSA_NO_SCRATCH_RECLAIM=1 SGLANG_USE_AITER="${SGLANG_USE_AITER:-1}"
export MC_GID_INDEX
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT="${SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT:-3600}"
export INFERA_ENGINE_READY_TIMEOUT="${INFERA_ENGINE_READY_TIMEOUT:-5400}"

# Must mirror launch_prefill.sh: the two legs exchange KV, so the attention
# parallelism has to be the same shape on both ends.
ATTN_ARGS=(--tp-size "$TP" --dp-size "$DP")
[[ "$DP" -gt 1 ]] && ATTN_ARGS+=(--enable-dp-attention)

pkill -f "(infera.engine.sglang|sglang.launch_server) .*--port ${PORT}( |$)" 2>/dev/null || true
sleep 5

# Both legs are the two ends of one Mooncake transfer, so IB_DEVICE, the MTP shape
# and the KV dtype must match launch_prefill.sh.
nohup python3 -m infera.engine.sglang \
  --model-path "$MODEL" --host 0.0.0.0 --port "$PORT" --advertise-host "$HOST_IP" \
  --etcd-endpoint "$ETCD_ENDPOINT" --discovery-backend etcd \
  --request-transport http "${KV_EVENT_ARGS[@]}" \
  "${ATTN_ARGS[@]}" \
  --trust-remote-code --kv-cache-dtype fp8_e4m3 \
  --reasoning-parser glm45 --tool-call-parser glm47 \
  --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
  --mem-fraction-static "$MEM_FRAC" --max-running-requests "$MAX_RUNNING" \
  --chunked-prefill-size "$CHUNK" --watchdog-timeout 1200 \
  --disable-custom-all-reduce --enable-cache-report \
  "${MTP_ARGS[@]}" "${METRICS_ARGS[@]}" ${EXTRA_ARGS:-} \
  --weight-loader-prefetch-checkpoints \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 32}' \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake --disaggregation-ib-device "$IB_DEVICE" \
  > "$LOG" 2>&1 &

echo "[decode] loading on ${HOST_IP}:${PORT}, TP=$TP DP=$DP MTP=$MTP($MTP_STEPS/$MTP_TOPK/$MTP_DRAFT_TOKENS)"
echo "[decode] rail=$IB_DEVICE metrics=$ENGINE_METRICS; log=$LOG"
echo "[decode] cold start can take 15-25 min; follow with: tail -f $LOG"
