#!/usr/bin/env bash
# Launch the SGLang decode leg on the decode node.
# Run inside the engine container:
#   bash launch/launch_decode.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
source "$HERE/env.sh"

LOG="${LOG:-$LOG_DIR/decode.log}"
HOST_IP="${HOST_IP:-$DECODE_IP}"
PORT="${PORT:-$DECODE_PORT}"

# Keep kv events off on the decode leg with MTP. Prefill-side prefix locality is
# the large win; decode events can make SGLang reject speculative disagg flags.
KV_EVENT_ARGS=(--no-enable-kv-events --kv-events off)
if [[ "${KV_EVENTS:-0}" == "1" ]]; then
  KV_EVENT_ARGS=(--enable-kv-events --kv-events on --kv-event-transport zmq)
fi

MTP_ARGS=()
if [[ "$MTP" != "0" ]]; then
  MTP_ARGS=(--speculative-algorithm EAGLE --speculative-num-steps 3
            --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
            --json-model-override-args '{"index_share_for_mtp_iteration":false}')
fi

IB_ARGS=()
if [[ -n "${IB_DEVICE:-}" ]]; then
  IB_ARGS=(--disaggregation-ib-device "$IB_DEVICE")
fi

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
export SGLANG_HOST_IP="$HOST_IP" HOST_IP
export SGLANG_DSA_TRITON_PREFILL=1 SAFETENSORS_FAST_GPU=1
export HSA_NO_SCRATCH_RECLAIM=1 SGLANG_USE_AITER="${SGLANG_USE_AITER:-1}"
export MC_GID_INDEX
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT="${SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT:-3600}"
export INFERA_ENGINE_READY_TIMEOUT="${INFERA_ENGINE_READY_TIMEOUT:-5400}"

pkill -f "(infera.engine.sglang|sglang.launch_server) .*--port ${PORT}( |$)" 2>/dev/null || true
sleep 5

nohup python3 -m infera.engine.sglang \
  --model-path "$MODEL" --host 0.0.0.0 --port "$PORT" --advertise-host "$HOST_IP" \
  --etcd-endpoint "$ETCD_ENDPOINT" --discovery-backend etcd \
  --request-transport http "${KV_EVENT_ARGS[@]}" \
  --tp-size "$TP" --dp-size "$DP" --enable-dp-attention \
  --trust-remote-code --kv-cache-dtype fp8_e4m3 \
  --reasoning-parser glm45 --tool-call-parser glm47 \
  --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
  --mem-fraction-static "$MEM_FRAC" --max-running-requests "$MAX_RUNNING" \
  --chunked-prefill-size "$CHUNK" --watchdog-timeout 1200 \
  --disable-custom-all-reduce --enable-cache-report \
  "${MTP_ARGS[@]}" ${EXTRA_ARGS:-} \
  --weight-loader-prefetch-checkpoints \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 32}' \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake "${IB_ARGS[@]}" \
  > "$LOG" 2>&1 &

echo "[decode] loading on ${HOST_IP}:${PORT}, TP=$TP DP=$DP MTP=$MTP; log=$LOG"
echo "[decode] cold start can take 15-25 min; follow with: tail -f $LOG"
