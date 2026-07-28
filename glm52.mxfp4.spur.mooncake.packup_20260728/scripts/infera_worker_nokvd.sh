#!/bin/bash
# Single-node GLM-5.2 sglang worker via the INFERA engine wrapper, with etcd self-registration +
# KV-event publishing for KV-AWARE routing, using the engine's NATIVE radix prefix cache
# (NO hicache/kvd — that write-back path GPU-faults on gfx950). --enable-cache-report so the
# response usage carries cached_tokens. Runs INSIDE the pd_spur container.
set -u
MODEL="${MODEL:-/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4}"
SERVED="${SERVED:-glm5.2-mxfp4}"
MY_IP="${MY_IP:?MY_IP}"
PORT="${PORT:-30000}"
ETCD="${ETCD:?ETCD=host:2379}"
KV_EVENTS_PORT="${KV_EVENTS_PORT:-5557}"
SNAP_PORT="${SNAP_PORT:-8801}"
LOG="${LOG:-/home/yihou/glm52_spur/logs/infera_worker.log}"

export SGLANG_USE_AITER=1 SGLANG_ROCM_FUSED_DECODE_MLA=0
export SGLANG_OPT_USE_TILELANG_INDEXER=1 SGLANG_OPT_USE_TOPK_V2=0 SGLANG_OPT_USE_JIT_NORM=0
export SAFETENSORS_FAST_GPU=1 HIP_FORCE_DEV_KERNARG=1
export SGLANG_HOST_IP="$MY_IP" HOST_IP="$MY_IP"

HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m infera.engine.sglang \
  --discovery-backend etcd --etcd-endpoint "$ETCD" --etcd-prefix /infera/workers/ \
  --request-transport http \
  --advertise-host "$MY_IP" \
  --enable-kv-events \
  --kv-events on \
  --kv-events-bind "tcp://0.0.0.0:${KV_EVENTS_PORT}" \
  --kv-event-transport zmq \
  --kv-snapshot-host 0.0.0.0 --kv-snapshot-port "$SNAP_PORT" \
  --index-block-size 64 \
  --model-path "$MODEL" --served-model-name "$SERVED" --tp-size 8 --trust-remote-code \
  --host "$MY_IP" --port "$PORT" \
  --nsa-prefill-backend tilelang --nsa-decode-backend tilelang \
  --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85 --context-length 32768 \
  --enable-cache-report \
  --max-running-requests 128 --cuda-graph-max-bs 128 --watchdog-timeout 3600 \
  > "$LOG" 2>&1
