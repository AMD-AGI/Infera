#!/bin/bash
# Launch ONE Kimi-K3 TP8 mixed worker on this host, for the two-node kv-aware
# split reproduction. Run it on each of the two nodes.
#
#   WORKER_ID=w0 MODELS=/mnt/nvme3-bench/models ETCD=chi2832:2379 bash _kv_aware_k3_launch_w.sh
#   WORKER_ID=w1 MODELS=/mnt/nvme-raid/models   ETCD=chi2832:2379 bash _kv_aware_k3_launch_w.sh
#
# TP8 is not a tuning choice, it is forced twice over:
#   * the DSpark latent-MoE tail kernel refuses anything else
#     ("requires TP=8, got TP=4")
#   * AITER MLA needs 1-15 heads or a multiple of 16; K3's 96 heads over TP4
#     gives 24, which is neither
#   * 1.5T of weights over 4x288GB MI355X does not fit regardless
# So one worker occupies a whole 8-GPU node, and two workers means two nodes.
#
# Mirrors examples/recipes/kimi-k3-optimized/aggregated/deploy.yaml: same image
# digest, same env block, same engine flags. Run as plain docker rather than
# through the operator because these nodes are not in the k8s cluster.
set -euo pipefail

WORKER_ID="${WORKER_ID:?set WORKER_ID (w0|w1)}"
MODELS="${MODELS:?set MODELS (host dir containing moonshotai/Kimi-K3)}"
ETCD="${ETCD:?set ETCD (host:2379)}"
PORT="${PORT:-30000}"
ADVERTISE="${ADVERTISE:-$(hostname)}"

ENGINE=johnqin2025/kimi-k3-dspark@sha256:5f3007aff1bc231eceb9f024e56ee80e44f9ca101a521aa50fe6bfa6c979d6b8
OVERLAY="${OVERLAY:-inferaimage/infera-overlay:v0.2.5}"

docker rm -f "k3-$WORKER_ID" >/dev/null 2>&1 || true
rm -rf /tmp/k3-overlay && mkdir -p /tmp/k3-overlay
docker run --rm -v /tmp/k3-overlay:/out "$OVERLAY" >/dev/null
echo "overlay staged"

# --kv-event-transport zmq is REQUIRED: the default is nats, and with no broker
# the worker never finishes registering. The ZMQ port is auto-allocated by the
# launcher and self-advertised in the etcd record, so it is not pinned here.
# --advertise-host must be the routable hostname: the router and the peer node
# both dial it, and 127.0.0.1 would only work same-host.
docker run -d --name "k3-$WORKER_ID" --network host --ipc host --shm-size 32g \
  --device /dev/kfd --device /dev/dri --group-add video \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -e VLLM_ROCM_USE_AITER=1 \
  -e VLLM_ROCM_USE_AITER_FP4BMM=1 \
  -e AITER_SITUV2_A8W4=1 \
  -e AITER_BF16_FP8_MOE_BOUND=0 \
  -e HIP_FORCE_DEV_KERNARG=1 \
  -e HSA_NO_SCRATCH_RECLAIM=1 \
  -e HSA_ENABLE_IPC_MODE_LEGACY=1 \
  -e SAFETENSORS_FAST_GPU=1 \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
  -e VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION=1 \
  -e VLLM_ROCM_USE_KIMI_K3_PREROUTE_BF16=0 \
  -e VLLM_ROCM_USE_KIMI_K3_PREROUTE_FP8=1 \
  -e VLLM_ROCM_USE_KIMI_K3_LATENT_TAIL_FP8=1 \
  -e KIMI_K3_DUAL_PROJ_FP8_WEIGHT_CACHE_MODIFIER=2 \
  -e KIMI_K3_SHARED_DOWN_FP8_WEIGHT_CACHE_MODIFIER=2 \
  -e INFERA_ENGINE_READY_TIMEOUT=7200 \
  -e HF_HUB_OFFLINE=1 \
  -v /tmp/k3-overlay:/overlay:ro -v "$MODELS":/models:ro \
  "$ENGINE" \
  /overlay/bin/infera-exec python3 -m infera.engine.vllm \
    --host 0.0.0.0 --port "$PORT" \
    --advertise-host "$ADVERTISE" \
    --model /models/moonshotai/Kimi-K3 --served-model-name kimi-k3 \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.88 \
    --trust-remote-code --load-format auto \
    --kv-cache-dtype auto --enable-prefix-caching \
    --block-size 16 \
    --max-num-seqs 64 --max-num-batched-tokens 4096 \
    --max-model-len 131072 \
    --kv-event-transport zmq \
    --request-transport http \
    --discovery-backend etcd --etcd-endpoint "$ETCD"

echo "k3-$WORKER_ID up on $ADVERTISE:$PORT (TP8), registering into etcd at $ETCD"
echo "1.5T of weights: first load takes several minutes. docker logs -f k3-$WORKER_ID"
