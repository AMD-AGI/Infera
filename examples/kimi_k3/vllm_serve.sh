#!/bin/bash
# Kimi-K3 on AMD Instinct — vLLM serve (TP8), MXFP4/A8W4 runtime quant via aiter.
# Base image + flags per the AMD "Kimi-K3 on AMD Instinct GPUs" guide. Run ON an
# 8-GPU node. The native moonshotai/Kimi-K3 checkpoint (~1.5 TB) is cached on the
# shared mount; the aiter A8W4 env flags do the low-bit quantization at load.
set -u
IMG=${IMG:-vllm/vllm-openai-rocm:kimi-k3}
# Local cached checkpoint (plain dir); override MODEL to use a different path/repo.
MODEL=${MODEL:-/mnt/vast/yaocheng/models/moonshotai/Kimi-K3}
PORT=${PORT:-8000}
TP=${TP:-8}
NAME=${NAME:-kimi_k3_vllm}

docker rm -f "$NAME" >/dev/null 2>&1
# Image ENTRYPOINT is /bin/bash (dev image), so serve via the vllm CLI:
# `vllm serve <model> <flags>` (override the entrypoint).
docker run -d --name "$NAME" \
  --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined --group-add video \
  --privileged --ipc=host --shm-size 32g -p "$PORT":8000 \
  -v /mnt/vast:/mnt/vast \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_ROCM_USE_AITER=1 \
  -e SAFETENSORS_FAST_GPU=1 \
  -e AITER_SITUV2_A8W4=1 \
  -e AITER_BF16_FP8_MOE_BOUND=0 \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
  --entrypoint vllm \
  "$IMG" serve "$MODEL" \
  --served-model-name kimi-k3 \
  --trust-remote-code \
  --moe-backend auto \
  --tensor-parallel-size "$TP" \
  --load-format auto \
  --gpu-memory-utilization 0.95 \
  --mm-encoder-tp-mode data \
  --max-num-seqs 128 \
  --max-num-batched-tokens 4096 \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k3 \
  --reasoning-parser kimi_k3

echo "launched $NAME (TP$TP) on port $PORT; follow: docker logs -f $NAME"
