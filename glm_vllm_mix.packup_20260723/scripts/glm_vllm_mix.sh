#!/bin/bash
# GLM-5.1-FP8 on vLLM — single-node PD-mix (one server, prefill+decode together).
# Verified 2026-07-23 on chi2879 (MI355X gfx950), card0-3, temp=0 probes ALL PASS.
#
# Single-node mix => NO kv-transfer connector (no MoRIIO / Mooncake / cross-node).
# The MoRIIO page-len fix in this image is a no-op here (no KV-transfer path). Run
# from the HOST (it does docker run).
set -u

MODEL=${MODEL:-/mnt/vast/xiaobo/models/GLM-5.1-FP8}
IMG=${IMG:-infera/engine-vllm:test-local}     # id e91a6d7d3a91 (contains the moriio pagelen fix)
CTR=${CTR:-glm_vllm_mix_c_hf}
PORT=${PORT:-8000}
LOG=${LOG:-/mnt/vast/c_huggingface/glm_vllm_mix.log}

docker rm -f "$CTR" 2>/dev/null || true

# card0-3 (chi2879 was fully free). aiter is enabled via env (not --moe-backend).
docker run -d --name "$CTR" --privileged --ipc host --shm-size 32g \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  -e HIP_VISIBLE_DEVICES=0,1,2,3 \
  -e VLLM_USE_V1=1 -e VLLM_ROCM_USE_AITER=1 -e AITER_BF16_FP8_MOE_BOUND=0 \
  -e PYTHONHASHSEED=0 -e VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1 \
  -v /mnt/vast:/mnt/vast --entrypoint bash "$IMG" -lc \
  "vllm serve $MODEL --served-model-name GLM-5.1-FP8 \
    --host 0.0.0.0 --port $PORT --tensor-parallel-size 4 --trust-remote-code \
    --kv-cache-dtype fp8 --reasoning-parser glm45 --no-enable-prefix-caching \
    --gpu-memory-utilization 0.85 --max-model-len 9472 --max-num-batched-tokens 8192 \
    --distributed-executor-backend mp 2>&1 | tee $LOG"

echo "[launch] container $CTR started; follow: docker logs -f $CTR"
echo "[launch] wait for 'Application startup complete' / health 200"
echo "[launch] CUDA-graph capture / torch.compile ~3-7 min; a silent window is normal."
echo "[note] if host port $PORT collides, drop -p and probe via docker exec localhost."
