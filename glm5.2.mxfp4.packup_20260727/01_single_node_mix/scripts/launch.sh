set -uo pipefail
IMAGE=rocm/infera:sglang-v0.1.0-rc6
MODEL=/mnt/vast/xiaobo/models/GLM-5.2-MXFP4
SERVED=glm5.2-mxfp4
PORT=30000
NAME=glm52-mix-p1
TP=8
MAXLEN=400000
docker rm -f $NAME >/dev/null 2>&1
docker run -d --name $NAME --network host --ipc host --shm-size 64g \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -v /mnt/vast:/mnt/vast \
  -e SGLANG_USE_AITER=1 -e SGLANG_ROCM_FUSED_DECODE_MLA=0 \
  -e SGLANG_OPT_USE_TILELANG_INDEXER=1 -e SGLANG_OPT_USE_TOPK_V2=0 \
  -e SGLANG_OPT_USE_JIT_NORM=0 \
  -e ROCM_QUICK_REDUCE_QUANTIZATION=INT4 -e SAFETENSORS_FAST_GPU=1 \
  -e HIP_FORCE_DEV_KERNARG=1 -e HSA_NO_SCRATCH_RECLAIM=1 \
  --entrypoint python3 $IMAGE \
  -m sglang.launch_server \
    --model-path "$MODEL" --served-model-name "$SERVED" \
    --host 0.0.0.0 --port $PORT --tp-size $TP --trust-remote-code \
    --nsa-prefill-backend tilelang --nsa-decode-backend tilelang \
    --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85 \
    --context-length $MAXLEN --chunked-prefill-size 8192 \
    --max-running-requests 64 --cuda-graph-max-bs 64
echo "started $NAME"; docker ps --filter name=$NAME --format '{{.Names}} {{.Status}}'
