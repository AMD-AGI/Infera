#!/bin/bash
# start.sh <variant: mx|bf16> <port> [tag] ; env EXTRA_FLAGS replaces the optional kit flags
set -u
v=$1; port=$2; tag=${3:-}
EXTRA_FLAGS=${EXTRA_FLAGS---enable-fused-qk-norm-rope --enable-aiter-allreduce-fusion}
D=/shared_nfs/yihou_glm52/llrepro_20261023/mxdraft.yihou
P=/sgl-workspace/sglang/python/sglang/srt
M=/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4
name=yihou-mxdraft-test-$v$tag
df -h / /var/lib/docker 2>/dev/null | tail -2
mounts=(-v $M:$M:ro -v yihou-mxdraft-cache:/root/.cache -v yihou-mxdraft-aiter-jit:/sgl-workspace/aiter/aiter/jit)
q=()
if [ "$v" = mx ]; then
  mounts+=(-v $D/patched/model_config.py:$P/configs/model_config.py:ro -v $D/patched/weight_utils.py:$P/model_loader/weight_utils.py:ro)
  q=(--speculative-draft-model-quantization quark_mxfp4)
fi
docker run -d --name $name --network host --ipc host --shm-size 32g \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined --ulimit memlock=-1:-1 --ulimit nofile=65536:65536 \
  "${mounts[@]}" \
  -e HIP_VISIBLE_DEVICES=0,1,2,3 -e NCCL_IB_DISABLE=1 -e NCCL_IGNORE_CPU_AFFINITY=1 -e HSA_NO_SCRATCH_RECLAIM=1 -e HIP_FORCE_DEV_KERNARG=1 \
  -e AITER_USE_FLYDSL_MOE_SORTING=1 -e PYTHONHASHSEED=0 -e SAFETENSORS_FAST_GPU=1 -e SGLANG_DP_USE_GATHERV=1 \
  -e SGLANG_OPT_USE_TOPK_V2=true -e SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=0 \
  --entrypoint python3 infera-sglang:v0522-fdbafb60 -m sglang.launch_server \
  --model-path $M --trust-remote-code --host 127.0.0.1 --port $port \
  --tp-size 4 --dp-size 4 --enable-dp-attention --ep-size 1 \
  --kv-cache-dtype fp8_e4m3 --dsa-prefill-backend triton --dsa-decode-backend triton \
  --chunked-prefill-size 32768 --max-running-requests 256 \
  --json-model-override-args "${JSON_OVR:-{\"index_share_for_mtp_iteration\":true}}" \
  $EXTRA_FLAGS \
  --reasoning-parser glm45 --tool-call-parser glm47 --watchdog-timeout 3600 \
  --speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 \
  "${q[@]}" --mem-fraction-static 0.85 --cuda-graph-max-bs-decode 64
echo started $name
