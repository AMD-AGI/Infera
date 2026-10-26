#!/bin/bash
# start.sh <K steps> <port> <gpus e.g. 0,1,2,3>
# Real-acceptance measurement server: TP4 DP4 dp-attention, kv fp8, triton DSA, EAGLE topk1,
# bf16 draft, NO fused-qk-norm-rope / aiter-allreduce-fusion, custom AR disabled, no SGLANG_SIMULATE_ACC_LEN.
# Derived from ../mxdraft.yihou/start.sh (bf16 variant).
set -u
K=$1; port=$2; gpus=$3
M=/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4
name=yihou-acc-k$K
D=/shared_nfs/yihou_glm52/llrepro_20261023/acc.yihou
df -h / | tail -1
docker run -d --name $name --network host ${IPCOPT---ipc host} --shm-size 32g \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined --ulimit memlock=-1:-1 --ulimit nofile=65536:65536 \
  -v $M:$M:ro -v $D:$D -v yihou-mxdraft-cache:/root/.cache -v yihou-mxdraft-aiter-jit:/sgl-workspace/aiter/aiter/jit \
  -e ${VISENV:-HIP_VISIBLE_DEVICES}=$gpus -e NCCL_DEBUG=${NCCL_DEBUG:-WARN} -e NCCL_IB_DISABLE=1 -e NCCL_IGNORE_CPU_AFFINITY=1 -e HSA_NO_SCRATCH_RECLAIM=1 -e HIP_FORCE_DEV_KERNARG=1 \
  -e AITER_USE_FLYDSL_MOE_SORTING=1 -e PYTHONHASHSEED=0 -e SAFETENSORS_FAST_GPU=1 -e SGLANG_DP_USE_GATHERV=1 \
  -e SGLANG_OPT_USE_TOPK_V2=true -e SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=0 \
  --entrypoint python3 infera-sglang:v0522-fdbafb60 -m sglang.launch_server \
  --model-path $M --trust-remote-code --host 127.0.0.1 --port $port --nccl-port $((port+77)) \
  --tp-size 4 --dp-size 4 --enable-dp-attention --ep-size 1 \
  --kv-cache-dtype fp8_e4m3 --dsa-prefill-backend triton --dsa-decode-backend triton \
  --chunked-prefill-size 32768 --max-running-requests 256 \
  --json-model-override-args '{"index_share_for_mtp_iteration":true}' \
  --disable-custom-all-reduce \
  --reasoning-parser glm45 --tool-call-parser glm47 --watchdog-timeout 3600 \
  --speculative-algorithm EAGLE --speculative-num-steps $K --speculative-eagle-topk 1 --speculative-num-draft-tokens $((K+1)) \
  --mem-fraction-static 0.85 --cuda-graph-max-bs-decode 64
echo started $name
