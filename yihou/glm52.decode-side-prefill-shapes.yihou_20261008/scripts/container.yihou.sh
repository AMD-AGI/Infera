#!/usr/bin/env bash
# Create the decode-leg container on the held node, mirroring the deployment's decode
# container env (bench-harness/engine.sh + config.sh as resolved in runtime.env).
# Runs as root (as the deployment did); outputs go to node-local scratch, synced later.
set -euo pipefail
WS=/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.decode-side-prefill-shapes.yihou_20261008
IMAGE=${IMAGE:-infera-sglang:v0519-yihou-0917-nextnfix-hicache}
# The deployment's ID was fd7220a5...; `docker load` on this node's containerd store
# reports the manifest digest instead. All 72 RootFS DiffIDs were verified identical
# to the reference packup's environment.md (working_process.md R1).
EXPECT_ID=sha256:291f148aab0a3bc77eb3e21ca431dfd7e3af562362de075d53f82029bff0fead
NAME=${NAME:-glm52-dsp-yihou}
MODEL=/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4
SCRATCH=/mnt/m2m_nobackup/yihou/dsp-yihou
GPUS=${GPUS:-4,5,6,7}   # the deployment's decode leg ran on GPUs 4-7

[[ "$(docker image inspect -f '{{.Id}}' "$IMAGE")" == "$EXPECT_ID" ]] || { echo "image id mismatch" >&2; exit 1; }
docker container inspect "$NAME" >/dev/null 2>&1 && { echo "exists: $NAME" >&2; exit 1; }
mkdir -p "$SCRATCH/out" "$SCRATCH/aiter-jit"

docker run -d --name "$NAME" --label owner=yihou \
    --network host --ipc=host --shm-size 64g \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
    -v "$WS:$WS:ro" -v "$MODEL:$MODEL:ro" -v "$SCRATCH:$SCRATCH" \
    -e HIP_VISIBLE_DEVICES="$GPUS" \
    -e AITER_JIT_DIR="$SCRATCH/aiter-jit" \
    -e NCCL_IGNORE_CPU_AFFINITY=1 -e HSA_NO_SCRATCH_RECLAIM=1 \
    -e SGLANG_USE_AITER=1 -e SGLANG_OPT_USE_TOPK_V2=false \
    -e SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=0 \
    -e PYTHONNOUSERSITE=1 -e PYTHONUNBUFFERED=1 \
    -e AITER_USE_FLYDSL_MOE_SORTING=1 -e SAFETENSORS_FAST_GPU=1 \
    -e HIP_FORCE_DEV_KERNARG=1 -e PYTHONHASHSEED=0 \
    -e SGLANG_DP_USE_GATHERV=1 -e HF_HUB_OFFLINE=1 \
    --entrypoint sleep "$IMAGE" infinity
echo "created $NAME from $IMAGE on GPUs $GPUS"
