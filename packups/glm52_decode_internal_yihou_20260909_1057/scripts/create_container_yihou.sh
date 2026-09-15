#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
: "${JOB_ID:?Set existing authorized JOB_ID}" "${NODE:?Set assigned NODE}" "${CONTAINER:?Set unique yihou CONTAINER}"
[[ "$JOB_ID" =~ ^[0-9]+$ && "$NODE" =~ ^crsuse2-m2m-[0-9]+$ && "$CONTAINER" =~ ^yihou-[A-Za-z0-9_-]+$ ]] || exit 2
case "$NODE" in crsuse2-m2m-234|crsuse2-m2m-036|crsuse2-m2m-249) printf 'Forbidden or peer-owned node: %s\n' "$NODE" >&2; exit 2;; esac
IMAGE=sha256:b9a83742f631d7321c6b75aceb031a8b347c70f97c62ef8c62370c686853de7d
GPU_DEVICES=${GPU_DEVICES:-0,1,2,3}
[[ "$GPU_DEVICES" =~ ^[0-7](,[0-7])*$ ]] || exit 2
STATE=$(squeue -j "$JOB_ID" -h -o '%u %T %N')
[[ "$STATE" == "$USER RUNNING $NODE" ]] || { printf 'Unexpected allocation: %s\n' "$STATE" >&2; exit 1; }
CMD=(docker run -d --name "$CONTAINER" --label owner="$USER" --label task=glm52-internal-decode
    -w / --network host --ipc=host --shm-size 32g
    --device=/dev/kfd --device=/dev/dri --group-add video --group-add render
    --security-opt seccomp=unconfined --ulimit memlock=-1:-1
    -v "$ROOT:$ROOT" -v /shared_nfs/models/GLM-5.2-MXFP4:/shared_nfs/models/GLM-5.2-MXFP4:ro
    -e PYTHONNOUSERSITE=1 -e PYTHONUNBUFFERED=1 -e HIP_VISIBLE_DEVICES="$GPU_DEVICES"
    -e AITER_JIT_DIR=/tmp/yihou-aiter-b9a83742f631
    -e HF_HOME=/tmp/yihou-hf-cache -e HF_HUB_OFFLINE=1
    -e SGLANG_OPT_USE_TOPK_V2=false -e SGLANG_TIMEOUT_KEEP_ALIVE=900
    -e AITER_USE_FLYDSL_MOE_SORTING=1 -e SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=1
    "$IMAGE" sleep infinity)
printf -v REMOTE '%q ' "${CMD[@]}"
spur exec "$JOB_ID" bash -lc "set -euo pipefail; test \"\$(hostname)\" = $NODE; if docker container inspect $CONTAINER >/dev/null 2>&1; then printf 'Refusing existing container\\n' >&2; exit 1; fi; $REMOTE"
