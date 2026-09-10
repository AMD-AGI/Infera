#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
JOB=126175
NODE=crsuse2-m2m-055
NAME=yihou-glm52-internal-20260909
IMAGE=sha256:b9a83742f631d7321c6b75aceb031a8b347c70f97c62ef8c62370c686853de7d
STATE=$(squeue -j "$JOB" -h -o '%u %T %N')
[[ "$STATE" == "yihou RUNNING $NODE" ]] || { printf 'Unexpected allocation: %s\n' "$STATE" >&2; exit 1; }
CMD=(docker run -d --name "$NAME" --label owner=yihou --label task=glm52-internal-decode
    -w / --network host --ipc=host --shm-size 32g
    --device=/dev/kfd --device=/dev/dri --group-add video --group-add render
    --security-opt seccomp=unconfined --ulimit memlock=-1:-1
    -v "$ROOT:$ROOT" -v /shared_nfs/models/GLM-5.2-MXFP4:/shared_nfs/models/GLM-5.2-MXFP4:ro
    -e PYTHONNOUSERSITE=1 -e PYTHONUNBUFFERED=1
    -e AITER_JIT_DIR=/tmp/yihou-aiter-b9a83742f631
    -e HF_HOME=/tmp/yihou-hf-cache -e HF_HUB_OFFLINE=1
    -e SGLANG_OPT_USE_TOPK_V2=false -e SGLANG_TIMEOUT_KEEP_ALIVE=900
    -e AITER_USE_FLYDSL_MOE_SORTING=1 -e SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=1
    "$IMAGE" sleep infinity)
printf -v REMOTE '%q ' "${CMD[@]}"
spur exec "$JOB" bash -lc "set -euo pipefail; test \"\$(hostname)\" = $NODE; if docker container inspect $NAME >/dev/null 2>&1; then printf 'Refusing existing container\\n' >&2; exit 1; fi; $REMOTE"
