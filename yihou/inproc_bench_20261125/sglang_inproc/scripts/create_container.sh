#!/bin/bash
# Create (or reuse) the long-lived bench container of the sglang kit. Idempotent; deletes nothing.
# Env: MODEL_DIR (host model dir, mounted ro), WORK_DIR and CACHE_DIR (both under a yihou path),
#      optional CONTAINER (default yihou-inproc-sglang), IMAGE (default below), GPUS (ROCR_VISIBLE_DEVICES),
#      ENV_FILES (preset env files, default presets/dsv41.env), EXTRA_ENV ("K=V K2=V2"), PIDHOST (default 1).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../../common/lib.sh"
CONTAINER=${CONTAINER:-yihou-inproc-sglang}
IMAGE=${IMAGE:-lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261111}
read -r -a FILES <<< "${ENV_FILES:-$HERE/../presets/dsv41.env}"
kit_create_container "$CONTAINER" "$IMAGE" "${FILES[@]}"
