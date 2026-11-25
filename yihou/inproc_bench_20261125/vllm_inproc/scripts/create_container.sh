#!/bin/bash
# Create (or reuse) the long-lived bench container of the vllm kit. Idempotent; deletes nothing.
# Env: MODEL_DIR (host model dir, mounted ro), WORK_DIR and CACHE_DIR (both under a yihou path),
#      optional CONTAINER (default yihou-inproc-vllm), IMAGE (default below), GPUS (ROCR_VISIBLE_DEVICES),
#      ENV_FILES (preset env files, default presets/dsv41.env), EXTRA_ENV ("K=V K2=V2"), PIDHOST (default 1).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../../common/lib.sh"
CONTAINER=${CONTAINER:-yihou-inproc-vllm}
IMAGE=${IMAGE:-infera-vllm-dsv41:0007cd8087f1-nightly-9c1d45c2}
read -r -a FILES <<< "${ENV_FILES:-$HERE/../presets/dsv41.env}"
kit_create_container "$CONTAINER" "$IMAGE" "${FILES[@]}"
