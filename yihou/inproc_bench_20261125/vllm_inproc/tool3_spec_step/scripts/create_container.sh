#!/usr/bin/env bash
# Create (or reuse) the owned vLLM tool-3 container on this host. Wraps common/lib.sh kit_create_container (label
# owner=yihou, model read-only at /models/<name>, kit read-only at /kit, work dir at /out, caches at /cache,
# --pid host for a 2nd RCCL container on the node, ROCR_VISIBLE_DEVICES). Idempotent; deletes nothing.
# Port note: the GLM-5.2 script's four host fixes (uid/gid for NFS root_squash, aiter_configs copy, synthesized passwd,
# explicit PATH) are replaced by running as root with the kit copied to a local yihou dir (the docker daemon cannot
# read the root-squashed NFS home) and all caches under /cache.
# Env: MODEL_DIR WORK_DIR CACHE_DIR KIT_DIR (host paths; WORK/CACHE under a yihou dir), CONTAINER, IMAGE, GPUS.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
KIT_ROOT=$(cd "$HERE/../../.." && pwd)
source "$KIT_ROOT/common/lib.sh"
CONTAINER=${CONTAINER:-yihou-inproc-v3}
IMAGE=${IMAGE:-infera-vllm-dsv41:0007cd8087f1-nightly-9c1d45c2}
PINNED_DIGEST=${PINNED_DIGEST:-sha256:51ceb17daa7d00c40a9978fd7c98337ffc3cb538719422fcdd9c085c1329db37}
GPUS=${GPUS:?set GPUS (ROCR_VISIBLE_DEVICES), e.g. 2,3 for DP2 TP1}
[[ "$CONTAINER" =~ ^yihou-inproc-v3[A-Za-z0-9_-]*$ ]] || kit_die "container name must start with yihou-inproc-v3"
[ "$(docker image inspect -f '{{.Id}}' "$IMAGE")" = "$PINNED_DIGEST" ] || kit_die "$IMAGE is not the pinned digest $PINNED_DIGEST"
export GPUS
kit_create_container "$CONTAINER" "$IMAGE" "$HERE/../presets/dsv41_decode.env"
