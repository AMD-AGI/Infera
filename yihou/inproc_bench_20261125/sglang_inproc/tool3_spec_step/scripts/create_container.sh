#!/usr/bin/env bash
# Create (or reuse) the owned tool-3 container on this host. Wraps common/lib.sh kit_create_container, which carries
# the host fixes this kit needs (label owner=yihou, model read-only at /models/<name>, kit read-only at /kit, work dir
# at /out, caches at /cache, --pid host for a 2nd RCCL container, ROCR_VISIBLE_DEVICES). Idempotent; deletes nothing.
# Port note: the GLM-5.2 script's four host fixes (uid/gid for NFS root_squash, aiter_configs copy, synthesized passwd,
# explicit PATH) are replaced by running as root with the kit copied to a local yihou dir (the docker daemon cannot
# read the root-squashed NFS home) and all caches under /cache.
# Env: MODEL_DIR WORK_DIR CACHE_DIR KIT_DIR (all host paths; WORK/CACHE under a yihou dir), CONTAINER, IMAGE, GPUS.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
KIT_ROOT=$(cd "$HERE/../../.." && pwd)
source "$KIT_ROOT/common/lib.sh"
CONTAINER=${CONTAINER:-yihou-inproc-t3}
IMAGE=${IMAGE:-lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261111}
PINNED_DIGEST=${PINNED_DIGEST:-sha256:6a0b79528f8f899365368e4f569bc3262466973a984e76ffa787dddad321dad8}
GPUS=${GPUS:-0,1,2,3}
[[ "$CONTAINER" =~ ^yihou-inproc-[A-Za-z0-9_-]+$ ]] || kit_die "container name must start with yihou-inproc-"
[ "$(docker image inspect -f '{{.Id}}' "$IMAGE")" = "$PINNED_DIGEST" ] || kit_die "$IMAGE is not the pinned digest $PINNED_DIGEST"
export GPUS
kit_create_container "$CONTAINER" "$IMAGE" "$KIT_ROOT/sglang_inproc/presets/dsv41.env"
