#!/usr/bin/env bash
# Verify the reverted HOST_RDMA_MOUNT design: the kit no longer hardcodes the
# image-specific path, so the wrapper must supply it and the container must still
# see its RDMA devices.
#
# Runs ON a node. Does NOT touch the live engines — it starts a THROWAWAY container
# under a different name, checks it, and removes it.
set -u
KIT=/mnt/vast/c_huggingface/glm52_example_verify/kit
CTR=glm52_mountcheck
export CTR
export INFERA_IMAGE=infera/engine-sglang:merged-e
export MODEL_MOUNT=/mnt/vast
source "$KIT/common.sh"

echo "===== A. wrapper supplies HOST_RDMA_MOUNT (the correct configuration) ====="
HOST_RDMA_LIB=/usr/lib/x86_64-linux-gnu/libionic.so.1 \
HOST_RDMA_MOUNT=/host-libionic/libionic.so \
ENTRYPOINT_KEEP=1 \
  start_container
docker rm -f "$CTR" >/dev/null 2>&1

echo
echo "===== B. HOST_RDMA_MOUNT omitted -> kit default -> the SILENT failure, now warned ====="
HOST_RDMA_LIB=/usr/lib/x86_64-linux-gnu/libionic.so.1 \
ENTRYPOINT_KEEP=1 \
  start_container
docker rm -f "$CTR" >/dev/null 2>&1
echo
echo "A must show a nonzero device count; B must show 0 AND print the warning."
