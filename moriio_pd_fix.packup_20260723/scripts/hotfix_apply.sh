#!/bin/bash
# Apply the MoRIIO page-len fix to a RUNNING pre-fix engine image (v0.25.1 built
# before patch_moriio_pagelen.py landed). Run on EACH node's container BEFORE
# launching the engines. If you build a fresh image from the repo, the Dockerfile
# patch loop bakes this in and you DO NOT need this script.
# Usage: docker exec <ctr> python3 patch_moriio_pagelen.py   (the durable patch,
# which is idempotent and self-locating). This wrapper just points at it.
set -u
CTR=${1:-glm_pd}
PATCH=${2:-/mnt/vast/c_huggingface/vllm_patch_verify/patch_moriio_pagelen.py}
docker exec "$CTR" python3 "$PATCH"
