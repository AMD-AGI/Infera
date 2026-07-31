#!/bin/bash
# Start the `final` container from the image built by Dockerfile.sglang.dmabuf,
# and health-gate the GPUs (spur has nodes that enumerate 8 GPUs but report
# torch.cuda.is_available() == False).
#
# Unlike the dbg2 containers used during debugging, this one gets NO in-container
# patching: the patches are baked into the image at build time. That is the point
# of the final validation.
set -eu
JOB="${1:?job}"
IMG=infera.yihou.sglang.final:1.0
CTR=final
export DOCKER_CONFIG=/tmp/dockercfg

spur exec "$JOB" bash -c "export DOCKER_CONFIG=/tmp/dockercfg
docker rm -f $CTR 2>/dev/null || true
docker run -d --name $CTR --network=host --ipc=host --shm-size=32G \
  --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
  --group-add video --group-add render --cap-add=SYS_PTRACE --cap-add=IPC_LOCK \
  --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
  -v /shared_nfs:/shared_nfs \
  --entrypoint '' $IMG sleep infinity
docker exec $CTR python3 -c 'import torch;print(\"GPUGATE\", torch.cuda.is_available(), torch.cuda.device_count())'
docker exec $CTR bash -c 'cd /sgl-workspace/sglang && echo PATCHED_FILES=\$(git status --short --untracked-files=no python/sglang/srt | wc -l)'
" 2>&1 | grep -v libtinfow
