#!/bin/bash
# Start the dbg2 container on a spur job and health-gate the GPUs.
# Usage: start_ctr.sh <job>
set -eu
JOB="${1:?job}"
IMG=infera.yihou.sglang.1.0
CTR=dbg2
export DOCKER_CONFIG=/tmp/dockercfg

spur exec "$JOB" bash -c "export DOCKER_CONFIG=/tmp/dockercfg
docker rm -f $CTR 2>/dev/null || true
docker run -d --name $CTR --network=host --ipc=host --shm-size=32G \
  --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
  --group-add video --group-add render --cap-add=SYS_PTRACE --cap-add=IPC_LOCK \
  --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
  -v /shared_nfs:/shared_nfs -v /home/yihou:/home/yihou \
  --entrypoint '' $IMG sleep infinity
docker exec $CTR python3 -c 'import torch;print(\"GPUGATE\", torch.cuda.is_available(), torch.cuda.device_count())'
" 2>&1 | grep -v libtinfow
