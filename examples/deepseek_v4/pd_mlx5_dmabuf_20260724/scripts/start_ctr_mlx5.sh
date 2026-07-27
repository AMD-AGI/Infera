#!/bin/bash
# Start the PD container on the current node from the dmabuf-enabled image.
# Forces mlx5 path; still exposes /dev/infiniband so mlx5 uverbs is visible.
set -x
IMG="${IMG:-dsv4-sgl-dmabuf:mlx5}"
NAME="${1:-pd_mlx5}"
docker rm -f "$NAME" 2>/dev/null || true
# bind host libionic (harmless; entrypoint no-ops if unused since we force mlx5)
HOST_IONIC=$(readlink -f /usr/lib/x86_64-linux-gnu/libionic.so.1 2>/dev/null || true)
IONIC_MOUNT=""
[ -n "$HOST_IONIC" ] && IONIC_MOUNT="-v $HOST_IONIC:/host-libionic/libionic.so:ro"
docker run -d --name "$NAME" \
  --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
  --group-add video --group-add render \
  --cap-add=SYS_PTRACE --cap-add=IPC_LOCK --security-opt seccomp=unconfined \
  --ipc=host --shm-size=32G --network=host \
  -v /shared_nfs:/shared_nfs -v /home/yihou:/home/yihou \
  $IONIC_MOUNT \
  --entrypoint "" "$IMG" sleep infinity
echo "$NAME: $(docker ps --filter name=$NAME --format '{{.Status}}')"
