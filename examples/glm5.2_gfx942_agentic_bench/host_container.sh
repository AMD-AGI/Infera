#!/usr/bin/env bash
# Create/remove the long-lived engine container. Run on the host shell of both
# nodes; service scripts are then run with docker exec inside this container.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

case "${1:-}" in
  --status)
    docker ps -a --filter "name=^/${CONTAINER}$" \
      --format 'name={{.Names}} state={{.State}} image={{.Image}} up={{.Status}}'
    exit 0 ;;
  --rm)
    docker rm -f "$CONTAINER" >/dev/null 2>&1 && echo "[container] removed $CONTAINER" \
      || echo "[container] $CONTAINER not present"
    exit 0 ;;
  "") ;;
  *) echo "usage: bash $0 [--status|--rm]" >&2; exit 2 ;;
esac

if [[ -n "$(docker ps -q --filter "name=^/${CONTAINER}$")" ]]; then
  echo "[container] $CONTAINER already running"
  exit 0
fi

docker image inspect "$IMAGE" >/dev/null \
  || { echo "[container] image missing: $IMAGE; run ./build_image.sh or docker pull it" >&2; exit 1; }
[[ -d "$MODEL" ]] || { echo "[container] model not found: $MODEL" >&2; exit 1; }
# Pre-create it here so the bind mount does not land as a root-owned directory.
mkdir -p "$DATA_DIR" \
  || { echo "[container] cannot create DATA_DIR: $DATA_DIR; set it for your cluster" >&2; exit 1; }

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

IONIC_ARGS=()
if [[ -e "${HOST_LIBIONIC:-/usr/lib/x86_64-linux-gnu/libionic.so}" ]]; then
  IONIC_ARGS=(-v "${HOST_LIBIONIC:-/usr/lib/x86_64-linux-gnu/libionic.so}:/host-libionic/libionic.so:ro")
fi

docker run -d --name "$CONTAINER" \
  --network host --ipc host --shm-size 128g \
  --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
  --group-add video --group-add render \
  --cap-add=IPC_LOCK --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576 \
  "${IONIC_ARGS[@]}" \
  -v "$REPO:$REPO" \
  -v "$MODEL:$MODEL:ro" \
  -v "$DATA_DIR:$DATA_DIR" \
  -w "$HERE" \
  "$IMAGE" sleep infinity >/dev/null

sleep 2
docker exec "$CONTAINER" bash -lc \
  'python3 -c "import infera, sglang, torch; print(\"infera ok\", \"sglang\", sglang.__version__, \"torch\", torch.__version__)"' \
  || { echo "[container] import check failed"; docker logs "$CONTAINER" | tail -40; exit 1; }

echo "[container] $CONTAINER up on $(hostname); enter with: docker exec -it $CONTAINER bash"
