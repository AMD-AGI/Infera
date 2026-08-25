#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Create or remove the long-lived engine container. Run on the HOST shell of both
# nodes; the launch, verify and bench scripts then run inside it.
#   bash host_container.sh            # create
#   bash host_container.sh --status
#   bash host_container.sh --rm
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

# A HuggingFace cache is a tree of symlinks -- every file in a snapshot dir is a
# relative link into a sibling blobs/, and MODEL is often itself a link into that
# snapshot. `-v $MODEL:$MODEL` makes docker resolve the outer link and mount the
# snapshot at MODEL's path, which leaves every inner link dangling one directory
# short of blobs/. The symptom is not a missing file but transformers refusing the
# model with "Should have a `model_type` key in its config.json".
# So when MODEL is a link, mount what the links actually point at, each at its own
# path, and let MODEL resolve inside the container the same way it does outside.
MODEL_ARGS=(-v "$MODEL:$MODEL:ro")
MODEL_REAL="$(readlink -f "$MODEL")"
if [[ "$MODEL_REAL" != "$MODEL" ]]; then
  HF_REPO="$MODEL_REAL"
  [[ "$HF_REPO" == */snapshots/* ]] && HF_REPO="${HF_REPO%/snapshots/*}"   # keep blobs/
  MODEL_ARGS=(-v "$(dirname "$MODEL"):$(dirname "$MODEL"):ro" -v "$HF_REPO:$HF_REPO:ro")
  echo "[container] $MODEL -> $MODEL_REAL; mounting $HF_REPO so its blobs resolve"
fi

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

IONIC_ARGS=()
if [[ -e "${HOST_LIBIONIC:-/usr/lib/x86_64-linux-gnu/libionic.so}" ]]; then
  IONIC_ARGS=(-v "${HOST_LIBIONIC:-/usr/lib/x86_64-linux-gnu/libionic.so}:/host-libionic/libionic.so:ro")
fi

# Nothing when KVD=0, which is the default. When on, kvd's L3 must land on the
# host's NVMe rather than the container's overlay, so bind-mount it. Harmless on
# the decode node, which runs no daemon; mounting it there anyway keeps one
# container command for both nodes. The socket stays inside the container --
# only the daemon and the prefill engine, both in here, ever open it.
KVD_ARGS=()
if [[ "$KVD" == "1" ]]; then
  mkdir -p "$KVD_L3_DIR" \
    || { echo "[container] cannot create KVD_L3_DIR: $KVD_L3_DIR; set it for your cluster" >&2; exit 1; }
  KVD_ARGS=(-v "$KVD_L3_DIR:$KVD_L3_DIR")
fi

# EXTRA_DOCKER_ARGS is for cluster-specific additions this script cannot know
# about, e.g. EXTRA_DOCKER_ARGS=--privileged where the fabric's RDMA registration
# path needs more than IPC_LOCK.
read -r -a EXTRA_ARGS_ARR <<< "${EXTRA_DOCKER_ARGS:-}"

docker run -d --name "$CONTAINER" \
  --network host --ipc host --shm-size 128g \
  "${EXTRA_ARGS_ARR[@]}" \
  --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
  --group-add video --group-add render \
  --cap-add=IPC_LOCK --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576 \
  "${IONIC_ARGS[@]}" "${KVD_ARGS[@]}" \
  -v "$REPO:$REPO" \
  "${MODEL_ARGS[@]}" \
  -w "$HERE" \
  "$IMAGE" sleep infinity >/dev/null

sleep 2
docker exec "$CONTAINER" bash -lc \
  'python3 -c "import infera, sglang, torch; print(\"infera ok\", \"sglang\", sglang.__version__, \"torch\", torch.__version__)"' \
  || { echo "[container] import check failed"; docker logs "$CONTAINER" | tail -40; exit 1; }

# Read the config the way the engine will. A mount that leaves the weights'
# symlinks dangling otherwise surfaces minutes into engine startup as a
# transformers error about model_type, far from its cause.
docker exec -e M="$MODEL" "$CONTAINER" bash -lc \
  'python3 -c "import json, os; print(\"model ok\", json.load(open(os.environ[\"M\"] + \"/config.json\"))[\"model_type\"])"' \
  || { echo "[container] cannot read $MODEL/config.json inside the container -- check the mounts above" >&2; exit 1; }

echo "[container] $CONTAINER up on $(hostname); enter with: docker exec -it $CONTAINER bash"
