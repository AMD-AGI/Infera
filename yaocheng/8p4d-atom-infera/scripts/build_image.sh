#!/usr/bin/env bash
# Build the Infera ATOM image on the control node, copy it to the prefill node
# and print both image IDs. Usage: scripts/build_image.sh [KEY=VALUE ...]
source "$(dirname "$0")/common.sh" "$@"

docker build --network host -f "$KIT_DIR/docker/Dockerfile" \
    --build-arg "ATOM_BASE_IMAGE=$IMAGE_BASE" -t "$IMAGE" "$REPO_DIR"
docker save "$IMAGE" | on "$PREFILL_NODE" docker load
for node in "$CONTROL_NODE" "$PREFILL_NODE"; do
    echo "$node $(on "$node" docker image inspect -f '{{.Id}}' "$IMAGE")"
done
