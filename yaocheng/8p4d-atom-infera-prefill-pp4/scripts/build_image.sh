#!/usr/bin/env bash
# Build the Infera ATOM image of ../8p4d-atom-infera (its docker/Dockerfile and
# patch/, unchanged) on the control and prefill nodes in parallel and print
# both image IDs; the IDs differ per node, the content does not.
# Usage: scripts/build_image.sh [KEY=VALUE ...]
source "$(dirname "$0")/common.sh" "$@"

build() {
    on "$1" docker pull -q "$IMAGE_BASE"
    on "$1" docker build -q --network host -f "$REPO_DIR/yaocheng/8p4d-atom-infera/docker/Dockerfile" \
        --build-arg "ATOM_BASE_IMAGE=$IMAGE_BASE" -t "$IMAGE" "$REPO_DIR"
}
for node in "$CONTROL_NODE" "$PREFILL_NODE"; do build "$node" & done
wait
for node in "$CONTROL_NODE" "$PREFILL_NODE"; do
    echo "$node $(on "$node" docker image inspect -f '{{.Id}}' "$IMAGE")"
done
