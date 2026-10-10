#!/usr/bin/env bash
# Build the Infera ATOM image of ../8p4d-atom-infera (its docker/Dockerfile and
# patch/, unchanged) on the control and prefill nodes in parallel and print
# both image IDs. Compare package versions as dependencies resolve at build time.
# Usage: scripts/build_image.sh [KEY=VALUE ...]
source "$(dirname "$0")/common.sh" "$@"

build() {
    on "$1" docker pull -q "$IMAGE_BASE"
    # Send only build inputs; experiment logs and caches can be very large.
    tar -C "$REPO_DIR" -cf - pyproject.toml README.md infera \
        deploy/docker/scripts/infera_inject_host_ionic.sh \
        yaocheng/8p4d-atom-infera/docker yaocheng/8p4d-atom-infera/patch |
        on "$1" docker build --network host -f yaocheng/8p4d-atom-infera/docker/Dockerfile \
            --build-arg "ATOM_BASE_IMAGE=$IMAGE_BASE" -t "$IMAGE" -
}
pids=()
for node in "$CONTROL_NODE" "$PREFILL_NODE"; do build "$node" & pids+=("$!"); done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
(( status == 0 )) || exit "$status"
for node in "$CONTROL_NODE" "$PREFILL_NODE"; do
    echo "$node $(on "$node" docker image inspect -f '{{.Id}}' "$IMAGE")"
done
