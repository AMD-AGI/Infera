#!/usr/bin/env bash
# Build the pinned ROCm optimization base, layer Infera on it, then distribute.
set -euo pipefail
COMPONENT=build-image
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/lib/common.sh"

mode=all
if [[ "${1:-}" =~ ^(build|distribute|verify|all)$ ]]; then
    mode="$1"
    shift
fi
load_config "$@"
init_ssh
start_log
mapfile -t nodes < <(topology_nodes)
builder="${BUILDER_NODE:-$CONTROL_NODE}"
: "${ROCM_LLM_BENCH_DIR:?set ROCM_LLM_BENCH_DIR in config.sh}"
: "${ROCM_OPT_BASE_IMAGE:?set ROCM_OPT_BASE_IMAGE in config.sh}"

build_images() {
    local -a base_build final_build
    base_build=(
        docker build --network "${BUILD_NETWORK:-host}"
        --tag "$ROCM_OPT_BASE_IMAGE"
        --file "$ROCM_LLM_BENCH_DIR/Dockerfile"
    )
    final_build=(
        docker build --network "${BUILD_NETWORK:-host}"
        --build-arg "SGLANG_BASE_IMAGE=$ROCM_OPT_BASE_IMAGE"
        --build-arg "APPLY_SGLANG_PD_ROCM_REJECTION_PATCH=${APPLY_SGLANG_PD_ROCM_REJECTION_PATCH:-1}"
        --tag "$IMAGE"
        --file "${BUILD_DOCKERFILE:-$BUILD_CONTEXT/deploy/docker/Dockerfile.sglang}"
    )
    if [[ "$(bool01 "${BUILD_NO_CACHE:-0}")" == 1 ]]; then
        base_build+=(--no-cache)
        final_build+=(--no-cache)
    fi
    base_build+=("$ROCM_LLM_BENCH_DIR")
    final_build+=("$BUILD_CONTEXT")

    log "building optimized ROCm base on $builder: $ROCM_OPT_BASE_IMAGE"
    print_command ssh "${SSH_ARGS[@]}" "$builder" "$(_remote_command "${base_build[@]}")"
    ssh_exec "$builder" "${base_build[@]}"

    log "building Infera PD engine on $builder: $IMAGE"
    print_command ssh "${SSH_ARGS[@]}" "$builder" "$(_remote_command "${final_build[@]}")"
    ssh_exec "$builder" "${final_build[@]}"
}

distribute_image() {
    ssh_exec "$builder" docker image inspect "$IMAGE" >/dev/null
    for node in "${nodes[@]}"; do
        [[ "$node" == "$builder" ]] && continue
        log "copying exact engine image $builder -> $node: $IMAGE"
        ssh_exec "$builder" docker save "$IMAGE" | ssh_exec "$node" docker load
    done
}

verify_node() {
    local node="$1"
    log "verifying $IMAGE on $node"
    ssh_exec "$node" docker image inspect \
        --format 'image_id={{.Id}},created={{.Created}},size={{.Size}}' "$IMAGE"
    ssh_exec "$node" docker run -i --rm \
        --device=/dev/kfd --device=/dev/dri \
        --group-add video --group-add render \
        --entrypoint /bin/bash "$IMAGE" <<'VERIFY_IMAGE'
set -e
python3 - <<'PY'
import aiter
import infera
import sglang
print("sglang_module=" + sglang.__file__)
print("aiter_module=" + aiter.__file__)
print("infera_module=" + infera.__file__)
PY
sglang_root="$(python3 -c 'import pathlib, sglang; print(pathlib.Path(sglang.__file__).resolve().parents[2])')"
echo "sglang_commit=$(git -C "$sglang_root" rev-parse HEAD)"
echo "aiter_commit=$(git -C /aiter rev-parse HEAD)"
/usr/local/bin/infera-router --help >/dev/null
VERIFY_IMAGE
}

verify_images() {
    local node image_id expected=""
    for node in "${nodes[@]}"; do
        verify_node "$node"
        image_id="$(ssh_exec "$node" docker image inspect --format '{{.Id}}' "$IMAGE")"
        log "image node=$node ref=$IMAGE id=$image_id"
        if [[ -z "$expected" ]]; then
            expected="$image_id"
        elif [[ "$image_id" != "$expected" ]]; then
            die "image IDs differ: $node has $image_id, expected $expected"
        fi
    done
    log "PASS: every topology node has Image ID $expected"
}

case "$mode" in
    build) build_images ;;
    distribute) distribute_image ;;
    verify) verify_images ;;
    all)
        build_images
        distribute_image
        verify_images
        ;;
    *) die "usage: $0 [build|distribute|verify|all] [VAR=value ...]" ;;
esac
