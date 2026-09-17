#!/usr/bin/env bash
# Purpose: Build, distribute, and verify the configured Infera engine image.
# Usage: ./build_image.sh [build|distribute|verify|all] [KEY=VALUE ...]
# Artifacts: Docker images in node-local image stores; no result files.
# Artifact paths: image names come from SGLANG_BASE_IMAGE and IMAGE.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$DIR/../.." && pwd)"

mode=all
if [[ "${1:-}" =~ ^(build|distribute|verify|all)$ ]]; then
    mode="$1"
    shift
fi
for assignment in "$@"; do
    [[ "$assignment" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] ||
        { echo "expected KEY=VALUE, got '$assignment'" >&2; exit 2; }
    export "$assignment"
done
CONFIG="${CONFIG:-$DIR/config.sh}"
[[ -r "$CONFIG" ]] || { echo "config is not readable: $CONFIG" >&2; exit 1; }
set -a
source "$CONFIG"
set +a
TOPOLOGY="${TOPOLOGY:-$DIR/topology.tsv}"
BUILD_CONTEXT="${BUILD_CONTEXT:-$REPO}"
BUILD_DOCKERFILE="${BUILD_DOCKERFILE:-$BUILD_CONTEXT/deploy/docker/Dockerfile.sglang}"
: "${IMAGE:?set IMAGE}"
: "${SGLANG_BASE_IMAGE:?set SGLANG_BASE_IMAGE}"
: "${BUILDER_NODE:?set BUILDER_NODE}"
[[ -r "$TOPOLOGY" ]] || { echo "topology is not readable: $TOPOLOGY" >&2; exit 1; }
[[ -d "$BUILD_CONTEXT" ]] || { echo "build context is not a directory: $BUILD_CONTEXT" >&2; exit 1; }
[[ -r "$BUILD_DOCKERFILE" ]] || { echo "Dockerfile is not readable: $BUILD_DOCKERFILE" >&2; exit 1; }

read -r -a ssh_args <<< "${SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=10}"
remote_command() {
    local result="" item quoted
    for item in "$@"; do
        printf -v quoted '%q' "$item"
        result+="${result:+ }$quoted"
    done
    echo "$result"
}
ssh_run() {
    local node="$1"
    shift
    ssh "${ssh_args[@]}" "$node" "$(remote_command "$@")"
}
node_lines="$(python3 "$DIR/tools/topology.py" nodes "$TOPOLOGY")"
mapfile -t nodes <<<"$node_lines"

build() {
    local -a command extra
    command=(
        docker build --network "${BUILD_NETWORK:-host}"
        --build-arg "SGLANG_BASE_IMAGE=$SGLANG_BASE_IMAGE"
        --tag "$IMAGE" --file "$BUILD_DOCKERFILE"
    )
    if [[ -n "${BUILD_ARGS:-}" ]]; then
        read -r -a extra <<< "$BUILD_ARGS"
        for assignment in "${extra[@]}"; do
            [[ "$assignment" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] ||
                { echo "invalid BUILD_ARGS item: $assignment" >&2; exit 2; }
            command+=(--build-arg "$assignment")
        done
    fi
    [[ "${BUILD_NO_CACHE:-0}" == 1 ]] && command+=(--no-cache)
    command+=("$BUILD_CONTEXT")
    echo "building $IMAGE on $BUILDER_NODE from $BUILD_DOCKERFILE"
    ssh_run "$BUILDER_NODE" "${command[@]}"
}

distribute() {
    ssh_run "$BUILDER_NODE" docker image inspect "$IMAGE" >/dev/null
    for node in "${nodes[@]}"; do
        [[ "$node" == "$BUILDER_NODE" ]] && continue
        echo "copying $IMAGE from $BUILDER_NODE to $node"
        ssh_run "$BUILDER_NODE" docker save "$IMAGE" | ssh_run "$node" docker load
    done
}

verify() {
    local node image_id expected=""
    for node in "${nodes[@]}"; do
        image_id="$(ssh_run "$node" docker image inspect --format '{{.Id}}' "$IMAGE")"
        echo "$node image_id=$image_id"
        [[ -z "$expected" || "$image_id" == "$expected" ]] ||
            { echo "image IDs differ across nodes" >&2; exit 1; }
        expected="${expected:-$image_id}"
        ssh_run "$node" docker run --rm --entrypoint /bin/bash "$IMAGE" -lc \
            'python3 -c "import infera, sglang; print(infera.__file__); print(sglang.__file__)" && /usr/local/bin/infera-router --help >/dev/null'
    done
    echo "all topology nodes use $expected"
}

case "$mode" in
    build) build ;;
    distribute) distribute ;;
    verify) verify ;;
    all) build; distribute; verify ;;
esac
