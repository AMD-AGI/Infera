#!/usr/bin/env bash
# Rebuild from pinned Infera source, then apply this kit's overlay.
# --check validates the source pin and required files without using SSH/Docker.
set -euo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(git -C "$KIT" rev-parse --show-toplevel)"
BUILDER_NODE="${BUILDER_NODE:-smci355-ccs-aus-n01-33}"
TARGET_NODE="${TARGET_NODE:-smci355-ccs-aus-n02-21}"
EXPECTED_INFERA_REV="${EXPECTED_INFERA_REV:-83e0f6c86cce34718f369d8669b750fa812c62d0}"
SGLANG_BASE_IMAGE="${SGLANG_BASE_IMAGE:-lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260916@sha256:eef7b70e015c94435d68dd92612229d52324b9e8d5edaac741802e7e78e66eb6}"
BASE_IMAGE="${BASE_IMAGE:-infera-sglang:v0519-llying-aus-base-83e0f6c8}"
IMAGE="${IMAGE:-infera-sglang:v0519-llying-aus-0922-nextnfix-hicache}"
SSH_OPTS="${SSH_OPTS:--F /dev/null -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/tmp/agentx_known_hosts}"
read -r -a ssh_args <<< "$SSH_OPTS"
revision="$(git -C "$REPO" rev-parse --verify "$EXPECTED_INFERA_REV^{commit}")"
source_paths=(.dockerignore pyproject.toml README.md infera rust deploy/docker)
for path in "${source_paths[@]}"; do
    git -C "$REPO" cat-file -e "$revision:$path"
done
for path in build/Dockerfile.yihou-overlay build/apply_nextn_patch.py scripts/verify_image.py patches/pr37152.sources.yihou.diff patches/nextn-fusion-fork-4350d37c5.patch; do
    [[ -r "$KIT/$path" ]] || { echo "missing $KIT/$path" >&2; exit 1; }
done
if [[ "${1:-}" == --check ]]; then
    echo "source pin and overlay files verified: $revision"
    exit 0
fi
(( $# == 0 )) || { echo "usage: $0 [--check]" >&2; exit 2; }
ssh_run() {
    local node="$1" command="" argument quoted
    shift
    for argument in "$@"; do
        printf -v quoted '%q' "$argument"
        command+="${command:+ }$quoted"
    done
    ssh "${ssh_args[@]}" "$node" "$command"
}
context="$(ssh_run "$BUILDER_NODE" mktemp -d /tmp/glm52-image-build.XXXXXXXX)"
[[ "$context" =~ ^/tmp/glm52-image-build\.[A-Za-z0-9]+$ ]] || {
    echo "unexpected build context: $context" >&2; exit 1;
}
echo "builder=$BUILDER_NODE source=$revision context=$context"
# Export only committed build inputs, independent of the kit's current HEAD.
git -C "$REPO" archive "$revision" "${source_paths[@]}" |
    ssh_run "$BUILDER_NODE" tar -xf - -C "$context"
ssh_run "$BUILDER_NODE" docker build --network host \
    --build-arg "SGLANG_BASE_IMAGE=$SGLANG_BASE_IMAGE" \
    --tag "$BASE_IMAGE" --file "$context/deploy/docker/Dockerfile.sglang" "$context"
# The kit checkout must be readable at this same path on the builder (shared NFS).
ssh_run "$BUILDER_NODE" docker build --network host \
    --build-arg "BASE_IMAGE=$BASE_IMAGE" --tag "$IMAGE" \
    --file "$KIT/build/Dockerfile.yihou-overlay" "$KIT"
ssh_run "$BUILDER_NODE" docker save "$IMAGE" |
    ssh_run "$TARGET_NODE" docker load
expected=""
for node in "$BUILDER_NODE" "$TARGET_NODE"; do
    image_id="$(ssh_run "$node" docker image inspect --format '{{.Id}}' "$IMAGE")"
    echo "$node image_id=$image_id"
    [[ -z "$expected" || "$image_id" == "$expected" ]] || {
        echo "image ID mismatch: $expected != $image_id" >&2; exit 1;
    }
    expected="${expected:-$image_id}"
    ssh_run "$node" docker run --rm -i --entrypoint python3 "$IMAGE" - \
        < "$KIT/scripts/verify_image.py"
done
echo "verified image: $IMAGE ($expected)"
echo "pinned source context retained for inspection: $BUILDER_NODE:$context"
