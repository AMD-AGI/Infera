#!/usr/bin/env bash
# Build the pinned GLM-5.2 v0.5.18 image once and copy that exact image to decode.
# Usage: bash build_v518_image.sh [build|distribute|verify|all]
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/config.sh"

MODE="${1:-all}"
OUT="${BUILD_OUT_DIR:-$RUN_ROOT/build}"
mkdir -p "$OUT"

run() {
    local node="$1"
    shift
    ssh $SSH_OPTS "$node" "$@"
}

build() {
    echo "[build] optimized ROCm base on $PREFILL_NODE: $ROCM_OPT_BASE_IMAGE"
    run "$PREFILL_NODE" \
        docker build --network host \
            --tag "$ROCM_OPT_BASE_IMAGE" \
            --file "$ROCM_LLM_BENCH_DIR/Dockerfile" \
            "$ROCM_LLM_BENCH_DIR" \
        2>&1 | tee "$OUT/rocm_opt_base.log"

    echo "[build] Infera PD engine on $PREFILL_NODE: $V518_IMAGE"
    run "$PREFILL_NODE" \
        docker build --network host \
            --build-arg "SGLANG_BASE_IMAGE=$ROCM_OPT_BASE_IMAGE" \
            --build-arg APPLY_SGLANG_PD_ROCM_REJECTION_PATCH=1 \
            --tag "$V518_IMAGE" \
            --file "$DIR/../../deploy/docker/Dockerfile.sglang" \
            "$DIR/../.." \
        2>&1 | tee "$OUT/infera_v518.log"
}

distribute() {
    echo "[build] copying the exact engine image $PREFILL_NODE -> $DECODE_NODE"
    run "$PREFILL_NODE" docker image inspect "$V518_IMAGE" >/dev/null
    run "$PREFILL_NODE" docker save "$V518_IMAGE" \
        | run "$DECODE_NODE" docker load \
        2>&1 | tee "$OUT/distribute.log"
}

verify_node() {
    local node="$1"
    {
        echo "node=$node"
        run "$node" docker image inspect \
            --format='image_id={{.Id}},created={{.Created}},size={{.Size}}' \
            "$V518_IMAGE"
        run "$node" docker run -i --rm \
            --device=/dev/kfd --device=/dev/dri \
            --group-add video --group-add render \
            --entrypoint /bin/bash "$V518_IMAGE" <<'VERIFY_IMAGE'
            set -e
            python3 - <<'PY'
import aiter
import infera
import sglang
print("sglang_module=" + sglang.__file__)
print("aiter_module=" + aiter.__file__)
print("infera_module=" + infera.__file__)
PY
            sglang_root="$(python3 -c "import pathlib, sglang; print(pathlib.Path(sglang.__file__).resolve().parents[2])")"
            git -C "$sglang_root" rev-parse HEAD
            git -C /aiter rev-parse HEAD
            /usr/local/bin/infera-router --help >/dev/null
VERIFY_IMAGE
    } | tee "$OUT/verify_$node.log"
}

verify() {
    verify_node "$PREFILL_NODE"
    verify_node "$DECODE_NODE"
    prefill_id="$(run "$PREFILL_NODE" docker image inspect -f '{{.Id}}' "$V518_IMAGE")"
    decode_id="$(run "$DECODE_NODE" docker image inspect -f '{{.Id}}' "$V518_IMAGE")"
    [[ "$prefill_id" == "$decode_id" ]] || {
        echo "[build] image IDs differ: prefill=$prefill_id decode=$decode_id" >&2
        return 1
    }
    echo "[build] PASS image_id=$prefill_id"
}

case "$MODE" in
    build) build ;;
    distribute) distribute ;;
    verify) verify ;;
    all)
        build
        distribute
        verify
        ;;
    *)
        echo "usage: $0 [build|distribute|verify|all]" >&2
        exit 64
        ;;
esac
