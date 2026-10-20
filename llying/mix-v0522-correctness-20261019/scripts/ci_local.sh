#!/usr/bin/env bash
# Runs the CI jobs of .github/workflows/ci.yml that touch the sglang image, on one GPU node,
# against a branch of the local Infera repository. Logs and pytest output go to OUT.
#   lint-unit   CI "lint" + "unit": python:3.10, pre-commit (cargo-fmt / gofmt skipped: no Rust or
#               Go in the branch, no toolchains in the container), pip install -e .[dev,gaie,projection],
#               pytest -m "not slow and not integration"
#   unit-image  tests/unit inside IMAGE, the way tests/lib/tiers.sh run_unit does it
#   engine      CI "engine", sglang half: build deploy/docker/Dockerfile.sglang as ENGINE_TAG,
#               then tests/engine/sglang per file (tests/lib/container_pytest.sh --per-file)
# Usage: ci_local.sh STEP REPO_DIR BRANCH OUT [IMAGE]
set -uo pipefail
step="$1" repo="$2" branch="$3" out="$4" image="${5:-}"
ENGINE_TAG="${ENGINE_TAG:-infera/engine-sglang:llying-ci-v0522}"
GPU_FLAGS=(--init --privileged --ipc host --shm-size 16gb --ulimit memlock=-1
    --device /dev/kfd --device /dev/dri --group-add video --group-add render -v /boot:/boot:ro)
mkdir -p "$out"

# A clean checkout of BRANCH from the local repository, so the work tree is never written to.
checkout() {
    rm -rf "$out/src" && git clone -q --branch "$branch" "$repo" "$out/src"
}

case "$step" in
lint-unit)
    checkout
    docker run --rm --network host -e CI=1 -e SKIP=cargo-fmt,gofmt -v "$out/src":/repo -w /repo \
        python:3.10 bash -c '
            git config --global --add safe.directory /repo
            pip install -q pre-commit && pre-commit run --all-files --show-diff-on-failure
            echo "LINT_RC=$?"
            pip install -q -e ".[dev,gaie,projection]" && pytest -m "not slow and not integration" -q -p no:cacheprovider
            echo "UNIT_RC=$?"'
    ;;
unit-image)
    checkout
    docker run --rm "${GPU_FLAGS[@]}" -v "$out/src":/workspace:ro -v "$out":/scratch -e HF_HOME=/scratch/hf \
        -e PYTHONDONTWRITEBYTECODE=1 -w /workspace --entrypoint bash "$image" \
        -lc 'exec bash tests/lib/container_pytest.sh "[unit]" -q tests/unit'
    echo "UNIT_IMAGE_RC=$?"
    ;;
engine)
    checkout
    docker build --network=host -f "$out/src/deploy/docker/Dockerfile.sglang" -t "$ENGINE_TAG" "$out/src" ||
        { echo "ENGINE_BUILD_FAILED"; exit 1; }
    docker run --rm "${GPU_FLAGS[@]}" -v "$out/src":/workspace:ro -v "$out":/scratch -e HF_HOME=/scratch/hf \
        -e PYTHONDONTWRITEBYTECODE=1 -w /workspace --entrypoint bash "$ENGINE_TAG" \
        -lc 'exec bash tests/lib/container_pytest.sh --per-file "[engine tests/engine/sglang]" tests/engine/sglang'
    echo "ENGINE_RC=$?"
    ;;
*) echo "unknown step: $step" >&2; exit 2 ;;
esac
