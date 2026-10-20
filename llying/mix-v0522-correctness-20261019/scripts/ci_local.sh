#!/usr/bin/env bash
# Runs the CI jobs of .github/workflows/ci.yml that touch the sglang image, on one GPU node,
# against a branch of the local Infera repository. Logs and pytest output go to OUT.
#   lint-unit   CI "lint" + "unit": python:3.10, pre-commit (cargo-fmt / gofmt skipped: no Rust or
#               Go in the branch, no toolchains in the container), pip install -e .[dev,gaie,projection],
#               pytest -m "not slow and not integration"
#   unit-image  tests/unit inside IMAGE, the way tests/lib/tiers.sh run_unit does it
#   engine      CI "engine", sglang half: build deploy/docker/Dockerfile.sglang as ENGINE_TAG,
#               then tests/engine/sglang per file (tests/lib/container_pytest.sh --per-file)
#   CI "e2e-disag" (sglang) without SLURM, over INFERA_E2E_TRANSPORT=ssh (feat/e2e-disag-ssh-transport):
#   disag-prep  checkout + cherry-pick PICK, orchestrator venv (pytest, pytest-asyncio, httpx) in OUT,
#               and the base image copied to PEER when it lacks it (docker save | ssh docker load)
#   disag-build build the e2e image on THIS node exactly as the harness does (run on both nodes;
#               the harness's own build is then a cache hit)
#   disag       tests/run_tests.sh e2e sglang disag on NODES (prefill first), fabric settings of ci.yml
#   For the disag steps OUT must be at the same path on both nodes (/home, not /mnt/m2m_nobackup):
#   the harness bind-mounts the checkout into every container and builds from it on each node.
# Usage: ci_local.sh STEP REPO_DIR BRANCH OUT [IMAGE]
set -uo pipefail
step="$1" repo="$2" branch="$3" out="$4" image="${5:-}"
ENGINE_TAG="${ENGINE_TAG:-infera/engine-sglang:llying-ci-v0522}"
PICK="${PICK:-a99fd8eb}"
PEER="${PEER:-crsuse2-m2m-140}"
NODES="${NODES:-crsuse2-m2m-138,crsuse2-m2m-140}"
NODE_IPS="${NODE_IPS:-crsuse2-m2m-138=10.245.157.237,crsuse2-m2m-140=10.245.159.30}"
BASE_IMAGE=lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261018
# tests/e2e/harness/images.py tag + launcher.py's build-arg suffix for INFERA_E2E_BUILD_ARGS below.
DISAG_BUILD_ARGS=MOONCAKE_HIP_DMABUF=1
DISAG_TAG=infera/engine-sglang:test-local-mooncake_hip_dmabuf1
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
disag-prep)
    checkout
    git -C "$out/src" -c user.name=ci-local -c user.email=ci-local@localhost cherry-pick "$PICK" ||
        { echo "CHERRY_PICK_FAILED"; exit 1; }
    [ -x "$out/venv/bin/python3" ] || python3 -m venv "$out/venv"
    "$out/venv/bin/pip" install -q pytest pytest-asyncio httpx || { echo "VENV_FAILED"; exit 1; }
    if ssh -n -o BatchMode=yes "$PEER" docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
        echo "$PEER already has $BASE_IMAGE"
    else
        docker save "$BASE_IMAGE" | ssh -o BatchMode=yes "$PEER" docker load || { echo "BASE_COPY_FAILED"; exit 1; }
    fi
    echo "DISAG_PREP_RC=0"
    ;;
disag-build)
    docker build --network=host --build-arg "$DISAG_BUILD_ARGS" \
        -f "$out/src/deploy/docker/Dockerfile.sglang" -t "$DISAG_TAG" "$out/src"
    echo "DISAG_BUILD_RC=$? ($(hostname -s))"
    ;;
disag)
    cd "$out/src" || exit 1
    mkdir -p "$out/logs"
    env PATH="$out/venv/bin:$PATH" \
        INFERA_E2E_TRANSPORT=ssh INFERA_E2E_NODES="$NODES" INFERA_E2E_NODE_IPS="$NODE_IPS" \
        INFERA_E2E_GFX_ARCH=gfx950 INFERA_E2E_MODEL_DIR=/shared_nfs/huggingface_models \
        INFERA_E2E_GID_INDEX=3 INFERA_E2E_WORKER_ENV=MC_TE_FILTERS=mlx5_0,MOONCAKE_DISABLE_HIP_DMABUF=0 \
        INFERA_E2E_BUILD_ARGS="$DISAG_BUILD_ARGS" INFERA_E2E_JOB_TAG=llying-v0522-disag \
        INFERA_DISPATCH_LOGDIR="$out/logs" \
        bash tests/run_tests.sh e2e sglang disag
    echo "DISAG_RC=$?"
    ;;
*) echo "unknown step: $step" >&2; exit 2 ;;
esac
