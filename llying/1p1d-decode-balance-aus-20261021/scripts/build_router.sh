#!/usr/bin/env bash
# Build and test infera-router from a source tree in the PR 210 base image on THIS node, the way
# deploy/docker/Dockerfile.sglang does (cargo from the base or rustup, LIBCLANG_PATH from ROCm), then run the
# launcher's Python unit tests. Output: <src>/rust/target/release/infera-router (glibc of the image's base).
# Usage: build_router.sh <source tree> [base image]
set -euo pipefail
src="${1:?source tree}"
base="${2:-lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261018}"
export DOCKER_CONFIG=/tmp/liyingli-docker-anon
mkdir -p "$DOCKER_CONFIG" && [[ -s "$DOCKER_CONFIG/config.json" ]] || echo '{}' > "$DOCKER_CONFIG/config.json"
docker run --rm --network host -v "$src:/src" -w /src --entrypoint /bin/bash "$base" -c '
    set -euo pipefail
    export CARGO_TARGET_DIR=/src/rust/target PATH="$HOME/.cargo/bin:$PATH"
    command -v cc >/dev/null || { apt-get update -qq && apt-get install -y -qq build-essential >/dev/null; }
    export LIBCLANG_PATH="$(dirname "$(find /opt/rocm* /usr/lib /usr/lib64 -name "libclang.so*" 2>/dev/null | head -1)")"
    command -v cargo >/dev/null || curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null
    cargo --version 2>/dev/null || rustup default stable >/dev/null
    cargo --version
    cd rust
    cargo test --release -p infera-router 2>&1 | tail -n 25
    cargo build --release --bin infera-router
    ls -la target/release/infera-router
    cd /src
    python3 -m pip install -q pytest >/dev/null 2>&1 || true
    python3 -m pytest -q tests/unit/server/test_launch_rust.py 2>&1 | tail -n 5
    chown -R '"$(id -u):$(id -g)"' /src/rust/target
'
