#!/usr/bin/env bash
# Build infera-router from SRC (fdbafb60's rust/ + 565461b3 --decode-request-weight, ../router-decode-request-weight.diff)
# inside IMAGE on THIS node (same glibc as the router container), with the router's unit tests.
# Output: SRC/rust/target/release/infera-router. Cargo is installed through rustup in the throw-away container
# (the image's Dockerfile removes it after building the bundled router).
# Usage: build_router.sh [SRC] [IMAGE]
set -euo pipefail
src="${1:-/perf_apps/liyingli/bench_agentx/opt-aus-20261103/router-src}"
image="${2:-infera-sglang:v0522-fdbafb60}"
docker run --rm --network host -v "$src:/src" -w /src --entrypoint /bin/bash "$image" -c '
    set -euo pipefail
    export CARGO_TARGET_DIR=/src/rust/target CARGO_BUILD_JOBS=32 PATH="$HOME/.cargo/bin:$PATH"
    command -v cc >/dev/null || { apt-get update -qq && apt-get install -y -qq build-essential >/dev/null; }
    export LIBCLANG_PATH="$(dirname "$(find /opt/rocm* /usr/lib /usr/lib64 -name "libclang.so*" 2>/dev/null | head -1)")"
    command -v cargo >/dev/null || curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null
    cargo --version
    cd rust
    cargo test --release -p infera-router 2>&1 | grep -E "^test result|FAILED|panicked" | tail -n 20
    cargo build --release --bin infera-router
    ls -la target/release/infera-router
    target/release/infera-router --help | grep -A2 decode-request-weight
    chown -R '"$(id -u):$(id -g)"' /src/rust/target
'
