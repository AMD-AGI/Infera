#!/usr/bin/env bash
set -Eeuo pipefail
root=/perf_apps/liyingli/bench_agentx/router-r234-31699-20260924
docker run --rm --name llying-adaptive-31699-build --cpus 8 --network host --entrypoint /bin/bash -v "$root:$root" infera-sglang:aus-0922-reqtrace -lc "export CARGO_HOME=$root/build/cargo RUSTUP_HOME=$root/build/rustup CARGO_TARGET_DIR=$root/build/target CARGO_BUILD_JOBS=8 LIBCLANG_PATH=/opt/rocm/llvm/lib; export PATH=\$CARGO_HOME/bin:\$PATH; cargo build --manifest-path $root/build/source/rust/Cargo.toml -p infera-router --release; cp $root/build/target/release/infera-router $root/artifacts/infera-router-r234; chmod 755 $root/artifacts/infera-router-r234; sha256sum $root/artifacts/infera-router-r234"
