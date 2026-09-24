#!/usr/bin/env bash
set -Eeuo pipefail
root=/perf_apps/liyingli/bench_agentx/r2-4k-31705-31706-20260924
tools=/perf_apps/liyingli/bench_agentx/router-r234-31699-20260924/build
docker run --rm --name llying-r2-31705-31706-build --cpus 8 --network host --entrypoint /bin/bash -v "$root:$root" -v "$tools:$tools" infera-sglang:aus-0922-reqtrace -lc "export CARGO_HOME=$tools/cargo RUSTUP_HOME=$tools/rustup CARGO_TARGET_DIR=$root/build/target CARGO_BUILD_JOBS=8 LIBCLANG_PATH=/opt/rocm/llvm/lib; export PATH=\$CARGO_HOME/bin:\$PATH; cargo build --locked --manifest-path $root/build/source/rust/Cargo.toml -p infera-router --release; cp $root/build/target/release/infera-router $root/artifacts/infera-router-r2; chmod 755 $root/artifacts/infera-router-r2; sha256sum $root/artifacts/infera-router-r2"
