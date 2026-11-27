#!/usr/bin/env bash
# CPU-only (runs on a compute node, in a rust container): (1) fmt / clippy / tests of the router B1 change on Infera main
# (branch llying/router-decode-dp-rank, the upstream PR candidate); (2) release build of infera-router from the recipe
# branch (0fcce33 + the same commit, llying/router-dp-on-vllm-glm52) for stage F: router-dp/bin/infera-router.
set -uo pipefail
. /perf_apps/liyingli/bench_agentx/vllm-p4d4-aus-20261127/site/env.sh
pull_image docker.io/library/rust:1-bookworm > /dev/null || { echo "rust image pull failed"; exit 1; }
mkdir -p "$W/router-dp/bin" "$NODE_CACHE" && chmod 777 "$NODE_CACHE"
run() {   # src_dir target_name script: as root (apt for protoc), sources read-only, target dir on the node (root
    # cannot write the root-squashed NFS), the binary copied out by this user afterwards
    docker run --rm --cpus 16 -e CARGO_HOME=/cache/cargo -e CARGO_TARGET_DIR="/cache/target-$2" -v "$NODE_CACHE:/cache" \
        -v "$1:/work:ro" -w /work/rust docker.io/library/rust:1-bookworm bash -c "$3"
}
# The router build needs protoc / clang for some crates on this branch; install into the throwaway container if missing.
deps='command -v protoc > /dev/null || (apt-get update -qq && apt-get install -y -qq protobuf-compiler libclang-dev > /dev/null) 2>/dev/null; rustup component add rustfmt clippy > /dev/null 2>&1;'
echo "$(ts) checks on Infera main + B1"
run /perf_apps/liyingli/bench_agentx/Infera-router-dp main "$deps cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test 2>&1 | tail -40" \
    > "$W/router-dp/checks.log" 2>&1
echo "$(ts) checks rc=$? (router-dp/checks.log)"
echo "$(ts) release build on the recipe branch"
run /perf_apps/liyingli/bench_agentx/Infera-router-dp-recipe recipe "$deps cargo build --release --bin infera-router && chmod -R a+rX /cache/target-recipe" \
    > "$W/router-dp/build.log" 2>&1
rc=$?
cp "$NODE_CACHE/target-recipe/release/infera-router" "$W/router-dp/bin/infera-router" 2>/dev/null || rc=1
echo "$(ts) build rc=$rc: $(ls -la "$W/router-dp/bin/infera-router" 2>&1)"
